from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from PIL import Image, ImageOps

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]
OutputMode = Literal["foreground", "mask"]

DEFAULT_MODEL_ID = "briaai/RMBG-2.0"
DEFAULT_IMAGE_SIZE = (1024, 1024)
ACCESS_ERROR_MARKERS = (
    "gated repo",
    "cannot access gated repo",
    "access to model",
    "401 client error",
    "please log in",
)


@dataclass(frozen=True)
class RemovalResult:
    image: Image.Image
    mask: Image.Image
    device: str
    model_id: str


class ModelAccessError(RuntimeError):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_access_error_message(model_id))


class BackgroundRemovalService:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: DevicePreference = "auto",
        hf_token: str | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.model_id = model_id
        self.device_preference = device
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.image_size = image_size
        self._model = None
        self._transform = None
        self._selected_device: str | None = None
        self._lock = Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def selected_device(self) -> str:
        if self._selected_device is None:
            self._selected_device = select_device(self.device_preference)
        return self._selected_device

    def load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            import torch
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation

            device = self.selected_device
            try:
                model = AutoModelForImageSegmentation.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    token=self.hf_token,
                )
            except OSError as exc:
                if is_model_access_error(exc):
                    raise ModelAccessError(self.model_id) from exc
                raise
            if device == "cuda":
                torch.set_float32_matmul_precision("high")
            model = model.eval().to(device)

            self._transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._model = model

    def remove_background(self, image: Image.Image) -> RemovalResult:
        self.load()

        import torch
        from torchvision import transforms

        original = ImageOps.exif_transpose(image).convert("RGB")
        input_tensor = self._transform(original).unsqueeze(0).to(self.selected_device)

        with torch.no_grad():
            preds = self._model(input_tensor)[-1].sigmoid().cpu()

        mask_tensor = preds[0].squeeze()
        mask = transforms.ToPILImage()(mask_tensor).resize(original.size)
        foreground = apply_alpha_mask(original, mask)

        return RemovalResult(
            image=foreground,
            mask=mask,
            device=self.selected_device,
            model_id=self.model_id,
        )

    def remove_file(
        self,
        input_path: Path,
        output_path: Path,
        mask_path: Path | None = None,
    ) -> RemovalResult:
        with Image.open(input_path) as image:
            result = self.remove_background(image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(output_path, format="PNG")

        if mask_path is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            result.mask.save(mask_path, format="PNG")

        return result


def select_device(preference: DevicePreference) -> str:
    import torch

    if preference == "cpu":
        return "cpu"

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda"

    if preference == "mps":
        if not _mps_available(torch):
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return "mps"

    if preference != "auto":
        raise ValueError(f"Unsupported device preference: {preference}")

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def is_model_access_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ACCESS_ERROR_MARKERS)


def model_access_error_message(model_id: str) -> str:
    return (
        f"Cannot access gated Hugging Face model '{model_id}'.\n\n"
        "Fix:\n"
        f"1. Open https://huggingface.co/{model_id} and accept the model access terms.\n"
        "2. Authenticate this project with one of these options:\n"
        "   - export HF_TOKEN=hf_your_token_here\n"
        "   - uv run hf auth login\n"
        "   - pass --hf-token hf_your_token_here to the rmbg command\n\n"
        "RMBG-2.0 is listed for non-commercial use unless you have a BRIA commercial agreement."
    )


def apply_alpha_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    foreground = ImageOps.exif_transpose(image).convert("RGBA")
    alpha = mask.convert("L").resize(foreground.size)
    foreground.putalpha(alpha)
    return foreground


def _mps_available(torch_module) -> bool:
    return bool(
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )
