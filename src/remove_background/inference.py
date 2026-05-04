from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from PIL import Image, ImageFilter, ImageOps

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]
OutputMode = Literal["foreground", "mask"]

DEFAULT_MODEL_ID = "briaai/RMBG-2.0"
DEFAULT_IMAGE_SIZE = (1024, 1024)
MASK_BACKGROUND_CUTOFF = 12
MASK_FOREGROUND_CUTOFF = 224
MASK_HOLE_MIN_AREA = 8
MASK_HOLE_MAX_AREA = 5000
MASK_HOLE_MAX_SPAN = 160
MASK_HOLE_DILATION_SIZE = 3
BACKGROUND_SAMPLE_WIDTH = 4
BACKGROUND_PALETTE_SIZE = 24
BACKGROUND_COLOR_DISTANCE = 18
FOREGROUND_BLACK_LUMA_CUTOFF = 35
FOREGROUND_WHITE_LUMA_CUTOFF = 245
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

        mask_tensor = normalize_prediction(preds[0].squeeze())
        raw_mask = transforms.ToPILImage()(mask_tensor)
        mask = refine_alpha_mask(raw_mask, original.size, original)
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
    alpha = mask.convert("L").resize(foreground.size, Image.Resampling.LANCZOS)
    foreground.putalpha(alpha)
    return foreground


def normalize_prediction(mask_tensor):
    minimum = mask_tensor.min()
    maximum = mask_tensor.max()
    scale = maximum - minimum
    if float(scale) <= 1e-6:
        return mask_tensor
    return (mask_tensor - minimum) / scale


def refine_alpha_mask(
    mask: Image.Image,
    size: tuple[int, int],
    original: Image.Image | None = None,
) -> Image.Image:
    alpha = mask.convert("L").resize(size, Image.Resampling.LANCZOS)
    alpha = ImageOps.autocontrast(alpha)
    refined = alpha.point(_refine_alpha_value)
    if original is None:
        return refined
    return clean_background_artifacts(refined, original)


def clean_background_artifacts(alpha: Image.Image, original: Image.Image) -> Image.Image:
    alpha = alpha.convert("L")
    original = ImageOps.exif_transpose(original).convert("RGB").resize(alpha.size, Image.Resampling.LANCZOS)
    background_palette = sample_background_palette(original)
    if not background_palette:
        return alpha

    alpha = clear_background_colored_edge_pixels(alpha, original, background_palette)
    alpha = preserve_high_contrast_foreground(alpha, original, background_palette)
    width, height = alpha.size
    transparent_pixels = bytearray(1 if value == 0 else 0 for value in alpha.tobytes())
    seen = bytearray(width * height)
    hole_mask = Image.new("L", alpha.size, 0)
    hole_pixels = hole_mask.load()

    for start, is_transparent in enumerate(transparent_pixels):
        if not is_transparent or seen[start]:
            continue

        stack = [start]
        seen[start] = 1
        component: list[int] = []
        touches_edge = False
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0

        while stack:
            index = stack.pop()
            component.append(index)
            x = index % width
            y = index // width
            touches_edge = touches_edge or x == 0 or y == 0 or x == width - 1 or y == height - 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            if x > 0:
                _push_transparent_neighbor(index - 1, transparent_pixels, seen, stack)
            if x < width - 1:
                _push_transparent_neighbor(index + 1, transparent_pixels, seen, stack)
            if y > 0:
                _push_transparent_neighbor(index - width, transparent_pixels, seen, stack)
            if y < height - 1:
                _push_transparent_neighbor(index + width, transparent_pixels, seen, stack)

        area = len(component)
        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1
        if (
            not touches_edge
            and MASK_HOLE_MIN_AREA <= area <= MASK_HOLE_MAX_AREA
            and span_x <= MASK_HOLE_MAX_SPAN
            and span_y <= MASK_HOLE_MAX_SPAN
        ):
            for index in component:
                hole_pixels[index % width, index // width] = 255

    dilated_holes = hole_mask.filter(ImageFilter.MaxFilter(MASK_HOLE_DILATION_SIZE))
    alpha_bytes = bytearray(alpha.tobytes())
    original_bytes = original.tobytes()
    for index, is_hole in enumerate(dilated_holes.tobytes()):
        color_index = index * 3
        if is_hole and is_background_like(
            original_bytes[color_index],
            original_bytes[color_index + 1],
            original_bytes[color_index + 2],
            background_palette,
        ):
            alpha_bytes[index] = 0
    return Image.frombytes("L", alpha.size, bytes(alpha_bytes))


def clear_background_colored_edge_pixels(
    alpha: Image.Image,
    original: Image.Image,
    background_palette: list[tuple[int, int, int]],
) -> Image.Image:
    width, height = alpha.size
    alpha_bytes = bytearray(alpha.tobytes())
    source_alpha_bytes = bytes(alpha_bytes)
    original_bytes = original.tobytes()

    for index, alpha_value in enumerate(alpha_bytes):
        if alpha_value == 0:
            continue

        color_index = index * 3
        if not is_background_like(
            original_bytes[color_index],
            original_bytes[color_index + 1],
            original_bytes[color_index + 2],
            background_palette,
        ):
            continue

        if alpha_value < 255 or has_transparent_neighbor(index, source_alpha_bytes, width, height):
            alpha_bytes[index] = 0

    return Image.frombytes("L", alpha.size, bytes(alpha_bytes))


def has_transparent_neighbor(index: int, alpha_bytes: bytes, width: int, height: int) -> bool:
    x = index % width
    y = index // width
    for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
        row = neighbor_y * width
        for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
            neighbor_index = row + neighbor_x
            if neighbor_index != index and alpha_bytes[neighbor_index] == 0:
                return True
    return False


def preserve_high_contrast_foreground(
    alpha: Image.Image,
    original: Image.Image,
    background_palette: list[tuple[int, int, int]],
) -> Image.Image:
    alpha_bytes = bytearray(alpha.tobytes())
    original_bytes = original.tobytes()

    for index in range(len(alpha_bytes)):
        color_index = index * 3
        red = original_bytes[color_index]
        green = original_bytes[color_index + 1]
        blue = original_bytes[color_index + 2]
        if is_background_like(red, green, blue, background_palette):
            continue

        luma = round(red * 0.299 + green * 0.587 + blue * 0.114)
        if luma <= FOREGROUND_BLACK_LUMA_CUTOFF or luma >= FOREGROUND_WHITE_LUMA_CUTOFF:
            alpha_bytes[index] = 255

    return Image.frombytes("L", alpha.size, bytes(alpha_bytes))


def _push_transparent_neighbor(
    index: int,
    transparent_pixels: bytearray,
    seen: bytearray,
    stack: list[int],
) -> None:
    if transparent_pixels[index] and not seen[index]:
        seen[index] = 1
        stack.append(index)


def sample_background_palette(image: Image.Image) -> list[tuple[int, int, int]]:
    from collections import Counter

    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    counter: Counter[tuple[int, int, int]] = Counter()

    for y in range(height):
        for offset in range(min(BACKGROUND_SAMPLE_WIDTH, width)):
            counter[_quantize_color(pixels[offset, y])] += 1
            counter[_quantize_color(pixels[width - offset - 1, y])] += 1

    for x in range(width):
        for offset in range(min(BACKGROUND_SAMPLE_WIDTH, height)):
            counter[_quantize_color(pixels[x, offset])] += 1
            counter[_quantize_color(pixels[x, height - offset - 1])] += 1

    return [
        color
        for color, _count in counter.most_common(BACKGROUND_PALETTE_SIZE)
        if is_neutral_mid_tone(color)
    ]


def is_background_like(
    red: int,
    green: int,
    blue: int,
    background_palette: list[tuple[int, int, int]],
) -> bool:
    return any(
        (red - bg_red) ** 2 + (green - bg_green) ** 2 + (blue - bg_blue) ** 2
        <= BACKGROUND_COLOR_DISTANCE**2
        for bg_red, bg_green, bg_blue in background_palette
    )


def is_neutral_mid_tone(color: tuple[int, int, int]) -> bool:
    red, green, blue = color
    luma = round(red * 0.299 + green * 0.587 + blue * 0.114)
    return max(color) - min(color) <= 12 and 24 <= luma <= 240


def _quantize_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = color
    return (
        round(red / 4) * 4,
        round(green / 4) * 4,
        round(blue / 4) * 4,
    )


def _refine_alpha_value(value: int) -> int:
    if value <= MASK_BACKGROUND_CUTOFF:
        return 0
    if value >= MASK_FOREGROUND_CUTOFF:
        return 255

    position = (value - MASK_BACKGROUND_CUTOFF) / (MASK_FOREGROUND_CUTOFF - MASK_BACKGROUND_CUTOFF)
    smoothed = position * position * (3 - 2 * position)
    return round(smoothed * 255)


def _mps_available(torch_module) -> bool:
    return bool(
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )
