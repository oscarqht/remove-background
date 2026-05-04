from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from remove_background.api import create_app
from remove_background.inference import ModelAccessError, RemovalResult, apply_alpha_mask


class StubRemovalService:
    model_id = "test/model"
    selected_device = "cpu"
    model_loaded = False

    def remove_background(self, image: Image.Image) -> RemovalResult:
        original = image.convert("RGB")
        mask = Image.new("L", original.size, color=200)
        return RemovalResult(
            image=apply_alpha_mask(original, mask),
            mask=mask,
            device=self.selected_device,
            model_id=self.model_id,
        )


class AccessErrorService(StubRemovalService):
    def remove_background(self, image: Image.Image) -> RemovalResult:
        raise ModelAccessError("briaai/RMBG-2.0")


def _png_bytes() -> bytes:
    image = Image.new("RGB", (5, 4), color=(0, 128, 255))
    payload = BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


def test_healthz_returns_model_state_without_loading_model() -> None:
    client = TestClient(create_app(service=StubRemovalService()))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_id": "test/model",
        "device": "cpu",
        "model_loaded": False,
    }


def test_remove_background_returns_foreground_png() -> None:
    client = TestClient(create_app(service=StubRemovalService()))

    response = client.post(
        "/v1/remove-background",
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    output = Image.open(BytesIO(response.content))
    assert output.mode == "RGBA"
    assert output.size == (5, 4)
    assert output.getpixel((0, 0))[3] == 200


def test_remove_background_returns_mask_png() -> None:
    client = TestClient(create_app(service=StubRemovalService()))

    response = client.post(
        "/v1/remove-background?mode=mask",
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    output = Image.open(BytesIO(response.content))
    assert output.mode == "L"
    assert output.size == (5, 4)
    assert output.getpixel((0, 0)) == 200


def test_remove_background_returns_401_for_gated_model_access() -> None:
    client = TestClient(create_app(service=AccessErrorService()))

    response = client.post(
        "/v1/remove-background",
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 401
    assert "Cannot access gated Hugging Face model" in response.json()["detail"]
