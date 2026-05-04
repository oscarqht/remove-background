from io import BytesIO
from zipfile import ZipFile

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


def _pixel_grid_png_bytes() -> bytes:
    image = Image.new("RGB", (4, 4))
    for y in range(4):
        for x in range(4):
            image.putpixel((x, y), (x * 50, y * 50, 25))

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


def test_slice_returns_zip_with_grid_cells_after_margin_crop() -> None:
    client = TestClient(create_app(service=StubRemovalService()))

    response = client.post(
        "/slice",
        files={"image": ("input.png", _pixel_grid_png_bytes(), "image/png")},
        data={"grid": "2", "margin": "1"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="slices.zip"'

    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == [
            "slice_1_1.png",
            "slice_1_2.png",
            "slice_2_1.png",
            "slice_2_2.png",
        ]
        top_left = Image.open(BytesIO(archive.read("slice_1_1.png")))
        bottom_right = Image.open(BytesIO(archive.read("slice_2_2.png")))

    assert top_left.size == (1, 1)
    assert bottom_right.size == (1, 1)
    assert top_left.getpixel((0, 0)) == (50, 50, 25)
    assert bottom_right.getpixel((0, 0)) == (100, 100, 25)


def test_slice_rejects_invalid_parameters() -> None:
    client = TestClient(create_app(service=StubRemovalService()))

    invalid_grid = client.post(
        "/slice",
        files={"image": ("input.png", _png_bytes(), "image/png")},
        data={"grid": "0", "margin": "0"},
    )
    invalid_margin = client.post(
        "/slice",
        files={"image": ("input.png", _png_bytes(), "image/png")},
        data={"grid": "2", "margin": "3"},
    )
    oversized_grid = client.post(
        "/slice",
        files={"image": ("input.png", _png_bytes(), "image/png")},
        data={"grid": "6", "margin": "0"},
    )

    assert invalid_grid.status_code == 400
    assert invalid_grid.json()["detail"] == "grid must be greater than or equal to 1."
    assert invalid_margin.status_code == 400
    assert invalid_margin.json()["detail"] == "margin is too large for the uploaded image."
    assert oversized_grid.status_code == 400
    assert oversized_grid.json()["detail"] == "grid is too large for the cropped image."
