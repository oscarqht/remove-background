from PIL import Image

from remove_background.inference import (
    BackgroundRemovalService,
    ModelAccessError,
    apply_alpha_mask,
    is_model_access_error,
)


def test_apply_alpha_mask_preserves_size_and_adds_alpha_channel() -> None:
    image = Image.new("RGB", (8, 6), color=(255, 0, 0))
    mask = Image.new("L", (4, 3), color=128)

    output = apply_alpha_mask(image, mask)

    assert output.mode == "RGBA"
    assert output.size == image.size
    assert output.getpixel((0, 0)) == (255, 0, 0, 128)


def test_service_reads_hf_token_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    service = BackgroundRemovalService()

    assert service.hf_token == "hf_test_token"


def test_gated_repo_error_is_detected_and_rewritten() -> None:
    error = OSError("401 Client Error. Cannot access gated repo. Please log in.")

    assert is_model_access_error(error)
    assert "accept the model access terms" in str(ModelAccessError("briaai/RMBG-2.0"))
