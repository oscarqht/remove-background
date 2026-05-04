from PIL import Image

from remove_background.inference import (
    BackgroundRemovalService,
    ModelAccessError,
    apply_alpha_mask,
    is_model_access_error,
    refine_alpha_mask,
)


def test_apply_alpha_mask_preserves_size_and_adds_alpha_channel() -> None:
    image = Image.new("RGB", (8, 6), color=(255, 0, 0))
    mask = Image.new("L", (4, 3), color=128)

    output = apply_alpha_mask(image, mask)

    assert output.mode == "RGBA"
    assert output.size == image.size
    assert output.getpixel((0, 0)) == (255, 0, 0, 128)


def test_refine_alpha_mask_hardens_confident_foreground_and_background() -> None:
    mask = Image.new("L", (5, 1))
    mask.putdata([0, 8, 128, 224, 250])

    refined = refine_alpha_mask(mask, mask.size)

    assert refined.getpixel((0, 0)) == 0
    assert refined.getpixel((1, 0)) == 0
    assert refined.getpixel((2, 0)) not in {0, 255}
    assert refined.getpixel((3, 0)) == 255
    assert refined.getpixel((4, 0)) == 255


def test_refine_alpha_mask_uses_requested_output_size() -> None:
    mask = Image.new("L", (2, 2), color=250)

    refined = refine_alpha_mask(mask, (8, 6))

    assert refined.mode == "L"
    assert refined.size == (8, 6)


def test_refine_alpha_mask_clears_artifacts_inside_enclosed_holes() -> None:
    mask = Image.new("L", (9, 9), color=250)
    original = Image.new("RGB", (9, 9), color=(255, 255, 255))
    for index in range(9):
        original.putpixel((index, 0), (80, 80, 80))
        original.putpixel((index, 8), (80, 80, 80))
        original.putpixel((0, index), (80, 80, 80))
        original.putpixel((8, index), (80, 80, 80))

    for y in range(3, 6):
        for x in range(3, 6):
            mask.putpixel((x, y), 0)
            original.putpixel((x, y), (80, 80, 80))
    original.putpixel((4, 2), (80, 80, 80))

    refined = refine_alpha_mask(mask, mask.size, original)

    assert refined.getpixel((4, 2)) == 0
    assert refined.getpixel((3, 2)) == 255
    assert refined.getpixel((1, 1)) == 255


def test_refine_alpha_mask_preserves_high_contrast_text_pixels() -> None:
    mask = Image.new("L", (5, 5), color=0)
    original = Image.new("RGB", (5, 5), color=(80, 80, 80))
    original.putpixel((2, 2), (0, 0, 0))
    original.putpixel((2, 1), (255, 255, 255))

    refined = refine_alpha_mask(mask, mask.size, original)

    assert refined.getpixel((2, 2)) == 255
    assert refined.getpixel((2, 1)) == 255
    assert refined.getpixel((1, 1)) == 0


def test_refine_alpha_mask_clears_background_colored_edge_pixels() -> None:
    mask = Image.new("L", (5, 5), color=250)
    mask.putpixel((2, 2), 128)
    mask.putpixel((1, 2), 0)
    original = Image.new("RGB", (5, 5), color=(255, 255, 255))
    for index in range(5):
        original.putpixel((index, 0), (80, 80, 80))
        original.putpixel((index, 4), (80, 80, 80))
        original.putpixel((0, index), (80, 80, 80))
        original.putpixel((4, index), (80, 80, 80))
    original.putpixel((2, 2), (80, 80, 80))
    original.putpixel((1, 1), (80, 80, 80))

    refined = refine_alpha_mask(mask, mask.size, original)

    assert refined.getpixel((2, 2)) == 0
    assert refined.getpixel((1, 1)) == 0
    assert refined.getpixel((2, 1)) == 255


def test_service_reads_hf_token_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    service = BackgroundRemovalService()

    assert service.hf_token == "hf_test_token"


def test_gated_repo_error_is_detected_and_rewritten() -> None:
    error = OSError("401 Client Error. Cannot access gated repo. Please log in.")

    assert is_model_access_error(error)
    assert "accept the model access terms" in str(ModelAccessError("briaai/RMBG-2.0"))
