from pathlib import Path

from remove_background.cli import default_output_path


def test_default_output_path_replaces_jpg_extension_with_png() -> None:
    assert default_output_path(Path("/tmp/photo.jpg")) == Path("/tmp/photo-bg-removed.png")


def test_default_output_path_keeps_input_directory_for_png() -> None:
    assert default_output_path(Path("images/product.png")) == Path("images/product-bg-removed.png")
