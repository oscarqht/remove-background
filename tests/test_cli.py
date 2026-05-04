from pathlib import Path

import pytest

from remove_background.cli import build_localtunnel_command, default_output_path


def test_default_output_path_replaces_jpg_extension_with_png() -> None:
    assert default_output_path(Path("/tmp/photo.jpg")) == Path("/tmp/photo-bg-removed.png")


def test_default_output_path_keeps_input_directory_for_png() -> None:
    assert default_output_path(Path("images/product.png")) == Path("images/product-bg-removed.png")


def test_localtunnel_command_prefers_installed_lt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/lt" if name == "lt" else None

    monkeypatch.setattr("remove_background.cli.shutil.which", fake_which)

    assert build_localtunnel_command(8000, "demo", None) == [
        "/usr/local/bin/lt",
        "--port",
        "8000",
        "--subdomain",
        "demo",
    ]


def test_localtunnel_command_falls_back_to_npx(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        return "/usr/bin/npx" if name == "npx" else None

    monkeypatch.setattr("remove_background.cli.shutil.which", fake_which)

    assert build_localtunnel_command(9000, None, "https://localtunnel.me") == [
        "npx",
        "--yes",
        "localtunnel",
        "--port",
        "9000",
        "--host",
        "https://localtunnel.me",
    ]


def test_localtunnel_command_supports_explicit_local_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("remove_background.cli.shutil.which", lambda name: "/usr/bin/npx" if name == "npx" else None)

    assert build_localtunnel_command(9000, local_host="127.0.0.1") == [
        "npx",
        "--yes",
        "localtunnel",
        "--port",
        "9000",
        "--local-host",
        "127.0.0.1",
    ]


def test_localtunnel_command_requires_cli_or_npx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("remove_background.cli.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="localtunnel CLI"):
        build_localtunnel_command(8000)
