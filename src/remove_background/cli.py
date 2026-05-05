from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from remove_background.api import create_app
from remove_background.inference import (
    DEFAULT_MODEL_ID,
    BackgroundRemovalService,
    DevicePreference,
    ModelAccessError,
)

app = typer.Typer(help="Run RMBG-2.0 locally for background removal.")


@app.command()
def remove(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Input image.")],
    output_path: Annotated[
        Path | None,
        typer.Argument(help="PNG output with transparent background. Defaults to INPUT-bg-removed.png."),
    ] = None,
    mask: Annotated[Path | None, typer.Option("--mask", help="Optional grayscale alpha matte output.")] = None,
    model: Annotated[str, typer.Option("--model", help="Hugging Face model id.")] = DEFAULT_MODEL_ID,
    device: Annotated[DevicePreference, typer.Option("--device", help="Device: auto, cuda, mps, or cpu.")] = "auto",
    hf_token: Annotated[
        str | None,
        typer.Option("--hf-token", envvar="HF_TOKEN", help="Hugging Face token for gated model access."),
    ] = None,
) -> None:
    """Remove the background from one image."""
    resolved_output_path = output_path or default_output_path(input_path)
    service = BackgroundRemovalService(model_id=model, device=device, hf_token=hf_token)
    try:
        result = service.remove_file(input_path=input_path, output_path=resolved_output_path, mask_path=mask)
    except ModelAccessError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote {resolved_output_path} using {result.model_id} on {result.device}.")
    if mask is not None:
        typer.echo(f"Wrote mask {mask}.")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Host to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind.")] = 8000,
    model: Annotated[str, typer.Option("--model", help="Hugging Face model id.")] = DEFAULT_MODEL_ID,
    device: Annotated[DevicePreference, typer.Option("--device", help="Device: auto, cuda, mps, or cpu.")] = "auto",
    hf_token: Annotated[
        str | None,
        typer.Option("--hf-token", envvar="HF_TOKEN", help="Hugging Face token for gated model access."),
    ] = None,
) -> None:
    """Serve the HTTP API."""
    import uvicorn

    api = create_app(model_id=model, device=device, hf_token=hf_token)
    uvicorn.run(api, host=host, port=port)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-bg-removed.png")


if __name__ == "__main__":
    app()
