from __future__ import annotations

from io import BytesIO
from typing import Annotated, Protocol

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from remove_background.inference import (
    DEFAULT_MODEL_ID,
    BackgroundRemovalService,
    DevicePreference,
    ModelAccessError,
    OutputMode,
    RemovalResult,
)


class RemovalEngine(Protocol):
    model_id: str
    selected_device: str
    model_loaded: bool

    def remove_background(self, image: Image.Image) -> RemovalResult:
        ...


def create_app(
    model_id: str = DEFAULT_MODEL_ID,
    device: DevicePreference = "auto",
    hf_token: str | None = None,
    service: RemovalEngine | None = None,
) -> FastAPI:
    app = FastAPI(title="RMBG-2.0 Background Removal API", version="0.1.0")
    app.state.service = service or BackgroundRemovalService(
        model_id=model_id,
        device=device,
        hf_token=hf_token,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        removal_service: RemovalEngine = app.state.service
        return {
            "status": "ok",
            "model_id": removal_service.model_id,
            "device": removal_service.selected_device,
            "model_loaded": removal_service.model_loaded,
        }

    @app.get("/v1/model")
    def model_info() -> dict[str, object]:
        removal_service: RemovalEngine = app.state.service
        return {
            "model_id": removal_service.model_id,
            "device": removal_service.selected_device,
            "model_loaded": removal_service.model_loaded,
        }

    @app.post("/v1/remove-background")
    async def remove_background(
        image: Annotated[UploadFile, File(description="Input image file.")],
        mode: Annotated[OutputMode, Query()] = "foreground",
    ) -> StreamingResponse:
        payload = await image.read()
        try:
            with Image.open(BytesIO(payload)) as input_image:
                result = app.state.service.remove_background(input_image)
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
        except ModelAccessError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        output = result.mask if mode == "mask" else result.image
        response_body = BytesIO()
        output.save(response_body, format="PNG")
        response_body.seek(0)

        filename = "mask.png" if mode == "mask" else "foreground.png"
        return StreamingResponse(
            response_body,
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    return app


app = create_app()
