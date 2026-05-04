from __future__ import annotations

from io import BytesIO
from typing import Annotated, Protocol
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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

    @app.post("/slice")
    async def slice_image(
        image: Annotated[UploadFile, File(description="Input image file.")],
        grid: Annotated[int, Form(description="Grid size. 3 means a 3x3 grid.")],
        margin: Annotated[int, Form(description="Outer margin in pixels to crop away before slicing.")] = 0,
    ) -> StreamingResponse:
        if grid < 1:
            raise HTTPException(status_code=400, detail="grid must be greater than or equal to 1.")
        if margin < 0:
            raise HTTPException(status_code=400, detail="margin must be greater than or equal to 0.")

        payload = await image.read()
        try:
            with Image.open(BytesIO(payload)) as input_image:
                source = input_image.copy()
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

        width, height = source.size
        if margin * 2 >= width or margin * 2 >= height:
            raise HTTPException(status_code=400, detail="margin is too large for the uploaded image.")

        cropped = source.crop((margin, margin, width - margin, height - margin))
        cropped_width, cropped_height = cropped.size
        if grid > cropped_width or grid > cropped_height:
            raise HTTPException(status_code=400, detail="grid is too large for the cropped image.")

        response_body = BytesIO()
        with ZipFile(response_body, mode="w", compression=ZIP_DEFLATED) as archive:
            for row in range(grid):
                top = round(row * cropped_height / grid)
                bottom = round((row + 1) * cropped_height / grid)
                for column in range(grid):
                    left = round(column * cropped_width / grid)
                    right = round((column + 1) * cropped_width / grid)
                    cell = cropped.crop((left, top, right, bottom))

                    cell_body = BytesIO()
                    cell.save(cell_body, format="PNG")
                    archive.writestr(f"slice_{row + 1}_{column + 1}.png", cell_body.getvalue())

        response_body.seek(0)
        return StreamingResponse(
            response_body,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="slices.zip"'},
        )

    return app


app = create_app()
