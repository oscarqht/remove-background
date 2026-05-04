# Local RMBG-2.0 Background Removal

Run BRIA AI's `briaai/RMBG-2.0` model locally and expose it through a CLI and HTTP API.

## Requirements

- Python 3.11+
- `uv`
- A Hugging Face account with access accepted for `briaai/RMBG-2.0`

RMBG-2.0 is a gated Hugging Face model. The model card lists the weights for non-commercial use under CC BY-NC 4.0; commercial use requires a BRIA commercial agreement.

## Setup

```bash
uv sync
```

Use `uv sync` for dependencies instead of running `pip install` in a global or Conda environment. The CLI is intended to run from this project's isolated `.venv` through `uv run`.

If Hugging Face asks for authentication, create an access token and export it before running the CLI or API:

```bash
export HF_TOKEN=hf_your_token_here
```

You can also log in once from this project environment:

```bash
uv run hf auth login
```

Or pass a token per command:

```bash
uv run rmbg remove input.jpg output.png --hf-token hf_your_token_here
```

The first real removal call downloads the model. Device selection defaults to `auto`, which tries CUDA, then Apple Silicon MPS, then CPU.

## CLI

Remove a background and write a transparent PNG:

```bash
uv run rmbg remove input.jpg
```

When the output path is omitted, the CLI writes `input-bg-removed.png` in the same folder as the input image.

To choose the output path explicitly:

```bash
uv run rmbg remove input.jpg output.png
```

Also write the grayscale alpha matte:

```bash
uv run rmbg remove input.jpg output.png --mask mask.png
```

Force a device or model id:

```bash
uv run rmbg remove input.jpg output.png --device mps --model briaai/RMBG-2.0
```

## HTTP API

Start the server:

```bash
uv run rmbg serve --host 127.0.0.1 --port 8000
```

The server also accepts `--hf-token` or the `HF_TOKEN` environment variable.

Expose the server through a public localtunnel HTTPS URL:

```bash
uv run rmbg serve --host 127.0.0.1 --port 8000 --localtunnel
```

The public URL is printed by localtunnel while the server is running. This requires either Node.js with `npx` available or a globally installed localtunnel CLI:

```bash
npm install -g localtunnel
```

You can request a subdomain or use another localtunnel server:

```bash
uv run rmbg serve --port 8000 --localtunnel --localtunnel-subdomain my-rmbg-api
uv run rmbg serve --port 8000 --localtunnel --localtunnel-host https://localtunnel.me
```

The CLI intentionally does not pass localtunnel's `--local-host` option by default. That option rewrites the `Host` header inside localtunnel's Node client and can corrupt binary multipart uploads from clients such as Postman. If localtunnel cannot reach your server through its default `localhost` target, prefer starting the API with `--host localhost`. Only use `--localtunnel-local-host` when you specifically need to override the tunnel client's local target.

Health and model metadata:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/model
```

Remove a background:

```bash
curl -X POST \
  -F image=@input.jpg \
  http://127.0.0.1:8000/v1/remove-background \
  --output output.png
```

Return only the alpha matte:

```bash
curl -X POST \
  -F image=@input.jpg \
  'http://127.0.0.1:8000/v1/remove-background?mode=mask' \
  --output mask.png
```

### Localtunnel troubleshooting

For public calls, use the `https://<subdomain>.loca.lt` URL printed by localtunnel. For local calls on the same machine, use `http://127.0.0.1:8000`. Do not call the local Uvicorn port with `https://127.0.0.1:8000`; Uvicorn is serving plain HTTP and will reject HTTPS bytes before the request reaches FastAPI.

The remove-background endpoint expects multipart form data with an `image` file field:

```bash
curl -X POST \
  -F image=@input.jpg \
  https://your-subdomain.loca.lt/v1/remove-background \
  --output output.png
```

If the response body is `Invalid HTTP request received.`, verify that the client is using the public `https://*.loca.lt` URL without adding a port, and that the API was not started with `--localtunnel-local-host` unless that override is required.

## Tests

The automated tests use synthetic images and a stub inference engine, so they do not download the gated model.

```bash
uv run pytest
```
