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

Expose the server through a public NPort HTTPS URL from a separate terminal:

```bash
scripts/start-nport.sh --port 8000
```

The public URL is printed by NPort while the tunnel is running. NPort cleans up tunnels after 4 hours; the script restarts NPort automatically until you stop it with Ctrl+C. The script uses a globally installed `nport` CLI when available, or falls back to `npx nport`. NPort requires Node.js 20+ and npm 10+:

```bash
node --version
npm --version
```

You can request a custom `nport.link` subdomain:

```bash
scripts/start-nport.sh --port 8000 --subdomain my-rmbg-api
```

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

Slice an image into an `n x n` grid after cropping away an outer margin:

```bash
curl -X POST \
  -F image=@input.jpg \
  -F grid=3 \
  -F margin=20 \
  http://127.0.0.1:8000/slice \
  --output slices.zip
```

The zip contains PNG files named `slice_<row>_<column>.png`, using 1-based row and column numbers.

## Tests

The automated tests use synthetic images and a stub inference engine, so they do not download the gated model.

```bash
uv run pytest
```
