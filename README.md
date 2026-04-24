# Orpheus TTS – Low-Latency Streaming Inference Server

FastAPI server for [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) with **vLLM** as the inference engine and **SNAC** audio decoding. Designed for CUDA GPUs with a focus on minimal time-to-first-byte (TTFB) and high concurrency.

## Architecture

```
Client ──▶ FastAPI ──▶ vLLM AsyncLLMEngine ──▶ SNAC Decoder ──▶ PCM audio stream
                       (continuous batching,    (24 kHz, 16-bit)
                        PagedAttention,
                        prefix caching)
```

The vLLM engine runs **embedded** in the same process — no HTTP hop to a
separate inference server — which shaves ~1-2 ms off every token and avoids
serialisation overhead.  Audio tokens are decoded into PCM chunks with
[SNAC](https://github.com/hubertsiuzdak/snac) and streamed back as they are
produced.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x+ (tested on RTX 4090 / A100 / H100)
- ~8 GB VRAM for a single 3B model at bfloat16 (≥16 GB if loading both
  English and Turkish models concurrently)

## Available models

The server can serve multiple Orpheus-compatible checkpoints at the same
time. Select one per request via the ``model`` field (or the model drop-down
in the UI). Built-in profiles:

| id           | Checkpoint                         | Language | Speakers        |
|--------------|------------------------------------|----------|-----------------|
| `orpheus-en` | `canopylabs/orpheus-3b-0.1-ft`     | English  | Multi (8 voices)|
| `orpheus-tr` | `yaltay/tmp_tmp_smp_vllm`          | Turkish  | Single-speaker  |

The Turkish profile is an Unsloth fine-tune of Orpheus 3B – it is a
single-speaker model, so the `voice` field is ignored for it.

Enable or disable models via `ENABLED_MODELS` (comma-separated ids) and
point `DEFAULT_MODEL` at the one you want to use when the request omits
a model id. Additional custom profiles can be declared in a JSON file
referenced by `MODELS_FILE`.

### VRAM planning for multi-model setups

Each Orpheus 3B engine needs roughly **8 GB of VRAM** for weights plus a
few GB for KV cache and CUDA graphs. Depending on your GPU:

| GPU VRAM       | Recommended setup                                                        |
|----------------|--------------------------------------------------------------------------|
| 40 GB+ (A100)  | `ENABLED_MODELS=orpheus-en,orpheus-tr`, default settings work            |
| 24 GB (4090)   | `ENABLED_MODELS=orpheus-en,orpheus-tr`, default settings work            |
| 16 GB or less  | Load **one** model at a time: `ENABLED_MODELS=orpheus-tr` (or `-en`)     |

`ENFORCE_EAGER=true` is now the default. It skips CUDA graph capture, which
can save several GB per engine and avoid `CUDA error: illegal memory access`
startup failures on some systems. If startup still fails, drop to a single
model or reduce `MAX_MODEL_LEN` / `MAX_NUM_SEQS`.

`PER_MODEL_GPU_MEMORY_UTILIZATION` lets you pin per-engine VRAM share
explicitly if the automatic split (`GPU_MEMORY_UTILIZATION / n_models`)
isn't ideal.

`PAD_VOCAB_TO_MULTIPLE=0` keeps the checkpoint's native vocab size on the
first startup attempt. If vLLM then fails during sampler warmup with the
known `CUDA error: illegal memory access` / engine-core init signature, the
server automatically retries once with a 64-token vocab alignment. Set
`PAD_VOCAB_TO_MULTIPLE=64` (or `128`) to force padding from the first try if
you already know your checkpoint needs it.

## Quick start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit config
cp .env.example .env

# 4. Launch
bash start.sh
```

The server starts on `http://0.0.0.0:8000` by default.

## API endpoints

### `POST /v1/audio/speech/stream`

Streams raw PCM audio (24 kHz, 16-bit, mono). Lowest latency option.

```bash
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you today?", "model": "orpheus-en", "voice": "tara"}' \
  --output speech.pcm
```

Turkish model (single-speaker – omit `voice`):

```bash
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Merhaba, bugün nasılsın?", "model": "orpheus-tr"}' \
  --output speech.pcm
```

Play the raw PCM:

```bash
ffplay -f s16le -ar 24000 -ac 1 speech.pcm
```

Response headers include `X-Sample-Rate`, `X-Bit-Depth`, and `X-Channels`.

### `POST /v1/audio/speech`

Returns a complete WAV file (blocks until generation is done).

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you today?", "voice": "tara"}' \
  --output speech.wav
```

### `POST /v1/audio/speech/metrics`

Runs the full generation pipeline and returns timing metrics instead of audio.
This is useful for concurrency testing and latency budgeting from the built-in UI.

```bash
curl -X POST http://localhost:8000/v1/audio/speech/metrics \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you today?", "voice": "tara"}'
```

Example response:

```json
{
  "request_id": "8e2d3f...",
  "status": "completed",
  "voice": "tara",
  "input_chars": 29,
  "token_deltas": 154,
  "codec_tokens": 617,
  "audio_chunks": 88,
  "audio_bytes": 360448,
  "ttft_ms": 74.2,
  "ttfa_ms": 191.6,
  "total_generation_ms": 1842.9
}
```

### `WS /v1/audio/speech/ws`

WebSocket endpoint for bidirectional streaming. Send a JSON message:

```json
{"input": "Hello!", "voice": "tara"}
```

The server streams back binary PCM frames, followed by a final JSON
`{"done": true}` message.

### `GET /v1/models`

List all loaded models with their voices and language metadata.

```bash
curl http://localhost:8000/v1/models
```

### `GET /v1/voices?model=<id>`

List available voice identifiers for a specific model. Returns an empty
list for single-speaker models.

### `GET /health`

Service health check.

## Request body

| Field               | Type    | Default           | Description                                           |
|---------------------|---------|-------------------|-------------------------------------------------------|
| `input`             | string  | —                 | Text to synthesize (required)                         |
| `model`             | string  | `DEFAULT_MODEL`   | Model id, e.g. `orpheus-en` or `orpheus-tr`           |
| `voice`             | string  | model default     | Speaker voice (ignored for single-speaker models)     |
| `temperature`       | float   | 0.4               | Sampling temperature                                  |
| `top_p`             | float   | 0.9               | Nucleus sampling threshold                            |
| `max_tokens`        | int     | 1200              | Max tokens to generate                                |
| `repetition_penalty`| float   | 1.1               | Repetition penalty                                    |

## Configuration

All settings are configurable via environment variables or a `.env` file.
See [`.env.example`](.env.example) for the full list.

### Multi-GPU setup

For maximum concurrency, run the SNAC decoder on a separate GPU:

```env
SNAC_DEVICE=cuda:1
```

This frees the primary GPU for vLLM inference and eliminates contention
during audio decoding.

## Voices

English (`orpheus-en`): `tara` · `zoe` · `jess` · `zac` · `leo` · `mia` · `julia` · `leah`

Turkish (`orpheus-tr`): single-speaker – omit the `voice` field.

## Built-in UI

The browser UI now includes:

- A model selector that switches voices / hides the voice field for
  single-speaker checkpoints
- Single-request streaming playback for listening tests
- A concurrency load-test panel with selectable parallel request count
- Aggregate P50/P95 latency stats and a per-request timing table
- Per-request sample visibility so you can see exactly what was sent
- Budget checks for TTFA and end-to-end generation latency

## License

Apache 2.0
