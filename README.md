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
- ~8 GB VRAM for the 3B model at bfloat16

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
  -d '{"input": "Hello, how are you today?", "voice": "tara"}' \
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

### `WS /v1/audio/speech/ws`

WebSocket endpoint for bidirectional streaming. Send a JSON message:

```json
{"input": "Hello!", "voice": "tara"}
```

The server streams back binary PCM frames, followed by a final JSON
`{"done": true}` message.

### `GET /v1/voices`

List available voice identifiers.

### `GET /health`

Service health check.

## Request body

| Field               | Type    | Default | Description                     |
|---------------------|---------|---------|---------------------------------|
| `input`             | string  | —       | Text to synthesize (required)   |
| `voice`             | string  | `tara`  | Speaker voice                   |
| `temperature`       | float   | 0.4     | Sampling temperature            |
| `top_p`             | float   | 0.9     | Nucleus sampling threshold      |
| `max_tokens`        | int     | 1200    | Max tokens to generate          |
| `repetition_penalty`| float   | 1.1     | Repetition penalty              |

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

`tara` · `zoe` · `jess` · `zac` · `leo` · `mia` · `julia` · `leah`

## License

Apache 2.0
