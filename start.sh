#!/usr/bin/env bash
set -euo pipefail

# ── Load .env if present ──────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "──────────────────────────────────────────────────"
echo "  Orpheus TTS Server"
echo "  Listening on ${HOST}:${PORT}"
echo "──────────────────────────────────────────────────"

exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --loop uvloop \
    --http httptools \
    --log-level "$LOG_LEVEL" \
    --timeout-keep-alive 120
