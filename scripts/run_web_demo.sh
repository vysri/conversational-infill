#!/usr/bin/env bash
# Launch the ConvFill web demo: FastAPI backend (:8000) + Vite frontend (:5173).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DEMO_DIR="$REPO_ROOT/web_demo"
cd "$REPO_ROOT"

export PYTHONPATH="$WEB_DEMO_DIR:${PYTHONPATH:-}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[warn] ffmpeg not found on PATH — Whisper transcription will fail. Install with: brew install ffmpeg" >&2
fi

if [ ! -d "$WEB_DEMO_DIR/frontend/node_modules" ]; then
  echo "[setup] installing frontend deps (first run)…"
  (cd "$WEB_DEMO_DIR/frontend" && npm install)
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "[shutdown] stopping…"
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[backend] starting FastAPI on http://127.0.0.1:8000"
python -m backend.run &
BACKEND_PID=$!

echo -n "[backend] waiting for :8000 to accept connections"
for i in $(seq 1 60); do
  if (echo >/dev/tcp/127.0.0.1/8000) >/dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 0.5
done

echo "[frontend] starting Vite on http://127.0.0.1:5173"
(cd "$WEB_DEMO_DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

echo
echo "ConvFill web demo running."
echo "  open: http://127.0.0.1:5173"
echo "  press Ctrl+C to stop both servers"
echo

wait
