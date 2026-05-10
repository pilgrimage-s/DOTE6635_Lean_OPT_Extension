#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/hpc_ollama_env.sh"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama command not found."
  echo "Run first: bash hpc_01_install_ollama_local.sh"
  exit 1
fi

mkdir -p "$OLLAMA_MODELS"

echo "==> Ollama URL: $OLLAMA_BASE_URL"
echo "==> Ollama models directory: $OLLAMA_MODELS"
echo "==> Ollama log file: $OLLAMA_LOG_FILE"

if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "==> Ollama server is already reachable."
  ollama list || true
  exit 0
fi

if [[ -f "$OLLAMA_PID_FILE" ]]; then
  OLD_PID="$(cat "$OLLAMA_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "==> Existing Ollama process found with PID $OLD_PID. Waiting for API..."
  else
    rm -f "$OLLAMA_PID_FILE"
  fi
fi

if [[ ! -f "$OLLAMA_PID_FILE" ]]; then
  echo "==> Starting Ollama server in the background..."
  nohup ollama serve > "$OLLAMA_LOG_FILE" 2>&1 &
  echo "$!" > "$OLLAMA_PID_FILE"
fi

for _ in $(seq 1 60); do
  if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "==> Ollama server is ready."
    echo "==> PID: $(cat "$OLLAMA_PID_FILE")"
    exit 0
  fi
  sleep 2
done

echo "Ollama server did not become reachable at $OLLAMA_BASE_URL"
echo "Check log file: $OLLAMA_LOG_FILE"
tail -n 80 "$OLLAMA_LOG_FILE" || true
exit 1
