#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/hpc_ollama_env.sh"

if [[ ! -f "$OLLAMA_PID_FILE" ]]; then
  echo "No Ollama PID file found: $OLLAMA_PID_FILE"
  exit 0
fi

PID="$(cat "$OLLAMA_PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
  rm -f "$OLLAMA_PID_FILE"
  echo "Empty PID file removed."
  exit 0
fi

if kill -0 "$PID" >/dev/null 2>&1; then
  echo "Stopping Ollama server PID $PID"
  kill "$PID"
  for _ in $(seq 1 15); do
    if ! kill -0 "$PID" >/dev/null 2>&1; then
      rm -f "$OLLAMA_PID_FILE"
      echo "Stopped."
      exit 0
    fi
    sleep 1
  done
  echo "PID $PID did not stop after SIGTERM. Use kill -9 $PID if needed."
else
  echo "PID $PID is not running."
  rm -f "$OLLAMA_PID_FILE"
fi
