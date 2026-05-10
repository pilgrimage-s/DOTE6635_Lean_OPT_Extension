#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/hpc_ollama_env.sh"

bash "$ROOT_DIR/hpc_02_start_ollama_server.sh"

echo "==> Pulling LLM model: $OLLAMA_LLM_MODEL"
ollama pull "$OLLAMA_LLM_MODEL"

echo "==> Pulling embedding model: $OLLAMA_EMBED_MODEL"
ollama pull "$OLLAMA_EMBED_MODEL"

echo "==> Installed Ollama models:"
ollama list
