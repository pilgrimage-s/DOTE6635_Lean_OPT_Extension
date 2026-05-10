#!/usr/bin/env bash
# Shared environment for running Ollama without sudo on an HPC account.
# Source this file from the project directory or from the other hpc_*.sh scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LEAN_OLLAMA_PROJECT_DIR="${LEAN_OLLAMA_PROJECT_DIR:-$SCRIPT_DIR}"
export OLLAMA_INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$LEAN_OLLAMA_PROJECT_DIR/.ollama-local}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$LEAN_OLLAMA_PROJECT_DIR/.ollama-models}"

# Ollama server settings. OLLAMA_HOST is used by `ollama serve`;
# OLLAMA_BASE_URL is used by the notebook/LangChain client.
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

export OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-gpt-oss:20b}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"

export OLLAMA_LOG_FILE="${OLLAMA_LOG_FILE:-$LEAN_OLLAMA_PROJECT_DIR/ollama-hpc-server.log}"
export OLLAMA_PID_FILE="${OLLAMA_PID_FILE:-$LEAN_OLLAMA_PROJECT_DIR/ollama-hpc-server.pid}"

export PATH="$OLLAMA_INSTALL_DIR/bin:$OLLAMA_INSTALL_DIR:$PATH"
