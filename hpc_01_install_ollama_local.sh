#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/hpc_ollama_env.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is intended for Linux HPC nodes only."
  exit 1
fi

case "$(uname -m)" in
  x86_64) OLLAMA_ARCH="amd64" ;;
  aarch64|arm64) OLLAMA_ARCH="arm64" ;;
  *)
    echo "Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac

for tool in curl tar; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool"
    exit 1
  fi
done

if ! command -v zstd >/dev/null 2>&1; then
  echo "Missing required tool: zstd"
  echo "Current Ollama Linux archives are .tar.zst files."
  echo "On HPC, try one of these before rerunning:"
  echo "  module avail zstd"
  echo "  module load zstd"
  echo "or ask the HPC admin to provide zstd."
  exit 1
fi

mkdir -p "$OLLAMA_INSTALL_DIR" "$OLLAMA_MODELS"

VER_PARAM=""
if [[ -n "${OLLAMA_VERSION:-}" ]]; then
  VER_PARAM="?version=$OLLAMA_VERSION"
fi

DOWNLOAD_URL="https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tar.zst${VER_PARAM}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ollama-install.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> Installing Ollama locally without sudo"
echo "==> Project directory: $ROOT_DIR"
echo "==> Install directory: $OLLAMA_INSTALL_DIR"
echo "==> Model directory: $OLLAMA_MODELS"
echo "==> Download URL: $DOWNLOAD_URL"

curl -fL --retry 3 --retry-delay 5 -o "$TMP_DIR/ollama-linux-${OLLAMA_ARCH}.tar.zst" "$DOWNLOAD_URL"
zstd -dc "$TMP_DIR/ollama-linux-${OLLAMA_ARCH}.tar.zst" | tar -xf - -C "$OLLAMA_INSTALL_DIR"

OLLAMA_BIN=""
if [[ -x "$OLLAMA_INSTALL_DIR/bin/ollama" ]]; then
  OLLAMA_BIN="$OLLAMA_INSTALL_DIR/bin/ollama"
elif [[ -x "$OLLAMA_INSTALL_DIR/ollama" ]]; then
  OLLAMA_BIN="$OLLAMA_INSTALL_DIR/ollama"
else
  OLLAMA_BIN="$(find "$OLLAMA_INSTALL_DIR" -type f -name ollama -perm -u+x | head -n 1 || true)"
fi

if [[ -z "$OLLAMA_BIN" ]]; then
  echo "Could not find an executable ollama binary under $OLLAMA_INSTALL_DIR"
  exit 1
fi

echo "==> Ollama binary: $OLLAMA_BIN"
"$OLLAMA_BIN" --version || true

echo
echo "Install complete."
echo "Use this environment in future shells:"
echo "  source \"$ROOT_DIR/hpc_ollama_env.sh\""
