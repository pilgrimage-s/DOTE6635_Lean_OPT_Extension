#!/usr/bin/env bash
set -euo pipefail

# Install/start Ollama, pull local models, and execute only the Large-Scale-OR
# section of LEAN_LLM_OPT_pure_ollama_Large-scale-or.ipynb.
#
# Usage:
#   chmod +x run_pure_ollama_large_scale.sh
#   ./run_pure_ollama_large_scale.sh
#
# Optional overrides:
#   OLLAMA_LLM_MODEL=gpt-oss:20b OLLAMA_EMBED_MODEL=embeddinggemma ./run_pure_ollama_large_scale.sh
#   SKIP_PIP_INSTALL=1 ./run_pure_ollama_large_scale.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-gpt-oss:20b}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"
NOTEBOOK="${NOTEBOOK:-LEAN_LLM_OPT_pure_ollama_Large-scale-or.ipynb}"
RUN_NOTEBOOK="${RUN_NOTEBOOK:-LEAN_LLM_OPT_pure_ollama_Large-scale-or.RUN_ONLY.ipynb}"
OUTPUT_NOTEBOOK="${OUTPUT_NOTEBOOK:-LEAN_LLM_OPT_pure_ollama_Large-scale-or.executed.ipynb}"

echo "==> Project directory: $ROOT_DIR"
echo "==> Ollama URL: $OLLAMA_BASE_URL"
echo "==> LLM model: $OLLAMA_LLM_MODEL"
echo "==> Embedding model: $OLLAMA_EMBED_MODEL"

if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Ollama not found. Installing..."
  case "$(uname -s)" in
    Linux)
      curl -fsSL https://ollama.com/install.sh | sh
      ;;
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install ollama
      else
        echo "Ollama is not installed and Homebrew is unavailable."
        echo "Install Ollama from https://ollama.com/download, then rerun this script."
        exit 1
      fi
      ;;
    *)
      echo "Unsupported OS for automatic Ollama install: $(uname -s)"
      echo "Install Ollama manually from https://ollama.com/download, then rerun this script."
      exit 1
      ;;
  esac
else
  echo "==> Ollama already installed: $(ollama --version || true)"
fi

echo "==> Starting Ollama if it is not already running..."
if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^ollama\.service'; then
    sudo systemctl enable ollama >/dev/null 2>&1 || true
    sudo systemctl restart ollama
  else
    nohup ollama serve > ollama-server.log 2>&1 &
  fi

  for _ in $(seq 1 30); do
    if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
fi

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama server is not reachable at $OLLAMA_BASE_URL."
  echo "Check ollama-server.log or run: ollama serve"
  exit 1
fi

echo "==> Pulling Ollama models..."
ollama pull "$OLLAMA_LLM_MODEL"
ollama pull "$OLLAMA_EMBED_MODEL"
ollama list

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  echo "==> Installing Python dependencies from requirements.txt..."
  python3 -m pip install -r requirements.txt
else
  echo "==> Skipping pip install because SKIP_PIP_INSTALL=1"
fi

echo "==> Creating Large-Scale-OR-only notebook: $RUN_NOTEBOOK"
python3 - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ.get("NOTEBOOK", "LEAN_LLM_OPT_pure_ollama_Large-scale-or.ipynb"))
dst = Path(os.environ.get("RUN_NOTEBOOK", "LEAN_LLM_OPT_pure_ollama_Large-scale-or.RUN_ONLY.ipynb"))

nb = json.loads(src.read_text(encoding="utf-8"))
cut_idx = None
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "Large-scale-or-Lean-Ollama-20b-orchestrated.csv" in source:
        cut_idx = i
        break

if cut_idx is None:
    raise SystemExit("Could not find the Large-Scale-OR execution cell.")

nb["cells"] = nb["cells"][: cut_idx + 1]
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {dst} with {len(nb['cells'])} cells.")
PY

echo "==> Executing notebook. This can take a long time for all 101 instances."
export OLLAMA_BASE_URL
export OLLAMA_LLM_MODEL
export OLLAMA_EMBED_MODEL

jupyter nbconvert \
  --to notebook \
  --execute "$RUN_NOTEBOOK" \
  --output "$OUTPUT_NOTEBOOK" \
  --ExecutePreprocessor.timeout=-1

echo "==> Done."
echo "Executed notebook: $OUTPUT_NOTEBOOK"
echo "CSV output: $ROOT_DIR/Large-scale-or-Lean-Ollama-20b-orchestrated.csv"
