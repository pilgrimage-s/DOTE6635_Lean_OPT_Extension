#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/hpc_ollama_env.sh"

PY_SCRIPT="${PY_SCRIPT:-LEAN_LLM_OPT_pure_ollama_Large-scale-or_incremental.py}"
RUN_PY_SCRIPT="${RUN_PY_SCRIPT:-LEAN_LLM_OPT_pure_ollama_Large-scale-or.HPC_RUN_ONLY.py}"

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  bash "$ROOT_DIR/hpc_02_start_ollama_server.sh"
fi

echo "==> Project directory: $ROOT_DIR"
echo "==> Ollama URL: $OLLAMA_BASE_URL"
echo "==> LLM model: $OLLAMA_LLM_MODEL"
echo "==> Embedding model: $OLLAMA_EMBED_MODEL"
echo "==> Python script: $PY_SCRIPT"

if [[ "${INSTALL_PYTHON_DEPS:-0}" == "1" ]]; then
  echo "==> Installing Python dependencies with pip in the current Python environment..."
  python3 -m pip install -r requirements.txt
else
  echo "==> Skipping pip install. Set INSTALL_PYTHON_DEPS=1 to install requirements.txt."
fi

echo "==> Creating HPC run-only Python script: $RUN_PY_SCRIPT"
export PY_SCRIPT RUN_PY_SCRIPT TEST_LIMIT
python3 - <<'PY'
import os
from pathlib import Path

src = Path(os.environ.get("PY_SCRIPT", "LEAN_LLM_OPT_pure_ollama_Large-scale-or_incremental.py"))
dst = Path(os.environ.get("RUN_PY_SCRIPT", "LEAN_LLM_OPT_pure_ollama_Large-scale-or.HPC_RUN_ONLY.py"))
test_limit = os.environ.get("TEST_LIMIT", "").strip()

if not src.exists():
    raise SystemExit(f"Python script not found: {src}")

source = src.read_text(encoding="utf-8")
if test_limit:
    target = "test = pd.read_csv('Test_Dataset/Large-scale-or/Large-scale-or-101.csv')"
    replacement = (
        "test = pd.read_csv('Test_Dataset/Large-scale-or/Large-scale-or-101.csv')\n"
        f"test = test.head({int(test_limit)})\n"
        f"print('Using TEST_LIMIT={int(test_limit)}')"
    )
    if target not in source:
        raise SystemExit("Could not patch TEST_LIMIT into the Python script.")
    source = source.replace(target, replacement, 1)

dst.write_text(source, encoding="utf-8")
print(f"Wrote {dst}")
PY

echo "==> Executing Python script."
export OLLAMA_BASE_URL
export OLLAMA_LLM_MODEL
export OLLAMA_EMBED_MODEL
export OLLAMA_HOST
export OLLAMA_MODELS

python3 "$RUN_PY_SCRIPT"

echo "==> Done."
echo "Executed script: $RUN_PY_SCRIPT"
echo "CSV output: $ROOT_DIR/Large-scale-or-Lean-Ollama-20b-orchestrated.csv"
