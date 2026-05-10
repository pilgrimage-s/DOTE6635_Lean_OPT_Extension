# Pure Ollama Orchestrator Workflow

This note documents the current workflow in `LEAN_LLM_OPT_pure_ollama_Large-scale-or.ipynb` after adding the lightweight Orchestrator layer.

## Scope

The notebook is a pure Ollama version of the Large-Scale-OR pipeline. It keeps the original LEAN-LLM-OPT notebook structure and adds a post-generation Orchestrator. It does not split the notebook into new Python modules.

Models used by default:

- Generation/review model: `gpt-oss:20b`
- Embedding model: `embeddinggemma`
- Ollama endpoint: `http://localhost:11434`

These can be overridden with:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_LLM_MODEL="gpt-oss:20b"
export OLLAMA_EMBED_MODEL="embeddinggemma"
```

## Relationship To `Plan.md`

`Plan.md` describes the original LEAN-LLM-OPT framework as a three-agent process:

1. Classification Agent
2. Workflow Generation Agent
3. Model Generation Agent

The pure Ollama notebook preserves that overall idea but keeps the implementation notebook-native:

- Classification is handled by `classify_problem`.
- Type-specific workflow and retrieval logic are embedded inside `get_NRM_response`, `get_RA_response`, `get_TP_response`, `get_FLP_response`, `get_AP_response`, `get_Others_response`, and `get_others_without_CSV_response`.
- Code generation is handled by `get_code`.
- The new Orchestrator is a fourth post-generation layer that reviews and optionally revises the generated model/code.

The main intentional difference from `Plan.md` is that the notebook does not introduce a separate pre-generation Workflow Agent abstraction. This avoids a large refactor and matches the current implementation strategy: keep the existing generation branches, then add model/code validation and at most one correction round afterward.

## End-To-End Workflow

For each row in `Test_Dataset/Large-scale-or/Large-scale-or-101.csv`:

1. Read `Query` and optional `Dataset_address`.
2. If a dataset exists, classify the problem with `classify_problem`.
3. Route the instance to the matching branch:
   - NRM: `get_NRM_response`
   - RA: `get_RA_response`
   - TP: `get_TP_response`
   - FLP/UFLP: `get_FLP_response`
   - AP: `get_AP_response`
   - Others with CSV: `get_Others_response`
   - Others without CSV: `get_others_without_CSV_response`
4. Generate the initial mathematical model.
5. Generate initial Gurobi code with `get_code`, except for `Others with CSV` when `get_Others_response` already returns Python/Gurobi code.
6. Run `orchestrate_generation`.
7. Save both initial and final outputs to `Large-scale-or-Lean-Ollama-20b-orchestrated.csv`.

## Orchestrator Steps

The Orchestrator is implemented in the notebook cell titled:

```python
# Lightweight post-generation Orchestrator for the pure Ollama pipeline.
```

Important configuration:

```python
ORCH_MAX_REFINEMENTS = 1
ORCH_ENABLE_CODE_EXECUTION = True
ORCH_REVIEW_MAX_CHARS = 7000
```

The Orchestrator performs:

1. Deterministic code check via `deterministic_code_check`.
2. LLM mathematical model review via `orchestrator_review_model`.
3. LLM Gurobi code review via `orchestrator_review_code`.
4. At most one refinement:
   - If the model has blocking flaws, call `refine_model_with_feedback`, then regenerate code with `get_code`.
   - If the model passes but code fails, call `refine_code_with_feedback`.
5. Final deterministic and LLM review after refinement.

The review output is normalized into a Python dict with:

```python
{
    "status": "PASS" | "NEEDS_REFINEMENT" | "FAILED",
    "critical_flaws": [],
    "major_flaws": [],
    "minor_flaws": [],
    "recommended_action": "none" | "refine_model" | "refine_code",
    "confidence": "Low" | "Medium" | "High"
}
```

## Deterministic Code Check

`deterministic_code_check`:

- Extracts a fenced Python block if present.
- Accepts raw code if it contains `gp.Model`, `gurobipy`, or `from gurobipy`.
- Runs `ast.parse` for syntax checking.
- Executes the generated code when `ORCH_ENABLE_CODE_EXECUTION = True`.
- Searches the execution environment for a `gurobipy.Model`.
- Optimizes the model if it is still in `GRB.LOADED`.
- Records model status and objective value when optimal.

This execution path uses Python `exec` and should only be used for trusted local experiment outputs.

## Output CSV

The Large-Scale-OR run writes:

```text
Large-scale-or-Lean-Ollama-20b-orchestrated.csv
```

Columns:

- `Query`
- `model_output_initial`
- `code_output_initial`
- `model_output_final`
- `code_output_final`
- `orchestrator_status`
- `orchestrator_report`
- `classification`

`orchestrator_report` is JSON text containing per-iteration model review, code review, deterministic check, and refinement count.

## Running The Pipeline

Recommended path:

```bash
cd "/Users/alanyu/Documents/CUHK/DOTE 6635/Project/lean-llm-opt-main"
./run_pure_ollama_large_scale.sh
```

Skip dependency installation if the Python environment is already ready:

```bash
SKIP_PIP_INSTALL=1 ./run_pure_ollama_large_scale.sh
```

Override models:

```bash
OLLAMA_LLM_MODEL=gpt-oss:20b \
OLLAMA_EMBED_MODEL=embeddinggemma \
./run_pure_ollama_large_scale.sh
```

The script will:

1. Check/install/start Ollama.
2. Pull the Ollama LLM and embedding models.
3. Install `requirements.txt` unless `SKIP_PIP_INSTALL=1`.
4. Create a run-only notebook ending at the Large-Scale-OR execution cell.
5. Execute the notebook with `jupyter nbconvert`.

## Static Tests

Run notebook and shell static checks:

```bash
cd "/Users/alanyu/Documents/CUHK/DOTE 6635/Project/lean-llm-opt-main"

python3 -Werror::SyntaxWarning - <<'PY'
import json
from pathlib import Path

for p in sorted(Path(".").rglob("*.ipynb")):
    nb = json.loads(p.read_text())
    for i, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{p}:cell-{i}", "exec")
print("all notebooks strict compile OK")
PY

bash -n run_pure_ollama_large_scale.sh
```

Check that the pure Ollama notebook has no OpenAI runtime dependency:

```bash
rg -n "OpenAIEmbeddings|ChatOpenAI|openai_api_key|user_api_key|import openai|openai\\.|classify_problem\\s*=\\s*llm1" \
  LEAN_LLM_OPT_pure_ollama_Large-scale-or.ipynb \
  run_pure_ollama_large_scale.sh
```

Expected result: no matches.

## Smoke Test

For a fast smoke test, temporarily run only the first two rows:

```python
test = pd.read_csv("Test_Dataset/Large-scale-or/Large-scale-or-101.csv").head(2)
(
    model_output_initial,
    code_output_initial,
    model_output_final,
    code_output_final,
    orchestrator_status,
    orchestrator_report,
    classification,
) = run_test(test, classify_problem, return_orchestrator_details=True)
```

Expected checks:

- Ollama responds for both generation and embedding.
- FAISS indexes are built.
- `run_test` returns lists with the same length as `test`.
- `orchestrator_status` contains only `PASS`, `NEEDS_REFINEMENT`, or `FAILED`.
- `orchestrator_report` is valid JSON text.

## Branch Tests

To cover major branches, select one row from each problem class when available:

- NRM
- RA
- TP
- FLP/UFLP
- AP
- Others with CSV
- Others without CSV

For each branch, confirm:

- The selected class routes to the expected `get_*_response` function.
- Initial model/code fields are populated.
- Final model/code fields are populated.
- The Orchestrator report contains at least one history entry.

## Targeted Orchestrator Tests

Test syntax-error code refinement:

```python
bad_code = "import gurobipy as gp\nm = gp.Model('x'\nm.optimize()"
result = orchestrate_generation(
    query="Maximize x subject to x <= 1.",
    selected_problem="Others without CSV",
    model_output="Maximize x subject to x <= 1, x >= 0.",
    code_output=bad_code,
)
print(result["status"])
print(result["report"])
```

Test model review on a deliberately incomplete model:

```python
review = orchestrator_review_model(
    query="Maximize profit with capacity limit 10 and demand limit 5.",
    selected_problem="Resource Allocation",
    model_output="Maximize profit.",
)
print(review)
```

## Full Run Evaluation

After running all 101 Large-Scale-OR instances, compare:

- `orchestrator_status` distribution.
- Gurobi executable rate from `deterministic_code_check` reports.
- Objective values in reports where available.
- Runtime before and after adding Orchestrator.
- Difference between `code_output_initial` and `code_output_final`.

## Requirements

The Orchestrator layer itself only adds standard-library imports: `ast`, `contextlib`, and `json`.

No new Python package is required beyond the existing pure Ollama notebook dependencies already present in `requirements.txt`, including:

- `langchain-ollama`
- `langchain-community`
- `langchain-classic`
- `langchain-core`
- `faiss-cpu`
- `gurobipy`
- `pandas`
- `numpy`
- `jupyter`
- `nbconvert`

Ollama itself is a system dependency, not a Python package. The run script handles installation/startup where possible.
