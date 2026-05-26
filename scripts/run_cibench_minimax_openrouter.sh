#!/usr/bin/env bash
# run_cibench_minimax_openrouter.sh
# ==================================
# Run cibench with MiniMax M2.5 via OpenRouter.
#
# Shared benchmark setup (run ONCE before any agent):
#   python scripts/prepare_shared_dataset.py   # → data/eval_dataset.jsonl + data/memory_seed.jsonl
#   python scripts/seed_memory.py              # → results/shared_memory/ (L1/L2/L3 bank)
#
# Usage:
#   scripts/run_cibench_minimax_openrouter.sh [--ablation LEVEL] [extra cibench args...]
#
# Ablation levels: baseline | L1 | L1+L2 | L1+L2+L3  (default: L1+L2+L3)
#
# Examples:
#   # Full run with memory (default)
#   scripts/run_cibench_minimax_openrouter.sh
#
#   # Baseline (no memory) — same eval issues, empty memory
#   scripts/run_cibench_minimax_openrouter.sh --ablation baseline
#
#   # Only first 10 instances
#   scripts/run_cibench_minimax_openrouter.sh --slice 0:10
#
# Environment variables (required):
#   MINIMAX_API_KEY   — your OpenRouter API key
#
# Environment variables (optional):
#   MINIMAX_BASE_URL  — defaults to https://openrouter.ai/api/v1
#   MEMCI_LLM_MODEL   — defaults to minimax/minimax-m2.5
#   PYTHON_BIN        — python launcher to use (default: python)
#   DATASET           — JSONL dataset (default: data/eval_dataset.jsonl — the shared TRS eval set)
#   MEMORY_ROOT       — shared memory directory  (default: results/shared_memory)
#   MSWEA_REPO_CACHE_ROOT — shared repo cache root (default: <project>/repo)
# ===========================================================================

set -euo pipefail

# ── Resolve project root (always run from there) ────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Load project .env if present ─────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "${PROJECT_ROOT}/.env"
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
: "${MINIMAX_API_KEY:?Set MINIMAX_API_KEY first (your OpenRouter API key)}"
: "${MINIMAX_BASE_URL:=https://openrouter.ai/api/v1}"
: "${MEMCI_LLM_MODEL:=minimax/minimax-m2.5}"
: "${PYTHON_BIN:=python}"
: "${DATASET:=data/eval_dataset.jsonl}"
: "${MEMORY_ROOT:=results/shared_memory}"

# ── Parse --ablation flag ─────────────────────────────────────────────────────
ABLATION="L1+L2+L3"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ablation)
      ABLATION="$2"; shift 2 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── Resolve output directory from ablation level ─────────────────────────────
case "${ABLATION}" in
  baseline)   OUTPUT_DIR="results/baseline" ;;
  L1)         OUTPUT_DIR="results/l1" ;;
  L1+L2)      OUTPUT_DIR="results/l1_l2" ;;
  L1+L2+L3)   OUTPUT_DIR="results/l1_l2_l3" ;;
  *)          OUTPUT_DIR="results/${ABLATION}" ;;
esac

# ── Validate dataset exists ──────────────────────────────────────────────────
if [[ ! -f "${DATASET}" ]]; then
  echo ""
  echo "ERROR: Dataset not found at '${DATASET}'"
  echo ""
  echo "Fetch it first:"
  echo "  pip install datasets huggingface_hub"
  echo "  python scripts/fetch_dataset.py"
  echo ""
  exit 1
fi

# ── Create output and memory directories ─────────────────────────────────────
mkdir -p "${OUTPUT_DIR}" "${MEMORY_ROOT}"

# ── Map MINIMAX_API_KEY → OPENROUTER_API_KEY ─────────────────────────────────
export OPENROUTER_API_KEY="${MINIMAX_API_KEY}"
export OPENROUTER_BASE_URL="${MINIMAX_BASE_URL}"

# ── Force PyTorch backend for sentence-transformers (avoids Keras 3 conflict) ─
export TRANSFORMERS_NO_TF=1

if [[ "${MINIMAX_BASE_URL}" != "https://openrouter.ai/api/v1" ]]; then
  echo "Warning: MINIMAX_BASE_URL=${MINIMAX_BASE_URL} (expected https://openrouter.ai/api/v1)" >&2
fi

# ── Print config ─────────────────────────────────────────────────────────────
echo "=============================================="
echo "  CI-REPAIR-BENCH run"
echo "  Dataset    : ${DATASET}"
echo "  Output     : ${OUTPUT_DIR}/"
echo "  Memory     : ${MEMORY_ROOT}/"
echo "  Ablation   : ${ABLATION}"
echo "  Model      : ${MEMCI_LLM_MODEL}"
echo "=============================================="
echo ""

# ── Build memory flags ────────────────────────────────────────────────────────
MEMORY_FLAGS=()
if [[ "${ABLATION}" == "baseline" ]]; then
  MEMORY_FLAGS+=(--no-memory-enabled --save-memory --memory-root "${MEMORY_ROOT}")
else
  MEMORY_FLAGS+=(--memory-enabled --memory-root "${MEMORY_ROOT}" --memory-ablation "${ABLATION}")
fi

# ── Run ───────────────────────────────────────────────────────────────────────
RUN_CMD=(
  "${PYTHON_BIN}" -m minisweagent.run.benchmarks.cibench
  --dataset "${DATASET}"
  --output "${OUTPUT_DIR}"
  -c src/minisweagent/config/benchmarks/cibench.yaml
  -c src/minisweagent/config/benchmarks/cibench_minimax_openrouter.yaml
  --context-model "${MEMCI_LLM_MODEL}"
)
RUN_CMD+=("${MEMORY_FLAGS[@]}")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  RUN_CMD+=("${EXTRA_ARGS[@]}")
fi
"${RUN_CMD[@]}"

echo ""
echo "Done. Results saved to:"
echo "  Predictions : ${OUTPUT_DIR}/preds.json"
echo "  Trajectories: ${OUTPUT_DIR}/<instance_id>/<instance_id>.traj.json"
echo "  Log         : ${OUTPUT_DIR}/cibench.log"
if [[ "${ABLATION}" != "baseline" ]]; then
  echo "  Memory L1   : ${MEMORY_ROOT}/failure_memory.json"
  echo "  Memory L2   : ${MEMORY_ROOT}/repo_memory.json"
  echo "  Memory L3   : ${MEMORY_ROOT}/cross_memory.json"
fi
