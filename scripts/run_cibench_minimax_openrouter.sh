#!/bin/bash
# Run one Mini-SWE-Agent CI-Bench ablation with MiniMax 2.5 via OpenRouter.
# Run from project root or from miniswe-agent/.

set -e

if [ -d "miniswe-agent" ]; then
    PROJECT_ROOT="$(pwd)"
else
    PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

MODEL_INPUT="${MODEL:-minimax2.5}"
MODEL="$(
    MODEL_INPUT="$MODEL_INPUT" PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" python3 -c \
        'from scripts.model_registry import resolve_model_alias; import os; print(resolve_model_alias(os.environ["MODEL_INPUT"]))' \
        2>/dev/null
)"
MODEL="${MODEL:-$MODEL_INPUT}"
MODEL_OUTPUT_NAME="$(
    MODEL="$MODEL" PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" python3 -c \
        'from scripts.model_registry import model_output_name; import os; print(model_output_name(os.environ["MODEL"]))' \
        2>/dev/null
)"
MODEL_OUTPUT_NAME="${MODEL_OUTPUT_NAME:-minimax-m2.5}"
DATASET="${DATASET:-$PROJECT_ROOT/data/trs/eval_set.jsonl}"
MEMORY_ROOT="${MEMORY_ROOT:-$PROJECT_ROOT/data/trs}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/results/miniswe-agent/$MODEL_OUTPUT_NAME}"
ABLATION="baseline"
SLICE_SPEC=""
FILTER_SPEC=""
WORKERS="${WORKERS:-1}"
REDO_EXISTING=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_cibench_minimax_openrouter.sh [options]

Options:
  --ablation baseline|L1|L1+L2|L1+L2+L3
  --slice START:END
  --filter REGEX
  --workers N
  --redo-existing

Environment overrides:
  MODEL, DATASET, MEMORY_ROOT, OUTPUT_BASE, WORKERS
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ablation)
            ABLATION="$2"
            shift 2
            ;;
        --slice)
            SLICE_SPEC="$2"
            shift 2
            ;;
        --filter)
            FILTER_SPEC="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --redo-existing)
            REDO_EXISTING="--redo-existing"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

case "$ABLATION" in
    baseline)
        OUTPUT_DIR="$OUTPUT_BASE/baseline"
        MEMORY_ARGS=(--no-memory-enabled)
        ;;
    L1)
        OUTPUT_DIR="$OUTPUT_BASE/L1"
        MEMORY_ARGS=(--memory-enabled --memory-root "$MEMORY_ROOT" --memory-ablation "L1")
        ;;
    L1+L2|L1_L2)
        OUTPUT_DIR="$OUTPUT_BASE/L1_L2"
        MEMORY_ARGS=(--memory-enabled --memory-root "$MEMORY_ROOT" --memory-ablation "L1+L2")
        ;;
    L1+L2+L3|L1_L2_L3)
        OUTPUT_DIR="$OUTPUT_BASE/L1_L2_L3"
        MEMORY_ARGS=(--memory-enabled --memory-root "$MEMORY_ROOT" --memory-ablation "L1+L2+L3")
        ;;
    *)
        echo "Unsupported ablation: $ABLATION"
        usage
        exit 2
        ;;
esac

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET"
    exit 1
fi

PYTHON="$PROJECT_ROOT/miniswe-agent/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Mini-SWE venv not found. Run: bash setup_environments.sh"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/miniswe-agent/src:$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

CMD=(
    "$PYTHON" -m minisweagent.run.benchmarks.cibench
    --dataset "$DATASET"
    --model "$MODEL"
    --output "$OUTPUT_DIR"
    --workers "$WORKERS"
    "${MEMORY_ARGS[@]}"
)

if [ -n "$SLICE_SPEC" ]; then
    CMD+=(--slice "$SLICE_SPEC")
fi

if [ -n "$FILTER_SPEC" ]; then
    CMD+=(--filter "$FILTER_SPEC")
fi

if [ -n "$REDO_EXISTING" ]; then
    CMD+=("$REDO_EXISTING")
fi

echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Ablation: $ABLATION"
echo "Output: $OUTPUT_DIR"
echo "Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
