# Mini-SWE-Agent

Mini-SWE-Agent implementation for CI-Repair-Bench with memory-guided repair.

## Installation

```bash
cd miniswe-agent
pip install -e .
```

## Usage

### Baseline (No Memory)

```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --output ../results/miniswe-agent/minimax/baseline
```

### With Memory (L1+L2+L3)

```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2_L3 \
    --output ../results/miniswe-agent/minimax/L1_L2_L3
```

### Ablation Studies

**L1 Only (Failure Memory):**
```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1 \
    --output ../results/miniswe-agent/minimax/L1
```

**L1+L2 (Failure + Repository Memory):**
```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2 \
    --output ../results/miniswe-agent/minimax/L1_L2
```

## Shared Resources

This agent uses shared resources from the parent directory:

- **Memory**: `../data/trs/` - Shared memory built from historical CI failures
- **Results**: `../results/miniswe-agent/` - All experiment results
- **Testbed repos**: `../repo/` - Cloned test repositories
- **Evaluation scripts**: `../scripts/` - Analysis and evaluation tools

## Path Configuration

The agent uses centralized path management via `src/minisweagent/config/paths.py`:

```python
from minisweagent.config.paths import (
    get_memory_root,
    get_results_dir,
    get_repo_dir
)

# Get shared memory
memory_root = get_memory_root()  # ../data/trs

# Get results directory
results_dir = get_results_dir("miniswe-agent", "minimax", "L1_L2_L3")
# Returns: ../results/miniswe-agent/minimax/L1_L2_L3
```

## Project Structure

```
miniswe-agent/
├── src/minisweagent/     # Source code
├── tests/                # Tests
├── .venv/                # Virtual environment
├── pyproject.toml        # Dependencies
└── README.md             # This file
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run linting
ruff check src/
```

## See Also

- [Project Root README](../README.md) - Multi-agent benchmark overview
- [Evaluation Scripts](../scripts/) - Analysis tools
- [OpenHands Integration](../openhands/) - Alternative agent scaffold
