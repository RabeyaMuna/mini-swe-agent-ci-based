# Run Your Own Agent/Harness with Any Model

Your system already has everything! No Codex, no proxy needed!

## Quick Start

```bash
# Single issue, baseline
python3 scripts/run_eval.py --issue-ids 125 --ablation BASELINE

# Single issue, full memory (L1+L2+L3)
python3 scripts/run_eval.py --issue-ids 125 --ablation L1+L2+L3

# Multiple issues
python3 scripts/run_eval.py --issue-ids 125,126,127 --ablation L1+L2+L3

# All issues from file
python3 scripts/run_eval.py --issue-ids-file data/eval_issue_ids.json --ablation BASELINE

# Specific repos
python3 scripts/run_eval.py --repos flower,camel --ablation L1+L2 --max-issues 10
```

## Available Modes

- `BASELINE` - No memory
- `L1` - File-level memory
- `L1+L2` - File-level + sequences
- `L1+L2+L3` - Full memory

## Change Model

Your agent uses `utilities/llm_model.py` which already supports:
- **glm5.2** (default, uses `GLM_API_KEY` from .env)
- **minimax** (uses `OPENROUTER_API_KEY` from .env)  
- Any model in `utilities/model_registry.py`

The model is automatically picked up from your `.env` file!

## Compare Models

Run with different models by setting environment variable:

```bash
# Test with GLM 5.2
GLM_MODEL_NAME=zai/glm-5.2 python3 scripts/run_eval.py \
    --issue-ids 125 \
    --ablation BASELINE

# Test with Minimax
MINIMAX_MODEL_NAME=openrouter/minimax/minimax-m2.5 python3 scripts/run_eval.py \
    --issue-ids 125 \
    --ablation BASELINE
```

## Run All Modes for Comparison

```bash
# Baseline
python3 scripts/run_eval.py --issue-ids 125,126,127 --ablation BASELINE

# L1 only
python3 scripts/run_eval.py --issue-ids 125,126,127 --ablation L1

# L1+L2
python3 scripts/run_eval.py --issue-ids 125,126,127 --ablation L1+L2

# L1+L2+L3 (full)
python3 scripts/run_eval.py --issue-ids 125,126,127 --ablation L1+L2+L3
```

## Results Location

```
results/
├── BASELINE/
│   └── <sha>/
│       ├── predictions.json
│       └── run_instance.log
├── L1/
│   └── <sha>/
├── L1_L2/
│   └── <sha>/
└── L1_L2_L3/
    └── <sha>/
```

## Examples

```bash
# Single issue, baseline, GLM 5.2
python3 scripts/run_eval.py --issue-ids 125 --ablation BASELINE

# Multiple issues, full memory
python3 scripts/run_eval.py --issue-ids 125,126,127,128 --ablation L1+L2+L3

# All flower repo issues, exclude memory seed
python3 scripts/run_eval.py --repos flower --exclude-memory --ablation L1+L2+L3

# Dry run (see what would execute)
python3 scripts/run_eval.py --issue-ids 125 --ablation BASELINE --dry-run
```

## That's It!

**No Codex, no LiteLLM proxy needed!** Your agent already works with GLM/Minimax directly via `utilities/llm_model.py`! 🎉
