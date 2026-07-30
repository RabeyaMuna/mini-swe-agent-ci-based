# Quick Reference Card

## 📋 TL;DR Commands

### Complete Workflow (Backward Decomposition)

```bash
# 1. Split dataset (once)
python scripts/split_before_decomposition.py --repos agno,flower,camel

# 2. Decompose memory
MODEL=minimax2.5 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model minimax2.5

# 3. Build memory
MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/back_trs/decomposed_issues.json \
  --output-dir data/back_trs \
  --model minimax2.5

# 4. Evaluate (Mini-SWE-Agent)
export PYTHONPATH="$PWD/miniswe-agent/src:$PWD"
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3
```

---

## 📁 File Paths

| File | Path | Purpose |
|------|------|---------|
| Eval issues | `data/eval_set.jsonl` | Issues for evaluation |
| Memory issues | `data/memory_set.jsonl` | Issues for memory building |
| Backward decomposed | `data/back_trs/decomposed_issues.json` | Decomposed memory (backward) |
| Forward decomposed | `data/fwr_trs/commit_decomposed_issue.json` | Decomposed memory (forward) |
| L1 memory (backward) | `data/back_trs/failure_memory.json` | File-level memory |
| L2 memory (backward) | `data/back_trs/repo_memory.json` | Sequence memory |
| L3 memory (backward) | `data/back_trs/cross_memory.json` | Pattern memory |
| Split metadata | `data/split_metadata.json` | Temporal safety info |

---

## 🔧 Common Commands

### Split Dataset
```bash
# All repos
python scripts/split_before_decomposition.py

# Specific repos
python scripts/split_before_decomposition.py --repos agno,flower,camel

# Custom ratio
python scripts/split_before_decomposition.py --repos agno --memory-ratio 0.2
```

### Decompose (Backward)
```bash
# MiniMax
MODEL=minimax2.5 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model minimax2.5

# GLM
MODEL=glm5.2 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model glm5.2
```

### Decompose (Forward/Commit-Based)
```bash
# MiniMax
MODEL=minimax2.5 python commit_decomposition/run_commit_decomposition.py \
  --dataset data/memory_set.jsonl \
  --output data/fwr_trs/commit_decomposed_issue.json \
  --model minimax2.5

# GLM
MODEL=glm5.2 python commit_decomposition/run_commit_decomposition.py \
  --dataset data/memory_set.jsonl \
  --output data/fwr_trs/commit_decomposed_issue.json \
  --model glm5.2
```

### Build Memory
```bash
# Backward
MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/back_trs/decomposed_issues.json \
  --output-dir data/back_trs \
  --model minimax2.5

# Forward
MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/fwr_trs/commit_decomposed_issue.json \
  --output-dir data/fwr_trs \
  --model minimax2.5
```

### Evaluate (Mini-SWE-Agent)
```bash
export PYTHONPATH="$PWD/miniswe-agent/src:$PWD"

# Baseline
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5

# With memory (backward)
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3

# With memory (forward)
MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3
```

### Evaluate (OpenHands)
```bash
# Baseline
MODEL=minimax2.5 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --model minimax2.5 \
  --mode baseline

# With memory (backward)
MODEL=minimax2.5 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --decomposed-issues data/back_trs/decomposed_issues.json \
  --model minimax2.5 \
  --mode memory \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3

# With memory (forward)
MODEL=minimax2.5 openhands/.venv/bin/python openhands/ci_bench_runner.py \
  --eval-issues data/eval_set.jsonl \
  --decomposed-issues data/fwr_trs/commit_decomposed_issue.json \
  --model minimax2.5 \
  --mode memory \
  --memory-root data/fwr_trs \
  --memory-ablation L1+L2+L3
```

---

## 🎯 Model Selection

| Model | Value | API Key |
|-------|-------|---------|
| MiniMax M2.5 | `minimax2.5` | `OPENROUTER_API_KEY` |
| GLM 5.2 | `glm5.2` | `GLM_API_KEY` |

---

## 🧪 Ablation Levels

| Level | Flag | What's Included |
|-------|------|-----------------|
| Baseline | (none) | No memory |
| L1 | `--memory-ablation L1` | File-level memory only |
| L1+L2 | `--memory-ablation L1+L2` | File + sequence memory |
| L1+L2+L3 | `--memory-ablation L1+L2+L3` | Full memory |

---

## 🔍 Verification

```bash
# Check split files
ls -lh data/*.jsonl data/*.json

# Check backward decomposition
ls -lh data/back_trs/

# Check forward decomposition
ls -lh data/fwr_trs/

# Verify temporal safety
cat data/split_metadata.json | jq '.temporal_leakage_prevented'
# Should be: true

# Check split stats
cat data/split_metadata.json | jq '{total: .total_issues, memory: .memory_size, eval: .eval_size}'
```

---

## ⚠️ Common Mistakes

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `data/trs/eval_set.jsonl` | `data/eval_set.jsonl` |
| `data/trs/memory_set.jsonl` | `data/memory_set.jsonl` |
| `data/trs/decomposed_issues.json` | `data/back_trs/decomposed_issues.json` |
| `data/eval_issue_ids.json` | `data/eval_set.jsonl` |
| `--memory-root data/trs` | `--memory-root data/back_trs` |
| Split after decomposition | Split BEFORE decomposition |

---

## 🚀 Quick Start (Copy-Paste)

```bash
# Setup
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
export PYTHONPATH="$PWD/miniswe-agent/src:$PWD"

# Complete workflow
python scripts/split_before_decomposition.py --repos agno,flower,camel

MODEL=minimax2.5 python scripts/decompose_ci_failure.py \
  --dataset data/memory_set.jsonl \
  --output-dir data/back_trs \
  --output-file decomposed_issues.json \
  --model minimax2.5

MODEL=minimax2.5 python memory_plugin/ci_memory_llm_analysis.py \
  --decomposed data/back_trs/decomposed_issues.json \
  --output-dir data/back_trs \
  --model minimax2.5

MODEL=minimax2.5 miniswe-agent/.venv/bin/python -m minisweagent.run.benchmarks.cibench \
  --dataset data/eval_set.jsonl \
  --model minimax2.5 \
  --memory-enabled \
  --memory-root data/back_trs \
  --memory-ablation L1+L2+L3
```

---

## 📚 Full Documentation

- [README.md](README.md) - Complete setup and usage
- [DATA_ORGANIZATION.md](DATA_ORGANIZATION.md) - Data structure guide
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage guide

---

**Last updated:** 2026-07-24
