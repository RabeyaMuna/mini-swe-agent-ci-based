# 🚀 START HERE - Multi-Agent CI-Bench Framework

**Welcome!** This document will get you started quickly.

---

## ⚡ 30-Second Quick Start

```bash
# 1. Setup (automated)
bash setup_environments.sh

# 2. Test it works
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent --help

# 3. Run first experiment (5 issues)
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test
```

**That's it!** You'll have results in ~30 minutes.

---

## 📚 Documentation Map

### 🎯 **Start With These**

1. **[FINAL_SETUP_SUMMARY.md](FINAL_SETUP_SUMMARY.md)** ⭐ **Read This First!**
   - Complete framework overview
   - What we built and why
   - Quick start guide
   - Experiment workflows
   - **Everything you need to know**

2. **[VIRTUAL_ENVIRONMENT_SETUP.md](VIRTUAL_ENVIRONMENT_SETUP.md)** ⭐
   - Explains 3-venv architecture
   - Which venv for what task
   - Complete setup steps
   - Usage patterns

3. **[README.md](README.md)**
   - Project overview
   - Quick setup commands
   - Structure diagram

### 🔧 **Agent-Specific Guides**

#### Mini-SWE-Agent (Ready Now ✅)
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup for both agents
- `miniswe-agent/README.md` - Agent-specific docs

#### OpenHands (For Later ⏳)
- **[OPENHANDS_QUESTIONS_ANSWERED.md](OPENHANDS_QUESTIONS_ANSWERED.md)** ⭐ - Start here for OpenHands
- **[OPENHANDS_DETAILED_SETUP.md](OPENHANDS_DETAILED_SETUP.md)** - Deep dive
- **[OPENHANDS_QUICK_SETUP.md](OPENHANDS_QUICK_SETUP.md)** - Quick reference

### 📋 **Reference & Advanced**
- **[RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md)** - What changed
- **[PUSH_INSTRUCTIONS.md](PUSH_INSTRUCTIONS.md)** - Git workflow
- `setup_environments.sh` - Automated setup script
- `requirements-shared.txt` - ROOT venv dependencies

---

## 🎓 What Problem Are We Solving?

**CI-Repair-Bench** evaluates agents on **pull-request level CI failure repair**.

Unlike SWE-bench (one issue, one patch):
- ✅ Multiple commits per PR
- ✅ Multiple failure types (tests, lint, config, deps)
- ✅ Multi-stage CI verification
- ✅ Real-world complexity

**Memory System**: Three layers (L1/L2/L3) provide context from:
- Similar failures (L1)
- Repository patterns (L2)
- Universal principles (L3)

---

## 🏗️ What Did We Build?

### 3 Virtual Environments

| Environment | Purpose | Use For |
|-------------|---------|---------|
| **`.venv/`** (ROOT) | Shared tools | Memory building, evaluation, plots |
| **`miniswe-agent/.venv/`** | Agent 1 | Running mini-swe-agent |
| **`openhands/.venv/`** | Agent 2 | Running OpenHands |

### Multi-Agent Structure

```
mini-swe-agent-ci-based/
├── .venv/              # Shared tools
├── miniswe-agent/      # Agent 1 ✅
├── openhands/          # Agent 2 ⏳
├── data/trs/          # Shared memory
├── results/           # Organized by agent/model/ablation
└── scripts/           # Shared evaluation
```

### Complete Documentation

10 documents covering:
- Setup & architecture
- Usage patterns
- Troubleshooting
- Research workflows
- Git operations

---

## 🚀 Your Path to Results

### Week 1: Get Working Results ✅

**Goal**: Have results to analyze

```bash
# Day 1: Setup
bash setup_environments.sh

# Day 2-3: Run experiments
cd miniswe-agent
source .venv/bin/activate

# Baseline
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --output ../results/miniswe-agent/minimax/baseline

# With memory
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2_L3 \
    --output ../results/miniswe-agent/minimax/L1_L2_L3

# Day 4-5: Evaluate
cd ..
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/baseline/preds.json \
    results/miniswe-agent/minimax/L1_L2_L3/preds.json
```

**Output**: Success rates, comparison tables, analysis

### Week 2: More Models & Analysis

- Test GLM model
- Run ablation studies (L1, L1_L2, L1_L2_L3)
- Analyze failure types
- Generate plots

### Week 3+: OpenHands Comparison (Optional)

- Setup OpenHands
- Run comparison experiments
- Multi-agent analysis

---

## 🎯 Common Tasks

### Check Which Venv is Active

```bash
which python
# or
echo $VIRTUAL_ENV
```

### Switch Between Venvs

```bash
# Shared tools
cd /path/to/mini-swe-agent-ci-based
source .venv/bin/activate

# Mini-SWE-Agent
cd miniswe-agent
source .venv/bin/activate

# OpenHands
cd openhands
source .venv/bin/activate
```

### Test Installation

```bash
# Test mini-swe-agent
cd miniswe-agent
source .venv/bin/activate
python -c "from minisweagent.config.paths import get_memory_root; print(get_memory_root())"
# Should print: /path/to/mini-swe-agent-ci-based/data/trs
```

### Build Memory (One Time)

```bash
cd /path/to/mini-swe-agent-ci-based
source .venv/bin/activate
python scripts/decompose_ci_failure.py \
    --eval-issues data/trs/eval_issues.json \
    --output-dir data/trs
```

---

## 🆘 Troubleshooting

### Issue: "Which venv should I use?"

See [VIRTUAL_ENVIRONMENT_SETUP.md](VIRTUAL_ENVIRONMENT_SETUP.md) - Quick Reference table

### Issue: "Module not found"

Make sure you're in the right venv:
- Scripts → ROOT `.venv/`
- Mini-swe-agent → `miniswe-agent/.venv/`
- OpenHands → `openhands/.venv/`

### Issue: "Setup script fails"

Run manually:
1. Create venvs: `python3 -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install: `pip install -r requirements-shared.txt`

### Need More Help?

Read: [FINAL_SETUP_SUMMARY.md](FINAL_SETUP_SUMMARY.md) - Comprehensive guide

---

## 📊 Expected Results

After running experiments:

```
results/
└── miniswe-agent/
    └── minimax/
        ├── baseline/
        │   ├── preds.json          ← Predictions
        │   └── cibench.log         ← Run log
        └── L1_L2_L3/
            ├── preds.json
            └── cibench.log
```

Evaluate:
```bash
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py results/miniswe-agent/minimax/*/preds.json
```

---

## ✅ Checklist

Before running experiments:

- [ ] Read FINAL_SETUP_SUMMARY.md
- [ ] Run `bash setup_environments.sh`
- [ ] Test mini-swe-agent: `python -m minisweagent --help`
- [ ] Check memory exists: `ls data/trs/*.json`
- [ ] API keys configured (in miniswe-agent/.env)

---

## 🎓 For Your Paper

### Key Points

1. **Problem**: PR-level CI repair (more complex than SWE-bench)
2. **Method**: Three-layer memory system
3. **Evaluation**: Multi-agent, multi-model comparison
4. **Results**: Baseline vs Memory, Model comparison

### Experiments to Run

- [x] Setup framework ✅
- [ ] Baseline (no memory)
- [ ] L1 (failure memory)
- [ ] L1+L2 (+ repo patterns)
- [ ] L1+L2+L3 (full memory)
- [ ] Model comparison (MiniMax, GLM)
- [ ] Agent comparison (mini-swe-agent, OpenHands)

---

## 🚀 Ready to Start?

### Right Now (2 minutes)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
bash setup_environments.sh
```

### Then Read (10 minutes)

[FINAL_SETUP_SUMMARY.md](FINAL_SETUP_SUMMARY.md) - Everything you need to know

### Then Run (30 minutes)

```bash
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test
```

**You'll have results!** 🎉

---

## 📖 Complete Document Index

1. **START_HERE.md** ← You are here
2. **FINAL_SETUP_SUMMARY.md** ← Read next
3. VIRTUAL_ENVIRONMENT_SETUP.md
4. SETUP_GUIDE.md
5. README.md
6. OPENHANDS_QUESTIONS_ANSWERED.md
7. OPENHANDS_DETAILED_SETUP.md
8. OPENHANDS_QUICK_SETUP.md
9. RESTRUCTURING_SUMMARY.md
10. PUSH_INSTRUCTIONS.md

Plus:
- setup_environments.sh
- requirements-shared.txt

---

**Status**: ✅ Framework complete, documented, ready  
**Next**: Run `bash setup_environments.sh`  
**Goal**: Get results for your thesis!

---

**Last Updated**: July 16, 2026
