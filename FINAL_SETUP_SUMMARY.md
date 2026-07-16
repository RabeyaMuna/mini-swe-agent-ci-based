# Final Setup Summary - Multi-Agent CI-Bench Framework

**Date**: July 16, 2026  
**Branch**: `restructure-multi-agent`  
**Status**: ✅ Complete and documented

---

## 🎯 What We've Built

A **multi-agent, multi-model benchmark framework** for evaluating CI failure repair with memory-guided assistance.

### Structure

```
mini-swe-agent-ci-based/
│
├── .venv/                           # 🔧 ROOT - Shared Tools
│   └── Memory building, evaluation, analysis
│
├── miniswe-agent/                   # 🤖 Agent 1 - Ready
│   ├── .venv/                       # Isolated environment
│   ├── src/minisweagent/
│   ├── tests/
│   └── README.md
│
├── openhands/                       # 🤖 Agent 2 - Setup documented
│   ├── .venv/                       # Isolated environment
│   ├── openhands/
│   ├── ci_bench_runner.py           # Template
│   └── README.md
│
├── data/                            # 📊 SHARED - Datasets & Memory
│   └── trs/
│       ├── failure_memory.json      # L1: Similar failures
│       ├── repo_memory.json         # L2: Repo patterns
│       ├── cross_memory.json        # L3: Universal principles
│       └── eval_set.jsonl           # Evaluation dataset
│
├── results/                         # 📈 SHARED - Organized Results
│   ├── miniswe-agent/
│   │   ├── minimax/{baseline,L1,L1_L2,L1_L2_L3}/
│   │   ├── glm/{baseline,L1_L2_L3}/
│   │   └── kimi/{baseline,L1_L2_L3}/
│   └── openhands/
│       ├── glm/{baseline,L1_L2_L3}/
│       └── minimax/{baseline,L1_L2_L3}/
│
├── repo/                            # 🔄 SHARED - Testbed repos
├── scripts/                         # 🛠️ SHARED - Evaluation tools
│
└── [Documentation Files]            # 📚 Complete guides
```

---

## 📚 Complete Documentation Created

### Setup & Architecture

1. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** ⭐
   - Complete setup for both agents
   - API key configuration
   - Running experiments
   - Troubleshooting

2. **[VIRTUAL_ENVIRONMENT_SETUP.md](VIRTUAL_ENVIRONMENT_SETUP.md)** ⭐
   - 3-venv architecture explanation
   - Which venv for what task
   - Complete setup commands
   - Usage patterns

3. **[RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md)**
   - What changed and why
   - Migration details
   - Verification tests

### OpenHands Specific

4. **[OPENHANDS_QUESTIONS_ANSWERED.md](OPENHANDS_QUESTIONS_ANSWERED.md)** ⭐ **Start Here**
   - What benchmarks OpenHands accepts
   - How it accepts problems
   - Should I use it now?
   - Clear recommendations

5. **[OPENHANDS_DETAILED_SETUP.md](OPENHANDS_DETAILED_SETUP.md)**
   - Full installation guide
   - Architecture explanation
   - Implementation roadmap

6. **[OPENHANDS_QUICK_SETUP.md](OPENHANDS_QUICK_SETUP.md)**
   - Quick reference
   - Current status

### Git & Deployment

7. **[PUSH_INSTRUCTIONS.md](PUSH_INSTRUCTIONS.md)**
   - How to push to GitHub
   - Verification steps
   - Rollback instructions

---

## ✅ 3 Virtual Environment Architecture

### Why 3 Separate Venvs?

| Environment | Purpose | When to Use |
|-------------|---------|-------------|
| **ROOT `.venv/`** | Shared tools | Building memory, evaluation, plotting |
| **miniswe-agent/.venv/** | Agent 1 | Running mini-swe-agent experiments |
| **openhands/.venv/** | Agent 2 | Running OpenHands experiments |

### Benefits

✅ **Clean Separation**
- Shared tools independent of agents
- Agent dependencies isolated
- No version conflicts

✅ **Consistency**
- Memory built with same environment
- Evaluation uses same tools
- Reproducible results

✅ **Flexibility**
- Update agents independently
- Add new agents easily
- Upgrade tools without breaking

---

## 🚀 Quick Start Guide

### Step 1: Setup ROOT Environment (Shared Tools)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based

# Create ROOT venv
python3 -m venv .venv
source .venv/bin/activate

# Install shared tools
pip install sentence-transformers numpy pandas matplotlib jupyter
pip freeze > requirements-shared.txt

deactivate
```

### Step 2: Setup Mini-SWE-Agent

```bash
cd miniswe-agent

# Activate its venv (already exists)
source .venv/bin/activate

# Verify installation
python -m minisweagent --help

deactivate
```

### Step 3: Build Memory (Use ROOT venv)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Build memory
python scripts/decompose_ci_failure.py \
    --eval-issues data/trs/eval_issues.json \
    --output-dir data/trs

deactivate
```

### Step 4: Run First Experiment

```bash
cd miniswe-agent
source .venv/bin/activate

# Small test (5 issues)
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test

deactivate
```

### Step 5: Evaluate Results

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/test/preds.json

deactivate
```

---

## 📊 Experiment Workflow

### Full Experimental Pipeline

```bash
# 1. Build Memory (Once) - Use ROOT
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
python scripts/decompose_ci_failure.py --eval-issues data/trs/eval_issues.json --output-dir data/trs
deactivate

# 2. Run Mini-SWE-Agent Baseline
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --output ../results/miniswe-agent/minimax/baseline
deactivate

# 3. Run with Memory (L1+L2+L3)
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2_L3 \
    --output ../results/miniswe-agent/minimax/L1_L2_L3
deactivate

# 4. Evaluate - Use ROOT
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/baseline/preds.json \
    results/miniswe-agent/minimax/L1_L2_L3/preds.json
deactivate

# 5. Compare - Use ROOT
python scripts/compare_runs.py \
    --baseline results/miniswe-agent/minimax/baseline \
    --memory results/miniswe-agent/minimax/L1_L2_L3
```

---

## 🎓 Research Questions & Experiments

### Planned Experiments

| Experiment | Purpose | Agent | Models |
|------------|---------|-------|--------|
| **Baseline** | No memory performance | mini-swe-agent | MiniMax, GLM |
| **L1 Only** | Failure memory effect | mini-swe-agent | MiniMax |
| **L1+L2** | + Repo patterns | mini-swe-agent | MiniMax |
| **L1+L2+L3** | Full memory | mini-swe-agent | MiniMax, GLM |
| **OpenHands** | Agent comparison | openhands | GLM |

### Analysis Planned

1. **Memory Ablation** - Which layers help most?
2. **Model Comparison** - MiniMax vs GLM vs Kimi
3. **Agent Comparison** - mini-swe-agent vs OpenHands
4. **Failure Type Analysis** - Which issues benefit from memory?
5. **Dataset Statistics** - Characterize CI-Repair-Bench

---

## 📈 Expected Results Structure

After experiments, you'll have:

```
results/
├── miniswe-agent/
│   ├── minimax/
│   │   ├── baseline/
│   │   │   ├── preds.json              # Predictions
│   │   │   ├── cibench.log             # Run log
│   │   │   └── {sha}/                  # Individual trajectories
│   │   ├── L1/
│   │   ├── L1_L2/
│   │   └── L1_L2_L3/
│   └── glm/
│       ├── baseline/
│       └── L1_L2_L3/
└── openhands/
    └── glm/
        ├── baseline/
        └── L1_L2_L3/
```

---

## 🔧 Common Commands Reference

### Activate Correct Venv

```bash
# Shared tools (memory, evaluation)
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Mini-SWE-Agent
cd miniswe-agent
source .venv/bin/activate

# OpenHands
cd openhands
source .venv/bin/activate
```

### Check Which Venv is Active

```bash
which python
# Shows path to active Python

echo $VIRTUAL_ENV
# Shows active venv path
```

### Quick Test

```bash
# Test mini-swe-agent works
cd miniswe-agent
source .venv/bin/activate
python -c "from minisweagent.config.paths import get_memory_root; print(get_memory_root())"
# Should print: /Users/.../data/trs
```

---

## 📋 Git Status

### Current Branch: `restructure-multi-agent`

**Commits**:
1. Main restructuring (moved code to miniswe-agent/)
2. Restructuring summary
3. OpenHands fix + setup guide
4. Push instructions
5. OpenHands documentation (Q&A, detailed, quick)
6. 3-venv architecture

**Files Created**:
- ✅ 7 documentation files
- ✅ Path configuration
- ✅ CI-Bench runner template
- ✅ README updates

### To Push to GitHub

```bash
# Make sure you're on the branch
git branch
# Should show: * restructure-multi-agent

# Push
git push origin restructure-multi-agent

# Or use GitHub Desktop
```

---

## 📖 Documentation Index

### Essential Reading (Start Here)

1. **[VIRTUAL_ENVIRONMENT_SETUP.md](VIRTUAL_ENVIRONMENT_SETUP.md)** - Which venv for what
2. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions
3. **[OPENHANDS_QUESTIONS_ANSWERED.md](OPENHANDS_QUESTIONS_ANSWERED.md)** - OpenHands Q&A

### Reference Documents

4. [RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md) - What changed
5. [OPENHANDS_DETAILED_SETUP.md](OPENHANDS_DETAILED_SETUP.md) - Deep dive
6. [OPENHANDS_QUICK_SETUP.md](OPENHANDS_QUICK_SETUP.md) - Quick reference
7. [PUSH_INSTRUCTIONS.md](PUSH_INSTRUCTIONS.md) - Git workflow

### Code Files

- `miniswe-agent/src/minisweagent/config/paths.py` - Shared paths
- `openhands/ci_bench_runner.py` - OpenHands template

---

## ✅ Verification Checklist

Before running experiments, verify:

- [ ] Git backup created (tag exists)
- [ ] All 3 venvs created
- [ ] Mini-swe-agent installed and working
- [ ] Memory files exist in `data/trs/`
- [ ] Results directories created
- [ ] API keys configured

### Quick Verification

```bash
# 1. Check venvs exist
ls -d .venv miniswe-agent/.venv openhands/.venv

# 2. Test mini-swe-agent
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent --help
deactivate

# 3. Check memory
ls data/trs/*.json

# 4. Check results structure
ls -d results/*/
```

---

## 🎯 Recommended Next Steps

### This Week (Priority 1) ✅

**Goal**: Get working results with mini-swe-agent

```bash
# 1. Setup ROOT venv
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy pandas
deactivate

# 2. Run small test
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test
deactivate

# 3. Check results
cd ..
source .venv/bin/activate
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/test/preds.json
```

### Next Week (Priority 2)

**Goal**: Full experiments

- Run complete baseline
- Run L1, L1_L2, L1_L2_L3 ablations
- Test GLM model
- Generate comparison tables

### Future (Priority 3)

**Goal**: Add OpenHands

- Complete `ci_bench_runner.py`
- Run OpenHands experiments
- Compare agents
- Update paper

---

## 🏆 What Makes This Framework Special

### Multi-Agent Support ✅
- Clean separation of agents
- Shared resources
- Easy to add new agents

### Memory System ✅
- Three-layer hierarchy (L1, L2, L3)
- Built once, used by all
- Consistent across experiments

### Organized Results ✅
- Hierarchical structure
- Easy comparisons
- Clear naming

### Reproducibility ✅
- 3 isolated venvs
- Clear documentation
- Complete workflow

### Scalability ✅
- Add models easily
- Add agents easily
- Add ablation levels easily

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Structure** | ✅ Complete | 3-venv architecture |
| **Documentation** | ✅ Complete | 7 guides created |
| **Mini-SWE-Agent** | ✅ Ready | Can run now |
| **Memory System** | ✅ Ready | L1/L2/L3 built |
| **OpenHands** | ⬜ Template | Needs implementation |
| **Results** | ⬜ Pending | Run experiments |
| **Paper** | ⬜ Pending | After results |

---

## 🎓 For Your Thesis/Paper

### Key Contributions

1. **Problem**: CI failure repair at PR-level (multi-commit, multi-issue)
2. **Dataset**: CI-Repair-Bench (different from SWE-bench)
3. **Method**: Three-layer memory system
4. **Evaluation**: Multi-agent, multi-model comparison
5. **Analysis**: Failure type, ablation, model comparison

### Dataset Characteristics (To Collect)

- Number of PRs
- Commits per PR
- Files modified per PR
- Lines changed
- Failure types distribution
- Complexity metrics

### Baseline Comparisons

- vs. No memory (baseline)
- vs. Different models
- vs. Different agents
- vs. Different ablations

---

## 💡 Final Thoughts

You now have:
- ✅ Complete multi-agent framework
- ✅ Clean architecture
- ✅ Comprehensive documentation
- ✅ Ready to run experiments

**Next action**: Follow [VIRTUAL_ENVIRONMENT_SETUP.md](VIRTUAL_ENVIRONMENT_SETUP.md) to set up the 3 venvs, then run your first experiment!

---

**Last Updated**: July 16, 2026  
**Status**: ✅ Framework complete, ready for experiments  
**Branch**: `restructure-multi-agent`
