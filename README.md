# CI-Repair-Bench: Multi-Agent Evaluation Framework

Benchmark for evaluating agent scaffolds on CI failure repair with memory-guided repair at the pull-request level.

---

## 🚀 Quick Setup (Start Here!)

### Prerequisites

- **Python**: 3.10+ (3.12 recommended)
- **Git**: For repository management  
- **Disk Space**: ~10GB for dependencies
- **API Keys**: OpenRouter (for MiniMax), GLM, or other models

### One-Command Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/RabeyaMuna/mini-swe-agent-ci-based.git
cd mini-swe-agent-ci-based

# Run automated setup
bash setup_environments.sh
```

This creates **3 virtual environments**:
- `.venv/` - ROOT (shared tools: memory, evaluation, plotting)
- `miniswe-agent/.venv/` - Mini-SWE-Agent  
- `openhands/.venv/` - OpenHands (optional)

### Configure API Keys

```bash
cd miniswe-agent
cat > .env <<'EOF'
# MiniMax via OpenRouter
OPENROUTER_API_KEY=your_key_here
MINIMAX_API_KEY=your_key_here
MINIMAX_BASE_URL=https://openrouter.ai/api/v1
MEMCI_LLM_MODEL=minimax/minimax-m2.5

# GLM (optional)
GLM_API_KEY=your_glm_key_here

# Python binary
PYTHON_BIN=.venv/bin/python
EOF

# Edit with your actual API keys
nano .env
```

### Verify Installation

```bash
# Test ROOT environment (shared tools)
source .venv/bin/activate
python -c "import sentence_transformers, numpy, pandas; print('✓ ROOT OK')"
deactivate

# Test Mini-SWE-Agent
cd miniswe-agent
source .venv/bin/activate
mini --help  # Should show usage
deactivate
```

### Run Your First Experiment (5 Issues)

```bash
cd miniswe-agent
source .venv/bin/activate

# Quick test on 5 issues
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:5

# Results in: ../results/miniswe-agent/minimax/baseline/
```

**That's it!** You're ready to run experiments.

---

## 📖 Manual Setup (If Automated Fails)

### Step 1: ROOT Environment (Shared Tools)

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip (with SSL workaround if needed)
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install shared tools
pip install -r requirements-shared.txt

deactivate
```

**Installs**: sentence-transformers, numpy, pandas, matplotlib, jupyter

### Step 2: Mini-SWE-Agent Environment

```bash
cd miniswe-agent

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install agent
pip install -e .

# Install memory backend
pip install sentence-transformers

deactivate
cd ..
```

### Step 3: OpenHands Environment (Optional)

```bash
cd openhands

# Create environment (needs Python 3.12+)
python3.12 -m venv .venv
source .venv/bin/activate

# Install
pip install poetry
poetry install

deactivate
cd ..
```

---

## 🧪 Running Experiments

### Understanding Virtual Environments

**Which environment for what?**

| Task | Directory | Activate | Purpose |
|------|-----------|----------|---------|
| **Build memory** | Root | `source .venv/bin/activate` | Create L1/L2/L3 memory |
| **Run experiments** | `miniswe-agent/` | `source .venv/bin/activate` | Run Mini-SWE-Agent |
| **Evaluate results** | Root | `source .venv/bin/activate` | Calculate metrics, plots |
| **Run OpenHands** | `openhands/` | `source .venv/bin/activate` | Run OpenHands (optional) |

### Experiment 1: Quick Test (5 Issues)

```bash
cd miniswe-agent
source .venv/bin/activate

bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:5
```

### Experiment 2: Full Baseline (No Memory)

```bash
cd miniswe-agent
source .venv/bin/activate

bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline
```

**Results**: `../results/miniswe-agent/minimax/baseline/preds.json`

### Experiment 3: With Memory (L1+L2+L3)

```bash
cd miniswe-agent
source .venv/bin/activate

bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3
```

**Results**: `../results/miniswe-agent/minimax/L1_L2_L3/preds.json`

### Experiment 4: Ablation Studies

```bash
# Test each memory layer
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3
```

### Evaluating Results

```bash
# Return to root
cd /path/to/mini-swe-agent-ci-based

# Activate ROOT environment
source .venv/bin/activate

# Evaluate single run
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/baseline/preds.json

# Compare baseline vs memory
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/baseline/preds.json \
    results/miniswe-agent/minimax/L1_L2_L3/preds.json

# Detailed comparison
python scripts/compare_runs.py \
    --baseline results/miniswe-agent/minimax/baseline \
    --memory results/miniswe-agent/minimax/L1_L2_L3
```

---

## 🐛 Troubleshooting

### Issue: `pip` SSL Certificate Error

```bash
pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### Issue: Wrong Virtual Environment

Check which environment is active:
```bash
which python
# Should show correct .venv/bin/python path

# Fix: deactivate and reactivate from correct directory
deactivate
cd miniswe-agent  # or cd to root
source .venv/bin/activate
```

### Issue: `mini` Command Not Found

```bash
cd miniswe-agent
source .venv/bin/activate
which mini  # Should show: .../miniswe-agent/.venv/bin/mini
```

### Issue: Memory Retrieval Disabled

Logs show "No embedding model available":
```bash
# In miniswe-agent venv
pip install sentence-transformers
python -c "import sentence_transformers; print('OK')"
```

### Issue: Module Not Found

Each venv is isolated:
- ROOT venv: Has numpy, pandas, jupyter (NO minisweagent)
- miniswe-agent venv: Has minisweagent, mini command (NO jupyter)
- Make sure you're in the RIGHT venv for your task!

---

## 🏗️ Project Architecture

### Directory Structure

```
mini-swe-agent-ci-based/
├── .venv/                    # ROOT - Shared tools
│   └── (sentence-transformers, numpy, pandas, matplotlib, jupyter)
│
├── miniswe-agent/            # Agent 1 - Mini-SWE-Agent
│   ├── .venv/                # Isolated environment
│   ├── src/minisweagent/     # Source code
│   ├── tests/                # Tests
│   └── .env                  # API keys HERE
│
├── openhands/                # Agent 2 - OpenHands (optional)
│   └── .venv/                # Separate environment
│
├── data/trs/                 # SHARED - Three-layer memory
│   ├── failure_memory.json   # L1: Similar CI failures
│   ├── repo_memory.json      # L2: Repository patterns
│   ├── cross_memory.json     # L3: Universal principles
│   └── eval_set.jsonl        # Evaluation dataset
│
├── results/                  # SHARED - Organized results
│   ├── miniswe-agent/
│   │   └── {model}/{ablation}/
│   │       ├── preds.json    # Predictions
│   │       ├── cibench.log   # Run log
│   │       └── {sha}/        # Trajectories
│   └── openhands/
│       └── {model}/{ablation}/
│
├── repo/                     # SHARED - Testbed repositories
│   └── {owner}__{repo}/
│
└── scripts/                  # SHARED - Evaluation tools
    ├── decompose_ci_failure.py
    ├── evaluate_ablation_preds.py
    └── compare_runs.py
```

### Three Virtual Environments

**Why 3 separate venvs?**

1. **ROOT `.venv/`** - Shared tools (agent-agnostic)
   - Use for: Building memory, evaluation, plotting
   - Tools: sentence-transformers, numpy, pandas, matplotlib, jupyter

2. **miniswe-agent/.venv/** - Mini-SWE-Agent only
   - Use for: Running mini-swe-agent experiments
   - Tools: minisweagent package, litellm, typer

3. **openhands/.venv/** - OpenHands only  
   - Use for: Running OpenHands experiments (optional)
   - Tools: OpenHands package, fastapi, poetry

**Benefits**:
- No dependency conflicts
- Consistent memory building
- Isolated agent dependencies
- Clean, reproducible

---

## 🎯 Problem Statement

Unlike existing benchmarks (SWE-bench, SWE-bench Verified, SWE-bench Pro) that focus on resolving single, atomic issues, **CI-Repair-Bench targets CI failure repair at the pull-request level**, where:

- One PR may contain **multiple commits**
- Multiple **types of CI failures** (style checks, dependency issues, test failures, configuration errors)
- Multiple **verification stages** (linting, tests, builds, deployment checks)
- **Merge-related problems** (conflicts, integration issues)

This is closer to real-world software development workflows.

---

## 📊 Three-Layer Memory System

| Layer | Name | Description | Location |
|-------|------|-------------|----------|
| **L1** | Failure Memory | Similar CI failures from same repo | `data/trs/failure_memory.json` |
| **L2** | Repository Memory | Repository-specific patterns | `data/trs/repo_memory.json` |
| **L3** | Cross-Repository | Common failures across projects | `data/trs/cross_memory.json` |

Memory is retrieved **one problem at a time** to reduce hallucination.

**Ablation Levels**:
- `baseline` — No memory (0 layers)
- `L1` — Failure memory only (1 layer)
- `L1+L2` / `L1_L2` — Failure + Repository (2 layers)
- `L1+L2+L3` / `L1_L2_L3` — All three layers

---

## 📁 Results Organization

Results are automatically organized hierarchically for easy comparison:

```
results/
├── miniswe-agent/
│   ├── minimax/
│   │   ├── baseline/         # No memory
│   │   ├── L1/               # Failure memory only
│   │   ├── L1_L2/            # + Repo patterns
│   │   └── L1_L2_L3/         # Full memory
│   ├── glm/
│   │   ├── baseline/
│   │   └── L1_L2_L3/
│   └── kimi/
└── openhands/
    └── glm/
        ├── baseline/
        └── L1_L2_L3/
```

**Easy comparisons**:
- Same agent, different models: `results/miniswe-agent/{minimax,glm,kimi}/`
- Same model, different agents: `results/{miniswe-agent,openhands}/glm/`
- Ablation levels: `results/miniswe-agent/minimax/{baseline,L1,L1_L2,L1_L2_L3}/`

---

## 🔧 Advanced Usage

### Running on Server (Background)

```bash
cd miniswe-agent
source .venv/bin/activate

# Background execution with nohup
nohup bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline > baseline.log 2>&1 &
nohup bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3 > memory.log 2>&1 &

# Monitor progress
tail -f baseline.log
```

### Running Specific Issues

```bash
# Slice by index
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 10:20

# Filter by instance ID
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline --filter '^(abc123|def456)'
```

### Building Memory from Scratch

```bash
# Activate ROOT environment
source .venv/bin/activate

# Build memory layers
python scripts/decompose_ci_failure.py \
    --eval-issues data/trs/eval_issues.json \
    --output-dir data/trs

# Verify
ls data/trs/*.json
```

### Stop Background Jobs

```bash
pkill -f 'run_cibench_minimax_openrouter.sh'
pkill -f 'minisweagent.run.benchmarks.cibench'
```

---

## 📊 Expected Results

**Current Performance** (MiniMax model):
- Baseline (no memory): ~13.48%
- L1+L2+L3 (full memory): ~17.98%
- **Improvement**: +4.5 percentage points

**Your experiments will test**:
- Multiple models (MiniMax, GLM, Kimi)
- Multiple ablation levels (baseline, L1, L1_L2, L1_L2_L3)
- Multiple agents (mini-swe-agent, OpenHands)
- Different failure types

---

## 🎓 For Researchers

### Comparison with SWE-bench

| Aspect | SWE-bench | CI-Repair-Bench (This Work) |
|--------|-----------|------------------------------|
| **Scope** | Single issue | Pull request (multi-issue) |
| **Commits** | One patch | Multiple commits |
| **Verification** | Single test | Multi-stage CI (lint, test, build) |
| **Problem Types** | Code bugs | Style, config, deps, tests, merge |
| **Complexity** | Atomic | Compositional |
| **Agents Tested** | Single | Multi-agent comparison |

### Dataset Statistics to Collect

```bash
# Activate ROOT venv
source .venv/bin/activate

# Analyze dataset
python scripts/analyze_dataset.py data/trs/eval_set.jsonl
```

**Collect**:
- Number of PRs
- Commits per PR (distribution)
- Modified files per PR
- Lines changed per PR
- CI failure types (distribution)
- Multiple issues per PR

### Adding New Agents

To add another agent (e.g., AutoCodeRover, CodeAct):

1. Create directory: `mkdir newagent`
2. Create venv: `cd newagent && python3 -m venv .venv`
3. Install agent: `source .venv/bin/activate && pip install ...`
4. Access shared memory: `../data/trs/`
5. Save results: `../results/newagent/{model}/{ablation}/`

---

## 📖 Additional Documentation

For more details:
- `FINAL_SETUP_SUMMARY.md` - Complete framework overview
- `VIRTUAL_ENVIRONMENT_SETUP.md` - 3-venv architecture explained
- `OPENHANDS_QUESTIONS_ANSWERED.md` - OpenHands integration Q&A
- `START_HERE.md` - Quick navigation guide
- `miniswe-agent/README.md` - Agent-specific docs

---

## 🆘 Need Help?

**Common Issues**:
1. SSL errors → Use `--trusted-host` flag
2. Wrong venv → Check `which python`
3. Module not found → Verify correct venv is active
4. Memory disabled → Install `sentence-transformers`

**Getting Help**:
- GitHub Issues: https://github.com/RabeyaMuna/mini-swe-agent-ci-based/issues
- Email: rabeykhatunmuna@gmail.com

---

## 📄 License

See [LICENSE.md](LICENSE.md)

---

## 🤝 Contributing

This framework supports multi-agent, multi-model, multi-ablation experiments with:
- Shared memory and resources
- Isolated dependencies
- Fair comparisons
- Reproducible results

Contributions welcome! Fork, branch, change, and submit a PR.

---

**Last Updated**: July 16, 2026  
**Version**: 2.3.0  
**Status**: ✅ Production Ready

---

## ⚡ Quick Reference

### Commands Cheat Sheet

```bash
# Setup
bash setup_environments.sh

# Run experiment
cd miniswe-agent && source .venv/bin/activate
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:5

# Evaluate
cd .. && source .venv/bin/activate
python scripts/evaluate_ablation_preds.py results/miniswe-agent/minimax/baseline/preds.json

# Check which venv
which python
echo $VIRTUAL_ENV
```

### File Locations

- **API keys**: `miniswe-agent/.env`
- **Memory**: `data/trs/*.json`
- **Results**: `results/miniswe-agent/{model}/{ablation}/`
- **Evaluation scripts**: `scripts/`
- **Setup script**: `setup_environments.sh`
- **Shared requirements**: `requirements-shared.txt`
