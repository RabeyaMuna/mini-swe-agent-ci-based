# Virtual Environment Setup - Multi-Agent Benchmark

**Recommended Structure**: 3 separate virtual environments for clean separation

---

## Architecture Overview

```
mini-swe-agent-ci-based/
├── .venv/                           # 1️⃣ ROOT - Shared Tools
│   ├── sentence-transformers        # Memory building
│   ├── numpy, pandas                # Data processing
│   ├── matplotlib, seaborn          # Visualization
│   └── jupyter                      # Analysis notebooks
│
├── miniswe-agent/
│   └── .venv/                       # 2️⃣ Agent 1
│       ├── minisweagent             # Agent code
│       ├── litellm                  # LLM interface
│       └── typer, rich              # CLI tools
│
├── openhands/
│   └── .venv/                       # 3️⃣ Agent 2
│       ├── openhands                # Agent code
│       ├── fastapi                  # Web server
│       └── anthropic, openai        # LLM clients
│
├── scripts/                         # Use ROOT .venv
│   ├── decompose_ci_failure.py      # Memory building
│   ├── evaluate_ablation_preds.py   # Evaluation
│   └── compare_runs.py              # Comparison
│
└── data/trs/                        # Built with ROOT .venv
```

---

## Why 3 Virtual Environments?

### 1️⃣ **ROOT `.venv/`** - Shared Tools (Agent-Agnostic)

**Purpose**: Memory building, evaluation, analysis

**When to use**:
- Building memory from CI failures
- Running evaluation scripts
- Comparing results across agents
- Generating plots and tables
- Data preprocessing

**Activate**:
```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
```

**Benefits**:
- ✅ Consistent environment for memory
- ✅ No agent-specific dependencies
- ✅ Same evaluation for all agents
- ✅ Clean, reproducible

---

### 2️⃣ **miniswe-agent/.venv/** - Mini-SWE-Agent Only

**Purpose**: Running Mini-SWE-Agent experiments

**When to use**:
- Running mini-swe-agent on issues
- Testing mini-swe-agent
- Developing mini-swe-agent code

**Activate**:
```bash
cd miniswe-agent
source .venv/bin/activate
```

**Benefits**:
- ✅ Isolated from OpenHands
- ✅ Agent-specific versions
- ✅ No conflicts

---

### 3️⃣ **openhands/.venv/** - OpenHands Only

**Purpose**: Running OpenHands experiments

**When to use**:
- Running OpenHands on issues
- Testing OpenHands
- Developing OpenHands adapter

**Activate**:
```bash
cd openhands
source .venv/bin/activate
```

**Benefits**:
- ✅ Isolated from mini-swe-agent
- ✅ Different Python version OK (3.12+)
- ✅ Complex dependencies isolated

---

## Complete Setup Guide

### Step 1: Create ROOT Virtual Environment

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based

# Create root venv (Python 3.10+ is fine)
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install shared tools
pip install sentence-transformers  # For memory building
pip install numpy pandas           # Data processing
pip install matplotlib seaborn     # Plotting
pip install jupyter notebook       # Analysis
pip install jsonlines              # JSON handling
pip install pyyaml                 # Config files
pip install tqdm                   # Progress bars

# Save requirements
pip freeze > requirements-shared.txt

echo "✓ ROOT environment setup complete"
deactivate
```

### Step 2: Create Mini-SWE-Agent Environment

```bash
cd miniswe-agent

# Create venv (already done in your case)
python3 -m venv .venv
source .venv/bin/activate

# Install mini-swe-agent
pip install -e .

# Install embedding backend for memory retrieval
pip install sentence-transformers

echo "✓ Mini-SWE-Agent environment setup complete"
deactivate
```

### Step 3: Create OpenHands Environment

```bash
cd ../openhands

# Create venv (needs Python 3.12+)
python3.12 -m venv .venv
source .venv/bin/activate

# Install OpenHands
pip install poetry
poetry install

echo "✓ OpenHands environment setup complete"
deactivate
```

---

## Usage Patterns

### Building Memory (Use ROOT .venv)

```bash
# Activate ROOT environment
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Build memory
python scripts/decompose_ci_failure.py \
    --eval-issues data/trs/eval_issues.json \
    --output-dir data/trs

# Verify memory files created
ls data/trs/*.json

deactivate
```

### Running Mini-SWE-Agent (Use miniswe-agent/.venv)

```bash
# Activate agent environment
cd miniswe-agent
source .venv/bin/activate

# Run experiment
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2_L3 \
    --output ../results/miniswe-agent/minimax/L1_L2_L3

deactivate
```

### Running OpenHands (Use openhands/.venv)

```bash
# Activate agent environment
cd openhands
source .venv/bin/activate

# Run experiment
python ci_bench_runner.py \
    --issue-id <sha> \
    --model glm \
    --ablation L1_L2_L3

deactivate
```

### Evaluating Results (Use ROOT .venv)

```bash
# Activate ROOT environment
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Evaluate
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/baseline/preds.json \
    results/miniswe-agent/minimax/L1_L2_L3/preds.json

# Compare agents
python scripts/compare_runs.py \
    --baseline results/miniswe-agent/glm/baseline \
    --memory results/miniswe-agent/glm/L1_L2_L3

deactivate
```

---

## Dependency Management

### ROOT `requirements-shared.txt`

```txt
# Memory Building
sentence-transformers==2.7.0
fastembed==0.8.0

# Data Processing
numpy==1.26.4
pandas==2.2.2
jsonlines==4.0.0

# Evaluation
scikit-learn==1.5.0
scipy==1.13.0

# Visualization
matplotlib==3.9.0
seaborn==0.13.2

# Utilities
tqdm==4.66.4
pyyaml==6.0.1
python-dotenv==1.0.1

# Analysis
jupyter==1.0.0
notebook==7.2.0
```

### miniswe-agent `requirements.txt`

*(Already exists in miniswe-agent/requirements.txt)*

### openhands `pyproject.toml`

*(Already exists in openhands/pyproject.toml)*

---

## .gitignore Updates

Add to `.gitignore`:

```gitignore
# Root virtual environment
/.venv/

# Agent virtual environments (already added)
miniswe-agent/.venv/
openhands/.venv/

# Python cache
__pycache__/
*.pyc
*.pyo
```

---

## Quick Reference

### Which venv for what?

| Task | Virtual Environment | Location |
|------|---------------------|----------|
| **Build memory** | ROOT `.venv/` | Project root |
| **Run mini-swe-agent** | `miniswe-agent/.venv/` | miniswe-agent/ |
| **Run OpenHands** | `openhands/.venv/` | openhands/ |
| **Evaluate results** | ROOT `.venv/` | Project root |
| **Compare agents** | ROOT `.venv/` | Project root |
| **Generate plots** | ROOT `.venv/` | Project root |
| **Run analysis notebooks** | ROOT `.venv/` | Project root |

### Activation Commands

```bash
# ROOT (shared tools)
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
# Shows:
# /.../mini-swe-agent-ci-based/.venv/bin/python           (ROOT)
# /.../miniswe-agent/.venv/bin/python                     (Agent 1)
# /.../openhands/.venv/bin/python                         (Agent 2)
```

---

## Common Workflows

### 1. First-Time Setup

```bash
# 1. Setup ROOT
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy pandas matplotlib jupyter
deactivate

# 2. Setup mini-swe-agent
cd miniswe-agent
source .venv/bin/activate
pip install -e .
deactivate

# 3. Setup OpenHands
cd ../openhands
python3.12 -m venv .venv
source .venv/bin/activate
poetry install
deactivate
```

### 2. Build Memory Once

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

python scripts/decompose_ci_failure.py \
    --eval-issues data/trs/eval_issues.json \
    --output-dir data/trs

deactivate
```

### 3. Run Experiments with Both Agents

```bash
# Mini-SWE-Agent experiment
cd miniswe-agent
source .venv/bin/activate
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --output ../results/miniswe-agent/minimax/baseline
deactivate

# OpenHands experiment
cd ../openhands
source .venv/bin/activate
python ci_bench_runner.py \
    --issue-id <sha> \
    --model glm \
    --ablation baseline
deactivate
```

### 4. Evaluate Both

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate

# Evaluate mini-swe-agent
python scripts/evaluate_ablation_preds.py \
    results/miniswe-agent/minimax/baseline/preds.json

# Evaluate OpenHands
python scripts/evaluate_ablation_preds.py \
    results/openhands/glm/baseline/preds.json

# Compare
python scripts/compare_runs.py \
    --miniswe results/miniswe-agent/minimax/baseline \
    --openhands results/openhands/glm/baseline

deactivate
```

---

## Troubleshooting

### Issue: "Which venv am I in?"

```bash
# Check active Python
which python

# Check pip location
which pip

# List installed packages
pip list
```

### Issue: Wrong venv activated

```bash
# Deactivate current
deactivate

# Activate correct one
source <correct-path>/.venv/bin/activate
```

### Issue: Scripts can't find modules

**Symptom**: `ModuleNotFoundError` when running scripts

**Solution**: Make sure you're using ROOT venv for scripts:
```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based
source .venv/bin/activate
python scripts/decompose_ci_failure.py ...
```

### Issue: Agent can't find minisweagent/openhands

**Solution**: Use the agent's venv, not ROOT:
```bash
# For mini-swe-agent
cd miniswe-agent
source .venv/bin/activate

# For OpenHands
cd openhands
source .venv/bin/activate
```

---

## Benefits Summary

✅ **Clean Separation**
- Shared tools independent of agents
- Agent dependencies isolated
- No version conflicts

✅ **Consistency**
- Memory built with same environment always
- Evaluation uses same tools for all agents
- Reproducible results

✅ **Flexibility**
- Can update agent dependencies independently
- Can add new agents without affecting existing
- Can upgrade shared tools without breaking agents

✅ **Clarity**
- Clear what depends on what
- Easy to document in paper
- Easy for others to reproduce

---

## Summary

**3 Virtual Environments**:
1. **ROOT** - Memory building, evaluation, analysis (agent-agnostic)
2. **miniswe-agent** - Running Mini-SWE-Agent experiments
3. **openhands** - Running OpenHands experiments

**Best Practice**:
- Build memory once with ROOT
- Run each agent in its own venv
- Evaluate all results with ROOT
- Keep dependencies separated and clean

---

**Last Updated**: July 16, 2026  
**Status**: Recommended structure for multi-agent benchmark
