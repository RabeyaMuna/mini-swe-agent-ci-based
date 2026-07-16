# Setup Guide: Mini-SWE-Agent & OpenHands

Complete guide for setting up both agent scaffolds in the multi-agent benchmark framework.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure Overview](#project-structure-overview)
3. [Setup Mini-SWE-Agent](#setup-mini-swe-agent)
4. [Setup OpenHands](#setup-openhands)
5. [Shared Resources](#shared-resources)
6. [Running Experiments](#running-experiments)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher (3.12 recommended)
- **Git**: For cloning repositories
- **Disk Space**: ~10GB for dependencies and repos
- **RAM**: 8GB minimum, 16GB recommended

### API Keys

You'll need API keys for the models you want to test:

- **MiniMax**: OpenRouter API key
- **GLM**: Direct API key or OpenRouter
- **Kimi**: Direct API key or OpenRouter

---

## Project Structure Overview

```
mini-swe-agent-ci-based/
├── miniswe-agent/          # Agent 1: Your implementation
│   ├── src/
│   ├── tests/
│   ├── .venv/              # Isolated virtual environment
│   ├── pyproject.toml
│   └── README.md
│
├── openhands/              # Agent 2: OpenHands integration
│   ├── openhands/
│   ├── .venv/              # Separate virtual environment
│   └── pyproject.toml
│
├── data/                   # SHARED: datasets and memory
│   └── trs/               # Three-layer memory system
│
├── results/                # SHARED: experiment outputs
│   ├── miniswe-agent/
│   └── openhands/
│
├── repo/                   # SHARED: testbed repositories
└── scripts/                # SHARED: evaluation tools
```

---

## Setup Mini-SWE-Agent

### Step 1: Navigate to Directory

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/miniswe-agent
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Mini-SWE-Agent

```bash
pip install -e .
```

This will install:
- Core dependencies from `pyproject.toml`
- The `minisweagent` package in editable mode

### Step 5: Install Embedding Backend

For memory retrieval, install one of:

**Option A: sentence-transformers (Recommended)**
```bash
pip install sentence-transformers
```

**Option B: fastembed (Lightweight alternative)**
```bash
pip install fastembed
```

### Step 6: Configure API Keys

Create a `.env` file in the miniswe-agent directory:

```bash
cat > .env <<'EOF'
# MiniMax via OpenRouter
MINIMAX_API_KEY=your_openrouter_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
MINIMAX_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MEMCI_LLM_MODEL=minimax/minimax-m2.5

# GLM (if using)
GLM_API_KEY=your_glm_key_here

# Kimi (if using)
KIMI_API_KEY=your_kimi_key_here

# Python binary
PYTHON_BIN=.venv/bin/python
EOF
```

**Important**: Replace `your_openrouter_key_here` with your actual API key!

### Step 7: Verify Installation

```bash
# Test import
python -c "import minisweagent; print(f'✓ Version: {minisweagent.__version__}')"

# Test CLI
python -m minisweagent --help

# Test path configuration
python -c "
from minisweagent.config.paths import get_memory_root, get_results_dir
print(f'✓ Memory root: {get_memory_root()}')
print(f'✓ Results dir: {get_results_dir(\"miniswe-agent\", \"minimax\", \"baseline\")}')
"

# Test embedding backend
python -c "import sentence_transformers; print('✓ sentence-transformers installed')"
# or
python -c "import fastembed; print('✓ fastembed installed')"
```

### Step 8: Verify Shared Resources Access

```bash
# Check memory exists
ls ../data/trs/failure_memory.json
ls ../data/trs/repo_memory.json
ls ../data/trs/cross_memory.json

# Check results directory
ls ../results/miniswe-agent/
```

---

## Setup OpenHands

### Step 1: Navigate to OpenHands Directory

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands
```

### Step 2: Create Separate Virtual Environment

**Important**: OpenHands needs its own isolated environment!

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install OpenHands

Follow OpenHands official installation:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

**If there's a requirements.txt:**
```bash
pip install -r requirements.txt
```

### Step 4: Create CI-Bench Adapter

Create a file to bridge OpenHands to our benchmark:

```bash
cat > ci_bench_adapter.py <<'ADAPTER'
"""
CI-Bench Adapter for OpenHands
Allows OpenHands to use shared memory and results structure.
"""

import sys
from pathlib import Path

# Add parent directory to path for shared utilities
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import shared paths (create this if needed)
class SharedPaths:
    """Paths to shared resources."""
    
    @staticmethod
    def get_memory_root():
        """Get shared memory directory."""
        return PROJECT_ROOT / "data" / "trs"
    
    @staticmethod
    def get_results_dir(agent_name: str, model_name: str, ablation_level: str):
        """Get results directory."""
        results_dir = PROJECT_ROOT / "results" / agent_name / model_name / ablation_level
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    @staticmethod
    def get_repo_dir(repo_identifier: str):
        """Get testbed repository directory."""
        return PROJECT_ROOT / "repo" / repo_identifier


def load_memory(issue_id: str, ablation_level: str = "L1_L2_L3"):
    """
    Load memory context for an issue.
    
    Args:
        issue_id: Issue identifier
        ablation_level: Memory layers to use (baseline, L1, L1_L2, L1_L2_L3)
    
    Returns:
        dict: Memory context or None if baseline
    """
    if ablation_level == "baseline":
        return None
    
    memory_root = SharedPaths.get_memory_root()
    
    # Load memory layers based on ablation level
    memory_context = {}
    
    if "L1" in ablation_level:
        # Load failure memory
        import json
        with open(memory_root / "failure_memory.json") as f:
            failure_memory = json.load(f)
            memory_context["failure_memory"] = failure_memory.get(issue_id, [])
    
    if "L2" in ablation_level:
        # Load repo memory
        import json
        with open(memory_root / "repo_memory.json") as f:
            repo_memory = json.load(f)
            memory_context["repo_memory"] = repo_memory.get(issue_id, [])
    
    if "L3" in ablation_level:
        # Load cross-repo memory
        import json
        with open(memory_root / "cross_memory.json") as f:
            cross_memory = json.load(f)
            memory_context["cross_memory"] = cross_memory.get(issue_id, [])
    
    return memory_context


def run_openhands_on_issue(
    issue_id: str,
    model_name: str = "glm",
    ablation_level: str = "baseline",
    **kwargs
):
    """
    Run OpenHands on a single CI failure issue.
    
    Args:
        issue_id: Issue identifier
        model_name: Model to use (glm, minimax, kimi)
        ablation_level: Memory ablation (baseline, L1, L1_L2, L1_L2_L3)
        **kwargs: Additional OpenHands parameters
    """
    # Get paths
    results_dir = SharedPaths.get_results_dir("openhands", model_name, ablation_level)
    
    # Load memory if not baseline
    memory_context = load_memory(issue_id, ablation_level)
    
    # TODO: Implement OpenHands execution
    # This is where you'd call OpenHands with:
    # - The issue data
    # - Memory context (if available)
    # - Model configuration
    # - Save results to results_dir
    
    print(f"Running OpenHands:")
    print(f"  Issue: {issue_id}")
    print(f"  Model: {model_name}")
    print(f"  Ablation: {ablation_level}")
    print(f"  Memory: {'Enabled' if memory_context else 'Disabled'}")
    print(f"  Results: {results_dir}")
    
    # Placeholder - implement actual OpenHands call
    raise NotImplementedError("OpenHands integration pending - add OpenHands execution here")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OpenHands on CI-Bench")
    parser.add_argument("--issue-id", required=True, help="Issue ID to process")
    parser.add_argument("--model", default="glm", choices=["glm", "minimax", "kimi"])
    parser.add_argument("--ablation", default="baseline", 
                       choices=["baseline", "L1", "L1_L2", "L1_L2_L3"])
    
    args = parser.parse_args()
    
    run_openhands_on_issue(
        issue_id=args.issue_id,
        model_name=args.model,
        ablation_level=args.ablation
    )
ADAPTER
```

### Step 5: Configure API Keys

Create `.env` file in openhands directory:

```bash
cat > .env <<'EOF'
# Model API keys
GLM_API_KEY=your_glm_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# OpenHands specific config
# Add any OpenHands-specific environment variables here
EOF
```

### Step 6: Verify OpenHands Installation

```bash
# Test OpenHands import
python -c "import openhands; print('✓ OpenHands installed')"

# Test adapter
python ci_bench_adapter.py --help
```

### Step 7: Verify Shared Access

```bash
# Test shared resources from OpenHands
python -c "
from ci_bench_adapter import SharedPaths
print(f'✓ Memory root: {SharedPaths.get_memory_root()}')
print(f'✓ Memory exists: {SharedPaths.get_memory_root().exists()}')
print(f'✓ Results dir: {SharedPaths.get_results_dir(\"openhands\", \"glm\", \"baseline\")}')
"
```

---

## Shared Resources

Both agents use these shared resources:

### 1. Memory System (`data/trs/`)

```bash
ls data/trs/
```

**Files:**
- `failure_memory.json` - L1: Similar CI failures
- `repo_memory.json` - L2: Repository patterns
- `cross_memory.json` - L3: Cross-repo failures
- `eval_set.jsonl` - Evaluation dataset
- `memory_set.jsonl` - Memory building dataset

### 2. Results Directory (`results/`)

**Structure:**
```
results/
├── miniswe-agent/
│   ├── minimax/{baseline,L1,L1_L2,L1_L2_L3}/
│   ├── glm/{baseline,L1_L2_L3}/
│   └── kimi/{baseline,L1_L2_L3}/
└── openhands/
    ├── glm/{baseline,L1_L2_L3}/
    └── minimax/{baseline,L1_L2_L3}/
```

### 3. Repository Cache (`repo/`)

Shared testbed repository clones:
```bash
ls repo/
# Shows: owner__reponame directories
```

### 4. Evaluation Scripts (`scripts/`)

Common analysis tools:
```bash
ls scripts/
```

**Key scripts:**
- `decompose_ci_failure.py` - Build memory
- `evaluate_ablation_preds.py` - Calculate metrics
- `compare_runs.py` - Compare experiments
- `run_eval.py` - Run evaluations

---

## Running Experiments

### Mini-SWE-Agent Experiments

#### Baseline (No Memory)

```bash
cd miniswe-agent
source .venv/bin/activate

python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --output ../results/miniswe-agent/minimax/baseline
```

#### With Memory (L1+L2+L3)

```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2_L3 \
    --output ../results/miniswe-agent/minimax/L1_L2_L3
```

#### Small Test (First 5 issues)

```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test
```

#### Ablation Studies

**L1 Only:**
```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1 \
    --output ../results/miniswe-agent/minimax/L1
```

**L1+L2:**
```bash
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2 \
    --output ../results/miniswe-agent/minimax/L1_L2
```

### OpenHands Experiments

#### Once Adapter is Complete

```bash
cd openhands
source .venv/bin/activate

# Baseline
python ci_bench_adapter.py \
    --issue-id <issue_id> \
    --model glm \
    --ablation baseline

# With memory
python ci_bench_adapter.py \
    --issue-id <issue_id> \
    --model glm \
    --ablation L1_L2_L3
```

### Evaluating Results

```bash
cd scripts

# Single run evaluation
python evaluate_ablation_preds.py \
    ../results/miniswe-agent/minimax/baseline/preds.json

# Compare two runs
python compare_runs.py \
    --baseline ../results/miniswe-agent/minimax/baseline \
    --memory ../results/miniswe-agent/minimax/L1_L2_L3

# Compare multiple models
python evaluate_ablation_preds.py \
    ../results/miniswe-agent/minimax/L1_L2_L3/preds.json \
    ../results/miniswe-agent/glm/L1_L2_L3/preds.json \
    ../results/openhands/glm/L1_L2_L3/preds.json
```

---

## Troubleshooting

### Mini-SWE-Agent Issues

#### Issue: `ModuleNotFoundError: No module named 'minisweagent'`

**Solution:**
```bash
cd miniswe-agent
source .venv/bin/activate
pip install -e .
```

#### Issue: `No embedding model available`

**Solution:**
```bash
pip install sentence-transformers
# or
pip install fastembed
```

**Verify:**
```bash
python -c "import sentence_transformers; print('✓ OK')"
```

#### Issue: `API key not found`

**Solution:**
```bash
# Check .env file exists
cat miniswe-agent/.env

# Verify API key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENROUTER_API_KEY'))"
```

#### Issue: `Path not found` errors

**Solution:**
```bash
# Verify you're in miniswe-agent directory
pwd
# Should show: .../mini-swe-agent-ci-based/miniswe-agent

# Test path configuration
python -c "
from minisweagent.config.paths import get_memory_root
print(get_memory_root())
print(get_memory_root().exists())
"
```

### OpenHands Issues

#### Issue: Import errors

**Solution:**
```bash
cd openhands
source .venv/bin/activate
pip install -e .
```

#### Issue: Adapter not working

**Solution:**
```bash
# Verify adapter exists
ls ci_bench_adapter.py

# Test paths
python -c "
from ci_bench_adapter import SharedPaths
print(SharedPaths.get_memory_root())
"
```

### Shared Resource Issues

#### Issue: Memory files not found

**Solution:**
```bash
# Check memory files exist
ls data/trs/

# Should see:
# - failure_memory.json
# - repo_memory.json
# - cross_memory.json
# - eval_set.jsonl
```

#### Issue: Results directory creation fails

**Solution:**
```bash
# Create manually
mkdir -p results/miniswe-agent/minimax/baseline
mkdir -p results/openhands/glm/baseline

# Or use path helpers (they auto-create)
python -c "
from minisweagent.config.paths import get_results_dir
print(get_results_dir('miniswe-agent', 'minimax', 'baseline'))
"
```

### General Issues

#### Issue: Python version mismatch

**Solution:**
```bash
# Check Python version
python --version
# Should be 3.10+

# If wrong, specify version:
python3.12 -m venv .venv
```

#### Issue: Virtual environment conflicts

**Solution:**
```bash
# Deactivate current venv
deactivate

# Remove and recreate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quick Reference

### Activate Virtual Environments

```bash
# Mini-SWE-Agent
cd miniswe-agent
source .venv/bin/activate

# OpenHands
cd openhands
source .venv/bin/activate
```

### Check Installation Status

```bash
# Mini-SWE-Agent
cd miniswe-agent
source .venv/bin/activate
python -c "import minisweagent; print('✓ Installed')"
python -m minisweagent --help

# OpenHands
cd openhands
source .venv/bin/activate
python -c "import openhands; print('✓ Installed')"
```

### Access Shared Resources

```bash
# From miniswe-agent
python -c "
from minisweagent.config.paths import get_memory_root
print(get_memory_root())
"

# From openhands
python -c "
from ci_bench_adapter import SharedPaths
print(SharedPaths.get_memory_root())
"
```

---

## Next Steps

### For Mini-SWE-Agent
1. ✅ Run small test (5 issues)
2. ✅ Verify memory retrieval works
3. ⬜ Run full baseline
4. ⬜ Run L1, L1_L2, L1_L2_L3 ablations
5. ⬜ Test with GLM model

### For OpenHands
1. ✅ Complete ci_bench_adapter.py implementation
2. ⬜ Test on single issue
3. ⬜ Run baseline experiments
4. ⬜ Compare with mini-swe-agent

### For Paper
1. ⬜ Collect dataset statistics
2. ⬜ Generate comparison tables
3. ⬜ Analyze failure types
4. ⬜ Create visualizations

---

## Support

For issues or questions:
- See: `RESTRUCTURING_SUMMARY.md` for migration details
- Check: `README.md` for project overview
- Review: Individual agent READMEs

**Last Updated**: July 16, 2026
