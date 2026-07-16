# CI-Repair-Bench: Multi-Agent Evaluation Framework

Benchmark for evaluating agent scaffolds on CI failure repair with memory-guided repair at the pull-request level.

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

| Layer | Name | Description |
|-------|------|-------------|
| **L1** | Failure Memory | Similar CI failures from the same repository |
| **L2** | Repository Memory | Repository-specific patterns and conventions |
| **L3** | Cross-Repository Memory | Common failures across different projects |

Memory is retrieved **one problem at a time** to reduce hallucination.

**Ablation Levels**:
- `baseline` — No memory
- `L1` — Failure memory only
- `L1_L2` — Failure + Repository memory
- `L1_L2_L3` — All three layers

---

## 🏗️ Project Structure

```
mini-swe-agent-ci-based/
├── .venv/                    # ROOT - Shared tools (memory, evaluation)
├── miniswe-agent/            # Agent 1 - Mini-SWE-Agent
│   ├── .venv/                # Isolated environment
│   ├── src/minisweagent/     # Source code
│   └── tests/                # Tests
├── openhands/                # Agent 2 - OpenHands (optional)
│   └── .venv/                # Isolated environment
├── data/trs/                 # SHARED - Three-layer memory
│   ├── failure_memory.json   # L1
│   ├── repo_memory.json      # L2
│   ├── cross_memory.json     # L3
│   └── eval_set.jsonl        # Evaluation dataset
├── results/                  # SHARED - Organized results
│   ├── miniswe-agent/
│   │   └── {model}/{ablation}/
│   └── openhands/
│       └── {model}/{ablation}/
├── repo/                     # SHARED - Testbed repositories
└── scripts/                  # SHARED - Evaluation tools
```

**Key Concept**: 3 isolated virtual environments allow multi-agent, multi-model experiments with shared resources but no dependency conflicts.

---

## 🚀 Complete Setup Guide

### Prerequisites

- **Python**: 3.10+ (3.12 recommended)
- **Git**: For repository management
- **API Keys**: OpenRouter (for MiniMax), GLM, or other model providers

### Step 1: Clone Repository

```bash
git clone https://github.com/RabeyaMuna/mini-swe-agent-ci-based.git
cd mini-swe-agent-ci-based
```

### Step 2: Automated Setup (Recommended)

```bash
bash setup_environments.sh
```

This creates all 3 virtual environments and installs dependencies.

**What it does**:
1. Creates ROOT `.venv/` with shared tools
2. Creates `miniswe-agent/.venv/` with Mini-SWE-Agent
3. Creates `openhands/.venv/` with OpenHands (optional)

### Step 3: Manual Setup (If Automated Fails)

#### 3.1 ROOT Environment (Shared Tools)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip (with SSL workaround if needed)
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install shared tools
pip install -r requirements-shared.txt

# Deactivate
deactivate
```

**Installed tools**:
- `sentence-transformers` - Memory retrieval
- `numpy`, `pandas` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `jupyter` - Analysis notebooks

#### 3.2 Mini-SWE-Agent Environment

```bash
# Navigate to agent directory
cd miniswe-agent

# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install mini-swe-agent
pip install -e .

# Install memory retrieval backend
pip install sentence-transformers

# Deactivate
deactivate

# Return to root
cd ..
```

#### 3.3 OpenHands Environment (Optional)

```bash
# Navigate to openhands directory
cd openhands

# Create virtual environment (needs Python 3.12+)
python3.12 -m venv .venv

# Activate
source .venv/bin/activate

# Install poetry
pip install poetry

# Install OpenHands
poetry install

# Deactivate
deactivate

# Return to root
cd ..
```

### Step 4: Configure API Keys

Create `.env` file in `miniswe-agent/` directory:

```bash
cd miniswe-agent
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

# Edit with your actual keys
nano .env
```

**Important**: Replace `your_openrouter_key_here` with your actual API key!

### Step 5: Verify Installation

```bash
# Test ROOT environment
source .venv/bin/activate
python -c "import sentence_transformers, numpy, pandas; print('✓ ROOT venv OK')"
deactivate

# Test Mini-SWE-Agent
cd miniswe-agent
source .venv/bin/activate
mini --help  # Should show usage
python -c "from minisweagent.config.paths import get_memory_root; print('✓ Paths OK:', get_memory_root())"
deactivate
cd ..
```

**Expected output**: No errors, confirmation messages

---

## 🧪 Running Experiments

### Understanding Virtual Environments

**Which venv to use?**

| Task | Directory | Command | Purpose |
|------|-----------|---------|---------|
| **Build memory** | Root | `source .venv/bin/activate` | Memory building (L1/L2/L3) |
| **Run experiments** | `miniswe-agent/` | `source .venv/bin/activate` | Run Mini-SWE-Agent |
| **Evaluate results** | Root | `source .venv/bin/activate` | Calculate metrics, plots |
| **Run OpenHands** | `openhands/` | `source .venv/bin/activate` | Run OpenHands (optional) |

### Experiment 1: Quick Test (5 Issues)

```bash
# Activate mini-swe-agent environment
cd miniswe-agent
source .venv/bin/activate

# Run baseline on 5 issues
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline --slice 0:5

# Results saved to: ../results/miniswe-agent/minimax/baseline/
```

### Experiment 2: Full Baseline

```bash
# Activate mini-swe-agent environment
cd miniswe-agent
source .venv/bin/activate

# Run full baseline (no memory)
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline
```

### Experiment 3: With Memory (L1+L2+L3)

```bash
# Activate mini-swe-agent environment
cd miniswe-agent
source .venv/bin/activate

# Run with full memory
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3
```

### Experiment 4: Ablation Studies

```bash
# L1 only
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1

# L1 + L2
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2

# L1 + L2 + L3
bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3
```

### Evaluating Results

```bash
# Return to root directory
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

# Compare multiple ablations
python scripts/compare_runs.py \
    --baseline results/miniswe-agent/minimax/baseline \
    --memory results/miniswe-agent/minimax/L1_L2_L3
```

---

## 📁 Results Organization

Results are automatically organized hierarchically:

```
results/
├── miniswe-agent/
│   ├── minimax/
│   │   ├── baseline/
│   │   │   ├── preds.json          # Predictions
│   │   │   ├── cibench.log         # Run log
│   │   │   └── {sha}/              # Individual trajectories
│   │   ├── L1/
│   │   ├── L1_L2/
│   │   └── L1_L2_L3/
│   ├── glm/
│   │   ├── baseline/
│   │   └── L1_L2_L3/
│   └── kimi/
└── openhands/
    └── glm/
        ├── baseline/
        └── L1_L2_L3/
```

This makes it easy to:
- Compare same agent, different models
- Compare same model, different agents
- Compare ablation levels

---

## 🔧 Advanced Usage

### Running on Server (Background)

```bash
# Use nohup for background execution
cd miniswe-agent
source .venv/bin/activate

nohup bash ../scripts/run_cibench_minimax_openrouter.sh --ablation baseline > baseline.log 2>&1 &
nohup bash ../scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2+L3 > memory.log 2>&1 &

# Monitor progress
tail -f baseline.log
```

### Running Specific Issues

```bash
# Filter by instance ID
bash ../scripts/run_cibench_minimax_openrouter.sh \
    --ablation baseline \
    --filter '^(abc123|def456)'

# Slice by index
bash ../scripts/run_cibench_minimax_openrouter.sh \
    --ablation baseline \
    --slice 10:20  # Issues 10-19
```

### Building Memory (Advanced)

If you need to rebuild memory from scratch:

```bash
# Activate ROOT environment
source .venv/bin/activate

# Build memory layers
python scripts/decompose_ci_failure.py \
    --eval-issues data/trs/eval_issues.json \
    --output-dir data/trs

# Verify memory created
ls data/trs/*.json
```

---

## 🐛 Troubleshooting

### Issue: `pip` SSL Certificate Error

**Solution**: Use trusted hosts flag:
```bash
pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### Issue: Wrong Virtual Environment Active

**Symptom**: `ModuleNotFoundError`

**Solution**: Check which venv is active:
```bash
which python
# Should show correct .venv/bin/python path

# If wrong, deactivate and reactivate
deactivate
source .venv/bin/activate  # From correct directory
```

### Issue: `mini` command not found

**Solution**: Make sure you're in miniswe-agent and venv is activated:
```bash
cd miniswe-agent
source .venv/bin/activate
which mini  # Should show: .../miniswe-agent/.venv/bin/mini
```

### Issue: Memory retrieval disabled

**Symptom**: Logs show "No embedding model available"

**Solution**: Install sentence-transformers:
```bash
# In miniswe-agent venv
pip install sentence-transformers

# Verify
python -c "import sentence_transformers; print('OK')"
```

### Issue: API key not found

**Solution**: Check `.env` file exists and has correct keys:
```bash
cat miniswe-agent/.env
# Should show API keys

# Re-edit if needed
nano miniswe-agent/.env
```

---

## 📊 Expected Results

After running experiments, you should see:

**Current Performance**:
- Baseline (no memory): ~13.48%
- L1+L2+L3 (full memory): ~17.98%
- **Improvement**: +4.5 percentage points

**Your experiments will**:
- Test multiple models (MiniMax, GLM, Kimi)
- Compare ablation levels (baseline, L1, L1_L2, L1_L2_L3)
- Analyze failure types
- Generate comparison tables

---

## 🎓 For Researchers

### Dataset Statistics

Collect these for your paper:
- Number of PRs
- Commits per PR
- Modified files per PR
- Lines changed per PR
- CI failure types distribution

```bash
# Activate ROOT venv
source .venv/bin/activate

# Run analysis
python scripts/analyze_dataset.py data/trs/eval_set.jsonl
```

### Comparing with SWE-bench

| Aspect | SWE-bench | CI-Repair-Bench |
|--------|-----------|-----------------|
| **Scope** | Single issue | Pull request (multi-issue) |
| **Commits** | One patch | Multiple commits |
| **Verification** | Single test | Multi-stage CI |
| **Problem Types** | Code bugs | Style, config, deps, tests, merge |
| **Complexity** | Atomic | Compositional |

### Adding New Agents

To add a new agent (e.g., CodeAct, AutoCodeRover):

1. Create directory: `mkdir newagent`
2. Create venv: `python3 -m venv newagent/.venv`
3. Install agent: `cd newagent && pip install ...`
4. Adapt to use shared memory from `../data/trs/`
5. Save results to `../results/newagent/{model}/{ablation}/`

---

## 📖 Additional Documentation

For more detailed information, see:
- `FINAL_SETUP_SUMMARY.md` - Complete framework overview
- `VIRTUAL_ENVIRONMENT_SETUP.md` - 3-venv architecture details
- `OPENHANDS_QUESTIONS_ANSWERED.md` - OpenHands integration Q&A
- `miniswe-agent/README.md` - Agent-specific docs

---

## 🤝 Contributing

This framework supports:
- Multiple agent scaffolds
- Multiple models
- Multiple ablation levels
- Shared memory and resources
- Fair comparisons

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

See [LICENSE.md](LICENSE.md)

---

## 📞 Contact

For questions or issues:
- GitHub Issues: https://github.com/RabeyaMuna/mini-swe-agent-ci-based/issues
- Email: rabeykhatunmuna@gmail.com

---

**Last Updated**: July 16, 2026  
**Version**: 2.3.0  
**Status**: Production Ready
