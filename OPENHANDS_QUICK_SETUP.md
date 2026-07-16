# OpenHands Quick Setup for CI-Bench

**For**: Running OpenHands on CI failure repair tasks with shared memory

---

## What is OpenHands?

OpenHands is a **Node.js + Python** application that provides a full UI for running coding agents. However, for our CI-Bench experiments, we don't need the full UI - we just need the agent backend.

## Current Status

⚠️ **OpenHands integration is IN PROGRESS**

We have:
- ✅ OpenHands source code cloned
- ✅ Shared resources configured (data/, results/, scripts/)
- ✅ CI-Bench runner template created (`ci_bench_runner.py`)
- ⬜ OpenHands agent backend installation (TODO)
- ⬜ CI-Bench adapter implementation (TODO)

---

## Quick Setup (Simplified - Python Only)

For CI-Bench, we only need the OpenHands **agent backend**, not the full UI.

### Step 1: Navigate to OpenHands Directory

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands
```

### Step 2: Check Python Version

```bash
python3 --version
# Should be 3.12 or 3.13
```

If you don't have Python 3.12+:
```bash
# macOS (using Homebrew)
brew install python@3.12

# Or download from python.org
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install OpenHands (Attempt 1 - Direct)

```bash
pip install --upgrade pip
pip install -e .
```

**If this fails** (likely due to poetry), try Attempt 2:

### Step 4: Install OpenHands (Attempt 2 - Using Poetry)

```bash
# Install poetry first
pip install poetry

# Install OpenHands
poetry install
```

### Step 5: Configure API Keys

Create `.env` file:

```bash
cat > .env <<'EOF'
# GLM (if using)
GLM_API_KEY=your_glm_key_here

# OpenRouter (for MiniMax, Kimi)
OPENROUTER_API_KEY=your_openrouter_key_here

# Anthropic (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here

# OpenAI (if using GPT)
OPENAI_API_KEY=your_openai_key_here
EOF
```

### Step 6: Test CI-Bench Runner

```bash
python ci_bench_runner.py --help
```

Expected output:
```
usage: ci_bench_runner.py [-h] --issue-id ISSUE_ID [--model {glm,minimax,kimi}]
                          [--ablation {baseline,L1,L1_L2,L1_L2_L3}] [--output OUTPUT]
```

---

## Alternative: Full OpenHands Setup (With UI)

If you want the full OpenHands experience:

### Prerequisites

- **Node.js**: 22.12.x or later
- **npm**: Comes with Node.js
- **uv**: Python package manager

### Install Node.js

```bash
# macOS
brew install node@22

# Or download from nodejs.org
```

### Install uv

```bash
pip install uv
```

### Install OpenHands Full

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands

# Install Node.js dependencies
npm install

# Run development server
npm run dev
```

Access UI at: http://localhost:8000

---

## Using OpenHands with CI-Bench

### Baseline (No Memory)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands
source .venv/bin/activate

python ci_bench_runner.py \
    --issue-id <sha_fail> \
    --model glm \
    --ablation baseline
```

### With Memory (L1+L2+L3)

```bash
python ci_bench_runner.py \
    --issue-id <sha_fail> \
    --model glm \
    --ablation L1_L2_L3
```

### Results Location

Results will be saved to:
```
../results/openhands/{model}/{ablation}/
```

Example:
```
../results/openhands/glm/baseline/
../results/openhands/glm/L1_L2_L3/
```

---

## Shared Resources

OpenHands uses the same shared resources as mini-swe-agent:

### Memory
```bash
ls ../data/trs/
```

Files:
- `failure_memory.json` - L1: Similar CI failures
- `repo_memory.json` - L2: Repository patterns
- `cross_memory.json` - L3: Cross-repo failures
- `eval_set.jsonl` - Evaluation dataset

### Testbed Repositories
```bash
ls ../repo/
```

### Evaluation Scripts
```bash
ls ../scripts/
```

---

## What Needs to be Implemented

The `ci_bench_runner.py` is currently a **placeholder**. To complete the integration:

### 1. Install OpenHands Agent Components

Figure out which parts of OpenHands we actually need:
- Agent core logic
- Repository sandbox setup
- LLM integration
- Patch generation

### 2. Implement `run_openhands_agent()` Function

In `ci_bench_runner.py`, implement:
```python
def run_openhands_agent(issue_data, memory_context, model_name, results_dir):
    # 1. Set up repository sandbox
    # 2. Construct prompt with issue + memory
    # 3. Run OpenHands agent
    # 4. Collect generated patch
    # 5. Save results
    pass
```

### 3. Test on Single Issue

```bash
# Get a test issue ID
head -1 ../data/trs/eval_set.jsonl | python -c "import sys, json; print(json.load(sys.stdin)['sha_fail'])"

# Run on that issue
python ci_bench_runner.py --issue-id <issue_id> --model glm --ablation baseline
```

### 4. Run Batch Evaluation

Once working, create a batch script similar to mini-swe-agent's runner.

---

## Troubleshooting

### Issue: `ModuleNotFoundError` when running `ci_bench_runner.py`

**Solution:**
```bash
# Make sure you're in openhands directory
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands

# Activate venv
source .venv/bin/activate

# Install OpenHands
pip install -e .
```

### Issue: Poetry installation fails

**Solution:**
```bash
# Try with pip directly
pip install --no-deps -e .

# Or install minimal dependencies
pip install fastapi anthropic google-genai httpx
```

### Issue: Can't load memory files

**Solution:**
```bash
# Verify shared memory exists
ls ../data/trs/failure_memory.json

# Test path resolution
python -c "
from ci_bench_runner import SharedPaths
print(SharedPaths.get_memory_root())
print(SharedPaths.get_memory_root().exists())
"
```

---

## Next Steps

### For Now (Recommended Approach)

**Focus on mini-swe-agent first:**

1. Get mini-swe-agent fully working
2. Run complete experiments with multiple models
3. Generate baseline results
4. THEN come back to OpenHands

### When Ready for OpenHands

1. Study OpenHands agent code to understand how it works
2. Extract just the agent logic (not the full UI)
3. Implement `ci_bench_runner.py` properly
4. Test on small dataset
5. Run full experiments

---

## Comparison: Mini-SWE-Agent vs OpenHands

| Aspect | Mini-SWE-Agent | OpenHands |
|--------|----------------|-----------|
| **Setup** | ✅ Simple (Python only) | ⚠️ Complex (Node.js + Python) |
| **Status** | ✅ Ready to use | ⬜ Needs implementation |
| **Memory Integration** | ✅ Built-in | ⬜ TODO |
| **CI-Bench** | ✅ Native support | ⬜ Needs adapter |

**Recommendation**: Start with mini-swe-agent, then add OpenHands later for comparison.

---

## Quick Reference

### Check Setup Status

```bash
# OpenHands
cd openhands
source .venv/bin/activate
python ci_bench_runner.py --help  # Should show usage

# Shared resources
ls ../data/trs/  # Should show memory files
ls ../results/openhands/  # Should exist
```

### Run Test

```bash
python ci_bench_runner.py \
    --issue-id test_id \
    --model glm \
    --ablation baseline
```

---

## Contact

For implementation help:
- Check: `ci_bench_runner.py` - Template/placeholder
- See: `SETUP_GUIDE.md` - Full multi-agent setup
- Review: OpenHands official docs at https://docs.openhands.dev/

**Status**: OpenHands integration is **in progress** - focus on mini-swe-agent first!

**Last Updated**: July 16, 2026
