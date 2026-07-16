# OpenHands Setup for CI-Bench - Complete Guide

## What is OpenHands?

**OpenHands** is a full-featured AI agent platform with:
- **Python backend** (`openhands/` directory) - The agent logic
- **React frontend** (`frontend/` directory) - Web UI
- **Server components** - REST API for agents

**For CI-Bench**, we need to understand:
1. How OpenHands accepts tasks/problems
2. How to integrate with our CI failure dataset
3. How to use shared memory

---

## OpenHands Task Format

Based on the structure, OpenHands likely accepts tasks through its **server API**. Unlike mini-swe-agent which has direct CLI support for benchmarks, OpenHands is designed as a **server-based system**.

### How OpenHands Works

```
User/Script → API Request → OpenHands Server → Agent → Repository → Generate Fix
                                ↓
                          Sandbox Environment
```

---

## Setup Options

### Option 1: Simplified (What We're Doing for CI-Bench)

**Goal**: Use OpenHands agent logic without the full UI

**Approach**:
1. Extract the agent components we need
2. Create a direct Python runner
3. Integrate with CI-Bench dataset
4. Use shared memory

**Status**: ⬜ TODO - Needs implementation

### Option 2: Full Installation (For Testing/Understanding)

**Goal**: Run complete OpenHands to understand how it works

**Approach**:
1. Install full OpenHands
2. Use UI to test on single issues
3. Then adapt for batch processing

**Status**: Can try now

---

## Full OpenHands Installation (Option 2)

### Prerequisites

```bash
# Check versions
python3 --version  # Need 3.12+
node --version     # Need 22.12+

# If you don't have them:
# Python 3.12+
brew install python@3.12

# Node.js 22+
brew install node@22

# uv (Python package manager)
brew install uv
# or
pip install uv
```

### Step-by-Step Installation

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands
```

#### 1. Install Python Backend

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install with poetry (OpenHands uses poetry)
pip install poetry
poetry install
```

**Alternative if poetry fails:**
```bash
# Install directly (might have dependency conflicts)
pip install -e .
```

#### 2. Install Frontend (Optional - Only if you want UI)

```bash
cd frontend
npm install
npm run build
cd ..
```

#### 3. Build Everything

```bash
make build
```

#### 4. Run OpenHands Server

```bash
# Set environment variables
export INSTALL_DOCKER=0  # Don't require Docker
export RUNTIME=local     # Run locally

# Start server
make run FRONTEND_PORT=8000 BACKEND_HOST=0.0.0.0
```

Access at: http://localhost:8000

---

## Understanding OpenHands Task Format

OpenHands accepts tasks through its **API**. Let's understand the format:

### Typical OpenHands Task Structure

```json
{
  "task": "Fix the CI failure in this repository",
  "repository": "https://github.com/owner/repo",
  "base_commit": "abc123",
  "context": {
    "error_log": "...",
    "failed_tests": ["test1", "test2"],
    "instructions": "..."
  }
}
```

### How This Maps to Our CI-Bench Dataset

Our dataset (`data/trs/eval_set.jsonl`) has:
```json
{
  "instance_id": "owner__repo__sha_fail",
  "sha_fail": "abc123",
  "repo": "owner/repo",
  "problem_statement": "...",
  "validation_type": "...",
  "validation_command": "..."
}
```

We need to **convert** our format → OpenHands format!

---

## Creating CI-Bench Adapter for OpenHands

### Approach 1: Direct Python API (Recommended)

Instead of going through the web UI/API, directly use OpenHands Python modules:

```python
# Pseudocode - needs actual OpenHands imports
from openhands.server import Agent
from openhands.runtime import LocalRuntime

# Create agent
agent = Agent(model="glm", runtime=LocalRuntime())

# Load our CI-Bench issue
issue = load_ci_bench_issue("sha_fail_id")

# Add memory context
memory_prompt = format_memory_for_prompt(memory_context)

# Create task
task = {
    "instruction": f"{issue['problem_statement']}\n\n{memory_prompt}",
    "repository": issue["repo_url"],
    "base_commit": issue["sha_fail"]
}

# Run agent
result = agent.run(task)

# Save patch
save_patch(result["patch"], results_dir)
```

### Approach 2: HTTP API

Use OpenHands server API:

```python
import requests

# Start OpenHands server first
response = requests.post("http://localhost:8000/api/v1/agent/run", json={
    "task": task_description,
    "repo": repo_url,
    "commit": sha_fail
})

patch = response.json()["patch"]
```

---

## Step-by-Step: Implementing CI-Bench for OpenHands

### Phase 1: Understand OpenHands Agent (Current Phase)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands

# Explore the agent code
ls openhands/server/
ls openhands/

# Find agent entry points
find openhands -name "*agent*.py" | head -10
```

### Phase 2: Create Minimal Runner

1. **Study** how OpenHands runs agents
2. **Extract** the minimum code needed
3. **Create** `ci_bench_openhands_runner.py`
4. **Test** on 1 issue

### Phase 3: Add Memory Integration

1. **Format** L1/L2/L3 memory for OpenHands prompts
2. **Inject** into agent context
3. **Test** memory retrieval works

### Phase 4: Batch Processing

1. **Loop** over eval_set.jsonl
2. **Run** each issue
3. **Save** results in shared results/ directory

---

## Recommended Immediate Next Steps

### Step 1: Install OpenHands (Try Full Install First)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/openhands

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install poetry
poetry install
```

**Expected output:**
```
Installing dependencies from lock file
...
Package operations: 200+ installs
```

### Step 2: Explore OpenHands Code

```bash
# Find how tasks are defined
grep -r "class Task" openhands/ | head -5

# Find agent implementation
grep -r "class Agent" openhands/ | head -5

# Find how to run agents
grep -r "def run" openhands/server/ | head -10
```

### Step 3: Test with Simple Task

Create `test_openhands.py`:
```python
# Try to import OpenHands
try:
    import openhands
    print(f"✓ OpenHands version: {openhands.__version__}")
    
    # Try to find Agent class
    from openhands import server
    print(f"✓ Server module loaded")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
```

Run:
```bash
python test_openhands.py
```

### Step 4: Study One Example

Look for OpenHands examples or tests:
```bash
# Find example usage
find . -name "*example*.py" -o -name "*test*.py" | grep -v node_modules | head -10

# Read an example
cat tests/integration/<some_test>.py
```

---

## What to Ask/Research

To complete OpenHands integration, we need to find:

1. **How to create an agent instance**
   - Which class to import?
   - What parameters needed?

2. **How to run agent on a repository**
   - What's the task format?
   - How to specify repository + commit?

3. **How to get the generated patch**
   - What does agent.run() return?
   - How to extract the diff/patch?

4. **How to inject custom context**
   - Where to add our memory prompts?
   - Can we modify the system prompt?

---

## Comparison Table

| Feature | Mini-SWE-Agent | OpenHands |
|---------|----------------|-----------|
| **Installation** | ✅ Simple (pip install) | ⚠️ Complex (poetry + node) |
| **CLI Support** | ✅ Direct CLI | ⬜ Server-based |
| **Benchmark Integration** | ✅ Built-in (cibench command) | ⬜ Need adapter |
| **Memory Support** | ✅ Native | ⬜ Need to add |
| **Current Status** | ✅ Working | ⬜ Setup in progress |
| **Documentation** | ✅ Good | ⚠️ For web UI, not batch |
| **Best For** | Direct experimentation | Production UI/server |

---

## Recommendation

### For Your Thesis/Paper

**Priority 1**: Get Mini-SWE-Agent working completely
- ✅ It's ready to use
- ✅ Has memory integration
- ✅ Easy to run experiments

**Priority 2**: Add OpenHands later for comparison
- ⬜ More complex setup
- ⬜ Needs adapter implementation
- ⬜ Good for showing multi-agent comparison

### Timeline Suggestion

**Week 1-2**: Mini-SWE-Agent
- Run baseline
- Run L1, L1_L2, L1_L2_L3 ablations
- Test multiple models (MiniMax, GLM)
- Generate initial results

**Week 3-4**: Paper writing + OpenHands exploration
- Write up Mini-SWE-Agent results
- Meanwhile, study OpenHands code
- Create basic adapter

**Week 5+**: OpenHands experiments (if time)
- Run OpenHands baseline
- Compare with Mini-SWE-Agent
- Add to paper as multi-agent comparison

---

## Current Files Created

We've created:
1. `ci_bench_runner.py` - Template/placeholder for OpenHands runner
2. `OPENHANDS_QUICK_SETUP.md` - Quick reference
3. `OPENHANDS_DETAILED_SETUP.md` - This file (complete guide)

---

## Next Concrete Actions

### If You Want to Understand OpenHands:

```bash
cd openhands
source .venv/bin/activate
poetry install
python test_openhands.py
```

### If You Want to Focus on Results First:

```bash
# Go to mini-swe-agent instead!
cd ../miniswe-agent
source .venv/bin/activate

# Run your first experiment
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --slice 0:5 \
    --output ../results/miniswe-agent/minimax/test
```

---

## Summary

**OpenHands** is a powerful but complex system. For CI-Bench:

✅ **Do First**: Mini-SWE-Agent (it's ready!)  
⏳ **Do Later**: OpenHands (needs research & implementation)  
📊 **End Goal**: Compare both agents on same benchmark  

**OpenHands accepts tasks through**:
- API endpoints (when running as server)
- Direct Python calls (when used as library)

**We need to**:
1. Understand OpenHands agent API
2. Convert our dataset format
3. Integrate memory into prompts
4. Run and collect results

---

**Last Updated**: July 16, 2026  
**Status**: OpenHands exploration phase - mini-swe-agent recommended first
