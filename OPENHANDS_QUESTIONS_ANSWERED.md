# OpenHands: Your Questions Answered

## Q1: What kind of benchmark or issues does OpenHands accept?

### Answer:

OpenHands is a **general-purpose coding agent platform** that can work on any software development task. Unlike mini-swe-agent which has specific benchmark support built-in, OpenHands is more flexible but requires custom integration.

### What OpenHands Can Handle:

1. **Bug Fixes** - Given a bug description, generate a fix
2. **Feature Implementation** - Implement new features from description
3. **Code Refactoring** - Refactor code based on instructions
4. **Test Writing** - Generate tests
5. **Documentation** - Write/update documentation
6. **CI Failures** - Fix CI/CD pipeline issues (← **This is us!**)

### OpenHands is NOT Benchmark-Specific

Unlike mini-swe-agent which has:
```bash
mini-swe-agent cibench --dataset ...
mini-swe-agent swebench --dataset ...
```

OpenHands doesn't have built-in benchmark commands. It's designed as a **server/API** that accepts generic tasks.

---

## Q2: How does OpenHands accept problems/projects?

### Answer: Through Its API/Server Interface

OpenHands works as a **server-based system**. Here's how it accepts tasks:

### Method 1: Web UI (Interactive)

```bash
# Start OpenHands server
npm install -g @openhands/agent-canvas
agent-canvas

# Open browser: http://localhost:8000
# Paste task in UI
# Agent runs in background
```

**Task Input Format (via UI)**:
```
Task: "Fix the failing CI test in this repository"

Repository: https://github.com/owner/repo
Branch: main
Commit: abc123def

Additional Context:
- The test test_feature_x is failing
- Error: AssertionError in line 45
- Please fix the issue and ensure all tests pass
```

### Method 2: HTTP API (Programmatic)

```python
import requests

# Send task to OpenHands server
response = requests.post("http://localhost:8000/api/v1/agent/run", json={
    "task": "Fix CI failure",
    "repository": "https://github.com/owner/repo",
    "base_commit": "abc123",
    "context": {
        "error_log": "...",
        "additional_info": "..."
    }
})

result = response.json()
patch = result["patch"]
```

### Method 3: Python API (Direct - What We Need)

```python
# This is what we need to figure out for CI-Bench
from openhands.server import Agent  # (approximate - need to verify)

# Create agent
agent = Agent(model="glm")

# Run task
result = agent.run({
    "instruction": "Fix the CI failure...",
    "repository_url": "https://github.com/owner/repo",
    "base_commit": "abc123"
})

# Get patch
patch = result.get_patch()
```

---

## Q3: How do we adapt CI-Bench for OpenHands?

### Current Challenge:

**Our CI-Bench Format** (`data/trs/eval_set.jsonl`):
```json
{
  "instance_id": "owner__repo__sha_fail",
  "sha_fail": "abc123",
  "repo": "owner/repo",
  "problem_statement": "CI test is failing...",
  "validation_command": "pytest tests/",
  "base_sha": "def456"
}
```

**OpenHands Expected Format** (needs research, but likely):
```json
{
  "task": "CI test is failing...",
  "repository": "https://github.com/owner/repo",
  "commit": "abc123",
  "context": { ... }
}
```

### Solution: Create an Adapter

We've created `ci_bench_runner.py` which:
1. Reads CI-Bench format
2. Converts to OpenHands format
3. Runs OpenHands agent
4. Saves results to shared `results/` directory

**Current Status**: Template created, needs implementation

---

## Q4: How do we add memory to OpenHands?

### Answer: Inject into Task Prompt

Since OpenHands doesn't have built-in memory like mini-swe-agent, we need to **manually inject memory into the task description**:

```python
def create_task_with_memory(issue_data, memory_context):
    """
    Combine issue description + memory guidance
    """
    
    # Base problem
    problem = issue_data["problem_statement"]
    
    # Memory guidance
    memory_prompt = ""
    
    if memory_context:
        memory_prompt = "\n\n## Relevant Past Solutions\n\n"
        
        # L1: Similar failures
        if "failure_memory" in memory_context:
            memory_prompt += "### Similar CI Failures:\n"
            for mem in memory_context["failure_memory"]:
                memory_prompt += f"- {mem['description']}\n"
                memory_prompt += f"  Solution: {mem['solution']}\n\n"
        
        # L2: Repository patterns
        if "repo_memory" in memory_context:
            memory_prompt += "### Repository Conventions:\n"
            for mem in memory_context["repo_memory"]:
                memory_prompt += f"- {mem['pattern']}\n\n"
        
        # L3: Universal principles
        if "cross_memory" in memory_context:
            memory_prompt += "### General Best Practices:\n"
            for mem in memory_context["cross_memory"]:
                memory_prompt += f"- {mem['principle']}\n\n"
    
    # Combined prompt
    full_task = f"{problem}{memory_prompt}"
    
    return {
        "task": full_task,
        "repository": issue_data["repo_url"],
        "commit": issue_data["sha_fail"]
    }
```

---

## Q5: What's the difference between mini-swe-agent and OpenHands?

### Comparison:

| Aspect | Mini-SWE-Agent | OpenHands |
|--------|----------------|-----------|
| **Type** | CLI tool | Web server + API |
| **Installation** | Simple (Python only) | Complex (Node.js + Python) |
| **Benchmark Support** | ✅ Built-in (cibench, swebench) | ⬜ Need adapter |
| **Memory System** | ✅ Native support | ⬜ Manual injection |
| **Usage** | Command-line | Web UI or API |
| **Best For** | Research, experiments | Production, team usage |
| **Setup Time** | 5 minutes | 30+ minutes |
| **Current Status** | ✅ Ready | ⬜ Needs work |

### When to Use Which?

**Use Mini-SWE-Agent when:**
- ✅ You want quick results
- ✅ Running benchmark experiments
- ✅ Command-line is fine
- ✅ Research/thesis work

**Use OpenHands when:**
- ⬜ You need a web UI
- ⬜ Want to show comparisons in paper
- ⬜ Have time to implement adapter
- ⬜ Building production system

---

## Q6: Can we use both mini-swe-agent and OpenHands?

### Answer: YES! That's the whole point of our restructuring!

Our new structure supports **both agents** with **shared resources**:

```
mini-swe-agent-ci-based/
├── miniswe-agent/          # Agent 1 - Ready to use
├── openhands/              # Agent 2 - Setup in progress
├── data/trs/              # SHARED memory
├── results/
│   ├── miniswe-agent/     # Mini-SWE-Agent results
│   └── openhands/         # OpenHands results
└── scripts/               # SHARED evaluation
```

### Benefits:

1. **Compare Agents**: See which agent performs better
2. **Shared Memory**: Both use same L1/L2/L3 memory
3. **Fair Comparison**: Same dataset, same evaluation
4. **Paper Contribution**: "Multi-agent comparison on CI-repair"

---

## Q7: What should I do first?

### Recommended Path:

### Phase 1: Get Results with Mini-SWE-Agent ✅ (Week 1-2)

```bash
cd miniswe-agent

# Baseline
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --output ../results/miniswe-agent/minimax/baseline

# With Memory
python -m minisweagent cibench \
    --dataset ../data/trs/eval_set.jsonl \
    --model minimax \
    --memory-root ../data/trs \
    --memory-layers L1_L2_L3 \
    --output ../results/miniswe-agent/minimax/L1_L2_L3

# Evaluate
cd ../scripts
python evaluate_ablation_preds.py \
    ../results/miniswe-agent/minimax/baseline/preds.json \
    ../results/miniswe-agent/minimax/L1_L2_L3/preds.json
```

**Output**: You'll have results to write about!

### Phase 2: Study OpenHands (Week 2-3)

```bash
cd openhands

# Install
poetry install

# Explore code
grep -r "class Agent" openhands/
grep -r "def run" openhands/server/

# Understand how it works
```

### Phase 3: Implement OpenHands Adapter (Week 3-4)

```bash
# Complete ci_bench_runner.py
# Test on 1 issue
# Run on small batch
# Compare with mini-swe-agent
```

### Phase 4: Write Paper (Ongoing)

```
Section 1: Problem (CI failure repair at PR level)
Section 2: Method (Three-layer memory system)
Section 3: Experiments
  - Mini-SWE-Agent results (baseline vs memory)
  - Model comparison (MiniMax vs GLM)
  - [Optional] OpenHands comparison
Section 4: Analysis
  - Failure type analysis
  - Memory layer ablation
Section 5: Related Work (compare with SWE-bench)
```

---

## Summary: Quick Answers

1. **What benchmarks?** → Any coding task, but needs adapter for specific benchmarks
2. **How to accept problems?** → Via API/server, not direct CLI
3. **Current setup?** → In progress, template created
4. **Should I use it?** → Use mini-swe-agent first, OpenHands later for comparison
5. **When ready?** → After getting mini-swe-agent results

---

## Files to Read

1. `OPENHANDS_DETAILED_SETUP.md` - Complete setup instructions
2. `OPENHANDS_QUICK_SETUP.md` - Quick reference
3. `SETUP_GUIDE.md` - Both agents setup
4. `ci_bench_runner.py` - Template (needs implementation)

---

**TL;DR**: 
- OpenHands accepts tasks via API/server
- Needs adapter for CI-Bench
- Template created but not implemented
- **Recommendation**: Use mini-swe-agent first, add OpenHands later

---

**Last Updated**: July 16, 2026
