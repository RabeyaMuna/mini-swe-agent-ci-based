# OpenHands Agent Algorithm

## What is OpenHands?

**OpenHands** (formerly OpenDevin) is an open-source AI software engineer platform that uses **CodeAct** - a framework for agents to act through code execution.

---

## Core Algorithm: CodeAct Framework

### Overview

OpenHands is based on the **CodeAct** paradigm, which consolidates LLM agents' **actions into a unified code action space**.

**Key Concept**: Instead of having separate tools for different tasks, CodeAct allows the agent to:
- Execute Python code
- Execute bash commands
- Interact with the environment through code

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│ 1. Task/Problem Given                                   │
│    "Fix the failing test in test_utils.py"             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Observation & Understanding                           │
│    - Read code                                          │
│    - Read error messages                                │
│    - Read test outputs                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Action Generation (CodeAct)                          │
│    Agent generates CODE to:                             │
│    - Explore the codebase (Python/bash)                │
│    - Run tests (Python/bash)                           │
│    - Make edits (Python/bash)                          │
│    - Verify fixes (Python/bash)                        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Code Execution in Sandbox                            │
│    - Safe, isolated environment                         │
│    - Docker container or VM                             │
│    - Captures stdout, stderr, return codes             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Observation of Results                                │
│    - Execution output                                   │
│    - Test results                                       │
│    - File changes                                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Loop Until Done or Max Steps                         │
│    Repeat 2-5 until:                                    │
│    - Task completed successfully                        │
│    - Max iterations reached                             │
│    - Agent decides to terminate                         │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Algorithm Steps

### Step 1: Task Understanding

```python
# Input: Task description
task = "Fix the failing CI test in repository X"

# Agent receives:
- Task description
- Repository access
- Initial state/context
```

### Step 2: Observation Phase

The agent can execute code to observe:

```python
# Example observation actions (in Python)
import os
import subprocess

# Read files
with open('test_utils.py') as f:
    content = f.read()

# Run tests
result = subprocess.run(['pytest', 'tests/'], capture_output=True)

# Check CI logs
ci_output = get_ci_logs()
```

### Step 3: Reasoning & Planning

Agent thinks:
- "What is the error?"
- "Which files are involved?"
- "What needs to be changed?"

This happens in the LLM's reasoning, not explicitly coded.

### Step 4: Action Execution (CodeAct)

Agent generates **executable code** as actions:

```python
# Example action: Edit file
code = '''
with open('utils.py', 'r') as f:
    content = f.read()

# Fix the bug
content = content.replace('old_code', 'new_code')

with open('utils.py', 'w') as f:
    f.write(content)
'''

# This code is executed in sandbox
execute(code)
```

Or in bash:

```bash
# Search for pattern
grep -r "function_name" src/

# Run specific test
pytest tests/test_utils.py::test_specific -v

# Check diff
git diff
```

### Step 5: Verification

```python
# Verify fix
result = subprocess.run(['pytest', 'tests/test_utils.py'], capture_output=True)

if result.returncode == 0:
    return "FIXED"
else:
    # Try different approach
    continue
```

### Step 6: Iteration

The agent loops through observe → act → verify until:
- Success (tests pass, CI green)
- Max iterations (typically 10-30)
- Explicit termination

---

## Key Differences from Mini-SWE-Agent

| Aspect | OpenHands (CodeAct) | Mini-SWE-Agent |
|--------|---------------------|----------------|
| **Action Space** | Unified code execution (Python + bash) | Predefined commands + bash |
| **Tool Use** | Everything through code | Specific tools (edit, search, etc.) |
| **Sandbox** | Docker/VM environment | Docker/local environment |
| **Interface** | Web UI + API | CLI-focused |
| **Flexibility** | Very flexible (any Python/bash) | Structured command set |
| **Complexity** | Higher (agent writes arbitrary code) | Lower (predefined actions) |

---

## CodeAct Advantages

1. **Unified Action Space**: No need to define many separate tools
2. **Flexibility**: Agent can do anything Python/bash can do
3. **Composability**: Can combine operations in complex ways
4. **Natural**: LLMs are good at generating code

## CodeAct Challenges

1. **Safety**: Arbitrary code execution needs strong sandboxing
2. **Debugging**: Harder to debug when agent writes complex code
3. **Error Handling**: More ways for things to go wrong
4. **Efficiency**: May take more steps than specialized tools

---

## For CI Failure Repair

### How OpenHands Would Approach CI Repair

```python
# Pseudocode of OpenHands approach

# 1. Understand the failure
ci_logs = read_ci_logs()
parse_errors(ci_logs)

# 2. Locate the problem
files = find_files_with_errors()
for file in files:
    read(file)
    analyze()

# 3. Generate fix
for file in files:
    code = generate_fix(file, error)
    apply_edit(file, code)

# 4. Verify
run_tests()
if tests_pass:
    return patch
else:
    rollback()
    try_different_approach()
```

### With Memory (Our Extension)

To add L1/L2/L3 memory to OpenHands:

```python
# Load memory
memory = load_memory_from_trs(issue_id, layers=['L1', 'L2', 'L3'])

# Inject into context
task_with_memory = f"""
{original_task}

## Relevant Past Solutions (Memory):
{memory['L1']}  # Similar failures
{memory['L2']}  # Repo patterns
{memory['L3']}  # Universal principles
"""

# Run OpenHands with enhanced context
openhands.run(task_with_memory)
```

---

## Research Paper

OpenHands is based on the **CodeAct** paper:

**"Executable Code Actions Elicit Better LLM Agents"**
- Authors: Xingyao Wang, et al.
- Key idea: Consolidate agent actions into executable code
- Shows better performance than tool-based approaches

---

## Comparison for CI-Bench

### Mini-SWE-Agent Approach
```
Task → Search files → Edit → Run tests → Verify
```

### OpenHands Approach
```python
Task → Execute Python/bash code to:
  - Search files (os.walk, grep)
  - Edit (file I/O)
  - Run tests (subprocess)
  - Verify (parse output)
All in one unified code action space
```

---

## Summary

**OpenHands Algorithm** = **CodeAct Framework**:
1. Observe through code execution
2. Reason about next step
3. Act through code generation & execution
4. Verify results
5. Loop until success/max steps

**For CI-Bench**: Would need adapter to:
- Load our CI failure dataset
- Inject L1/L2/L3 memory into prompts
- Run in their sandbox
- Collect patches and evaluate

**Key Insight**: OpenHands is more **flexible** but **complex**. Mini-SWE-Agent is more **structured** but **simpler**. Both can work for CI repair, but with different strengths.

---

## References

- OpenHands GitHub: https://github.com/OpenHands/OpenHands
- CodeAct Paper: https://arxiv.org/abs/2402.01030
- Documentation: https://docs.openhands.dev/

---

**Last Updated**: July 16, 2026
