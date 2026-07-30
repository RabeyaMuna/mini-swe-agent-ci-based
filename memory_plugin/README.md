# Memory Plugin - Shared CI-Bench Memory System

## Overview

This is the **complete memory plugin system** for CI-Bench, shared by BOTH mini-swe-agent and OpenHands.

**Moved from:** `miniswe-agent/src/minisweagent/run/benchmarks/utils/`
**To:** `memory_plugin/` (root directory)

**Why:** So both agents can import and use the same memory logic without duplication.

---

## Files

| File | Purpose |
|------|---------|
| `memory_plugin.py` | Core MemoryPlugin class - L1/L2/L3 retrieval, semantic similarity, top-K ranking |
| `ci_memory_system.py` | Integration system - wires memory into CI context pipeline |
| `ci_memory_llm_analysis.py` | LLM-based analysis for L1 problem identification |
| `ci_memory_l2_analysis.py` | L2 consecutive pattern analysis |
| `ci_memory_staged_analysis.py` | Staged L3 cross-repo analysis |
| `ci_memory_llm_analysis_multistage.py` | Multi-stage LLM analysis |

---

## Features

###  Three-Layer Memory

- **L1 (failure_memory.json)**: Similar CI failures from same repo
- **L2 (repo_memory.json)**: Repository-specific patterns
- **L3 (cross_memory.json)**: Cross-repo general principles

###  Top-K Retrieval

- Configurable top-K (default: 3, supports up to 10)
- Semantic similarity using sentence-transformers
- Cosine similarity ranking

###  Analysis & Selection

- Fetches top-10 candidates from each layer
- Analyzes for relevance, commonality, consecutiveness
- Selects most relevant CI failure patterns
- Organizes by source (L1/L2/L3)
- Generates repair plan

###  Ablation Support

- **Baseline**: No memory (layers=None)
- **L1**: Failure memory only
- **L1+L2**: Failure + Repository patterns
- **L1+L2+L3**: Full memory (all three layers)

###  Performance Optimizations

- Embedding caching (no re-embedding)
- Pre-loaded stored embeddings
- ChromaDB backend option

---

## Usage

### For Mini-SWE-Agent

```python
from memory_plugin import MemoryPlugin

# Configure
class Config:
    memory_enabled = True
    memory_top_k = 3
    memory_ablation_levels = "L1+L2+L3"  # or "L1" or "L1+L2" or "baseline"
    project_result_dir = "data/trs"

# Initialize
plugin = MemoryPlugin(config, result_dir="data/trs", llm=None)

# Retrieve
retrieved = plugin.retrieve(
    repo="pytest-dev/pytest",
    workflow=".github/workflows/test.yml",
    files=["tests/test_collection.py"],
    problem="CI test failing - Expected 1 test, found 0"
)

# retrieved is a list of ranked memory items from L1/L2/L3
```

### For OpenHands

```python
from memory_plugin import MemoryPlugin

# Same configuration
class Config:
    memory_enabled = True
    memory_top_k = 3
    memory_ablation_levels = "L1+L2+L3"
    project_result_dir = "../data/trs"

# Same initialization
plugin = MemoryPlugin(config, result_dir="../data/trs", llm=None)

# Same retrieval
retrieved = plugin.retrieve(
    repo=issue_data['repo'],
    workflow=issue_data.get('workflow', '.github/workflows'),
    files=issue_data.get('files', []),
    problem=issue_data['problem_statement']
)

# Format for OpenHands prompt
repair_plan = format_repair_plan(retrieved)
```

---

## Memory Retrieval Flow

```
1. Load Memory Files
   ├─ L1: failure_memory.json
   ├─ L2: repo_memory.json
   └─ L3: cross_memory.json

2. Build Query
   └─ repo + workflow + files + problem

3. Semantic Search
   ├─ Embed query
   ├─ Compute cosine similarity
   └─ Rank by similarity

4. Top-K Selection
   ├─ L1: Top 10 similar failures (same repo)
   ├─ L2: Top 10 patterns (same repo)
   └─ L3: Top 10 principles (cross-repo)

5. Analysis
   ├─ Select relevant items
   ├─ Identify common patterns
   ├─ Find consecutive issues
   └─ Organize by source

6. Generate Repair Plan
   ├─ From L1: Similar fixes
   ├─ From L2: Repository patterns
   └─ From L3: General strategies
```

---

## Configuration

### Ablation Levels

```python
# Baseline - No memory
config.memory_ablation_levels = "baseline"
config.memory_enabled = False

# L1 only - Similar failures
config.memory_ablation_levels = "L1"

# L1+L2 - Failures + Patterns
config.memory_ablation_levels = "L1+L2"

# L1+L2+L3 - Full memory
config.memory_ablation_levels = "L1+L2+L3"
```

### Top-K

```python
# Top 3 (default)
config.memory_top_k = 3

# Top 10 (maximum detail)
config.memory_top_k = 10
```

### Memory Location

```python
# Default: data/trs/
config.project_result_dir = "data/trs"

# Custom location
config.project_result_dir = "/path/to/memory"
```

---

## Integration

### Mini-SWE-Agent Integration

**File:** `miniswe-agent/src/minisweagent/run/benchmarks/cibench.py`

```python
# Import from root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from memory_plugin import MemoryPlugin

# Use it
memory = MemoryPlugin(config, result_dir, llm)
retrieved = memory.retrieve(...)
```

### OpenHands Integration

**File:** `openhands/ci_bench_runner.py`

```python
# Import from root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_plugin import MemoryPlugin

# Use it
memory = MemoryPlugin(config, result_dir, llm)
retrieved = memory.retrieve(...)
```

---

## Output Format

### Retrieved Memory Structure

```python
[
    {
        "memory_level": "L1",  # or "L2" or "L3"
        "similarity_score": 0.87,
        "repo": "pytest-dev/pytest",
        "problem": "test collection fails - missing __init__.py",
        "fixes": "Added __init__.py file to test directory",
        "why_fix": "pytest requires package structure for test discovery",
        "files": ["tests/test_collection.py"],
        # ... other fields
    },
    {
        "memory_level": "L2",
        "similarity_score": 0.75,
        "pattern": "makepyfile() requires __init__='' for package creation",
        # ... other fields
    },
    # ... more items
]
```

### Formatted Repair Plan

```
Based on previous experiences, consider these approaches:

**From Similar Past Failures (L1):**
1. Added __init__.py file to test directory
   (Similar issue: test collection fails - missing __init__.py)
2. makepyfile needs __init__='' parameter
   (Similar issue: test_parametrize.py fails with 0 tests)

**Repository-Specific Patterns (L2):**
3. pytest uses testdir fixture for testing
4. makepyfile() requires __init__='' for packages

**General Debugging Strategies (L3):**
5. Test discovery failures often indicate missing __init__.py
6. Verify package structure before running tests
```

---

## Benefits

###  Shared Between Both Agents

- Same code, same logic, same results
- No duplication
- Single source of truth

###  No Breakage

- Mini-swe-agent continues to work
- Just updates import path
- All features preserved

###  Easy to Maintain

- Fix bug once → fixes for both agents
- Update once → updates for both
- Add feature once → available to both

###  Fair Comparison

- Both agents use identical memory system
- Only agent scaffold differs
- Research findings are valid

---

## Testing

```bash
# Test memory plugin directly
cd memory_plugin
python3 -c "
from memory_plugin import MemoryPlugin
print('Memory plugin imported successfully!')
"
```

---

## Migration from Original Location

**Original:**
```python
from memory_plugin import MemoryPlugin
```

**New:**
```python
from memory_plugin import MemoryPlugin
```

---

**Last Updated**: July 16, 2026
**Version**: 1.0.0
**Status**:  Ready for use by both agents
