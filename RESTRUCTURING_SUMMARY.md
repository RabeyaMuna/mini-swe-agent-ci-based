# Project Restructuring Summary

**Date**: July 16, 2026  
**Branch**: `restructure-multi-agent`  
**Status**: ✅ Complete and tested

## What Changed

Successfully restructured the project to support **multi-agent, multi-model** experiments with shared resources.

### Before (main branch)
```
mini-swe-agent-ci-based/
├── src/minisweagent/     # Agent code at root
├── tests/                # Tests at root
├── data/trs/            # Shared memory
├── results/
│   ├── BASELINE/        # Flat structure
│   └── L1_L2_L3/
└── scripts/
```

### After (restructure-multi-agent branch)
```
mini-swe-agent-ci-based/
├── miniswe-agent/              # 🔧 Isolated agent 1
│   ├── src/minisweagent/
│   ├── tests/
│   ├── .venv/
│   └── pyproject.toml
│
├── openhands/                  # 🔧 Isolated agent 2
│   └── (OpenHands codebase)
│
├── data/                       # ✅ SHARED
│   └── trs/                   # Three-layer memory
│
├── results/                    # ✅ SHARED (hierarchical)
│   ├── miniswe-agent/
│   │   ├── minimax/{baseline,L1,L1_L2,L1_L2_L3}/
│   │   ├── glm/{baseline,L1_L2_L3}/
│   │   └── kimi/{baseline,L1_L2_L3}/
│   └── openhands/
│       ├── glm/{baseline,L1_L2_L3}/
│       └── minimax/{baseline,L1_L2_L3}/
│
├── repo/                       # ✅ SHARED
├── scripts/                    # ✅ SHARED
└── README.md                   # Updated overview
```

## Key Files Created

### 1. Path Configuration
**File**: `miniswe-agent/src/minisweagent/config/paths.py`

Provides centralized access to shared resources:
```python
from minisweagent.config.paths import (
    get_memory_root,      # → ../data/trs/
    get_results_dir,      # → ../results/{agent}/{model}/{ablation}/
    get_repo_dir          # → ../repo/{owner}__{repo}/
)
```

### 2. Documentation
- `README.md` - Multi-agent framework overview
- `miniswe-agent/README.md` - Agent-specific documentation
- `.gitignore` - Updated for agent-specific venvs

## Migration Details

### Results Migration
- ✅ `results/BASELINE/` → `results/miniswe-agent/minimax/baseline/`
- ✅ `results/L1_L2_L3/` → `results/miniswe-agent/minimax/L1_L2_L3/`
- ✅ All trajectory files preserved
- ✅ `preds.json` migrated

### Code Migration
- ✅ `src/` → `miniswe-agent/src/`
- ✅ `tests/` → `miniswe-agent/tests/`
- ✅ `pyproject.toml` → `miniswe-agent/pyproject.toml`
- ✅ `.venv/` → `miniswe-agent/.venv/`

### OpenHands Setup
- ✅ Cloned from GitHub
- ✅ Removed `.git/` to avoid nested repo
- ✅ Ready for CI-Bench adapter

## Verification Tests Passed

### 1. Path Configuration ✅
```bash
cd miniswe-agent
python -c "from minisweagent.config.paths import get_memory_root; print(get_memory_root())"
# Output: /Users/.../mini-swe-agent-ci-based/data/trs
```

### 2. Memory Access ✅
```python
from minisweagent.config.paths import get_memory_root
memory_root = get_memory_root()
assert memory_root.exists()  # ✅ Passes
assert (memory_root / "failure_memory.json").exists()  # ✅ Passes
```

### 3. Results Access ✅
```bash
ls results/miniswe-agent/minimax/baseline/preds.json  # ✅ Exists
ls results/miniswe-agent/minimax/L1_L2_L3/           # ✅ 35 items
```

### 4. Installation ✅
```bash
cd miniswe-agent
pip install -e .  # ✅ Successful
python -m minisweagent --help  # ✅ Works
```

## Benefits

### 1. Agent Isolation
- ✅ Separate codebases
- ✅ Independent dependencies
- ✅ No version conflicts

### 2. Shared Resources
- ✅ Single memory build (`data/trs/`)
- ✅ Common testbed repos (`repo/`)
- ✅ Shared evaluation tools (`scripts/`)

### 3. Organized Results
Easy comparisons:
```bash
# Same agent, different models
results/miniswe-agent/{minimax,glm}/baseline/

# Same model, different agents
results/{miniswe-agent,openhands}/glm/baseline/

# Ablation study
results/miniswe-agent/minimax/{baseline,L1,L1_L2,L1_L2_L3}/
```

### 4. Scalability
- ✅ Easy to add new models (just create new dirs)
- ✅ Easy to add new agents (clone to new folder)
- ✅ Easy to add ablation levels (L4, L5, etc.)

## Git Safety

### Backup Points
1. **Git tag**: `backup-before-restructure-20260716_103106`
2. **Main branch**: Untouched, safe
3. **New branch**: `restructure-multi-agent`

### Rollback
If needed:
```bash
# Option 1: Switch back to main
git checkout main

# Option 2: Reset to backup tag
git reset --hard backup-before-restructure-20260716_103106
```

## Next Steps

### Immediate (this week)
1. ✅ Verify miniswe-agent works in new location
2. ⬜ Run small test experiment
3. ⬜ Update scripts to use new paths

### Short-term (1-2 weeks)
4. ⬜ Create OpenHands adapter for CI-Bench
5. ⬜ Test OpenHands with shared memory
6. ⬜ Run baseline experiments for both agents

### Medium-term (2-4 weeks)
7. ⬜ Add GLM model support
8. ⬜ Run full ablation study
9. ⬜ Generate comparison tables
10. ⬜ Merge to main when stable

## Testing Checklist

Before merging to `main`:

- [ ] Run baseline experiment with miniswe-agent
- [ ] Verify memory retrieval works
- [ ] Check results save to correct location
- [ ] Test evaluation scripts with new paths
- [ ] Run ablation study (L1, L1_L2, L1_L2_L3)
- [ ] OpenHands baseline (if ready)
- [ ] Update paper with new structure screenshots

## File Structure Summary

### What's SHARED (Root Level)
- `data/` - Datasets and three-layer memory
- `results/` - All experiment outputs (organized)
- `repo/` - Testbed repository clones
- `scripts/` - Evaluation and memory tools

### What's ISOLATED (Per-Agent)
- `miniswe-agent/` - Your implementation
  - Dependencies
  - Virtual environment
  - Source code
  - Tests
  
- `openhands/` - OpenHands integration
  - Separate dependencies
  - Separate venv
  - Own configuration

## Comparison with SWE-bench

| Aspect | SWE-bench | CI-Repair-Bench (This Work) |
|--------|-----------|------------------------------|
| **Scope** | Single issue | Pull request (multi-issue) |
| **Commits** | One patch | Multiple commits |
| **Verification** | Single test | Multi-stage CI |
| **Problem Types** | Code bugs | Style, config, deps, tests, merge |
| **Complexity** | Atomic | Compositional |
| **Agent Support** | Single | Multi-agent comparison |

## Success Metrics

### Current (MiniMax Baseline)
- Baseline: 13.48%
- L1+L2+L3: **17.98%** (+4.5% improvement)

### Target (After Multi-Agent Setup)
- Compare: MiniMax vs GLM vs Kimi
- Compare: mini-swe-agent vs OpenHands
- Ablation: Identify which memory layers help most
- Analysis: Which failure types benefit from memory

## Contact

For questions about this restructuring:
- See: `RESTRUCTURING_SUMMARY.md` (this file)
- Check: [Migration Plan](/private/tmp/.../restructure_migration_plan.md)
- Review: Git commit `a27a5fa`

---

**Status**: ✅ Ready for testing and experimentation
**Last Updated**: July 16, 2026
