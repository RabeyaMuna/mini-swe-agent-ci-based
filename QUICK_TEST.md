# Quick Test Guide

## ✅ What Was Fixed

1. **CI Log Parser** - No longer crashes on malformed JSON from LLM
2. **Repair Plan Generation** - Added `CIMemorySystem.plugin` property
3. **Problem Statement** - Already includes all memory guidance and repair plans

## 🧪 Verify Fixes Work

```bash
# Run simple test
python test_simple.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

## 🚀 Test with Real Benchmark

### Option 1: Automated Test Script

```bash
./test_issue_121.sh
```

This will:
1. Clear Python cache
2. Create test set with issue 121 (multi-problem case)
3. Check memory exists
4. Run benchmark
5. Analyze results

### Option 2: Manual Test

```bash
# 1. Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 2. Create test file with issue 121
python -c "
import json
with open('data/trs/eval_issues.json') as f:
    all_issues = json.load(f)
issue = [i for i in all_issues if str(i.get('id')) == '121'][0]
with open('data/trs/eval_issues_121.jsonl', 'w') as f:
    f.write(json.dumps(issue) + '\n')
print('Created test file')
"

# 3. Run test
python -m minisweagent.run.benchmarks.cibench \
  --instances data/trs/eval_issues_121.jsonl \
  --run_name test_fixes \
  --memory_root data/trs \
  --memory_enabled \
  --ablation_levels L1+L2+L3 \
  --max_workers 1
```

## 📊 Expected Results

### Issue 121 (Multi-Problem):
- **Problem 1**: numpy type annotation (visible)
- **Problem 2**: 88 RST files formatting (hidden)
- **Problem 3**: taplo in pyproject.toml (hidden)

### Agent Should Fix:
1. ✅ `ndarrays_arithmetic.py` - Replace DTypeLike with np.dtype[Any]
2. ✅ `docs/source/*.rst` - Add overline adornments (88 files)
3. ✅ `pyproject.toml` - Uncomment taplo (2 files)

### Check Results:
```bash
# View generated patch
cat results/test_fixes/preds.json

# Check files changed
python -c "
import json
preds = json.load(open('results/test_fixes/preds.json'))
for k, v in preds.items():
    diff = v.get('diff', '')
    files = [l for l in diff.split('\n') if l.startswith('diff --git')]
    print(f'Files changed: {len(files)}')
    print('Has numpy fix:', 'ndarrays_arithmetic.py' in diff)
    print('Has RST fixes:', '.rst' in diff)
    print('Has taplo fix:', 'pyproject.toml' in diff)
"
```

## 🔍 Debug if It Fails

### Check logs:
```bash
# CI context log
cat results/test_fixes/cibench.log | grep "Phase"

# Memory retrieval
cat results/test_fixes/memory_retrieval_debug.jsonl | jq .
```

### Key checkpoints:
1. **Phase A** - CI log analysis should succeed or fallback gracefully
2. **Phase C** - Memory retrieval should find L2 with similarity > 0.3
3. **Phase C.5** - Repair plan generation should succeed (no AttributeError!)
4. **Problem statement** - Should include "## Memory Context" and "## Suggested Repair Plan"

### If repair plan still fails:
```bash
# Check property exists
python -c "
import sys
sys.path.insert(0, 'src')
from minisweagent.run.benchmarks.utils.ci_memory_system import CIMemorySystem
ms = CIMemorySystem.create('data/trs', memory_enabled=True)
print('Has plugin:', hasattr(ms, 'plugin'))
print('Plugin accessible:', ms.plugin is not None)
"
```

## 📝 Files Changed

1. `src/minisweagent/run/benchmarks/utils/ci_log_analyzer.py`
   - Added `_clean_malformed_json()` function
   - Used in 2 JSON parsing locations

2. `src/minisweagent/run/benchmarks/utils/ci_memory_system.py`
   - Added `@property plugin` to expose `self._plugin`

3. No changes to mini-swe-agent core!
   - All fixes in problem statement generation
   - Agent receives complete context naturally

## 🎯 Success Criteria

✅ **Phase A**: CI log analysis succeeds (or graceful fallback)  
✅ **Phase C**: Memory retrieves L2 entries with trajectories  
✅ **Phase C.5**: Repair plan generates successfully  
✅ **Problem Statement**: Includes memory context + repair plan  
✅ **Agent Output**: Fixes all 3 problems (numpy + RST + taplo)  

If all 5 criteria pass, the system is working! 🚀
