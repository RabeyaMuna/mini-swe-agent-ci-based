# Quick Start Guide

## 🚀 Commands to Run

### 1. Build Memory from Decomposed Issues
```bash
python scripts/build_memory_from_decomposed.py \
  --decomposed data/trs/decomposed_issues.json \
  --output-dir data/trs
```

**Output**: Creates `failure_memory.json`, `repo_memory.json`, `cross_memory.json`

---

### 2. Pre-compute Embeddings (FAST RETRIEVAL)
```bash
python scripts/precompute_embeddings.py \
  --memory-root data/trs \
  --verbose
```

**What it does**:
- Embeds all `search_document` fields
- Saves `_embedding` to each entry
- Makes retrieval 10x+ faster (no recomputation)

**Output**: Updates JSON files with embeddings

---

### 3. Verify Embeddings
```bash
python verify_embeddings.py
```

**Expected output**:
```
L2: repo_memory.json
  Total entries: 5
  With embeddings: 5
  Coverage: 5/5 (100.0%)
  Embedding dimension: 384
  ✅ Has repair_trajectory in search_document: True

✅ SUCCESS: All memory entries have embeddings!
🚀 Ready for retrieval testing
```

---

## 📊 Check Memory Structure

### Quick Check
```bash
python -c "
import json
data = json.load(open('data/trs/repo_memory.json'))
print(f'Total L2 entries: {len(data)}')
print(f'Has embeddings: {\"_embedding\" in data[0]}')
print(f'Has repair_trajectory_summary: {\"repair_trajectory_summary\" in data[0]}')
print(f'Keys: {list(data[0].keys())}')
"
```

### View Sample Entry
```bash
python -c "
import json
data = json.load(open('data/trs/repo_memory.json'))
entry = data[0]
print('=== SAMPLE L2 ENTRY ===')
print(f'ID: {entry[\"id\"]}')
print(f'Repo: {entry[\"repo\"]}')
print(f'Problems: {len(entry[\"atomic_problems\"])}')
print(f'\nRepair Trajectory (first 300 chars):')
print(entry[\"repair_trajectory_summary\"][:300])
print(f'\nVerification Sequence: {len(entry[\"verification_sequence\"])} steps')
"
```

---

## 🧪 Test Memory Retrieval

### Create Test Script
```python
# test_retrieval.py
import sys
sys.path.insert(0, 'src')

from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin

# Initialize
memory_plugin = MemoryPlugin(
    l1_memory_path="data/trs/failure_memory.json",
    l2_memory_path="data/trs/repo_memory.json",
    l3_memory_path="data/trs/cross_memory.json",
    embedding_type="sentence-transformers"
)

# Test query
query = {
    "task_id": "test",
    "repo": "flower",
    "error_type": ["dependency_error"],
    "relevant_files": ["pyproject.toml"],
    "overall_failure_reason": "poetry install failed"
}

# Retrieve
result = memory_plugin.retrieve(query, top_k=3, filters={})

print(f"Found {len(result['matches'])} matches")
for i, match in enumerate(result['matches'][:3]):
    print(f"\nMatch {i+1}:")
    print(f"  Level: {match['memory_level']}")
    print(f"  Score: {match['similarity_score']:.3f}")
    print(f"  Repo: {match.get('repo')}")
    
    # Check trajectory
    if 'repair_trajectory_summary' in match:
        print(f"  ✅ Has trajectory ({len(match['repair_trajectory_summary'])} chars)")
    else:
        print(f"  ❌ No trajectory")
```

Run it:
```bash
python test_retrieval.py
```

**Expected**: Finds similar L2 entries with trajectories

---

## 🎯 Test LLM Synthesis (CRITICAL TEST)

```python
# test_synthesis.py
import sys
sys.path.insert(0, 'src')

from minisweagent.run.benchmarks.utils.ci_memory_system import CIMemorySystem
from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin
from langchain_openai import ChatOpenAI

# Initialize
memory_plugin = MemoryPlugin(
    l1_memory_path="data/trs/failure_memory.json",
    l2_memory_path="data/trs/repo_memory.json",
    l3_memory_path="data/trs/cross_memory.json",
    embedding_type="sentence-transformers"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory_system = CIMemorySystem(memory_plugin=memory_plugin)

# Test query
query = {
    "task_id": "test",
    "repo": "flower",
    "error_type": ["dependency_error"],
    "relevant_files": ["pyproject.toml"],
    "overall_failure_reason": "numpy.typing.mypy_plugin incompatible"
}

# Retrieve with synthesis
result = memory_system.build_and_retrieve(query, llm=llm, top_k=3)

guidance = result.get('guidance_document', {})

print("=== LLM SYNTHESIS RESULT ===")
print(f"\n1. Primary Files: {len(guidance.get('primary_files', []))}")

print(f"\n2. Additional Files (HIDDEN): {len(guidance.get('full_scope', {}).get('additional_files', []))}")
additional = guidance.get('full_scope', {}).get('additional_files', [])
if additional:
    print("   ✅ CRITICAL TEST PASSED: Hidden failures identified!")
    for f in additional[:3]:
        print(f"   - {f.get('file')}: {f.get('reason', '')[:80]}")
else:
    print("   ❌ CRITICAL TEST FAILED: No hidden failures")

print(f"\n3. Fix Approach: {len(guidance.get('fix_approach', []))} steps")
for i, step in enumerate(guidance.get('fix_approach', [])[:3], 1):
    print(f"   {i}. {step[:100]}")

print(f"\n4. Confidence: {guidance.get('confidence')}")
```

Run it:
```bash
python test_synthesis.py
```

**CRITICAL CHECK**: Does `additional_files` contain hidden failures?

---

## 📋 Full Workflow

```bash
# Step 1: Build memory
python scripts/build_memory_from_decomposed.py \
  --decomposed data/trs/decomposed_issues.json \
  --output-dir data/trs

# Step 2: Pre-compute embeddings
python scripts/precompute_embeddings.py \
  --memory-root data/trs \
  --verbose

# Step 3: Verify
python verify_embeddings.py

# Step 4: Test retrieval
python test_retrieval.py

# Step 5: Test synthesis (CRITICAL!)
python test_synthesis.py
```

---

## ✅ Success Criteria

Your system is working if:

1. ✅ Memory building completes without errors
2. ✅ All entries have `_embedding` field (verify_embeddings.py)
3. ✅ L2 entries have `repair_trajectory_summary` 
4. ✅ Retrieval finds similar memories
5. ✅ **LLM synthesis produces `additional_files` with hidden failures** ← CRITICAL!

---

## 🔍 Troubleshooting

### If precompute_embeddings fails:
```bash
# Check if sentence-transformers is installed
python -c "import sentence_transformers; print('OK')"

# If not:
pip install sentence-transformers
```

### If embeddings are missing after precompute:
```bash
# Check file permissions
ls -lh data/trs/*.json

# Re-run with verbose
python scripts/precompute_embeddings.py --memory-root data/trs --verbose
```

### If additional_files is empty in synthesis:
1. Check L2 has `repair_trajectory_summary`
2. Check `_compact_candidate()` includes trajectory (already fixed)
3. Check synthesis prompt asks for hidden failures (already fixed)
4. Try with a different query that matches your L2 entries better

---

## 📊 Expected Performance

### Before Embedding Pre-computation:
- First query: ~5-10 seconds (computes all embeddings)
- Subsequent queries: ~0.5 seconds (uses cached embeddings)

### After Embedding Pre-computation:
- First query: ~0.5 seconds (reads stored embeddings)
- Subsequent queries: ~0.5 seconds (same, already fast)

**10x+ speedup on first query!**

---

## 🎯 Next Steps

After verifying everything works:

1. Run full evaluation on your benchmark
2. Measure Pass@1 improvement
3. Check repair trajectories
4. Verify one-shot multi-file repairs

See [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md) for detailed testing.

---

## 💡 Key Files

- **Memory files**: `data/trs/*.json` - Contains L1/L2/L3 with embeddings
- **Build script**: `scripts/build_memory_from_decomposed.py` - Creates memory
- **Embed script**: `scripts/precompute_embeddings.py` - Pre-computes embeddings
- **Verify script**: `verify_embeddings.py` - Checks embedding coverage
- **Memory plugin**: `src/minisweagent/run/benchmarks/utils/memory_plugin.py` - Retrieval
- **Synthesis**: `src/minisweagent/run/benchmarks/utils/ci_memory_system.py` - LLM synthesis

---

## 🚀 Ready to Test!

Your system now has:
✅ Backward reasoning in L2 (repair_trajectory_summary)
✅ Embeddings for fast retrieval
✅ LLM synthesis with trajectory awareness
✅ Multi-problem CI repair capability

**Test it and see the one-shot repairs!** 🎉
