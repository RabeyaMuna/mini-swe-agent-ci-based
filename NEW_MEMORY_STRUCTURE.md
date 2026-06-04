# New Memory Structure (Professor's Approach)
## Multi-Problem Decomposition + Hierarchical Abstraction

---

## 🎯 **What Changed?**

### **Problem Identified by Professor:**
```
CI log shows:  → "dependency error" (FIRST failure only)
Ground truth:  → Fixes dependency + type errors + tests (ALL problems)

Old memory: Mixed all problems together → Too generic
New memory: Separate each problem → Better matching
```

### **Solution:**
1. **Reverse engineer** hidden problems from diff (O-CRD paper)
2. **Multi-problem** decomposition (one issue = multiple atomic problems)
3. **Hierarchical abstraction** at L3 (STAIR paper: 3 levels)
4. **Reasoning** at all levels (WHY + HOW + Verification)

---

## 📋 **New Memory Structure**

### **L1: Per-File Memory (within-repo)**

**One entry = One file in one atomic problem**

```json
{
  "file": "src/auth.py",
  "repo": "myrepo/project",
  "issue_id": "410",
  "atomic_problem_id": "410_p2",
  
  // WHY it failed
  "failure_symptom": "test_login failed with 401",
  "root_cause": "Token parsing doesn't handle None header",
  "why_this_file_failed": "Missing null check before .split()",
  
  // HOW it was fixed
  "fix_description": "Added null check line 42",
  "why_fix_works": "Prevents AttributeError when header missing",
  
  // CI context
  "ci_visibility": "hidden",  // or "visible_in_log"
  "ci_workflow_stage": "test",
  
  // Dependencies
  "file_dependencies": {
    "callers": ["src/api/endpoints.py"],
    "callees": ["src/db/session.py"]
  },
  
  // Verification
  "verification_strategy": "pytest tests/test_auth.py::test_login_no_header"
}
```

### **L2: Per-Issue Memory (within-repo) - WITH MULTI-PROBLEM DECOMPOSITION**

**One entry = One issue with MULTIPLE atomic problems**

```json
{
  "issue_id": "410",
  "repo": "fishaudio/fish-speech",
  
  // Visible failure (from CI log)
  "visible_ci_failure": "fish-audio-sdk version constraint invalid",
  "ci_stage_failed": "pip install",
  
  // CRITICAL: Multiple atomic problems!
  "total_atomic_problems": 2,
  "atomic_problems": [
    {
      "problem_id": "410_p1",
      "visibility": "visible_in_log",  // In CI log
      "problem_type": "dependency_error",
      "symptom": "Version constraint >=2024.12.5,<2025 invalid",
      "evidence_in_ci_log": "ERROR: No matching distribution...",
      "ci_workflow_stage": "install",
      
      // WHY + HOW
      "root_cause": "SDK changed from CalVer to SemVer",
      "how_fixed": "Updated constraint to >=1.0.0,<2",
      "why_fix_works": "Matches new SemVer scheme",
      
      // Dependencies
      "depends_on": null,
      "enables": ["410_p2"],
      
      // Verification
      "verification_after_fix": "pip install --dry-run",
      "next_failure_after_fix": "Type errors would appear"
    },
    {
      "problem_id": "410_p2",
      "visibility": "hidden",  // NOT in CI log! Inferred from diff
      "problem_type": "type_checking",
      "symptom": "Type errors in 11 files (INFERRED)",
      "evidence_in_ci_log": null,  // Hidden!
      "ci_workflow_stage": "type_check",
      
      // WHY + HOW
      "root_cause": "SDK v1.0 changed API from dict to typed objects",
      "how_fixed": "Updated type hints in 11 files",
      "why_fix_works": "Type hints match new SDK types",
      
      // Dependencies (KEY!)
      "depends_on": "410_p1",  // Must fix dependency first!
      "dependency_reason": "Can't check types until SDK installable",
      
      // Verification
      "verification_after_fix": "mypy src/"
    }
  ],
  
  // Inferred repair trajectory
  "repair_trajectory": [
    {
      "step": 1,
      "problem_fixed": "410_p1",
      "validation": "pip install --dry-run",
      "result": "success",
      "expected_next_failure": "Type errors"
    },
    {
      "step": 2,
      "problem_fixed": "410_p2",
      "validation": "mypy src/",
      "result": "success"
    }
  ]
}
```

### **L3: Cross-Repo Universal Principles - WITH HIERARCHICAL ABSTRACTION**

**One entry = One universal principle with 3 abstraction levels (STAIR)**

```json
{
  "principle_id": "dep_version_scheme_migration",
  "principle_name": "Dependency Version Scheme Change Pattern",
  
  // STAIR: 3-level hierarchical abstraction
  "abstraction_hierarchy": {
    
    // Level 1: Concrete (specific but generalized)
    "level_1_concrete": {
      "description": "Python package migrating from CalVer to SemVer",
      "specific_examples": ["fish-audio-sdk: 2024.12.5 → 1.0.0"],
      "reusable_strategy": "Check release history, update constraint syntax",
      "when_to_apply": "Python package with date-based → number-based version jump"
    },
    
    // Level 2: Pattern (general strategy)
    "level_2_pattern": {
      "description": "Package dependency versioning scheme evolution",
      "general_strategy": [
        "Identify upstream versioning change",
        "Understand new scheme",
        "Update constraint to match",
        "Verify installation"
      ],
      "when_to_apply": "Dependency resolution fails after version jump",
      "ecosystem_specific": {
        "Python": "Update pyproject.toml",
        "Node": "Update package.json"
      }
    },
    
    // Level 3: Universal (cross-language)
    "level_3_universal": {
      "description": "Version specification evolution pattern",
      "universal_principle": "When dependency versioning changes, update constraints to match new convention",
      "applies_to_languages": ["Python", "Node", "Rust", "Go", "Java"]
    }
  },
  
  // Evidence from L2
  "evidence_from_l2": [
    {
      "repo": "fishaudio/fish-speech",
      "issue": "410",
      "atomic_problem": "410_p1"
    }
  ]
}
```

---

## 🚀 **How to Build New Memory**

### **Step 1: Test on 1 Issue**

```bash
# This runs full pipeline on issue 0 only
./scripts/build_memory_pipeline.sh --test

# Check output
cat test_decomposed_issues.json | python -m json.tool | head -100
cat test_memory_v2/repo_memory.json | python -m json.tool | head -100
```

**Expected output:**
- Issue decomposed into 2-3 atomic problems
- L2 has `atomic_problems` array with `visibility` field
- L3 has `abstraction_hierarchy` with 3 levels

### **Step 2: Build Full Memory**

```bash
# Build for all eval issues (takes ~30 min for 15 issues)
./scripts/build_memory_pipeline.sh

# Or build for first 5 issues
./scripts/build_memory_pipeline.sh --limit 5
```

### **Step 3: Verify and Use**

```bash
# Precompute embeddings
python scripts/precompute_embeddings.py --memory-root data/trs_memory_v2

# Verify structure
python scripts/verify_memory_structure.py --memory-root data/trs_memory_v2

# Run evaluation
scripts/run_cibench_minimax_openrouter.sh --slice 0:15
```

---

## 📊 **Key Differences from Old Memory**

| Aspect | Old | New |
|--------|-----|-----|
| **Granularity** | One L2 per issue | Multiple atomic problems per L2 |
| **Visibility** | Not tracked | `visible_in_log` vs `hidden` |
| **Reasoning** | Minimal | WHY + HOW + verification at all levels |
| **Dependencies** | Not tracked | Explicit (p2 depends on p1) |
| **Trajectory** | Ground truth only | Inferred repair sequence |
| **L3 Abstraction** | Single level | 3 levels (concrete/pattern/universal) |
| **Evidence** | CI log only | CI log + diff + CI workflow |

---

## 🔍 **How Reverse Engineering Works**

### **Input:**
```
CI Log:    "ERROR: No matching distribution for fish-audio-sdk>=2024.12.5"
Diff:      Changes pyproject.toml + 11 .py files with type hints
Workflow:  install → lint → type-check → test
```

### **LLM Reasoning:**
```
Problem 1 (VISIBLE in CI log):
  - Symptom: Dependency constraint invalid
  - Evidence: CI log has "No matching distribution"
  - Files: pyproject.toml
  - Stage: pip install

Problem 2 (HIDDEN, inferred from diff):
  - Symptom: Type errors (not in CI log!)
  - Evidence: 11 .py files have type hint changes in diff
  - Files: src/*.py (11 files)
  - Stage: Would fail at "mypy src/" (after p1 fixed)
  - Depends on: p1 (can't check types until SDK installable)
```

---

## 🎓 **Alignment with Papers**

### **O-CRD Paper:**
- ✅ Backward reasoning from ground truth
- ✅ Stage-wise decomposition (install → lint → test)
- ✅ Adaptation hints for each problem

### **STAIR Paper:**
- ✅ Hierarchical abstraction (3 levels)
- ✅ Retrieval from multiple levels
- ✅ Evidence from multiple issues

### **Professor's Additions:**
- ✅ Multi-problem decomposition (one issue = multiple problems)
- ✅ Visibility tracking (visible vs hidden)
- ✅ CI workflow mapping (which stage would fail)
- ✅ Dependencies between problems (p2 depends on p1)
- ✅ Repair trajectory inference (step-by-step)

---

## ✅ **What to Check**

After building memory, verify:

1. **L2 has multi-problem structure:**
   ```bash
   cat data/trs_memory_v2/repo_memory.json | python -c "
   import json, sys
   l2 = json.load(sys.stdin)
   print(f'Total L2 entries: {len(l2)}')
   total_problems = sum(m['total_atomic_problems'] for m in l2)
   print(f'Total atomic problems: {total_problems}')
   print(f'Avg problems per issue: {total_problems/len(l2):.1f}')
   "
   ```

2. **Problems have visibility markers:**
   ```bash
   cat data/trs_memory_v2/repo_memory.json | python -c "
   import json, sys
   l2 = json.load(sys.stdin)
   visible = sum(sum(1 for p in m['atomic_problems'] if p['visibility']=='visible_in_log') for m in l2)
   hidden = sum(sum(1 for p in m['atomic_problems'] if p['visibility']=='hidden') for m in l2)
   print(f'Visible problems: {visible}')
   print(f'Hidden problems: {hidden}')
   "
   ```

3. **L3 has hierarchical abstraction:**
   ```bash
   cat data/trs_memory_v2/cross_memory.json | python -c "
   import json, sys
   l3 = json.load(sys.stdin)
   with_hierarchy = sum(1 for p in l3 if 'abstraction_hierarchy' in p)
   print(f'L3 principles: {len(l3)}')
   print(f'With 3-level hierarchy: {with_hierarchy}')
   "
   ```

---

## 🐛 **Troubleshooting**

### **Issue: LLM returns malformed JSON**
```bash
# Check the raw output
cat test_decomposed_issues.json | python -m json.tool

# If fails, the decompose script shows raw content in error
# Check for markdown fences or extra text
```

### **Issue: No hidden problems detected**
```bash
# Check if diff has multiple file types
cat data/trs/eval_issues.json | python -c "
import json, sys
issues = json.load(sys.stdin)
for i in issues[:5]:
    files = i.get('changed_files', [])
    print(f\"Issue {i['id']}: {len(files)} files - {files}\")
"

# If all issues change only 1 file, may legitimately have no hidden problems
```

### **Issue: Memory too slow to build**
```bash
# Use smaller model
./scripts/build_memory_pipeline.sh --model openai/gpt-4o-mini

# Or build incrementally
./scripts/build_memory_pipeline.sh --limit 5  # First 5
./scripts/build_memory_pipeline.sh --limit 10  # Then first 10
```

---

## 📈 **Expected Improvements**

With new memory structure:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| **Similarity (407→410)** | 0.35 | 0.75-0.85 |
| **Memory injection rate** | 20-30% | 65-80% |
| **Success rate** | 28.57% | 38-43% |

---

## 🎯 **Next Steps**

1. **Today:** Test on 1 issue, verify structure
2. **Tomorrow:** Build full memory for 15 eval issues
3. **Day 3:** Run evaluation, compare results
4. **Day 4:** Analysis and presentation for professor

**Start here:**
```bash
./scripts/build_memory_pipeline.sh --test
```

Good luck! 🚀
