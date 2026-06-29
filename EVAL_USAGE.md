# Evaluation Script Usage Guide

## **Script: `scripts/run_eval.py`**

Clean, simple evaluation script that follows mini-swe-agent's default directory structure.

---

## **Directory Structure**

Results are saved to:
```
results/{ablation}/{sha}/
```

**Example:**
```
results/L1_L2_L3/5f98672f68c7f4acdf3e6f68fd79e66b56d0b529/
├── preds.json
├── run_instance.log
├── patch.diff
└── ...
```

**No timestamps, no extra folders - just clean SHA-based directories!**

---

## **Quick Examples**

### **1. Run Specific Issue IDs**
```bash
python3 scripts/run_eval.py --issue-ids 111
```

**Output:** `results/L1_L2_L3/5f98672f.../`

---

### **2. Run Multiple Issues**
```bash
python3 scripts/run_eval.py --issue-ids 111,121,145
```

---

### **3. Run with Different Ablation**
```bash
# L1 only
python3 scripts/run_eval.py --issue-ids 111 --ablation L1

# L1+L2
python3 scripts/run_eval.py --issue-ids 111 --ablation L1+L2

# L1+L2+L3 (default)
python3 scripts/run_eval.py --issue-ids 111 --ablation L1+L2+L3
```

**Output:**
- L1: `results/L1/5f98672f.../`
- L1+L2: `results/L1_L2/5f98672f.../`
- L1+L2+L3: `results/L1_L2_L3/5f98672f.../`

---

### **4. Run Camel/Flower Repos (Exclude Memory)**
```bash
python3 scripts/run_eval.py --repos camel,flower --exclude-memory --max-issues 10
```

**What it does:**
- Filters only camel and flower repos
- Excludes issues used to build memory
- Limits to first 10 issues

---

### **5. Dry Run (See What Would Run)**
```bash
python3 scripts/run_eval.py --issue-ids 111,121 --dry-run
```

**Output:**
```
Selected issues: 2
Issues to process:
  1. Issue 111 - flower - 5f98672f
  2. Issue 121 - camel - 66718a25

Command:
  python3 -m minisweagent.run.benchmarks.cibench
  --dataset .eval_temp_dataset.json
  --output results/L1_L2_L3
  --memory-ablation L1+L2+L3
  --workers 1

DRY RUN - Not executing
```

---

## **All Options**

```bash
python3 scripts/run_eval.py --help
```

### **Filter Options:**

| Option | Description | Example |
|--------|-------------|---------|
| `--issue-ids` | Comma-separated issue IDs | `111,121,145` |
| `--repos` | Comma-separated repo names | `camel,flower` |
| `--exclude-memory` | Exclude issues used for memory | Flag |
| `--max-issues` | Limit number of issues | `10` |

### **Configuration Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--ablation` | Memory level | `L1+L2+L3` |
| `--workers` | Parallel workers | `1` |
| `--dry-run` | Show what would run | Flag |

---

## **Common Workflows**

### **Workflow 1: Ablation Study**

Run same issues with different memory levels:

```bash
# Baseline (L1 only)
python3 scripts/run_eval.py --issue-ids 111,121,145 --ablation L1

# L1+L2
python3 scripts/run_eval.py --issue-ids 111,121,145 --ablation L1+L2

# L1+L2+L3
python3 scripts/run_eval.py --issue-ids 111,121,145 --ablation L1+L2+L3
```

**Results:**
```
results/
├── L1/
│   ├── 5f98672f.../
│   ├── 66718a25.../
│   └── ...
├── L1_L2/
│   ├── 5f98672f.../
│   ├── 66718a25.../
│   └── ...
└── L1_L2_L3/
    ├── 5f98672f.../
    ├── 66718a25.../
    └── ...
```

---

### **Workflow 2: Evaluate Unseen Issues**

Run on camel/flower issues NOT in memory:

```bash
python3 scripts/run_eval.py \
  --repos camel,flower \
  --exclude-memory \
  --max-issues 20 \
  --ablation L1+L2+L3
```

**What happens:**
1. Loads all camel/flower issues
2. Excludes 238 memory issues
3. Selects first 20 remaining
4. Runs with full memory (L1+L2+L3)

---

### **Workflow 3: Quick Test**

Test one issue to verify setup:

```bash
python3 scripts/run_eval.py --issue-ids 111
```

**Check results:**
```bash
ls -la results/L1_L2_L3/5f98672f*/
cat results/L1_L2_L3/5f98672f*/preds.json
```

---

## **Output Files**

Each issue directory contains:

```
results/L1_L2_L3/5f98672f68c7f4acdf3e6f68fd79e66b56d0b529/
├── preds.json              # Predictions (patch)
├── run_instance.log        # Agent execution log
├── patch.diff              # Final diff
├── memory_retrieval_debug.jsonl  # Memory debug info
├── testbed/                # Cloned repo
└── trajectory.json         # Agent trajectory
```

---

## **Features**

### **✅ Improvements vs. Old Script**

| Feature | Old Script | This Script |
|---------|------------|-------------|
| **Directory structure** | `results/.../TIMESTAMP/...` | `results/L1_L2_L3/SHA/` |
| **Timestamps** | Yes (messy) | No (clean) |
| **Ablation in path** | No | Yes (`L1_L2_L3`) |
| **Temp dataset cleanup** | Manual | Automatic |
| **Per-problem timeout** | No | Yes (10 min) |
| **5s delay** | No | Yes |
| **Skip stuck problems** | No | Yes |
| **Memory exclusion** | Manual | `--exclude-memory` |
| **Dry run** | No | `--dry-run` |

---

## **Timeout Protection**

**Built-in protection from previous fix:**

✅ **Per-problem timeout:** 10 minutes max  
✅ **5-second delay** between problems  
✅ **Skip stuck problems** instead of failing entire run

**Example:**
```
Problem 1/27 → Success (2 min)
[5s delay]
Problem 2/27 → Success (3 min)
[5s delay]
Problem 3/27 → TIMEOUT (10 min) → SKIP
[5s delay]
Problem 4/27 → Success (1 min)
...
```

**Result:** 26/27 problems attempted (only 1 skipped)

---

## **Troubleshooting**

### **Issue: "No issues match the filters!"**

**Cause:** Issue IDs not found or all excluded

**Fix:**
```bash
# Check if issue exists
python3 -c "
from datasets import load_dataset
ds = load_dataset('ci-benchmark-user/ci-repair-bench', split='train')
ids = [str(item['id']) for item in ds]
print('111 in dataset:', '111' in ids)
"

# Or use dry-run to see what's selected
python3 scripts/run_eval.py --issue-ids 111 --dry-run
```

---

### **Issue: Results already exist**

**Mini-swe-agent behavior:**
- If `results/L1_L2_L3/SHA/` exists, it may skip or overwrite
- Delete directory to re-run:

```bash
rm -rf results/L1_L2_L3/5f98672f68c7f4acdf3e6f68fd79e66b56d0b529/
python3 scripts/run_eval.py --issue-ids 111
```

---

### **Issue: Too many problems per issue**

**If you see:**
```
Problem 1/27
Problem 2/27
...
Problem 27/27
```

**Solution:** See `PROFESSOR_UPDATE.md` for deduplication improvements

---

## **Comparison: Old vs. New Script**

### **Old: `test_memory_guided_repair.py`**
```bash
python3 scripts/test_memory_guided_repair.py --issue-ids 111

# Output:
results/memory_guided_ablation/20260629_130329/L1_L2_L3/5f98672f.../
# ❌ Timestamp in path
# ❌ Long nested structure
```

### **New: `run_eval.py`**
```bash
python3 scripts/run_eval.py --issue-ids 111

# Output:
results/L1_L2_L3/5f98672f.../
# ✅ Clean structure
# ✅ Ablation in path
# ✅ No timestamps
```

---

## **Summary**

**Script:** `scripts/run_eval.py`

**Key Features:**
- ✅ Clean directory structure (`results/{ablation}/{sha}/`)
- ✅ No timestamps
- ✅ Easy ablation studies
- ✅ Memory exclusion
- ✅ Dry run support
- ✅ Per-problem timeout (10 min)
- ✅ 5-second delays
- ✅ Skip stuck problems

**Most Common Usage:**
```bash
# Run specific issues
python3 scripts/run_eval.py --issue-ids 111,121,145

# Ablation study
python3 scripts/run_eval.py --issue-ids 111 --ablation L1
python3 scripts/run_eval.py --issue-ids 111 --ablation L1+L2
python3 scripts/run_eval.py --issue-ids 111 --ablation L1+L2+L3

# Unseen camel/flower
python3 scripts/run_eval.py --repos camel,flower --exclude-memory --max-issues 10
```

**Perfect for clean, reproducible evaluations!** 🎯
