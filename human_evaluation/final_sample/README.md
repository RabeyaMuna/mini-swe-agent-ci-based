# Human Evaluation - ReCAP Attribution Quality

## 📊 What You Have

**File to annotate:** `human_eval_recap_attribution_FIXED.csv`
- 26 instances (Agno: 23, CAMEL: 3)
- 83 rows (one per ReCAP problem)
- All with valid GitHub links ✅

## 🚀 How to Annotate

### Step 1: Open CSV
```bash
open human_eval_recap_attribution_FIXED.csv
```

### Step 2: For Each Row (83 rows)
1. **Click** the `github_link` column → Opens actual GitHub PR/commit
2. **Read** the PR: CI logs, code diff, commit message
3. **Compare** ReCAP's attribution with what you see
4. **Fill** these 8 columns:

| Column | Fill with |
|--------|-----------|
| `human_problem_exists` | YES / NO |
| `human_problem_description` | What YOU see in the PR |
| `correct_problem_description` | YES / PARTIAL / NO |
| `correct_root_cause` | YES / PARTIAL / NO |
| `correct_repair_attribution` | YES / PARTIAL / NO |
| `correct_files` | YES / NO |
| **`overall_attribution_correct`** | **YES / NO** ⭐ |
| `notes` | Any observations |

### Step 3: Calculate Metrics
```bash
python3 calculate_metrics.py
```

## 🎯 What This Evaluates

**ReCAP's Core Claim:** "Can accurately reconstruct which code changes fixed which problems"

**Your Job:** Verify if ReCAP's reconstruction is correct by checking actual PRs

**Output:** Precision, Recall, F1 scores for the paper

## ⏱️ Time Estimate

83 rows × 1-2 minutes = **1.5-3 hours total**

---

**Ready to start!** 🎯
