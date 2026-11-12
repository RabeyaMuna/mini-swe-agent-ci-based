# Quick Reference Card - Annotation

## 🎯 ONE QUESTION TO ANSWER:

**"Did ReCAP correctly identify which code changes fixed which problem?"**

---

## 📋 8 Columns - Quick Rules

| # | Column | Fill With | Quick Rule |
|---|--------|-----------|------------|
| 1 | `human_problem_exists` | YES/NO | See problem in CI logs? |
| 2 | `human_problem_description` | Text | What YOU see (1-2 sentences) |
| 3 | `correct_problem_description` | YES/PARTIAL/NO | ReCAP's description accurate? |
| 4 | `correct_root_cause` | YES/PARTIAL/NO | ReCAP's diagnosis correct? |
| 5 | `correct_repair_attribution` | YES/PARTIAL/NO | Changes fix THIS problem? ⭐ |
| 6 | `correct_files` | YES/NO | File paths correct? |
| 7 | **`overall_attribution_correct`** | **YES/NO** | **Whole link correct?** ⭐⭐ |
| 8 | `notes` | Text | Observations |

---

## ⚡ Quick Decision Tree

```
1. Problem exists in PR?
   ├─ NO → overall = NO ❌ (stop here)
   └─ YES → continue

2. Files in ReCAP match diff?
   ├─ NO → overall = NO ❌ (stop here)
   └─ YES → continue

3. Changes fix THIS problem?
   ├─ NO → overall = NO ❌ (stop here)
   └─ YES → overall = YES ✅
```

---

## ❌ Mark NO if:

- Problem doesn't exist (hallucinated)
- Wrong file paths
- Changes don't fix the problem
- Bundled multiple problems incorrectly
- Fix is for a DIFFERENT problem

---

## ✅ Mark YES if:

- Problem exists in PR
- Description accurate
- Files correct
- Changes actually fix it
- Link is correct

---

## 🔍 What to Check (30 seconds per row)

1. **Click github_link** → Opens PR
2. **Skim CI logs** → What failed?
3. **Look at diff** → What changed?
4. **Compare** → ReCAP vs reality
5. **Fill** → 8 columns
6. **Next row**

---

## ⚠️ Common NO Cases

| Scenario | Label |
|----------|-------|
| Problem doesn't exist | NO ❌ |
| Wrong file | NO ❌ |
| ReCAP bundled 2+ problems | NO ❌ |
| Fix is for different problem | NO ❌ |
| Hallucinated problem | NO ❌ |

---

## 💡 When in Doubt

**Ask:** "If I only read ReCAP's attribution, would I understand which code changes fixed which problem correctly?"

- ✅ YES → overall = YES
- ❌ NO → overall = NO

---

**See ANNOTATION_RULES.md for detailed examples!**
