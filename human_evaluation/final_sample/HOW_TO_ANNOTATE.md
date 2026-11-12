# How to Annotate - Step by Step Guide

## 🎯 What You're Checking

**Main Question:** Did ReCAP correctly identify which code changes fixed which problem?

**Check Against:** The ACTUAL GitHub PR (not another file, not a guess - the real commit!)

---

## 📋 For Each Row (207 times):

### **STEP 1: Open the GitHub Link**

```
Click the github_link column → Opens actual commit
```

You'll see:
- Commit message
- Files changed (the diff)
- What lines changed (green = added, red = removed)

---

### **STEP 2: Look at the Actual PR**

#### **A. Find the Problem Evidence:**

Look for:
- ✅ Error messages in commit message
- ✅ Test names that failed
- ✅ What the developer says was wrong
- ✅ The changes made (the diff)

#### **B. Understand What Was Fixed:**

From the diff:
- What files changed?
- What lines changed?
- What was the nature of the fix?

---

### **STEP 3: Compare ReCAP's Claim vs Reality**

Now compare what ReCAP says (in the CSV) vs what you just saw (in the GitHub PR).

---

## ✅ DECISION TREE - How to Fill Each Column

### **1. human_problem_exists**

```
Question: Do you see evidence this problem exists in the PR?

Decision Tree:
├─ Can you find this problem mentioned in commit message? → YES
├─ Can you see error about this in the diff? → YES
├─ Does the fix address this issue? → YES
└─ No evidence anywhere? → NO

Examples:
✅ YES: "ReCAP says 'mypy error' → Commit says 'fix mypy error'"
❌ NO: "ReCAP says 'f-string issue' → No mention anywhere"
```

---

### **2. human_problem_description**

```
Question: In YOUR words, what problem do you see?

Write 1-2 sentences describing what YOU observe:
- What failed?
- What was wrong?
- Based on commit message + diff

Example:
"Type checking error - function return type annotation doesn't 
match actual return value, causing mypy to fail"
```

---

### **3. correct_problem_description**

```
Question: Is ReCAP's description accurate?

Decision Tree:
├─ Matches what you see exactly? → YES
├─ Right general idea but vague/incomplete? → PARTIAL
└─ Wrong or different problem? → NO

Examples:
✅ YES: 
   ReCAP: "mypy return type error in chain.py"
   Reality: Commit shows mypy error, chain.py, return type
   
⚠️ PARTIAL:
   ReCAP: "format error"
   Reality: Black formatting - indentation issues (too vague)
   
❌ NO:
   ReCAP: "f-string issue"
   Reality: Indentation error (completely different)
```

---

### **4. correct_root_cause**

```
Question: Is ReCAP's explanation of WHY correct?

Decision Tree:
├─ Diagnosis matches what you see? → YES
├─ Generally right direction? → PARTIAL
└─ Wrong explanation? → NO

Examples:
✅ YES:
   ReCAP: "Return type annotation is str but returns RunResponse"
   Reality: Diff shows exactly this - annotation changed
   
⚠️ PARTIAL:
   ReCAP: "Type mismatch"
   Reality: Specific Optional[Any] vs str issue (too generic)
   
❌ NO:
   ReCAP: "Missing import"
   Reality: Function signature wrong (not import related)
```

---

### **5. correct_files**

```
Question: Are the file paths correct?

Decision Tree:
├─ All files ReCAP lists appear in the diff? → YES
└─ Any file wrong, missing, or not in diff? → NO

Examples:
✅ YES:
   ReCAP: "libs/agno/chain.py"
   Reality: Diff shows libs/agno/chain.py changed
   
❌ NO:
   ReCAP: "src/utils.py"
   Reality: Diff shows src/helper.py (wrong file!)
```

---

### **6. correct_repair_attribution**

```
Question: Do the changes in those files FIX this specific problem?

Decision Tree:
├─ Changes clearly fix this problem? → YES
├─ Some changes right, some wrong? → PARTIAL
├─ Bundled multiple unrelated problems? → NO
├─ Changes are for different problem? → NO
└─ Wrong changes entirely? → NO

Examples:
✅ YES:
   Problem: mypy return type error
   Changes: Function signature updated -> str to -> RunResponse
   Match: Changes directly fix the problem
   
⚠️ PARTIAL:
   Problem: Format errors in test_client.py
   Changes: Fixed test_client.py + also changed unrelated test_utils.py
   
❌ NO:
   Problem: Type error in chain.py
   Changes: Fixed completely different file
   OR: Bundled 3 different problems as one
```

---

### **7. overall_attribution_correct** ⭐ **MOST IMPORTANT**

```
Question: Taking everything together, is the problem→repair link correct?

Decision Tree:

Does problem exist in PR?
├─ NO → overall = NO ❌
└─ YES
   └─ Are files correct?
      ├─ NO → overall = NO ❌
      └─ YES
         └─ Do changes fix THIS problem?
            ├─ NO → overall = NO ❌
            └─ YES
               └─ Description accurate?
                  ├─ Completely wrong → overall = NO ❌
                  ├─ Too vague but right → overall = YES ✅
                  └─ Accurate → overall = YES ✅

Summary Rule:
✅ YES = Problem exists + Files correct + Fix works + Not bundled
❌ NO = Any major component wrong
```

---

### **8. notes**

```
Question: Any observations or reasoning?

Write down:
- Why you marked NO (if you did)
- Edge cases
- Interesting findings
- Problems ReCAP missed
- Multiple problems bundled

Examples:
"ReCAP bundled 2 separate mypy errors as one problem"
"Description too vague but fix is correct"
"ReCAP missed a third file that also had changes"
```

---

## 🎯 CONCRETE EXAMPLE

### **CSV Row:**
```
instance_id: 75
repo: agno
github_link: https://github.com/agno/commit/ca75b3dc...
recap_problem: "mypy return-value error: function declared str but returns RunResponse"
recap_files: "libs/agno/agno/utils/chain.py"
```

### **Your Process:**

**1. Click Link:** Opens GitHub commit

**2. What You See:**
```
Commit Message: "Fix mypy type errors"

Diff shows:
  File: libs/agno/agno/utils/chain.py
  
  - def run(self, input: str) -> str:
  + def run(self, input: str) -> RunResponse:
      return self._execute(input)
```

**3. Your Analysis:**
- ✅ Problem exists: Yes, mypy error visible
- ✅ Description accurate: Yes, matches exactly
- ✅ Files correct: Yes, chain.py in diff
- ✅ Fix correct: Yes, return type changed str → RunResponse

**4. Your Annotation:**
```
human_problem_exists: YES
human_problem_description: "Type annotation error - function return type is str but actually returns RunResponse object"
correct_problem_description: YES
correct_root_cause: YES
correct_repair_attribution: YES
correct_files: YES
overall_attribution_correct: YES
notes: "Perfect attribution - all components correct"
```

---

## ❌ WHEN TO MARK NO

### **Scenario 1: Problem Doesn't Exist**
```
ReCAP says: "f-string formatting issue"
Reality: Indentation error, no f-strings mentioned

→ human_problem_exists: NO
→ overall_attribution_correct: NO
```

### **Scenario 2: Wrong Files**
```
ReCAP says: Fixed in "test_utils.py"
Reality: Diff shows only "test_client.py" changed

→ correct_files: NO
→ overall_attribution_correct: NO
```

### **Scenario 3: Bundled Problems**
```
ReCAP says: "mypy errors"
Reality: 3 separate unrelated type errors bundled together

→ correct_repair_attribution: NO
→ overall_attribution_correct: NO
→ notes: "Bundled 3 distinct problems incorrectly"
```

### **Scenario 4: Vague but Correct**
```
ReCAP says: "Format error"
Reality: Specific Black indentation issue

→ correct_problem_description: PARTIAL
→ BUT if fix is correct: overall_attribution_correct: YES
→ notes: "Description vague but fix is correct"
```

---

## ⏱️ Time Per Row

- **Quick rows:** ~30 seconds (obvious YES or NO)
- **Normal rows:** ~1-2 minutes (need to read diff)
- **Complex rows:** ~3-5 minutes (many files, need careful check)

**Average: ~1-2 minutes per row**

---

## ✅ Quality Check

Before moving to next row, ask yourself:

1. Did I actually look at the GitHub PR? (not guess)
2. Can I defend my YES/NO decision?
3. Did I fill all 8 columns?
4. Is my reasoning in notes (if I marked NO)?

---

## 🎯 Summary

**What you're checking:** 
Did ReCAP correctly identify which code changes fixed which problem?

**Check against:** 
The ACTUAL GitHub commit (click the link!)

**Decision rule:**
- ✅ YES = Problem exists + Files right + Fix works
- ❌ NO = Any major component wrong

**Your judgment is the ground truth!**

---

**Ready to start!** Open the CSV and begin with row 1! 🚀
