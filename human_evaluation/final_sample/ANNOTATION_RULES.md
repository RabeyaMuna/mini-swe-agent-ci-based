# Annotation Rules for ReCAP Attribution Evaluation

## 🎯 Your Goal

For each row, verify if **ReCAP correctly identified which code changes fixed which problem** by comparing against the ACTUAL GitHub PR.

---

## 📋 The 8 Columns to Fill

### 1. `human_problem_exists` (YES / NO)

**Rule:** Does the problem ReCAP identified actually exist in the PR?

| Label | When to Use | Example |
|-------|-------------|---------|
| **YES** | You see evidence of this problem in CI logs or code | CI log shows "mypy error", ReCAP says "mypy error" ✅ |
| **NO** | No evidence of this problem anywhere | ReCAP says "f-string issue" but CI shows "indentation error" ❌ |

**Check:**
- ✅ CI failure logs
- ✅ Error messages
- ✅ Test failures
- ✅ Code that was changed

---

### 2. `human_problem_description` (Free Text)

**Rule:** In YOUR OWN WORDS, describe what problem you see in the actual PR.

**Guidelines:**
- Write 1-2 sentences
- Describe what failed/broke
- Based ONLY on what you observe (CI logs, diff, commit message)
- Don't copy ReCAP's text - write independently

**Example:**
```
"Type checking error - function return type annotation says 'str' 
but implementation returns RunResponse object"
```

---

### 3. `correct_problem_description` (YES / PARTIAL / NO)

**Rule:** Is ReCAP's problem description accurate compared to what you see?

| Label | When to Use | Example |
|-------|-------------|---------|
| **YES** | ReCAP's description matches reality exactly | ReCAP: "mypy return type error" → You see: mypy return type error ✅ |
| **PARTIAL** | ReCAP is mostly right but incomplete/imprecise | ReCAP: "format error" → You see: "Black format error - specific indentation issue" (too vague) ⚠️ |
| **NO** | ReCAP's description is wrong or irrelevant | ReCAP: "f-string issue" → You see: "indentation error" (completely different) ❌ |

**Check:**
- Does the problem type match? (type error vs format error vs test failure)
- Is the description specific enough?
- Does it match what CI logs say?

---

### 4. `correct_root_cause` (YES / PARTIAL / NO)

**Rule:** Is ReCAP's diagnosis of WHY the problem occurred correct?

| Label | When to Use | Example |
|-------|-------------|---------|
| **YES** | Root cause analysis is accurate | ReCAP: "return type annotation is str but returns RunResponse" → Diff shows exactly this ✅ |
| **PARTIAL** | General direction right but details off | ReCAP: "type mismatch" → You see: specific type mismatch with more context ⚠️ |
| **NO** | Completely wrong explanation | ReCAP: "missing import" → You see: wrong function signature ❌ |

**Check:**
- Does ReCAP identify the actual source of the problem?
- Is the causal relationship correct?
- Does the explanation match what you see in the code?

---

### 5. `correct_repair_attribution` (YES / PARTIAL / NO) ⭐ **MOST IMPORTANT**

**Rule:** Did ReCAP correctly identify which code changes fixed THIS specific problem?

| Label | When to Use | Example |
|-------|-------------|---------|
| **YES** | The files/changes ReCAP lists actually fix this problem | ReCAP: "Updated return type in chain.py" → Diff shows `-> str` changed to `-> RunResponse` in chain.py ✅ |
| **PARTIAL** | Some files correct, some wrong/missing | ReCAP lists 2 files: 1 correct, 1 unrelated ⚠️ |
| **NO** | Wrong files OR bundled multiple problems OR unrelated changes | ReCAP: "Fixed in test_utils.py" → Actual fix in test_client.py ❌ |

**Check:**
- ✅ Are the file paths correct?
- ✅ Do the changes in those files actually fix THIS problem?
- ✅ Did ReCAP bundle multiple unrelated problems together?
- ✅ Are there changes in other files that also fix this problem but ReCAP missed?

**Common Issues:**
- ❌ ReCAP attributes the fix to wrong file
- ❌ ReCAP bundles fixes for 2+ different problems together
- ❌ ReCAP misses some files that contribute to the fix

---

### 6. `correct_files` (YES / NO)

**Rule:** Are the file paths ReCAP listed correct?

| Label | When to Use | Example |
|-------|-------------|---------|
| **YES** | All file paths match files in the diff | ReCAP: "libs/agno/chain.py" → Diff changes libs/agno/chain.py ✅ |
| **NO** | Wrong file path OR file not in diff | ReCAP: "src/utils.py" → Diff changes src/helper.py ❌ |

**Check:**
- Do all files ReCAP listed appear in the GitHub diff?
- Are the paths exactly correct? (not typos or wrong directory)

---

### 7. `overall_attribution_correct` (YES / NO) ⭐⭐ **KEY METRIC**

**Rule:** Taking everything into account, is the problem→repair link correct?

| Label | When to Use | Criteria |
|-------|-------------|----------|
| **YES** | Attribution is correct | ✅ Problem exists<br>✅ Description accurate<br>✅ Root cause correct<br>✅ Files correct<br>✅ Changes actually fix the problem |
| **NO** | Attribution has issues | ❌ Problem doesn't exist OR<br>❌ Wrong files OR<br>❌ Changes don't fix this problem OR<br>❌ Bundled multiple problems incorrectly |

**Decision Tree:**
```
Does the problem exist?
├─ NO → overall = NO ❌
└─ YES
   └─ Do the changes fix it?
      ├─ NO → overall = NO ❌
      └─ YES
         └─ Are files correct?
            ├─ NO → overall = NO ❌
            └─ YES → overall = YES ✅
```

**When to be strict:**
- If ANY major component is wrong → NO
- If files are wrong → NO
- If problem doesn't exist → NO
- If bundling errors → NO

**When to be lenient:**
- Minor wording differences in description → Still YES if substance correct
- Root cause slightly imprecise but directionally correct → Can still be YES if fix is right

---

### 8. `notes` (Free Text)

**Rule:** Record any observations, edge cases, or reasoning.

**What to note:**
- Interesting findings
- Why you marked something NO
- Edge cases or ambiguity
- Problems ReCAP missed
- Multiple problems bundled together

**Examples:**
```
"ReCAP bundled 2 separate mypy errors as one problem"
"Fix description is vague but technically correct"
"ReCAP missed a third file that also had changes for this problem"
```

---

## 🎯 Step-by-Step Annotation Process

### For Each Row (83 total):

```
1. Click github_link
   └─> Opens actual GitHub PR/commit

2. Read the ACTUAL evidence:
   ├─ CI failure logs (what error occurred?)
   ├─ Code diff (what changed?)
   └─ Commit message (what did developer say?)

3. Read ReCAP's claim (from CSV):
   ├─ recap_problem
   ├─ recap_root_cause
   ├─ recap_fix_strategy
   └─ recap_files

4. Compare ReCAP vs Reality:
   ├─ Does problem exist? → human_problem_exists
   ├─ Describe what YOU see → human_problem_description
   ├─ Is ReCAP's description accurate? → correct_problem_description
   ├─ Is root cause correct? → correct_root_cause
   ├─ Are the files correct? → correct_files
   ├─ Do changes fix this problem? → correct_repair_attribution
   ├─ Overall correct? → overall_attribution_correct
   └─ Any notes? → notes

5. Move to next row
```

---

## ⚠️ Common Pitfalls

### 1. **Bundling Multiple Problems**

❌ **Wrong:** ReCAP says "mypy errors in multiple files"
- Actual: 3 separate, unrelated type errors

✅ **How to annotate:**
- `overall_attribution_correct`: NO
- `notes`: "ReCAP bundled 3 distinct type errors as one problem"

---

### 2. **Hallucinated Problems**

❌ **Wrong:** ReCAP says "f-string formatting issue"
- Actual: No f-strings in the diff, it's indentation

✅ **How to annotate:**
- `human_problem_exists`: NO
- `overall_attribution_correct`: NO
- `notes`: "ReCAP hallucinated f-string issue - actual problem is indentation"

---

### 3. **Correct Problem, Wrong File**

⚠️ **Issue:** ReCAP identifies the problem correctly but attributes fix to wrong file
- ReCAP says: Fixed in "test_utils.py"
- Actual: Fixed in "test_client.py"

✅ **How to annotate:**
- `human_problem_exists`: YES
- `correct_problem_description`: YES
- `correct_repair_attribution`: NO
- `correct_files`: NO
- `overall_attribution_correct`: NO
- `notes`: "Problem description correct but wrong file - actual fix in test_client.py not test_utils.py"

---

### 4. **Vague but Technically Correct**

⚠️ **Issue:** ReCAP's description is generic but not wrong
- ReCAP says: "Format error"
- Actual: "Black formatting error - indentation and whitespace issues"

✅ **How to annotate:**
- `correct_problem_description`: PARTIAL
- `overall_attribution_correct`: YES (if files and fix are correct)
- `notes`: "Description too vague but technically correct"

---

## 🔍 Quality Checks

### Before you submit, verify:

- [ ] All 83 rows annotated
- [ ] `overall_attribution_correct` filled for every row (YES or NO)
- [ ] At least checked CI logs and diff for each GitHub link
- [ ] Notes explain any NO decisions
- [ ] Decisions are based on ACTUAL PR, not ReCAP's text

---

## 📊 What Good Annotation Looks Like

### Example Row:

**From CSV:**
```
instance_id: 75
github_link: https://github.com/agno/commit/ca75b3dc...
recap_problem: "mypy return-value error: function declared to return str 
                but actually returned RunResponse object"
recap_files: "libs/agno/agno/utils/chain.py"
```

**You check actual PR:**
```
CI Log: 
  libs/agno/agno/utils/chain.py:45: error: 
  Incompatible return value type (got "RunResponse", expected "str")

Diff:
  - def run(self, input: str) -> str:
  + def run(self, input: str) -> RunResponse:
```

**Your annotation:**
```
human_problem_exists: YES
human_problem_description: "Type annotation mismatch - function return type annotation is 'str' but implementation returns RunResponse object, causing mypy error"
correct_problem_description: YES
correct_root_cause: YES
correct_repair_attribution: YES
correct_files: YES
overall_attribution_correct: YES
notes: "Perfect attribution - problem, file, and fix all correct"
```

---

## 🎯 Summary of Rules

| Column | Values | Key Question |
|--------|--------|--------------|
| `human_problem_exists` | YES/NO | Does this problem exist in the PR? |
| `human_problem_description` | Text | What do YOU see? |
| `correct_problem_description` | YES/PARTIAL/NO | Is ReCAP's description accurate? |
| `correct_root_cause` | YES/PARTIAL/NO | Is ReCAP's diagnosis correct? |
| `correct_repair_attribution` | YES/PARTIAL/NO | Do the changes fix THIS problem? |
| `correct_files` | YES/NO | Are file paths correct? |
| **`overall_attribution_correct`** | **YES/NO** | **Is the problem→repair link correct?** ⭐ |
| `notes` | Text | Any observations? |

---

**BE STRICT with `overall_attribution_correct`:**
- If problem doesn't exist → NO
- If wrong files → NO
- If wrong fix → NO
- If bundled multiple problems → NO
- Only mark YES if the attribution is genuinely correct!

---

**Ready to annotate!** 🎯
