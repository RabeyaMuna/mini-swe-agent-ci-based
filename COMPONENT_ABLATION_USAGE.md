# Component Ablation Implementation - Usage Guide

## ✅ Implementation Complete

All component ablation flags have been successfully implemented and verified.

## 🔧 What Was Added

### 4 New Command-Line Flags:
```bash
--disable-dependency-expansion    # Skip STAGE 4: dependency extraction
--disable-recurring-recall        # Skip STAGE 5: common patterns detection
--disable-filtering              # Skip STAGE 2: applicability filtering
--disable-ordering               # Skip STAGE 6: dependency-based reordering
```

### Modified Files:
1. `scripts/run_miniswe_ci_bench.py` - Added flags and environment variables
2. `memory_plugin/memory_plugin.py` - Read env vars, pass to STAIRRetrieval
3. `memory_plugin/stair_retrieval.py` - Accept parameters, conditional execution

## ✅ Backward Compatibility Guaranteed

**Your existing commands will work EXACTLY as before:**

```bash
# This still works unchanged - all components enabled by default
./run_miniswe_direct.sh "" L1+L2+L3 bidirectional deepseek-v4-flash

# Or the Python script directly
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash
```

**Why it's backward compatible:**
- All parameters default to `True` (enabled)
- Environment variables default to `"1"` (enabled)
- Only when you explicitly pass `--disable-X` flags do components get skipped

## 📊 Reviewer Experiment Commands

### For jcJU Question 2 (4 separate component ablations on 50 instances):

#### 1. Without Dependency Expansion Only
```bash
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --issue_regex "^(43|63|37|...)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash \
  --disable-dependency-expansion \
  --output_root results/ablations/no_dependency \
  --workers 1
```

#### 2. Without Recurring-Problem Recall Only
```bash
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --issue_regex "^(43|63|37|...)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash \
  --disable-recurring-recall \
  --output_root results/ablations/no_recurring \
  --workers 1
```

#### 3. Without Applicability Filtering Only
```bash
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --issue_regex "^(43|63|37|...)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash \
  --disable-filtering \
  --output_root results/ablations/no_filtering \
  --workers 1
```

#### 4. Without Repair Ordering Only
```bash
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --issue_regex "^(43|63|37|...)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash \
  --disable-ordering \
  --output_root results/ablations/no_ordering \
  --workers 1
```

### For u2Pr Question 1 (Hidden-Problem Prediction - 100 instances):

#### Without BOTH Dependency Expansion AND Recurring Recall
```bash
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --issue_regex "^(43|63|37|...)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash \
  --disable-dependency-expansion \
  --disable-recurring-recall \
  --output_root results/ablations/no_hidden_prediction \
  --workers 1
```

### Full ReCAP Baseline (for comparison):
```bash
python scripts/run_miniswe_ci_bench.py \
  --dataset experimental_data/eval_set.jsonl \
  --issue_regex "^(43|63|37|...)$" \
  --ablation L1+L2+L3 \
  --direction bidirectional \
  --model deepseek-v4-flash \
  --output_root results/ablations/full_recap \
  --workers 1
```

## 📝 Log Messages

When a component is disabled, you'll see clear log messages:

```
[Memory]     SKIP Filtering (disabled by component ablation)
[Memory]     SKIP Dependency extraction (disabled by component ablation)
[Memory] STAGE 5: SKIP Common patterns detection (disabled by component ablation)
[Memory] STAGE 6: SKIP Reordering (disabled by component ablation)
```

## 🧪 Testing

### Quick Verification (already run):
```bash
./verify_ablation_quick.sh
```

### Full Test Suite (runs 6 tests with 2 instances each):
```bash
./test_component_ablation.sh
```

## 📊 Expected Results Tables

### Table for jcJU Q2 (Component Ablations - 50 instances)
```
Agent         | Model              | Configuration                              | Pass@1 (%)
--------------|--------------------|--------------------------------------------|------------
Minisweagent  | Deepseek-v4-Flash | Full ReCAP                                 | X
Minisweagent  | Deepseek-v4-Flash | ReCAP without Dependency Expansion         | Y
Minisweagent  | Deepseek-v4-Flash | ReCAP without Recurring-Problem Recall     | Z
Minisweagent  | Deepseek-v4-Flash | ReCAP without Applicability Filtering      | A
Minisweagent  | Deepseek-v4-Flash | ReCAP without Repair Ordering              | B
```

### Table for u2Pr Q1 (Hidden-Problem Prediction - 100 instances)
```
Model              | Configuration                              | Pass@1 (%)
-------------------|--------------------------------------------|------------
Deepseek-v4-Flash | Full ReCAP                                 | X
Deepseek-v4-Flash | ReCAP without Hidden-Problem Prediction    | Y
```

## 🎯 Component Mappings

| Reviewer Term                    | Flag                              | Stage   | Function                                  |
|----------------------------------|-----------------------------------|---------|-------------------------------------------|
| Dependency Expansion             | `--disable-dependency-expansion` | STAGE 4 | `_stage_4_extract_related_problems()`    |
| Recurring-Problem Recall         | `--disable-recurring-recall`     | STAGE 5 | `_stage_5_detect_common_patterns()`      |
| Applicability Filtering          | `--disable-filtering`            | STAGE 2 | `_stage_2_per_problem_filtering()`       |
| Repair Ordering                  | `--disable-ordering`             | STAGE 6 | `_stage_6_final_reorder()`               |
| Hidden-Problem Prediction        | Both `--disable-dependency-expansion` AND `--disable-recurring-recall` | STAGES 4+5 | Both functions |

## 💡 Tips

1. **Use `--slice` for testing**: Test with a small subset first
   ```bash
   --slice "0:5"  # First 5 instances
   ```

2. **Monitor logs**: Look for "SKIP" messages to confirm ablations are working

3. **Compare results**: Run Full ReCAP alongside each ablation for direct comparison

4. **Reuse results**: Hidden-problem prediction results can be shared between jcJU Q2 and u2Pr Q1

## 🔍 Troubleshooting

**Q: How do I know if the flags are working?**  
A: Look for "SKIP ... (disabled by component ablation)" messages in the logs

**Q: Will this break my existing experiments?**  
A: No! All existing commands work unchanged. Flags only take effect when explicitly passed.

**Q: Can I combine multiple flags?**  
A: Yes! For example: `--disable-dependency-expansion --disable-recurring-recall`

**Q: How do I see what flags are available?**  
A: Run: `python scripts/run_miniswe_ci_bench.py --help`
