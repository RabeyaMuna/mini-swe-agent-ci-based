# ReCAP Ablation Study: Component Breakdown

This document explains the different components of the ReCAP memory system and their corresponding ablation experiments.

## Overview of Memory Pipeline

The ReCAP memory system has several key components that work together:

1. **Memory Retrieval** - Fetch relevant historical problems
2. **Evidence Gate** (Log-based filtering) - Filter based on CI log evidence
3. **Source Validation** (LLM-based applicability) - LLM reviews if memory applies to current problem
4. **Recurring Detection** - Identify and retrieve recurring problems
5. **Deduplication** - Remove duplicate problems at the end

## Ablation Experiments

### 1. **No Applicability Filtering** (`no_applicability_filtering`)
- **Component removed**: Evidence Gate
- **Environment variable**: `RECAP_ENABLE_EVIDENCE_GATE=0`
- **What it does**: Removes the initial log-based filtering that checks if retrieved memories have evidence in the current CI failure log
- **Location in code**: `memory_plugin.py` - evidence gate filtering before source validation
- **Effect**: All retrieved memories pass through regardless of CI log evidence

### 2. **No Recurring** (`no_recurring`)
- **Component removed**: Recurring problem detection and recall
- **Environment variable**: `RECAP_ENABLE_RECURRING_RECALL=0`
- **What it does**: Disables the detection and retrieval of recurring/common problems from memory
- **Location in code**: `memory_plugin.py` - recurring problem detection
- **Effect**: Only directly-observed CI failures are used, no recurring patterns

### 3. **No Source Validation** (`no_source_validation`) ⭐ NEW
- **Component removed**: LLM-based applicability filtering
- **Environment variable**: `RECAP_ENABLE_SOURCE_VALIDATION=0`
- **What it does**: Removes the LLM review that determines if a retrieved memory is actually applicable based on source code evidence
- **Location in code**: `memory_plugin.py`, method `_review_candidate_source` (lines 639-756)
- **Effect**: All retrieved memories are passed to repair without LLM validation

### 4. **No Deduplication** (`no_deduplication`)
- **Component removed**: Final deduplication step
- **Environment variable**: `RECAP_ENABLE_DEDUPLICATION=0`
- **What it does**: Removes the final step that deduplicates similar/identical problems
- **Location in code**: `memory_plugin.py` - deduplication logic
- **Effect**: Duplicate problems may be sent to repair agent

## Filtering Pipeline Stages

Here's the order of filtering in the full system:

```
1. Memory Retrieval (STAIR)
   ↓
2. Evidence Gate [RECAP_ENABLE_EVIDENCE_GATE]
   - Checks CI log for evidence of retrieved problems
   ↓
3. Source Validation [RECAP_ENABLE_SOURCE_VALIDATION]
   - LLM reviews source code to confirm applicability
   - Returns: confirmed / rejected / unverifiable
   ↓
4. Deduplication [RECAP_ENABLE_DEDUPLICATION]
   - Removes duplicate problems
   ↓
5. Send to Repair Agent
```

## Key Differences

### Evidence Gate vs Source Validation

**Evidence Gate** (no_applicability_filtering):
- **Input**: Retrieved memories + CI failure log
- **Method**: Pattern matching / heuristic checks
- **Question**: "Is there evidence of this problem in the CI log?"
- **Fast**: No LLM call
- **Location**: Before source validation

**Source Validation** (no_source_validation):
- **Input**: Retrieved memories + source code excerpts
- **Method**: LLM review
- **Question**: "Does the source code actually contain this defect?"
- **Slow**: Requires LLM call for each candidate
- **Location**: After evidence gate

### Example Flow

For a retrieved memory about "ImportError: missing foo module":

1. **Evidence Gate**: Checks if CI log contains "ImportError" and "foo"
   - If no → reject (with `RECAP_ENABLE_EVIDENCE_GATE=1`)
   - If yes → pass to source validation

2. **Source Validation**: LLM reviews source code
   - Looks for actual import statements
   - Checks if "foo" is actually missing
   - Returns "confirmed" only if defect is visible in source
   - If disabled (`RECAP_ENABLE_SOURCE_VALIDATION=0`) → skip this check

## Running Ablations

```bash
# Run individual ablations
./scripts/run_ablation_no_applicability_filtering.sh
./scripts/run_ablation_no_recurring.sh
./scripts/run_ablation_no_source_validation.sh

# Compare results
python3 scripts/analyze_ablation_results.py
```

## Expected Outcomes

### No Applicability Filtering
- **More memories used**: All retrieved memories pass through
- **Lower precision**: May include irrelevant memories
- **Higher recall**: Won't miss relevant memories due to conservative filtering

### No Recurring
- **Fewer memories used**: Only current CI failures
- **Miss patterns**: Won't detect recurring problems
- **Faster**: Less retrieval overhead

### No Source Validation ⭐
- **More memories to repair**: No LLM filtering after retrieval
- **May help if LLM is too conservative**: Some "unverifiable" memories might actually work
- **May hurt if LLM is accurate**: More false positives sent to repair
- **Higher token usage**: More memories sent to repair agent
- **Faster filtering**: No LLM calls during validation stage

## Analysis Questions

1. Does source validation improve precision or is it too conservative?
2. How many "unverifiable" memories could have been useful?
3. What's the trade-off between filtering cost and repair cost?
4. Which filtering stage (evidence gate vs source validation) is more important?

## Configuration Files

- `config/ablation_no_applicability_filtering.yaml`
- `config/ablation_no_recurring.yaml`
- `config/ablation_no_source_validation.yaml`

## Environment Variables Reference

```bash
# All components (full system)
export RECAP_ENABLE_EVIDENCE_GATE=1              # Log-based filtering
export RECAP_ENABLE_SOURCE_VALIDATION=1          # LLM-based applicability
export RECAP_ENABLE_RECURRING_RECALL=1           # Recurring detection
export RECAP_ENABLE_DEDUPLICATION=1              # Final dedup
export RECAP_ENABLE_DEPENDENCY_EXPANSION=1       # Dependency problems
export RECAP_ENABLE_FILTERING=1                  # General filtering
export RECAP_ENABLE_ORDERING=1                   # Problem prioritization
export RECAP_ENABLE_MEMORY_OPTIMIZATION=1        # Memory optimization
```
