# Cold-Start Analysis - Repository History Stratification

**Reviewer u2Pr Q2**: Break down test results between repositories that have prior instances in the history bank versus cold-start repositories.

## Overview

This analysis partitions the 408 evaluation instances into two groups based on repository-level historical experience, then calculates Pass@1 for each group.

## Simple Flow

```
Step 1: Analyze Memory & Eval Instances → Extract Repos
   ↓
Step 2: Partition based on Repos → History-Available vs Cold-Start  
   ↓
Step 3: Calculate Pass@1 from Results → For each model/config
   ↓
Step 4: Generate Reports → Table R7
```

---

## How to Run

### Quick Start (All Models)

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/experiments/cold_start

# Run complete analysis
python run_analysis.py
```

### Step by Step

#### Step 1: Analyze Instances and Repos
```bash
python 01_analyze_instances_and_repos.py
```

**What it does:**
- Loads 157 memory instances from L1 memory
- Loads 408 eval instances from results
- Extracts repository information for each
- Identifies which repos have historical experience

**Output:**
- `results/cold_start/01_instance_repos.json`

---

#### Step 2: Partition Instances
```bash
python 02_partition_instances.py
```

**What it does:**
- Takes repos from Step 1
- Partitions 408 eval instances into:
  - **History-Available**: Eval instances from repos WITH historical experience
  - **Cold-Start**: Eval instances from repos WITHOUT historical experience

**Output:**
- `results/cold_start/02_partition.json`
- Shows: 375 History-Available, 33 Cold-Start

---

#### Step 3: Calculate Pass@1 for Each Model
```bash
python 03_calculate_results.py
```

**What it does:**
- For each model configuration:
  - Loads baseline (no-experience) results
  - Loads ReCAP (bidirectional) results
  - Calculates Pass@1 for History-Available group
  - Calculates Pass@1 for Cold-Start group

**Supported models:**
- mini-SWE-agent + DeepSeek-V4-Flash
- mini-SWE-agent + MiniMax-M2.5

**Output:**
- `results/cold_start/03_results_deepseek_v4_flash.json`
- `results/cold_start/03_results_minimax_m2_5.json`

---

#### Step 4: Generate Table R7
```bash
python 04_generate_table_r7.py
```

**What it does:**
- Combines results from all models
- Generates Table R7 in multiple formats (JSON, Markdown, LaTeX)

**Output:**
- `results/cold_start/table_r7.json`
- `results/cold_start/table_r7.md`
- `results/cold_start/table_r7.tex`

---

## File Structure

```
experiments/cold_start/
├── README.md                              # This file
├── 01_analyze_instances_and_repos.py      # Step 1
├── 02_partition_instances.py              # Step 2  
├── 03_calculate_results.py                # Step 3
├── 04_generate_table_r7.py                # Step 4
└── run_analysis.py                        # Run all steps

results/cold_start/
├── 01_instance_repos.json                 # Instance → repo mapping
├── 02_partition.json                      # History-Available vs Cold-Start
├── 03_results_deepseek_v4_flash.json      # DeepSeek results
├── 03_results_minimax_m2_5.json           # MiniMax results
├── table_r7.json                          # Final table (JSON)
├── table_r7.md                            # Final table (Markdown)
└── table_r7.tex                           # Final table (LaTeX)
```

---

## Expected Results

### Table R7: Repair Performance by Repository-History Availability

**DeepSeek-V4-Flash:**

| Repository Group | Instances | No-Exp Pass@1 | ReCAP Pass@1 | Gain |
|------------------|-----------|---------------|--------------|------|
| History-Available | 375 | 23.5% | 33.1% | +9.6 pp |
| Cold-Start | 33 | 21.2% | 30.3% | +9.1 pp |
| Overall | 408 | 23.3% | 32.8% | +9.6 pp |

**MiniMax-M2.5:**

| Repository Group | Instances | No-Exp Pass@1 | ReCAP Pass@1 | Gain |
|------------------|-----------|---------------|--------------|------|
| History-Available | TBD | TBD | TBD | TBD |
| Cold-Start | TBD | TBD | TBD | TBD |
| Overall | TBD | TBD | TBD | TBD |

---

## Customization

### Add New Model

Edit `03_calculate_results.py` and add to `MODELS` dict:

```python
MODELS = {
    "deepseek_v4_flash": {
        "name": "DeepSeek-V4-Flash",
        "baseline_path": "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl",
        "recap_path": "results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl",
    },
    "your_model": {
        "name": "Your Model Name",
        "baseline_path": "path/to/baseline/results.jsonl",
        "recap_path": "path/to/recap/results.jsonl",
    }
}
```

Then re-run Step 3 and Step 4.

---

## Verification

To verify calculations:
```bash
# Show detailed breakdown
python 03_calculate_results.py --verbose

# Shows:
# - Exact instance counts
# - Which instances passed
# - Which instances improved/regressed
```

---

## Key Findings

1. **Both groups benefit from ReCAP** - Similar improvements for History-Available (+9.6pp) and Cold-Start (+9.1pp)

2. **Cross-repository transfer works** - Cold-Start repos with NO historical experience still improve

3. **L3 patterns generalize** - Demonstrates transferable repair knowledge across repositories

---

## Troubleshooting

**Q: Missing instances in results?**
A: Some instances may not complete. Check console output for missing IDs.

**Q: Different partition counts?**
A: Verify L1 memory path in Step 1 is correct (`data/bidirect_trs/failure_memory.json`)

**Q: Need to re-run just one step?**
A: Steps are independent after their prerequisites. Re-run any step individually.
