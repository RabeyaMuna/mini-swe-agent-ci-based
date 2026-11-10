# Experiments

## Available Experiments

### 1. Cold-Start Analysis
**Directory**: `cold_start/`  
**Purpose**: Partition evaluation instances by repository-level historical experience

```bash
cd experiments/cold_start
python run_analysis.py
```

### 2. Cross-Repository Evaluation
**Directory**: `04_cross_repository_l3/`  
**Purpose**: Test ReCAP with same-repository experience excluded

```bash
# Setup (once)
bash experiments/04_cross_repository_l3/setup.sh
bash ./run_miniswe_direct.sh \  experimental_data/eval_issue_ids.json \
  L1+L2+L3 cross_repo deepseek-v4-flash \
  "" experimental_data/eval_set.jsonl 1
# Run experiments
bash experiments/04_cross_repository_l3/run_cross_repo_l1_l2_l3.sh
bash experiments/04_cross_repository_l3/run_cross_repo_l3_only.sh
```

### 3. Other Experiments
- `01_whole_repair_history/` - Repair history analysis
- `02_human_attribution/` - Human attribution study
- `03_unified_ablations/` - Unified ablation studies

See individual README.md files in each directory for details.
