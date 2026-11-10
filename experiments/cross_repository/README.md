# Cross-Repository Evaluation

**Question**: Can ReCAP work without same-repository historical experience?

## Experimental Design

### Evaluation Set: First 100 Instances
- Use first 100 instances from `data/eval_issue_ids.json`
- Fixed, reproducible evaluation set
- Same 100 instances used across all three configurations

### Memory Construction: Cross-Repository Only
- Identify which repos the 100 eval instances belong to → **TARGET repos**
- Build memory from all OTHER repos → **SOURCE repos**
- Select ~50 memory instances from SOURCE repos
- **Guarantee**: For each eval instance, its repo has ZERO instances in memory

### Three Configurations

| Configuration | Evaluation | Memory | Retrieval |
|---------------|-----------|--------|-----------|
| No-Experience | First 100 instances | Empty | N/A |
| ReCAP (Cross-Repository) | Same 100 instances | ~50 from SOURCE repos | L1/L2/L3 (all levels) |
| ReCAP (L3-only) | Same 100 instances | Same ~50 from SOURCE repos | L3 only |

### Key Insight: "Same-Repository Experience Excluded"
- **TARGET repos**: Repos containing the 100 eval instances
- **SOURCE repos**: All other repos (disjoint from TARGET)
- Memory built exclusively from SOURCE repos
- This tests pure cross-repository transfer

## Steps to Run

### Step 1: Analyze First 100 Instances
```bash
python 01_analyze_first_100.py
```
- Reads first 100 IDs from eval_issue_ids.json
- Identifies which repos they belong to (TARGET repos)
- Identifies SOURCE repos (all others with memory instances)
- Output: `results/cross_repository/01_first_100_analysis.json`

### Step 2: Build Cross-Repository Memory
```bash
python 02_build_cross_repo_memory.py
```
- Selects ~50 memory instances from SOURCE repos
- Ensures diversity across repos and failure types
- Output: `results/cross_repository/cross_repo_memory.json`

### Step 3: Prepare Evaluation Set
```bash
python 03_prepare_eval_set.py
```
- Creates evaluation set from first 100 instances
- Output: `results/cross_repository/cross_repo_eval_set.jsonl`

### Step 4: Run Experiments
```bash
# You will run these three configurations manually using your existing pipeline
# Each uses cross_repo_eval_set.jsonl (same 100 instances)

# Config 1: No-Experience
# Config 2: ReCAP Cross-Repository (memory from SOURCE repos, L1/L2/L3)
# Config 3: ReCAP L3-only (memory from SOURCE repos, L3 only)
```

### Step 5: Calculate Results
```bash
python 04_calculate_table_r6.py
```
- Compares Pass@1 across three configurations
- Generates Table R6 in JSON, Markdown, LaTeX
- Output: `results/cross_repository/table_r6.*`

## Expected Results: Table R6

| Configuration | N | Pass@1 (%) | Gain (pp) |
|---------------|---|------------|-----------|
| No-Experience | 100 | TBD | Reference |
| ReCAP (Cross-Repository) | 100 | TBD | TBD |
| ReCAP (L3-only) | 100 | TBD | TBD |

**Answers Reviewer Concerns:**
- **jcJU Q4**: L3-only vs No-Experience → Does L3 transfer across repos?
- **xm6z Q3**: Normal cross-repo vs No-Experience → Can ReCAP work without same-repo history?
