# Analysis Scripts

Scripts for analyzing mini-swe-agent CI repair results across different decomposition directions.

## compare_directions_v2.py

**Recommended**: Comprehensive comparison with detailed ID lists for re-running experiments.

### Quick Start

```bash
python analysis/compare_directions_v2.py
```

This generates **20+ ID list files** organized by comparison type.

### Generated ID Lists

#### 1. Common Issues (Solved by Multiple Directions)

- **`ids_common_all_three.txt`** (75 IDs) - Solved by ALL three directions (the "easy" ones)
- **`ids_forward_backward_only.txt`** (19 IDs) - Forward + Backward solved, Bidirectional missed
- **`ids_forward_bidirectional_only.txt`** (4 IDs) - Forward + Bidirectional solved, Backward missed
- **`ids_backward_bidirectional_only.txt`** (14 IDs) - Backward + Bidirectional solved, Forward missed

#### 2. Direction-Specific Issues (Only One Direction Solved)

- **`ids_only_forward.txt`** (13 IDs) - Only Forward succeeded
- **`ids_only_backward.txt`** (7 IDs) - Only Backward succeeded
- **`ids_only_bidirectional.txt`** (9 IDs) - Only Bidirectional succeeded

#### 3. Per-Direction Success/Failure

- **`ids_forward_success.txt`** (111 IDs) - All issues Forward solved
- **`ids_forward_failure.txt`** (297 IDs) - All issues Forward failed
- **`ids_backward_success.txt`** (115 IDs) - All issues Backward solved
- **`ids_backward_failure.txt`** (293 IDs) - All issues Backward failed
- **`ids_bidirectional_success.txt`** (102 IDs) - All issues Bidirectional solved
- **`ids_bidirectional_failure.txt`** (306 IDs) - All issues Bidirectional failed

#### 4. Comparison Lists (What One Direction Solved That Another Missed)

- **`ids_forward_beats_backward.txt`** (17 IDs) - Forward solved, Backward missed
- **`ids_backward_beats_forward.txt`** (21 IDs) - Backward solved, Forward missed
- **`ids_bidirectional_beats_forward.txt`** (23 IDs) - Bidirectional solved, Forward missed
- **`ids_bidirectional_beats_backward.txt`** (13 IDs) - Bidirectional solved, Backward missed
- **`ids_forward_and_backward_beat_bidirectional.txt`** (19 IDs) - Both Forward AND Backward solved, but Bidirectional missed

#### 5. All Failures

- **`ids_all_failed.txt`** (267 IDs) - None of the three directions succeeded (the "hard" problems)

### Usage Examples

**Re-run only forward-specific successes:**
```bash
python run_miniswe.py --issue-ids-file analysis/ids_only_forward.txt --direction forward
```

**Re-run issues where backward beat forward:**
```bash
python run_miniswe.py --issue-ids-file analysis/ids_backward_beats_forward.txt --direction backward
```

**Re-run all failures with improved logic:**
```bash
python run_miniswe.py --issue-ids-file analysis/ids_all_failed.txt
```

**Use with codex:**
```bash
python codex/scripts/run_codex_ci_repair.py \
  --issue-ids-file analysis/ids_only_backward.txt \
  --ablations L1+L2+L3
```

### Custom Analysis

```bash
# Different model results
python analysis/compare_directions_v2.py --model l1_l2_l3_gpt4

# Custom paths
python analysis/compare_directions_v2.py \
  --forward results/custom/forward/jobs.jsonl \
  --backward results/custom/backward/jobs.jsonl \
  --bidirectional results/custom/bidir/jobs.jsonl

# Custom output directory
python analysis/compare_directions_v2.py --output-dir /tmp/analysis
```

## compare_directions.py

Original detailed report with repository-level analysis.

```bash
python analysis/compare_directions.py
```

Includes:
- Venn diagram analysis
- Examples of direction-specific successes  
- Repository performance by direction
- Detailed console report

## Key Insights from Analysis

### Overall Performance
- **Backward** performs slightly better (28.2% vs 27.2% forward, 25.0% bidirectional)
- **~18%** of problems solved by all three (consensus = easy)
- **~65%** fail in all three (the hard problems)
- **~7%** show direction-specific success (direction matters!)

### When Direction Matters

**Forward-only successes (13)**:
- Primarily linting/formatting issues
- Works well for issues that follow code structure linearly
- Examples: axolotl, accelerate, deepcode linting

**Backward-only successes (7)**:
- Mixed: linting, tests, docs
- May catch dependency issues forward misses
- Examples: aws-cli tests, google-cloud-python docs

**Bidirectional-only successes (9)**:
- Complex lint issues requiring reconciliation
- Benefits from considering both directions
- Examples: browser-use, openai-python, robotframework

**Forward + Backward beat Bidirectional (19)**:
- Both simple directions work, but bidirectional adds complexity
- May indicate bidirectional reconciliation introduces conflicts
- Good candidates for debugging bidirectional logic

### Practical Applications

1. **Debugging Direction Logic**: Use `ids_forward_and_backward_beat_bidirectional.txt` to find cases where bidirectional reconciliation fails

2. **Testing Improvements**: Re-run `ids_only_X.txt` files after improving direction-specific logic

3. **Ablation Studies**: Compare success rates on specific ID subsets before/after code changes

4. **Focus on Hard Problems**: Use `ids_all_failed.txt` (267 IDs) to focus optimization efforts

5. **Validate Fixes**: After fixing patch merger bugs, re-run specific ID lists to verify improved success rates
