# Unified Ablations Experiment

**Covers**: 
- Reviewer jcJU Q2 (Component ablation)
- Reviewer u2Pr Q1 (Hidden-problem prediction ablation)

## Configurations

### Baseline Configs (Reuse Existing Results)
1. **baseline** (No-Experience) - Reuse from `results/miniswe-agent/baseline_*`
2. **full_recap** - Reuse from `results/miniswe-agent/bidirectional/l1_l2_l3_*`

### New Configs to Run (5 total)
3. **no_dependency_expansion** - Disable dependency expansion only
4. **no_recurring_recall** - Disable recurring-problem recall only
5. **no_applicability_filter** - Disable applicability filtering only
6. **no_ordering** - Disable dependency-based ordering only
7. **no_hidden_problems** - Disable BOTH dependency expansion + recurring recall

## Dataset
- **100 instances** for all configs
- **First 50 instances** used for component ablation table (Reviewer jcJU Q2)
- **All 100 instances** used for hidden-problem table (Reviewer u2Pr Q1)

## Implementation Strategy
- **Reuse**: `memory_plugin.stair_retrieval.STAIRRetrieval` base class
- **Modify**: Create wrapper classes that disable specific stages
- **No changes to core code**: All modifications via inheritance

## Output
- `results/03_unified_ablations/component_ablation_table.json` (50 instances)
- `results/03_unified_ablations/hidden_problem_table_r1.json` (100 instances)
