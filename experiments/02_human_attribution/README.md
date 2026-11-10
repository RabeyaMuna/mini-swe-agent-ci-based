# Human Attribution Evaluation (Reviewer jcJU Q1ii, xm6z Q1)

**Question**: How accurate are ReCAP's problem–repair associations?

## Approach

1. Select 50 historical repair episodes
2. LLM proposes candidate associations
3. Human annotator verifies each association
4. Compute precision, recall, F1

Compare three methods:
- Endpoint-only
- Development-only  
- Full ReCAP (bidirectional)

## Output

- `results/02_human_attribution/table_r2.json`
- Annotation data for 50 episodes
