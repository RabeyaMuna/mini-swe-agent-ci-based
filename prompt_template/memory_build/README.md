# Memory Build Prompts (L1, L2, L3)

All prompts for building memory from CI failures at three levels of abstraction.

## Structure

```
prompt_template/memory_build/
├── __init__.py                  # Main exports
├── l2_repair_sequence.py        # L2 repair sequencing prompts
├── l3_universal_patterns.py     # L3 pattern extraction prompts
└── README.md                    # This file
```

## Memory Levels

### L1: File-Level Problems
- **Generated**: Mechanically (not LLM-based)
- **Content**: Direct file-to-problem mapping
- **Purpose**: Preserve exact decomposition data

### L2: Repair Sequence
- **Generated**: LLM-based with optimal ordering
- **Content**: Ordered list of problems to fix
- **Purpose**: Actionable repair plan

### L3: Universal Patterns
- **Generated**: LLM-based pattern extraction
- **Content**: Generalizable problem patterns
- **Purpose**: Reusable fixes for similar issues

## Usage

### L2 Prompts

```python
from prompt_template.memory_build import l2_repair_sequence

# Full sequence (all problems at once)
prompt = l2_repair_sequence.format_full_sequence(
    issue_id="issue-123",
    repo="owner/repo",
    problems=atomic_problems,
    dependencies=dep_graph,
    strict_json_rules=STRICT_JSON_RULES,
)

# Validation group (one validation at a time)
prompt = l2_repair_sequence.format_validation_group(
    validation_cmd="black --check .",
    problems=problems_for_this_validation,
    strict_json_rules=STRICT_JSON_RULES,
)

result = llm.invoke(prompt)
```

### L3 Prompts

```python
from prompt_template.memory_build import l3_universal_patterns

# Full extraction (all problems at once)
prompt = l3_universal_patterns.format_full_extraction(
    l2_problems=repair_sequence["problems"],
    strict_json_rules=STRICT_JSON_RULES,
)

# Validation group (one validation at a time)
prompt = l3_universal_patterns.format_validation_group(
    validation_cmd="python -m mypy",
    problems=l2_problems_for_mypy,
    strict_json_rules=STRICT_JSON_RULES,
)

# Cross-validation dependencies
prompt = l3_universal_patterns.format_cross_validation_deps(
    patterns=all_extracted_patterns,
    strict_json_rules=STRICT_JSON_RULES,
)

patterns = llm.invoke(prompt)
```

## Available Prompts

### L2 Prompts (`l2_repair_sequence.py`)

#### `L2_REPAIR_SEQUENCE_FULL`
- **Purpose**: Generate complete repair sequence with optimal ordering
- **Strategy**: Single LLM call for all problems (Tier 1)
- **Returns**: JSON with ordered problems array
- **Used in**: `_l2_tier1_single_llm()`

#### `L2_VALIDATION_GROUP`
- **Purpose**: Generate repair entries for single validation
- **Strategy**: Grouped by validation (Tier 2)
- **Returns**: JSON with problems array
- **Used in**: `_l2_tier2_grouped_llm()`

### L3 Prompts (`l3_universal_patterns.py`)

#### `L3_FULL_EXTRACTION`
- **Purpose**: Extract all universal patterns at once
- **Strategy**: Single LLM call (small datasets)
- **Returns**: JSON array of pattern objects
- **Used in**: `_stage5_l3_single()`

#### `L3_VALIDATION_GROUP`
- **Purpose**: Extract patterns for single validation
- **Strategy**: Grouped by validation (large datasets)
- **Returns**: JSON array of patterns
- **Used in**: `_extract_patterns_for_validation_group()`

#### `L3_CROSS_VALIDATION_DEPS`
- **Purpose**: Identify dependencies between patterns
- **Strategy**: Post-processing after group extraction
- **Returns**: JSON array with dependencies added
- **Used in**: `_identify_cross_pattern_dependencies()`

## Ordering Rules (L2)

### Primary vs Hidden
1. **PRIMARY problems** go FIRST
   - Visible in CI logs
   - CI stopped here

2. **HIDDEN problems** go AFTER
   - Would have failed later
   - Never reached by CI

### Within Each Group
- Sort by `validation_order` (lower first)
- Use dependencies to break ties
- Independent problems can be in any order

### Example

```
Input: [
  {id: 1, validation_order: 8, problem_type: "hidden"},
  {id: 2, validation_order: 7, problem_type: "primary"},
  {id: 3, validation_order: 11, problem_type: "hidden"}
]

Output: [
  {problem_id: 2, validation_order: 7},  // Primary
  {problem_id: 1, validation_order: 8},  // Hidden, lower order
  {problem_id: 3, validation_order: 11}  // Hidden, higher order
]
```

## Pattern Independence Rules (L3)

### Separate Patterns
- Different validations = separate patterns
- Example: mypy ≠ black (different patterns)

### Linked Patterns
- Only link if ACTUAL dependency exists
- Example: Code change requires config change

### Do NOT Link
- Happened in same commit (timing ≠ dependency)
- Same failure type (similarity ≠ dependency)
- No clear cause-effect

## Integration Example

### Before (Inline Prompt)
```python
# In decompose_ci_failure.py
prompt = f"""Generate L2 REPAIR SEQUENCE...
{json.dumps(problems_for_llm, indent=2)}
...
"""
response = llm.invoke(prompt)
```

### After (Using Template)
```python
from prompt_template.memory_build import l2_repair_sequence

prompt = l2_repair_sequence.format_full_sequence(
    issue_id=issue_id,
    repo=repo,
    problems=problems_for_llm,
    dependencies=dependencies,
    strict_json_rules=STRICT_JSON_RULES,
)

response = llm.invoke(prompt)
```

## Output Schemas

### L2 Output
```json
{
  "problems": [
    {
      "problem_id": 1,
      "verification_cmd": "string",
      "failure_type": "string",
      "problem": "string",
      "root_cause": "string",
      "fix_strategy": "string (single paragraph)",
      "pattern_detected": {...} | null,
      "files": ["array"],
      "estimated_time_minutes": 5
    }
  ],
  "total_problems": 10
}
```

### L3 Output
```json
[
  {
    "pattern_id": "unique_id",
    "failure_type": "category",
    "verification_cmd": "command",
    "failure_pattern": "what breaks",
    "problem": "root cause",
    "universal_fix": {
      "approach": "strategy",
      "steps": ["array"],
      "applies_to": ["scenarios"]
    },
    "examples": [{
      "file": "path",
      "before": "code",
      "after": "code"
    }],
    "dependent_problems": [{
      "pattern_id": "other",
      "relationship": "type",
      "rationale": "why"
    }]
  }
]
```

## Testing

```python
# Test L2 formatting
from prompt_template.memory_build import l2_repair_sequence

prompt = l2_repair_sequence.format_full_sequence(
    issue_id="test",
    repo="test/repo",
    problems=[{"id": 1, "problem": "test"}],
    dependencies={},
    strict_json_rules="JSON only",
)

assert "test/repo" in prompt
assert "{issue_id}" not in prompt

# Test L3 formatting
from prompt_template.memory_build import l3_universal_patterns

prompt = l3_universal_patterns.format_full_extraction(
    l2_problems=[{"problem_id": 1}],
    strict_json_rules="JSON only",
)

assert "problem_id" in prompt
```

## Version History

- **v1.0** (2026-07-24): Initial extraction
  - l2_repair_sequence.py (2 prompts)
  - l3_universal_patterns.py (3 prompts)
