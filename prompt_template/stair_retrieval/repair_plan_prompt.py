"""
Repair Plan Generation Prompt

Generates detailed repair plans for each problem identified by STAIR retrieval.
This prompt takes a structured problem and generates step-by-step execution plan.
"""

import json
from typing import Dict, Any


def build_repair_plan_prompt(problem: Dict[str, Any], strict_json_rules: str = "") -> str:
    """
    Build repair plan generation prompt for a single problem.

    Args:
        problem: Structured problem from Stage 5 with:
            - problem: Description
            - root_cause: What caused it
            - repair_actions: Strategy, steps, files, validation_cmd
            - rationale: WHY/HOW/WHAT breakdown
            - signals: Error and config signals
        strict_json_rules: JSON formatting rules from llm_invoker

    Returns:
        Prompt string for LLM
    """

    prompt = f"""Generate DETAILED REPAIR PLAN for this CI failure problem.

# PROBLEM

{json.dumps(problem, indent=2)}

# YOUR TASK

Generate a detailed, step-by-step repair plan that an agent can execute.

## Required Analysis

1. **Understand the Problem**:
   - What: {problem.get('problem', 'N/A')}
   - Why: {problem.get('root_cause', 'N/A')}
   - Evidence: {json.dumps(problem.get('signals', {}), indent=2)}

2. **Review Repair Strategy**:
   - Strategy: {problem.get('repair_actions', {}).get('strategy', 'N/A')}
   - Files: {problem.get('repair_actions', {}).get('files', [])}
   - Validation: {problem.get('repair_actions', {}).get('validation_cmd', 'N/A')}

3. **Apply Rationale** (WHY → HOW → WHAT):
   - WHY (Universal Principle): {problem.get('rationale', {}).get('why', 'N/A')}
   - HOW (Pattern/Strategy): {problem.get('rationale', {}).get('how', 'N/A')}
   - WHAT (Concrete Action): {problem.get('rationale', {}).get('what', 'N/A')}

## Output Format

Generate a repair plan with:

1. **Pre-Checks**: What to verify before starting (files exist, current state)
2. **Repair Steps**: Detailed, ordered steps with:
   - Action type: CODE/CONFIG/DEPENDENCY/VALIDATION
   - File/command to modify
   - Specific change to make
   - Why this step is needed
3. **Validation**: How to verify the fix worked
4. **Rollback Plan**: What to do if fix fails

# OUTPUT (JSON)

{{
  "problem_id": "{problem.get('problem_id', 'unknown')}",
  "failure_type": "{problem.get('failure_type', 'unknown')}",
  "repair_plan": {{
    "summary": "One-sentence summary of the fix",
    "pre_checks": [
      {{
        "check": "Verify file exists",
        "command": "ls path/to/file.py",
        "expected": "File exists"
      }}
    ],
    "steps": [
      {{
        "step_number": 1,
        "action_type": "CODE|CONFIG|DEPENDENCY|VALIDATION",
        "file": "path/to/file.py",
        "description": "What to do",
        "change": {{
          "type": "replace|insert|delete|run_command",
          "target": "Specific line/section to change",
          "new_content": "What to change it to",
          "command": "Command to run (if type=run_command)"
        }},
        "reasoning": "Why this step is needed (from rationale)",
        "expected_outcome": "What should happen after this step"
      }}
    ],
    "validation": {{
      "command": "{problem.get('repair_actions', {}).get('validation_cmd', '')}",
      "expected_output": "What indicates success",
      "fallback_checks": [
        "Alternative way to verify if main validation fails"
      ]
    }},
    "rollback": {{
      "if_step_fails": {{
        "1": "How to undo step 1",
        "2": "How to undo step 2"
      }},
      "full_rollback": "How to completely undo all changes"
    }},
    "dependencies": {{
      "reveals_problems": {json.dumps(problem.get('dependencies', {}).get('reveals', []))},
      "must_fix_before": {json.dumps(problem.get('dependencies', {}).get('requires', []))},
      "consecutive_next": "problem_id of next in sequence (if consecutive)"
    }},
    "risk_assessment": {{
      "risk_level": "LOW|MEDIUM|HIGH",
      "potential_issues": ["What could go wrong"],
      "mitigation": ["How to reduce risk"]
    }}
  }},
  "estimated_time": "X minutes",
  "confidence": "{problem.get('confidence', 'MEDIUM')}"
}}

## IMPORTANT GUIDELINES

1. **Be Specific**: "Change line 42" not "Update the file"
2. **Be Sequential**: Steps must be in execution order
3. **Be Complete**: Include ALL steps from strategy
4. **Be Safe**: Include pre-checks and rollback
5. **Use Rationale**: Reference WHY/HOW/WHAT in reasoning
6. **Handle Dependencies**: Note if fixing this reveals other problems

{strict_json_rules}

Return ONLY the JSON, no additional text."""

    return prompt


def build_batch_repair_plans_prompt(
    problems: list[Dict[str, Any]],
    consecutive_sequences: list[Dict[str, Any]],
    strict_json_rules: str = ""
) -> str:
    """
    Build prompt to generate repair plans for multiple problems in optimal order.

    Args:
        problems: List of structured problems from Stage 5
        consecutive_sequences: Sequences from Stage 3 (pipeline cascades)
        strict_json_rules: JSON formatting rules

    Returns:
        Prompt string for batch repair planning
    """

    prompt = f"""Generate REPAIR PLANS for multiple CI problems in OPTIMAL EXECUTION ORDER.

# PROBLEMS

{json.dumps(problems, indent=2)}

# CONSECUTIVE SEQUENCES

{json.dumps(consecutive_sequences, indent=2)}

# YOUR TASK

1. **Determine Execution Order**:
   - Respect consecutive sequences (A must be fixed before B)
   - Respect dependencies (fixing X reveals Y)
   - Group independent problems (can be done in parallel)

2. **Generate Repair Plan for Each**:
   - Detailed steps (CODE/CONFIG/DEPENDENCY/VALIDATION)
   - Pre-checks and validation
   - Rollback plan
   - Time estimate

3. **Identify Blocking Relationships**:
   - Which problems block others
   - Which can be done in parallel
   - Which are revealed after fixes

# OUTPUT (JSON)

{{
  "execution_order": [
    {{
      "phase": 1,
      "problems": ["problem_0"],
      "reasoning": "Must fix first - blocks pipeline",
      "can_parallelize": false
    }},
    {{
      "phase": 2,
      "problems": ["problem_1", "problem_2"],
      "reasoning": "Independent, can run in parallel after phase 1",
      "can_parallelize": true
    }}
  ],
  "repair_plans": [
    {{
      "problem_id": "problem_0",
      "phase": 1,
      "repair_plan": {{
        "summary": "Fix description",
        "pre_checks": [...],
        "steps": [...],
        "validation": {{...}},
        "rollback": {{...}}
      }},
      "blocks": ["problem_1", "problem_2"],
      "estimated_time": "5 minutes"
    }}
  ],
  "total_estimated_time": "15 minutes",
  "critical_path": ["problem_0", "problem_1"]
}}

{strict_json_rules}

Return ONLY JSON."""

    return prompt
