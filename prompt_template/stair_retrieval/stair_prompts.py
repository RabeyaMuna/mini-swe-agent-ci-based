"""
LLM Prompts for STAIR Memory Retrieval System
All decision-making prompts for Stages 2-5
"""

import json
from typing import Dict, List, Any


def build_common_detection_prompt(l1_items: List[Dict], l2_items: List[Dict], l3_items: List[Dict]) -> str:
    """
    Stage 2: Prompt LLM to detect common/frequent problems across retrieved data.
    """

    prompt = f"""Analyze retrieved CI failure data and identify COMMON/FREQUENT problems.

# RETRIEVED DATA

## L1 (Specific Issues - {len(l1_items)} items)
{_format_l1_summary(l1_items)}

## L2 (Repo Patterns - {len(l2_items)} items)
{_format_l2_summary(l2_items)}

## L3 (Universal Patterns - {len(l3_items)} items)
{_format_l3_summary(l3_items)}

# TASK

Identify problems that are COMMON or FREQUENT across the retrieved data:

1. **Cross-Issue Common**: Problems appearing in multiple L1 issues (40%+ of issues OR 3+ occurrences)
2. **L2 Frequency**: Problems marked as frequent in L2 (e.g., "dependency_version - 4 problems")
3. **Config-Related**: Problems involving config files (.toml, .yml, .json)

For each common problem, extract:
- Failure type
- How many issues it appears in
- Repair actions (from L1/L2/L3)
- Why it's common (root cause pattern)

# OUTPUT (JSON)

{{
  "common_problems": [
    {{
      "failure_type": "dependency_version",
      "appears_in_issues": ["113", "115"],
      "total_occurrences": 4,
      "l2_frequency": 4,
      "config_related": true,
      "repair_action": "Remove version constraints, use poetry lock",
      "why_common": "Config changes break dependency resolution across issues",
      "evidence": "L2 shows 4 problems, appears in 2/3 L1 issues"
    }}
  ],
  "config_problems": [
    {{
      "failure_type": "dependency_version",
      "config_files": ["pyproject.toml"]
    }}
  ]
}}

Return ONLY JSON, no additional text."""

    return prompt


def build_filtering_prompt(
    l1_items: List[Dict],
    l2_items: List[Dict],
    l3_items: List[Dict],
    ci_failure: Dict,
    common_problems: List[Dict]
) -> str:
    """
    Stage 3: Prompt LLM to filter relevant problems and analyze dependencies.
    """

    prompt = f"""You are a CI repair expert. Filter relevant problems from retrieved data to fix current CI failure.

# CURRENT CI FAILURE

Repository: {ci_failure.get('repo')}
Workflow: {ci_failure.get('workflow')}
Failure Type: {ci_failure.get('failure_type', ci_failure.get('stage', 'unknown'))}

Problem:
{ci_failure.get('problem_statement')}

Error Signals:
{json.dumps(ci_failure.get('error_signals', []), indent=2)}

Config Signals:
{json.dumps(ci_failure.get('config_signals', []), indent=2)}

# RETRIEVED DATA

## L1 - Specific Issues ({len(l1_items)} items)
{_format_l1_detailed(l1_items[:3])}

## L2 - Repo Patterns ({len(l2_items)} items)
{_format_l2_detailed(l2_items[:3])}

## L3 - Universal Patterns ({len(l3_items)} items)
{_format_l3_detailed(l3_items[:3])}

## COMMON PROBLEMS (from Stage 2)
{json.dumps(common_problems, indent=2)}

# TASK

Select relevant problems and analyze dependencies:

1. **Filter Relevant Problems**:
   - Match current failure type OR common across issues
   - Error signals overlap with current failure
   - Could be root cause, dependent, or consecutive

2. **Combine Information** from L1/L2/L3:
   - Root Cause: Synthesize from all levels
   - Rationale: WHY (L3) → HOW (L2) → WHAT (L1)
   - Signals: Merge error_signals + config_signals
   - Repair Actions: Concrete steps from L1, strategy from L2, approach from L3

3. **Identify Dependencies**:
   - **Consecutive**: Pipeline cascade (A → B → C) from L2 causal_chain
   - **Dependent**: Hidden problems revealed after fix (from L3 dependent_changes)
   - **Common**: Frequent across issues (from Stage 2)

4. **Evidence**: Cite signal matches and explain relevance

# OUTPUT (JSON)

{{
  "problems": [
    {{
      "problem_id": "problem_0",
      "type": "ci_failure|common|dependent|consecutive",
      "failure_type": "dependency_version",
      "problem": "Clear description",
      "root_cause": "Synthesized from L1/L2/L3",
      "rationale": {{
        "why": "L3: Universal principle",
        "how": "L2: Pattern/strategy",
        "what": "L1: Concrete instance"
      }},
      "signals": {{
        "error_signals": ["signal1", "signal2"],
        "config_signals": ["config1"],
        "match_evidence": "Why these match current failure"
      }},
      "repair_actions": {{
        "strategy": "High-level approach",
        "steps": ["Step 1", "Step 2"],
        "files": ["file1.py"],
        "validation_cmd": "pytest"
      }},
      "source": {{
        "l1": ["issue_113"],
        "l2": ["issue_113"],
        "l3": ["pattern_id"]
      }},
      "confidence": "HIGH|MEDIUM|LOW",
      "priority": 1
    }}
  ],
  "consecutive_sequences": [
    {{
      "sequence": ["problem_0", "problem_1"],
      "reasoning": "Fix 0 → enables 1 to run",
      "source": "L2 causal_chain"
    }}
  ],
  "dependencies": [
    {{
      "primary": "problem_0",
      "reveals": ["problem_1"],
      "reasoning": "Fixing 0 surfaces 1"
    }}
  ]
}}

Be selective - only include ACTUALLY relevant problems. Return ONLY JSON."""

    return prompt


def build_clustering_prompt(problems: List[Dict]) -> str:
    """
    Stage 4: Prompt LLM to cluster similar problems.
    """

    prompt = f"""Cluster similar problems that should be merged.

# PROBLEMS

{json.dumps(problems, indent=2)}

# TASK

Group problems that:
- Have same failure_type
- Target same files
- Have similar fix strategies
- Should be solved together

For each cluster, decide if problems should be MERGED (combined into one) or kept SEPARATE.

# OUTPUT (JSON)

{{
  "clusters": [
    {{
      "cluster_id": "cluster_0",
      "problem_ids": ["problem_0", "problem_1"],
      "should_merge": true,
      "reasoning": "Same failure type, same files, can apply same fix"
    }},
    {{
      "cluster_id": "cluster_1",
      "problem_ids": ["problem_2"],
      "should_merge": false,
      "reasoning": "Standalone problem, different context"
    }}
  ]
}}

Return ONLY JSON."""

    return prompt


def build_final_generation_prompt(clusters: List[Dict], problems: List[Dict]) -> str:
    """
    Stage 5: Prompt LLM to generate final structured problem list.
    """

    prompt = f"""Generate final structured problems ready for agent execution.

# CLUSTERS

{json.dumps(clusters, indent=2)}

# ORIGINAL PROBLEMS

{json.dumps(problems, indent=2)}

# TASK

For each cluster:
- If should_merge=true: Merge problems into ONE with combined info
- If should_merge=false: Keep as single problem

Each final problem must have:
- Clear problem description
- Complete root cause
- Detailed repair actions with steps
- Files to modify
- Validation command
- Priority (1=highest)

Sort by priority: ci_failure (1) > common (2) > dependent (3) > consecutive (4)

# OUTPUT (JSON)

{{
  "final_problems": [
    {{
      "problem": "Clear description of what needs fixing",
      "root_cause": "What fundamentally caused this",
      "rationale": {{
        "why": "Universal principle (from L3)",
        "how": "Pattern/strategy (from L2)",
        "what": "Concrete case (from L1)"
      }},
      "signals": {{
        "error_signals": ["signal1"],
        "config_signals": ["config1"]
      }},
      "repair_actions": {{
        "strategy": "Overall approach",
        "steps": [
          "Step 1: ACTION - file.py - do X",
          "Step 2: ACTION - verify with cmd"
        ],
        "files": ["file1.py", "config.toml"],
        "validation_cmd": "pytest tests/"
      }},
      "type": "ci_failure|common|dependent|consecutive",
      "failure_type": "dependency_version",
      "confidence": "HIGH|MEDIUM|LOW",
      "priority": 1,
      "source": {{
        "l1": ["issue_id"],
        "l2": ["issue_id"],
        "l3": ["pattern_id"]
      }}
    }}
  ]
}}

Return ONLY JSON."""

    return prompt


# Helper formatting functions

def _format_l1_summary(items: List[Dict]) -> str:
    """Summarize L1 for common detection."""
    lines = []
    for item in items:
        issue = item['item']
        problems = issue.get('problems', [])
        failure_types = [p.get('failure_type') for p in problems]
        lines.append(f"Issue {issue.get('issue_id')}: {len(problems)} problems - {', '.join(set(failure_types))}")
    return '\n'.join(lines)


def _format_l2_summary(items: List[Dict]) -> str:
    """Summarize L2 for common detection."""
    lines = []
    for item in items:
        issue = item['item']
        failure_ids = issue.get('failure_identify', [])
        lines.append(f"Issue {issue.get('issue_id')}: {', '.join(failure_ids[:3])}")
    return '\n'.join(lines)


def _format_l3_summary(items: List[Dict]) -> str:
    """Summarize L3 for common detection."""
    lines = []
    for item in items:
        pattern = item['item']
        lines.append(f"Pattern {pattern.get('pattern_id')}: {pattern.get('failure_type')} - {pattern.get('failure_pattern', '')[:100]}")
    return '\n'.join(lines)


def _format_l1_detailed(items: List[Dict]) -> str:
    """Detailed L1 for filtering."""
    blocks = []
    for item in items:
        issue = item['item']
        problems_json = json.dumps(issue.get('problems', [])[:3], indent=2)
        blocks.append(f"""
Issue {issue.get('issue_id')} (score: {item['score']:.3f})
Repo: {issue.get('repo')}
Workflow: {issue.get('workflow')}
Problems: {problems_json}
""")
    return '\n---\n'.join(blocks)


def _format_l2_detailed(items: List[Dict]) -> str:
    """Detailed L2 for filtering."""
    blocks = []
    for item in items:
        issue = item['item']
        strategies_json = json.dumps(issue.get('repair_strategies', [])[:3], indent=2)
        blocks.append(f"""
Issue {issue.get('issue_id')} (score: {item['score']:.3f})
Failure Types: {json.dumps(issue.get('failure_identify', []), indent=2)}
Repair Strategies: {strategies_json}
""")
    return '\n---\n'.join(blocks)


def _format_l3_detailed(items: List[Dict]) -> str:
    """Detailed L3 for filtering."""
    blocks = []
    for item in items:
        pattern = item['item']
        blocks.append(f"""
Pattern {pattern.get('pattern_id')} (score: {item['score']:.3f})
Type: {pattern.get('failure_type')}
Pattern: {pattern.get('failure_pattern')}
Problem: {pattern.get('problem', '')[:200]}
Reasoning: {pattern.get('reasoning', '')[:200]}
Universal Fix: {json.dumps(pattern.get('universal_fix', {}), indent=2)}
Dependent Changes: {json.dumps(pattern.get('dependent_changes', []), indent=2)}
""")
    return '\n---\n'.join(blocks)
