#!/usr/bin/env python3
"""
Add technical dependency analysis to decomposed issues.
Analyzes how file changes depend on each other with technical reasoning.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from decompose_ci_failure import LitellmModel


def analyze_file_dependencies(issue: Dict[str, Any], llm: Any) -> List[Dict[str, Any]]:
    """
    Analyze which file changes cause/require other file changes.
    Returns technical dependency information.
    """
    
    problems = issue.get("problems", [])
    if not problems:
        return []
    
    # Group problems by file to identify cross-file dependencies
    file_to_problems = {}
    for problem in problems:
        for file_path in problem.get("affected_files", []):
            if file_path not in file_to_problems:
                file_to_problems[file_path] = []
            file_to_problems[file_path].append(problem)
    
    # Identify potential dependency patterns
    dependency_candidates = []
    
    # Pattern 1: Config/dependency files that affect other files
    config_files = ["pyproject.toml", "package.json", "requirements.txt", "Cargo.toml", ".taplo.toml"]
    
    for problem in problems:
        for affected_file in problem.get("affected_files", []):
            # Check if it's a config file
            is_config = any(cfg in affected_file for cfg in config_files)
            
            if is_config:
                # This might be a primary change that affects others
                dependency_candidates.append({
                    "primary_problem": problem,
                    "primary_file": affected_file,
                    "is_config": True
                })
    
    # Pattern 2: Type/import changes that propagate
    for problem in problems:
        how_fixed = problem.get("how_fixed", "").lower()
        if any(keyword in how_fixed for keyword in ["import", "union", "type", "from typing"]):
            dependency_candidates.append({
                "primary_problem": problem,
                "primary_file": problem.get("affected_files", [None])[0],
                "is_import_change": True
            })
    
    if not dependency_candidates:
        return []
    
    # Use LLM to analyze dependencies with technical detail
    prompt = f"""Analyze file change dependencies with technical reasoning.

You have {len(problems)} atomic problems from a CI failure fix.
Some changes cause or enable other changes. Identify these dependencies.

=== ATOMIC PROBLEMS ===

{json.dumps(problems, indent=2)}

=== TASK ===

For each primary change (config file, dependency, type change), identify:
1. What other files/problems depend on it
2. WHY they depend on it (technical reason with line numbers, error codes)
3. HOW the fixes are connected (exact before/after code changes)
4. The complete technical flow (step-by-step causal chain)

=== OUTPUT FORMAT ===

{{
  "file_dependencies": [
    {{
      "primary_file": "dev/pyproject.toml",
      "primary_problem_id": 3,
      "primary_change": {{
        "what_is_error": "Line 18: taplo dependency commented out '#taplo = \"==0.9.3\"'. Without this, poetry install skips taplo, causing validation order 14 ('taplo fmt --check') to fail with exit code 127 (command not found).",
        "fixes_done": "Line 18: changed from '#taplo = \"==0.9.3\"' (commented) to 'taplo = \"==0.9.3\"' (active). This tells poetry to download and install taplo==0.9.3 from PyPI during 'poetry install --all-extras'.",
        "validation_order": 14,
        "technical_context": "Poetry parses [tool.poetry.dev-dependencies] section. Commented lines are ignored. Active lines trigger pip install. Without taplo in venv, shell returns 'command not found' when CI runs 'taplo fmt --check'."
      }},
      "dependent_changes": [
        {{
          "affected_files": ["dev/pyproject.toml", "framework/pyproject.toml"],
          "affected_problem_ids": [8],
          "reason": "After taplo is installed (primary fix), 'taplo fmt --check' can execute. It reads .taplo.toml config (indent_string = 2 spaces, space_around_equals = true). It scans all .toml files, compares actual vs expected formatting. Violations found: dev/pyproject.toml:25 missing space in 'black=\"24.2.0\"' (expects 'black = \"24.2.0\"'), framework/pyproject.toml:180 uses 4-space indent instead of 2. Taplo exits code 1 (formatting differs), blocking CI.",
          "fixes": [
            {{
              "file": "dev/pyproject.toml",
              "line": 25,
              "before": "black=\"24.2.0\"",
              "after": "black = \"24.2.0\"",
              "rule_violated": ".taplo.toml: space_around_equals = true",
              "why_this_fix": "Taplo enforces spaces around '=' for readability. Parser sees 'black=\"' as violation. Adding spaces makes it compliant: 'black = \"'. This satisfies taplo's formatting rules."
            }}
          ]],
          "dependency_type": "tool_installation_enablement",
          "technical_flow": [
            "Step 1: dev/pyproject.toml:18 uncommented (taplo dependency activated)",
            "Step 2: CI runs 'poetry install --all-extras' in framework/",
            "Step 3: Poetry reads pyproject.toml [tool.poetry.dev-dependencies]",
            "Step 4: Finds 'taplo = \"==0.9.3\"', downloads from PyPI, installs to .venv/bin/taplo",
            "Step 5: CI runs 'taplo fmt --check' (validation order 14)",
            "Step 6: taplo binary available in PATH, executes successfully",
            "Step 7: taplo reads .taplo.toml config file",
            "Step 8: Scans all .toml files in repo, checks formatting rules",
            "Step 9: Detects violations (missing spaces, wrong indent)",
            "Step 10: Exits code 1 (would exit 0 if formatting matched)"
          ],
          "repair_order": "MUST fix primary (install taplo) BEFORE dependent (format .toml). Without taplo installed, formatting changes are meaningless - validation still fails with 'command not found'."
        }}
      ]]
    }}
  ]]
}}

=== REQUIREMENTS ===

1. **Technical precision**: Include line numbers, error codes, exit codes, tool versions
2. **Exact code quotes**: Show before/after for both primary and dependent changes
3. **Causal chain**: Explain step-by-step how primary change enables dependent changes
4. **Tools and mechanisms**: Mention poetry, pip, PATH, config files, validation commands
5. **Skip if no dependencies**: Return empty array if changes are independent

Only analyze actual dependencies (one change enables/requires another).
Don't invent dependencies - use evidence from problems.
"""

    try:
        result = llm.predict(prompt)
        
        # Parse JSON from response
        if isinstance(result, str):
            # Extract JSON from markdown or text
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return []
        
        if isinstance(result, dict):
            return result.get("file_dependencies", [])
        
        return []
        
    except Exception as e:
        print(f"  Warning: Dependency analysis failed: {e}")
        return []


def remove_redundancy_and_add_dependencies(input_file: str, output_file: str, llm: Any = None):
    """
    1. Remove duplicate validation_sequence
    2. Add technical dependency analysis
    """
    
    with open(input_file) as f:
        issues = json.load(f)
    
    print(f"Processing {len(issues)} issues...")
    
    for idx, issue in enumerate(issues, 1):
        issue_id = issue.get("original_issue_id", "?")
        print(f"\nIssue {idx}/{len(issues)}: {issue_id}")
        
        # 1. Remove redundant validation_sequence
        if "validation_sequence" in issue and "benchmark_ci_context" in issue:
            if "validation_sequence" in issue["benchmark_ci_context"]:
                # validation_sequence exists in both places - remove top-level one
                print(f"  ✓ Removing redundant validation_sequence (keeping in benchmark_ci_context)")
                del issue["validation_sequence"]
        
        # Also remove from workflow_validation_context if it duplicates
        if "benchmark_ci_context" in issue:
            ci_ctx = issue["benchmark_ci_context"]
            if "workflow_validation_context" in ci_ctx:
                wf_ctx = ci_ctx["workflow_validation_context"]
                if "validation_sequence" in wf_ctx and "validation_sequence" in ci_ctx:
                    # Duplicate - remove from workflow_validation_context
                    print(f"  ✓ Removing redundant workflow_validation_context.validation_sequence")
                    del wf_ctx["validation_sequence"]
        
        # 2. Add dependency analysis (only if LLM provided)
        if llm and issue.get("problems"):
            print(f"  Analyzing dependencies for {len(issue.get('problems', []))} problems...")
            dependencies = analyze_file_dependencies(issue, llm)
            
            if dependencies:
                issue["file_dependencies"] = dependencies
                print(f"  ✓ Found {len(dependencies)} dependency relationships")
            else:
                print(f"  ℹ️  No cross-file dependencies detected")
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(issues, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remove redundancy and add dependency analysis")
    parser.add_argument("--input", required=True, help="Input decomposed_issues.json")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--add-dependencies", action="store_true", help="Add dependency analysis (requires LLM)")
    parser.add_argument("--model", default="anthropic/claude-3.7-sonnet", help="LLM model for dependency analysis")
    
    args = parser.parse_args()
    
    llm = None
    if args.add_dependencies:
        print("Initializing LLM for dependency analysis...")
        llm = LitellmModel(model_name=args.model)
    
    remove_redundancy_and_add_dependencies(args.input, args.output, llm)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
