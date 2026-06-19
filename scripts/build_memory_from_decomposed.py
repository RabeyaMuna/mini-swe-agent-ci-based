#!/usr/bin/env python3
"""
Build L1/L2/L3 Memory from Decomposed Issues

L1: Per-file memory
L2: Per-issue memory (clean format)
L3: Universal patterns (LLM analyzes decomposition to extract patterns)

Usage:
    python scripts/build_memory_from_decomposed.py \
        --decomposed data/trs/decomposed_issues.json \
        --output-dir data/trs \
        --model openrouter/minimax/minimax-m2.5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Simple LLM wrapper for L3 analysis
import litellm
from dotenv import load_dotenv
load_dotenv()

class SimpleLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, prompt: str):
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        class Result:
            content = response.choices[0].message.content or ""
        return Result()


# ============================================================================
# L1 BUILDER - Per-File Memory
# ============================================================================

def extract_code_changes_deterministic(how_fixed: str, problem_desc: str) -> List[Dict]:
    """
    Deterministically extract code changes from how_fixed description.
    Handles multiple changes, numbered lists, and various patterns.
    """
    import re

    changes = []

    # Step 1: Split by numbered patterns like (1), (2), [1], [2], "1.", "2."
    numbered_parts = re.split(r'\(\d+\)|\[\d+\]|\d+\.', how_fixed)

    # If we found numbered parts, process each separately
    if len(numbered_parts) > 2:  # More than just the intro text
        parts_to_process = numbered_parts[1:]  # Skip first (before numbering)
    else:
        parts_to_process = [how_fixed]  # Process whole text

    for part in parts_to_process:
        if not part.strip():
            continue

        # Pattern 1: "from 'X' to 'Y'" or "from X to Y"
        from_to = re.findall(r"from\s+['\"]([^'\"]+?)['\"]\s+(?:to|leaving only)\s+['\"]([^'\"]+?)['\"]", part, re.IGNORECASE)
        for before, after in from_to:
            changes.append({
                "from": before.strip(),
                "to": after.strip()
            })

        # Pattern 2: "Changed/Replaced 'X' to 'Y'"
        changed = re.findall(r"(?:Changed|Replaced|Renamed)\s+(?:the\s+)?(?:\w+\s+)?(?:from\s+)?['\"]([^'\"]+?)['\"]\s+to\s+['\"]([^'\"]+?)['\"]", part, re.IGNORECASE)
        for before, after in changed:
            changes.append({
                "from": before.strip(),
                "to": after.strip()
            })

        # Pattern 3: Removed 'X' (deletion)
        removed = re.findall(r"Removed?\s+(?:the\s+)?(?:\w+\s+)?['\"]([^'\"]+?)['\"]", part, re.IGNORECASE)
        for rem in removed:
            changes.append({
                "from": rem.strip(),
                "to": "",
                "type": "deletion"
            })

        # Pattern 4: Added 'X' (addition)
        added = re.findall(r"Added?\s+(?:the\s+)?(?:\w+\s+)?['\"]([^'\"]+?)['\"]", part, re.IGNORECASE)
        for add in added:
            changes.append({
                "from": "",
                "to": add.strip(),
                "type": "addition"
            })

    # Pattern 5: Before/After blocks (if no changes found yet)
    if not changes:
        before_after = re.findall(r"Before:\s*['\"]?([^'\"]+?)['\"]?\s+After:\s*['\"]?([^'\"]+?)['\"]?(?:\n|$)", how_fixed, re.IGNORECASE | re.DOTALL)
        for before, after in before_after:
            changes.append({
                "from": before.strip(),
                "to": after.strip()
            })

    # Pattern 6: 'X' → 'Y' or 'X' became 'Y'
    if not changes:
        arrow = re.findall(r"['\"]([^'\"]+?)['\"]\s+(?:→|became)\s+['\"]([^'\"]+?)['\"]", how_fixed)
        for before, after in arrow:
            changes.append({
                "from": before.strip(),
                "to": after.strip()
            })

    # Clean up: Remove duplicates and empty changes
    seen = set()
    unique_changes = []
    for change in changes:
        # Create a key for deduplication
        key = (change.get("from", ""), change.get("to", ""))
        if key not in seen and (change.get("from") or change.get("to")):
            seen.add(key)
            unique_changes.append(change)

    return unique_changes


def build_l1_memory(decomposed_issues: List[Dict]) -> List[Dict]:
    """Build L1 (per-file) memory with deterministic code changes."""

    print("\n" + "="*80)
    print("BUILDING L1 (PER-FILE MEMORY)")
    print("="*80)

    l1_memories = []
    file_map = {}

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        repo = issue.get("repo", "unknown")
        problems = issue.get("problems") or issue.get("atomic_problems", [])

        for problem in problems:
            validation_cmd = problem.get("validation_cmd", "")
            failure_type = problem.get("failure_type", "")
            problem_desc = problem.get("problem", "")
            root_cause = problem.get("root_cause", "")
            how_fixed = problem.get("how_fixed", "")

            # Extract code changes deterministically
            changes = extract_code_changes_deterministic(how_fixed, problem_desc)

            # Extract files
            affected_files = problem.get("affected_files", [])
            if isinstance(affected_files, dict):
                files = affected_files.get("files", [])
            else:
                files = affected_files

            # Create one L1 entry per file
            for file_path in files[:10]:
                if not file_path:
                    continue

                key = f"{repo}::{file_path}::{failure_type}"
                if key in file_map:
                    continue
                file_map[key] = True

                # Build dependencies: other files in same problem (exclude current file)
                dependencies = []
                if len(files) > 1:
                    # Get other files (dependent files that must be changed together)
                    other_files = [f for f in files if f != file_path]

                    # If many files (pattern), sample a few
                    sample_size = 3 if len(other_files) > 5 else len(other_files)
                    for dep_file in other_files[:sample_size]:
                        dependencies.append({
                            "file": dep_file,
                            "what_changes": how_fixed if how_fixed else "Same fix as primary file",
                            "why_this_change": f"Part of same {failure_type} issue - all {len(files)} files need the same fix"
                        })

                    # If we sampled, add note about total
                    if len(other_files) > sample_size:
                        dependencies.append({
                            "file": f"... and {len(other_files) - sample_size} more files",
                            "what_changes": "Same pattern as above",
                            "why_this_change": f"Total {len(files)} files affected by this issue"
                        })

                l1_entry = {
                    "memory_level": "L1",
                    "file": file_path,
                    "repo": repo,
                    "issue_type": failure_type,
                    "validation_cmd": validation_cmd,
                    "problem": problem_desc,
                    "root_cause": root_cause,
                    "fix_strategy": how_fixed,
                    "dependencies": dependencies,  # ← NEW: Dependent files
                    "changes": changes  # ← NEW: Deterministically extracted changes
                }

                l1_memories.append(l1_entry)

    print(f"  Created {len(l1_memories)} L1 (per-file) entries")
    return l1_memories


# ============================================================================
# L2 BUILDER - Per-Issue Memory (Clean Format)
# ============================================================================

def extract_files_from_problem(problem: Dict) -> List[str]:
    """Extract file list from problem."""
    affected_files = problem.get("affected_files", [])

    if isinstance(affected_files, dict):
        return affected_files.get("files", [])
    elif isinstance(affected_files, list):
        return affected_files
    return []


def extract_fixes_from_problem(problem: Dict) -> List[Dict[str, str]]:
    """Extract code changes from problem."""

    fixes = []
    file_changes = problem.get("file_changes", [])

    for fc in file_changes:
        if not isinstance(fc, dict):
            continue

        change = fc.get("change")
        if change and isinstance(change, dict):
            from_code = change.get("from", "")
            to_code = change.get("to", "")
            why = change.get("why", "")

            if from_code and to_code and from_code != "..." and to_code != "...":
                fix = {"from": from_code, "to": to_code}
                if why:
                    fix["why_this_fix"] = why
                fixes.append(fix)

    if not fixes:
        how_fixed = problem.get("how_fixed", "")
        if how_fixed:
            fixes.append({"description": how_fixed})

    return fixes


def transform_problem_to_clean_l2(problem: Dict) -> Dict[str, Any]:
    """Transform to clean L2 format."""

    return {
        "problem_id": problem.get("problem_id", 0),
        "verification_cmd": problem.get("validation_cmd", ""),
        "issue_type": problem.get("failure_type", ""),
        "files": extract_files_from_problem(problem),
        "problem": {
            "what_wrong": problem.get("problem", ""),
            "root_cause": problem.get("root_cause", "")
        },
        "fixes": extract_fixes_from_problem(problem),
        "depends_on": problem.get("depends_on", []),
        "blocks": problem.get("blocks", []),
        "validation_order": problem.get("validation_order", 0)
    }


def build_l2_memory(decomposed_issues: List[Dict]) -> List[Dict]:
    """Build clean L2 memory."""

    print("\n" + "="*80)
    print("BUILDING L2 (PER-ISSUE MEMORY - CLEAN FORMAT)")
    print("="*80)

    l2_memories = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id", "unknown")
        repo = issue.get("repo", "unknown")
        sha_fail = issue.get("sha_fail", "")

        problems = issue.get("problems") or issue.get("atomic_problems", [])
        if not problems:
            continue

        clean_problems = [transform_problem_to_clean_l2(p) for p in problems]

        l2_entry = {
            "issue_id": issue_id,
            "repo": repo,
            "sha_fail": sha_fail,
            "atomic_problems": clean_problems
        }

        l2_memories.append(l2_entry)
        print(f"  Issue {issue_id}: {len(clean_problems)} atomic problems")

    total_problems = sum(len(l2.get("atomic_problems", [])) for l2 in l2_memories)
    print(f"  Created {len(l2_memories)} L2 entries with {total_problems} total problems")

    return l2_memories


# ============================================================================
# L3 BUILDER - Universal Patterns (LLM Analysis)
# ============================================================================

def extract_atomic_problems_from_decomposed(decomposed_issues: List[Dict]) -> List[Dict]:
    """Extract all atomic problems with context."""

    all_problems = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        repo = issue.get("repo", "unknown")
        issue_id = issue.get("original_issue_id", "unknown")

        problems = issue.get("problems") or issue.get("atomic_problems", [])

        for problem in problems:
            enriched = {
                **problem,
                "repo": repo,
                "issue_id": issue_id,
                "issue_type": problem.get("failure_type", "unknown"),
                "affected_files": extract_files_from_problem(problem)
            }
            all_problems.append(enriched)

    return all_problems


def group_problems_by_type(problems: List[Dict]) -> Dict[str, List[Dict]]:
    """Group problems by failure type."""

    groups = defaultdict(list)

    for problem in problems:
        issue_type = problem.get("issue_type") or problem.get("failure_type", "unknown")
        if issue_type == "unknown":
            continue
        groups[issue_type].append(problem)

    return dict(groups)


def analyze_pattern_with_llm(issue_type: str, problems: List[Dict], llm) -> Dict[str, Any]:
    """Use LLM to analyze ALL problems and extract universal pattern."""

    # Prepare ALL examples for comprehensive analysis
    # LLM sees all variations to extract what's truly universal
    # NOTE: We don't include specific file paths - only file types matter for universal patterns
    examples = []
    for p in problems:
        examples.append({
            "validation_cmd": p.get("validation_cmd", ""),
            "problem": p.get("problem", ""),
            "root_cause": p.get("root_cause", ""),
            "how_fixed": p.get("how_fixed", "")
        })

    # Get file extensions
    all_files = []
    for p in problems:
        all_files.extend(p.get("affected_files", []))

    from pathlib import Path
    extensions = list(set(Path(f).suffix for f in all_files if f and Path(f).suffix))
    file_type = ", ".join(sorted(extensions)) if extensions else "various"

    # Build LLM prompt
    prompt = f"""Analyze ALL {len(examples)} examples of "{issue_type}" CI failures and extract the universal pattern.

FAILURE TYPE: {issue_type}
FILE TYPES: {file_type}

ALL EXAMPLES:
{json.dumps(examples, indent=2)}

INSTRUCTIONS:
1. Analyze what is COMMON across ALL examples (not just one)
2. Identify the root cause that applies to ALL cases
3. Extract the code pattern that causes this failure universally
4. Determine the fix strategy that works for ALL cases

Extract a reusable pattern with this EXACT structure:

{{
  "problem": {{
    "root_cause": "<what is the root cause that applies to ALL examples? What common patterns lead to this failure?>",
    "code_pattern_that_fails": "<specific code pattern that causes this failure across ALL examples>",
    "why_this_fails": "<technical reason why this pattern fails validation>"
  }},
  "fix_pattern": {{
    "what_to_change": "<what needs to be modified in the code>",
    "how_to_fix": "<concrete steps to fix this issue - make it actionable>",
    "why_this_fix": "<why this fix resolves the issue>",
    "example_transform": "<show before → after pattern from the examples>"
  }}
}}

Be SPECIFIC and ACTIONABLE. Extract actual patterns from ALL the examples above.

Output ONLY valid JSON (no markdown, no explanations):"""

    try:
        response = llm.invoke(prompt)

        # Try to parse JSON
        content = response.content.strip()

        # Remove markdown code blocks if present
        import re
        # Remove ```json or ``` fences
        content = re.sub(r'^```(?:json)?\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        content = content.strip()

        # Try to find JSON object boundaries
        if not content.startswith('{'):
            # Try to find the first { and last }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]
                print(f"    ℹ️  Extracted JSON from position {start} to {end}")
            else:
                print(f"    ⚠️  Warning: No JSON braces found, trying to add them")
                # Try to add missing braces
                content = '{' + content + '}'

        result = json.loads(content)

        # Validate result has required fields
        if not result.get("problem") or not result.get("fix_pattern"):
            raise ValueError("LLM response missing problem or fix_pattern fields")

        return result

    except json.JSONDecodeError as e:
        print(f"    ❌ ERROR: JSON parsing failed for '{issue_type}'")
        print(f"       Error: {e}")
        print(f"       LLM response (first 200 chars): {content[:200] if 'content' in locals() else 'N/A'}")
        raise  # Re-raise to stop execution instead of silently returning empty

    except Exception as e:
        print(f"    ❌ ERROR: LLM analysis failed for '{issue_type}': {e}")
        if 'content' in locals():
            print(f"       LLM response (first 200 chars): {content[:200]}")
        raise  # Re-raise instead of silently returning empty


def build_l3_memory(decomposed_issues: List[Dict], llm=None) -> List[Dict]:
    """Build L3 universal patterns by analyzing decomposition with LLM."""

    print("\n" + "="*80)
    print("BUILDING L3 (UNIVERSAL PATTERNS)")
    print("="*80)

    if not llm:
        print("  WARNING: No LLM provided, L3 will be limited")
        return []

    # Extract and group problems
    all_problems = extract_atomic_problems_from_decomposed(decomposed_issues)
    print(f"  Extracted {len(all_problems)} atomic problems")

    groups = group_problems_by_type(all_problems)
    print(f"  Grouped into {len(groups)} failure types")

    # Analyze each group with LLM
    l3_patterns = []

    for issue_type, problems in groups.items():
        print(f"  Analyzing '{issue_type}' ({len(problems)} examples)...")

        # Get validation cmd and file types
        validation_cmds = list(set(p.get("validation_cmd", "") for p in problems if p.get("validation_cmd")))
        primary_cmd = validation_cmds[0] if validation_cmds else ""

        # Get file extensions
        all_files = []
        for p in problems:
            all_files.extend(p.get("affected_files", []))

        from pathlib import Path
        extensions = list(set(Path(f).suffix for f in all_files if f and Path(f).suffix))
        file_type_desc = ", ".join(sorted(extensions)) + " files" if extensions else "various files"

        # Analyze with LLM
        analysis = analyze_pattern_with_llm(issue_type, problems, llm)

        # Build pattern
        pattern = {
            "pattern_id": issue_type.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", ""),
            "issue_type": issue_type,
            "validation_cmd": primary_cmd,
            "file_type_pattern": file_type_desc,
            **analysis
        }

        l3_patterns.append(pattern)
        print(f"    ✓ Analyzed")

    print(f"  Created {len(l3_patterns)} L3 universal patterns")
    return l3_patterns


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build L1/L2/L3 memory from decomposed issues"
    )
    parser.add_argument(
        "--decomposed",
        default="data/trs/decomposed_issues.json",
        help="Path to decomposed issues"
    )
    parser.add_argument(
        "--output-dir",
        default="data/trs",
        help="Output directory for L1/L2/L3"
    )
    parser.add_argument(
        "--model",
        default="openrouter/minimax/minimax-m2.5",
        help="LLM model for L3 analysis"
    )
    args = parser.parse_args()

    # Load decomposed issues
    decomposed_path = Path(args.decomposed)
    if not decomposed_path.exists():
        print(f"ERROR: Decomposed issues not found: {decomposed_path}")
        return 1

    print("="*80)
    print("Building L1/L2/L3 Memory from Decomposed Issues")
    print("="*80)

    with open(decomposed_path) as f:
        decomposed_issues = json.load(f)

    print(f"Loaded {len(decomposed_issues)} decomposed issues")

    # Initialize LLM for L3
    llm = SimpleLLM(args.model)

    # Build L1
    l1_memories = build_l1_memory(decomposed_issues)

    # Build L2
    l2_memories = build_l2_memory(decomposed_issues)

    # Build L3 (with LLM analysis)
    l3_patterns = build_l3_memory(decomposed_issues, llm=llm)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)

    # Save L1
    l1_path = output_dir / "failure_memory.json"
    with open(l1_path, "w") as f:
        json.dump(l1_memories, f, indent=2)
    print(f"  ✓ L1: {l1_path} ({len(l1_memories)} entries)")

    # Save L2
    l2_path = output_dir / "repo_memory.json"
    with open(l2_path, "w") as f:
        json.dump(l2_memories, f, indent=2)
    total_l2_problems = sum(len(l2.get("atomic_problems", [])) for l2 in l2_memories)
    print(f"  ✓ L2: {l2_path} ({len(l2_memories)} entries, {total_l2_problems} problems)")

    # Save L3
    l3_path = output_dir / "cross_memory.json"
    with open(l3_path, "w") as f:
        json.dump(l3_patterns, f, indent=2)
    print(f"  ✓ L3: {l3_path} ({len(l3_patterns)} patterns)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  L1 (per-file):        {len(l1_memories)} entries")
    print(f"  L2 (per-issue):       {len(l2_memories)} entries ({total_l2_problems} atomic problems)")
    print(f"  L3 (universal):       {len(l3_patterns)} patterns (LLM-analyzed)")
    print(f"\n  All using CLEAN structure")
    print(f"  Output directory:     {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
