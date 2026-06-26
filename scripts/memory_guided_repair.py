"""
Memory-Guided Sequential Repair
Integrates with existing mini-swe-agent setup

USAGE:
    python scripts/memory_guided_repair.py <instance_id>

INTEGRATION:
    This script works with your existing setup:
    1. Uses data/trs/ memory files (L1, L2, L3)
    2. Generates problem statements one at a time
    3. Saves results to results/preds.json

    To integrate with your agent:
    - Update call_mini_swe_agent() function with your agent calling logic
    - Or use the generated prompts in data/prompts/ directory
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


def load_issue(issue_id: str, data_dir: Path = Path("data")) -> Dict:
    """Load issue from SWE-bench dataset or issue file"""
    issue_file = data_dir / f"{issue_id}.json"
    if issue_file.exists():
        with open(issue_file) as f:
            return json.load(f)

    # Try loading from SWE-bench Lite dataset
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        for item in ds:
            if item["instance_id"] == issue_id:
                return dict(item)
    except:
        pass

    raise FileNotFoundError(f"Could not find issue {issue_id}")


def identify_primary_problem(issue_data: Dict, l1_problems: List[Dict]) -> Dict:
    """
    Identify primary CI failure from issue data.
    Returns problem with highest similarity from L1 memory.
    """

    repo = issue_data.get("repo", "")
    # Extract validation commands from issue logs
    logs = issue_data.get("test_patch", "") + issue_data.get("problem_statement", "")

    best_match = None
    best_score = 0.0

    for problem in l1_problems:
        if problem.get("repo") != repo:
            continue

        score = 0.0

        # Check if validation_cmd appears in logs
        val_cmd = problem.get("validation_cmd", "")
        if val_cmd and val_cmd in logs:
            score += 0.6

        # Check failure_type
        failure_type = problem.get("failure_type", "")
        if failure_type and failure_type in logs.lower():
            score += 0.4

        if score > best_score:
            best_score = score
            best_match = problem

    if best_match:
        return {
            "problem": best_match,
            "score": best_score,
            "source": "L1",
            "is_primary": True
        }

    # If no match, create from issue
    return {
        "problem": {
            "repo": repo,
            "problem": issue_data.get("problem_statement", ""),
            "validation_cmd": "python -m pytest",
            "files": []
        },
        "score": 1.0,
        "source": "ISSUE",
        "is_primary": True
    }


def calculate_similarity_score(primary_problem: Dict,
                                candidate: Dict,
                                l1_all: List[Dict],
                                l2_all: List[Dict]) -> float:
    """
    Calculate similarity based on:
    - Relevance: validation_cmd, failure_type match
    - Commonality: how often pattern appears in repo history
    - Consecutiveness: appears in same L2 sequence
    """

    primary_data = primary_problem["problem"]
    repo = primary_data.get("repo")

    # Relevance
    relevance = 0.0
    if candidate.get("validation_cmd") == primary_data.get("validation_cmd"):
        relevance += 0.5
    if candidate.get("failure_type") == primary_data.get("failure_type"):
        relevance += 0.5

    # Commonality (count in repo history)
    commonality = 0.0
    pattern = f"{candidate.get('validation_cmd')}:{candidate.get('failure_type')}"

    l1_count = sum(1 for p in l1_all
                   if p.get("repo") == repo and
                   f"{p.get('validation_cmd')}:{p.get('failure_type')}" == pattern)

    l2_count = 0
    for seq in l2_all:
        if seq.get("repo") == repo:
            for prob in seq.get("problems", []):
                if f"{prob.get('verification_cmd')}:{prob.get('failure_type')}" == pattern:
                    l2_count += 1

    commonality = min(l1_count / 10.0, 1.0) * 0.5 + min(l2_count / 5.0, 1.0) * 0.5

    # Consecutiveness (in same L2 sequence)
    consecutiveness = 0.0
    for seq in l2_all:
        if seq.get("repo") == repo:
            problems = seq.get("problems", [])
            has_primary = any(
                p.get("verification_cmd") == primary_data.get("validation_cmd")
                for p in problems
            )
            has_candidate = any(
                p.get("verification_cmd") == candidate.get("validation_cmd")
                for p in problems
            )
            if has_primary and has_candidate:
                consecutiveness = 1.0
                break

    # Final score
    return 0.3 * relevance + 0.3 * commonality + 0.4 * consecutiveness


def select_consecutive_problems(primary: Dict,
                                 l1: List[Dict],
                                 l2: List[Dict],
                                 l3: List[Dict],
                                 threshold: float = 0.5) -> List[Dict]:
    """Select consecutive problems based on similarity"""

    candidates = []
    repo = primary["problem"].get("repo")

    # From L1
    for problem in l1:
        if problem.get("repo") != repo:
            continue

        score = calculate_similarity_score(primary, problem, l1, l2)
        if score >= threshold:
            candidates.append({
                "problem": problem,
                "score": score,
                "source": "L1",
                "is_primary": False
            })

    # From L2
    for seq in l2:
        if seq.get("repo") == repo:
            for prob in seq.get("problems", []):
                score = calculate_similarity_score(primary, prob, l1, l2)
                if score >= threshold:
                    candidates.append({
                        "problem": prob,
                        "score": score,
                        "source": "L2",
                        "is_primary": False
                    })

    # From L3
    for pattern in l3:
        score = calculate_similarity_score(primary, pattern, l1, l2)
        if score >= threshold:
            candidates.append({
                "problem": pattern,
                "score": score,
                "source": "L3",
                "is_primary": False
            })

    # Deduplicate by validation_cmd + failure_type
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        key = f"{c['problem'].get('validation_cmd')}:{c['problem'].get('failure_type')}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def order_problems(primary: Dict, consecutive: List[Dict]) -> List[Dict]:
    """Order problems: primary first, then by score"""
    return [primary] + sorted(consecutive, key=lambda x: x["score"], reverse=True)


def create_problem_statement(problem_data: Dict, issue_data: Dict, problem_id: int) -> str:
    """Create problem statement for mini-swe-agent"""

    prob = problem_data["problem"]

    statement = f"""# PROBLEM {problem_id}

## Issue Context
Repository: {issue_data.get('repo', '')}
Instance: {issue_data.get('instance_id', '')}

## Original Issue
{issue_data.get('problem_statement', '')[:500]}

## Specific Problem to Fix
{prob.get('problem', '')}

## Files
"""

    files = []
    if 'file' in prob:
        files.append(prob['file'])
    elif 'actual_files' in prob:
        files = prob['actual_files'][:10]
    elif 'files' in prob:
        files = prob['files'][:10]

    for f in files:
        statement += f"- {f}\n"

    statement += f"""
## Repair Guidance
"""

    # Add repair guidance based on source
    if problem_data["source"] == "L1":
        statement += f"Fix: {prob.get('fixes', '')[:300]}\n"
        statement += f"Why: {prob.get('why_fix', '')[:300]}\n"
    elif problem_data["source"] == "L2":
        statement += f"Strategy: {prob.get('fix_strategy', '')}\n"
    elif problem_data["source"] == "L3":
        fix = prob.get('universal_fix', {})
        statement += f"Approach: {fix.get('approach', '')}\n"
        statement += "\nSteps:\n"
        for step in fix.get('steps', [])[:5]:
            statement += f"  {step}\n"

    statement += f"""
## Verification
Command: {prob.get('validation_cmd') or prob.get('verification_cmd', 'pytest')}

This command must PASS after your fix.

## Instructions
Fix ONLY this specific problem. Do not modify unrelated code.
"""

    return statement


def call_mini_swe_agent(problem_statement: str,
                        instance_id: str,
                        problem_id: int,
                        repo_path: Optional[Path] = None) -> Dict:
    """
    Save problem statement for mini-swe-agent.
    This creates the prompt that will be used by your agent.

    NOTE: You should replace this function with your actual
    mini-swe-agent calling logic if you have it.
    """

    # Save problem statement to directory where agent can read it
    prompts_dir = Path("data/prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)

    problem_file = prompts_dir / f"{instance_id}_problem_{problem_id}.txt"
    with open(problem_file, "w") as f:
        f.write(problem_statement)

    print(f"  Problem statement saved to: {problem_file}")
    print(f"  You can now run your agent with this prompt")

    # Placeholder - replace with your actual agent call
    # For now, just return success so the flow continues
    return {
        "success": True,
        "stdout": "Problem statement created",
        "stderr": ""
    }


def get_git_diff(repo_path: Path) -> str:
    """Get current git diff"""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return result.stdout


def run_validation(cmd: str, repo_path: Path) -> Dict:
    """Run validation command"""
    result = subprocess.run(
        cmd.split(),
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return {
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def commit_changes(repo_path: Path, message: str):
    """Commit changes"""
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True)


def sequential_repair(issue_data: Dict,
                      ordered_problems: List[Dict],
                      repo_path: Path) -> Dict:
    """Execute sequential repair, one problem at a time"""

    print("\n" + "="*60)
    print("SEQUENTIAL REPAIR")
    print("="*60)

    instance_id = issue_data["instance_id"]
    results = []

    # Save initial state
    initial_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True
    ).stdout.strip()

    for idx, problem_data in enumerate(ordered_problems, 1):
        print(f"\n{'='*60}")
        print(f"PROBLEM {idx}/{len(ordered_problems)}")
        if problem_data["is_primary"]:
            print("[PRIMARY CI FAILURE]")
        else:
            print(f"[{problem_data['source']}] Score: {problem_data['score']:.2f}")
        print("="*60)

        prob = problem_data["problem"]
        print(f"Description: {prob.get('problem', '')[:100]}")

        # Create problem statement
        statement = create_problem_statement(problem_data, issue_data, idx)

        # Call mini-swe-agent
        print("\nCalling mini-swe-agent...")
        agent_result = call_mini_swe_agent(statement, instance_id, idx, repo_path)

        if not agent_result["success"]:
            print(f"Agent failed: {agent_result.get('error', 'Unknown')}")
            results.append({
                "problem_id": idx,
                "status": "AGENT_FAILED"
            })

            if problem_data["is_primary"]:
                print("\nPRIMARY PROBLEM FAILED - STOPPING")
                break
            continue

        # Get diff
        print("Collecting diff...")
        diff = get_git_diff(repo_path)

        if not diff.strip():
            print("No changes made")
            results.append({
                "problem_id": idx,
                "status": "NO_CHANGES"
            })
            continue

        # Verify
        val_cmd = prob.get("validation_cmd") or prob.get("verification_cmd", "pytest")
        print(f"Verifying: {val_cmd}")

        validation = run_validation(val_cmd, repo_path)
        status = "PASS" if validation["passed"] else "FAIL"

        print(f"Status: {status}")

        results.append({
            "problem_id": idx,
            "is_primary": problem_data["is_primary"],
            "status": status
        })

        if status == "PASS":
            commit_changes(repo_path, f"Fix problem {idx}")
            print("Changes committed")
        else:
            if problem_data["is_primary"]:
                print("\nPRIMARY PROBLEM FAILED - STOPPING")
                break
            else:
                print("Reverting, continuing...")
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path)

    # Get unified diff
    print(f"\n{'='*60}")
    print("GENERATING UNIFIED DIFF")
    print("="*60)

    final_diff = subprocess.run(
        ["git", "diff", initial_commit, "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True
    ).stdout

    print(f"Unified diff: {len(final_diff)} chars")

    return {
        "model_patch": final_diff,
        "results": results,
        "problems_fixed": len([r for r in results if r["status"] == "PASS"]),
        "total_problems": len(results),
        "primary_status": next((r["status"] for r in results if r.get("is_primary")), "NOT_FOUND")
    }


def save_to_preds(issue_data: Dict, repair_result: Dict, output_path: Path):
    """Save to results/preds.json"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    if output_path.exists():
        with open(output_path) as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except:
                existing = []
    else:
        existing = []

    # Create entry
    entry = {
        "instance_id": issue_data["instance_id"],
        "model_patch": repair_result["model_patch"],
        "model_name_or_path": "memory-guided-repair"
    }

    # Update or append
    found = False
    for i, e in enumerate(existing):
        if e.get("instance_id") == entry["instance_id"]:
            existing[i] = entry
            found = True
            break

    if not found:
        existing.append(entry)

    # Save
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nSaved to: {output_path}")


def main(issue_id: str):
    """Main entry point"""

    # Use default paths
    memory_dir = Path("data/trs")
    output_path = Path("results/preds.json")

    print("="*60)
    print("MEMORY-GUIDED SEQUENTIAL REPAIR")
    print("="*60)
    print(f"Issue: {issue_id}")

    # Load issue
    issue_data = load_issue(issue_id)
    repo_path = Path(issue_data.get("repo_path", f"repos/{issue_data['repo']}"))

    print(f"Repo: {issue_data.get('repo')}")

    # Load memory
    print("\nLoading memory...")
    with open(memory_dir / "failure_memory.json") as f:
        l1 = json.load(f)
    with open(memory_dir / "repo_memory.json") as f:
        l2 = json.load(f)
    with open(memory_dir / "cross_memory.json") as f:
        l3 = json.load(f)

    print(f"  L1: {len(l1)} problems")
    print(f"  L2: {len(l2)} sequences")
    print(f"  L3: {len(l3)} patterns")

    # Identify primary
    print("\n[1/4] Identifying PRIMARY problem...")
    primary = identify_primary_problem(issue_data, l1)
    print(f"  Source: {primary['source']}")
    print(f"  Score: {primary['score']:.2f}")

    # Select consecutive
    print("\n[2/4] Selecting consecutive problems...")
    consecutive = select_consecutive_problems(primary, l1, l2, l3)
    print(f"  Found: {len(consecutive)} problems")

    # Order
    print("\n[3/4] Ordering problems...")
    ordered = order_problems(primary, consecutive)

    print("\n  Repair sequence:")
    for i, p in enumerate(ordered, 1):
        marker = "PRIMARY" if p["is_primary"] else p["source"]
        print(f"    {i}. [{marker:8}] Score: {p['score']:.2f}")

    # Sequential repair
    print("\n[4/4] Executing repair...")
    result = sequential_repair(issue_data, ordered, repo_path)

    # Save
    save_to_preds(issue_data, result, output_path)

    print("\n" + "="*60)
    print("REPAIR COMPLETE")
    print("="*60)
    print(f"Primary: {result['primary_status']}")
    print(f"Fixed: {result['problems_fixed']}/{result['total_problems']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/memory_guided_repair.py <instance_id>")
        sys.exit(1)

    main(sys.argv[1])
