#!/usr/bin/env python3
"""
Memory-Guided Sequential Repair with TF-IDF Cosine Similarity

ARCHITECTURE:
1. Fetch CI failure for new issue
2. TF-IDF cosine similarity search (top-10 from L1, L2, L3)
3. Expand with dependency chains (enables/enabled_by)
4. Analyze memories (similar fixes, hidden problems, patterns)
5. Create problem list (primary + anticipated + patterns)
6. Sequential repair (one problem at a time to mini-swe-agent)
7. Merge patches into final diff

USAGE:
    python scripts/memory_guided_repair.py <instance_id>

EXAMPLE:
    python scripts/memory_guided_repair.py 110
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Try to import sklearn, provide helpful error if missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERROR: scikit-learn not installed!")
    print("Install with: pip install scikit-learn")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = PROJECT_ROOT / "data/trs"


# ============================================================================
# PHASE 1: Load Issue and Memories
# ============================================================================

def load_issue_from_cibench(instance_id: str) -> Dict:
    """Load issue from CIBench eval_issues.json dataset"""
    # Try eval_issues.json first (main dataset)
    eval_issues_file = PROJECT_ROOT / "data/trs/eval_issues.json"

    if eval_issues_file.exists():
        with open(eval_issues_file) as f:
            dataset = json.load(f)

        for issue in dataset:
            if str(issue.get("id")) == str(instance_id):
                return issue

    # Try memory_seed_issues.json as fallback
    seed_issues_file = PROJECT_ROOT / "data/trs/memory_seed_issues.json"
    if seed_issues_file.exists():
        with open(seed_issues_file) as f:
            dataset = json.load(f)

        for issue in dataset:
            if str(issue.get("id")) == str(instance_id):
                return issue

    raise ValueError(f"Issue {instance_id} not found in eval_issues.json or memory_seed_issues.json")


def load_memories() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load L1, L2, L3 memories"""
    l1_path = MEMORY_DIR / "failure_memory.json"
    l2_path = MEMORY_DIR / "repo_memory.json"
    l3_path = MEMORY_DIR / "cross_memory.json"

    with open(l1_path) as f:
        l1 = json.load(f)

    with open(l2_path) as f:
        l2 = json.load(f)

    with open(l3_path) as f:
        l3 = json.load(f)

    return l1, l2, l3


def extract_ci_features(issue: Dict) -> Dict:
    """Extract features from CI failure"""
    problem_statement = issue.get("problem_statement", "")
    test_patch = issue.get("test_patch", "")

    # Combine all text for analysis
    full_text = f"{problem_statement} {test_patch}"

    return {
        'instance_id': issue.get('id'),
        'repo': issue.get('repo', ''),
        'problem_description': problem_statement,
        'test_patch': test_patch,
        'full_text': full_text
    }


# ============================================================================
# PHASE 2: TF-IDF Cosine Similarity Search
# ============================================================================

def build_search_corpus(l1: List[Dict], l2: List[Dict], l3: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Build search corpus from L1, L2, L3 memories.

    Returns:
        corpus_texts: List of text strings for TF-IDF
        corpus_metadata: List of dicts with {source, level, data, index}
    """
    corpus_texts = []
    corpus_metadata = []

    # L1: Per-file problems
    for i, entry in enumerate(l1):
        text = f"{entry.get('problem', '')} {entry.get('root_cause', '')} {entry.get('how_fixed', '')} {entry.get('failure_type', '')} {entry.get('issue_type', '')} {entry.get('validation_cmd', '')}"
        corpus_texts.append(text)
        corpus_metadata.append({
            'level': 'L1',
            'source': 'per_file',
            'data': entry,
            'index': i
        })

    # L2: Repair trajectories (each step's problems)
    for i, trajectory in enumerate(l2):
        for step in trajectory.get('repair_trajectory', []):
            for prob in step.get('problems', []):
                text = f"{prob.get('problem', '')} {prob.get('root_cause', '')} {prob.get('how_fixed', '')} {step.get('failure_type', '')} {prob.get('issue_type', '')} {step.get('validation_cmd', '')}"
                corpus_texts.append(text)
                corpus_metadata.append({
                    'level': 'L2',
                    'source': 'trajectory',
                    'data': {
                        'trajectory': trajectory,
                        'step': step,
                        'problem': prob
                    },
                    'index': i
                })

    # L3: Universal patterns
    for i, pattern in enumerate(l3):
        failure_patterns_text = ' '.join(pattern.get('failure_patterns', []))
        fix_approach = pattern.get('fix_approach', [])
        # fix_approach can be list of strings or list of dicts
        if fix_approach and isinstance(fix_approach[0], dict):
            fix_approach_text = ' '.join([item.get('approach', '') for item in fix_approach])
        else:
            fix_approach_text = ' '.join(fix_approach) if isinstance(fix_approach, list) else str(fix_approach)

        text = f"{pattern.get('pattern_name', '')} {failure_patterns_text} {fix_approach_text}"
        corpus_texts.append(text)
        corpus_metadata.append({
            'level': 'L3',
            'source': 'pattern',
            'data': pattern,
            'index': i
        })

    return corpus_texts, corpus_metadata


def cosine_similarity_search(query: str, l1: List[Dict], l2: List[Dict], l3: List[Dict], top_k: int = 10) -> Dict:
    """
    Search using TF-IDF + Cosine Similarity.

    Returns top-k most similar from each level.
    """
    print(f"\n{'='*80}")
    print("PHASE 2: TF-IDF COSINE SIMILARITY SEARCH")
    print(f"{'='*80}")

    # Build corpus
    corpus_texts, corpus_metadata = build_search_corpus(l1, l2, l3)

    print(f"Corpus size: {len(corpus_texts)} entries")
    print(f"  L1: {len(l1)}")
    print(f"  L2: {sum(len(t['repair_trajectory']) for t in l2)} steps")
    print(f"  L3: {len(l3)}")

    # TF-IDF vectorization
    all_texts = [query] + corpus_texts
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    try:
        vectors = vectorizer.fit_transform(all_texts)
    except ValueError as e:
        print(f"WARNING: TF-IDF failed: {e}")
        print("Falling back to empty results")
        return {'top_l1': [], 'top_l2': [], 'top_l3': []}

    query_vector = vectors[0:1]
    corpus_vectors = vectors[1:]

    # Calculate similarities
    similarities = cosine_similarity(query_vector, corpus_vectors)[0]

    # Separate by level
    l1_results = []
    l2_results = []
    l3_results = []

    for i, (score, meta) in enumerate(zip(similarities, corpus_metadata)):
        result = {'score': float(score), **meta}

        if meta['level'] == 'L1':
            l1_results.append(result)
        elif meta['level'] == 'L2':
            l2_results.append(result)
        elif meta['level'] == 'L3':
            l3_results.append(result)

    # Sort and get top-k
    l1_results.sort(key=lambda x: x['score'], reverse=True)
    l2_results.sort(key=lambda x: x['score'], reverse=True)
    l3_results.sort(key=lambda x: x['score'], reverse=True)

    top_l1 = l1_results[:top_k]
    top_l2 = l2_results[:top_k]
    top_l3 = l3_results[:top_k]

    print(f"\nTop-{top_k} retrieval:")
    print(f"  L1: {len(top_l1)} entries (avg score: {np.mean([r['score'] for r in top_l1]):.3f})")
    print(f"  L2: {len(top_l2)} entries (avg score: {np.mean([r['score'] for r in top_l2]):.3f})")
    print(f"  L3: {len(top_l3)} entries (avg score: {np.mean([r['score'] for r in top_l3]):.3f})")

    return {
        'top_l1': top_l1,
        'top_l2': top_l2,
        'top_l3': top_l3
    }


# ============================================================================
# PHASE 3: Expand with Dependencies
# ============================================================================

def expand_with_dependencies(top_l1: List[Dict], l1_memory: List[Dict]) -> List[Dict]:
    """
    Expand L1 results by following enables/enabled_by chains.
    """
    print(f"\n{'='*80}")
    print("PHASE 3: EXPAND WITH DEPENDENCIES")
    print(f"{'='*80}")

    expanded = []
    seen_ids = set()

    # Create lookup by problem_id
    l1_by_problem_id = {}
    for entry in l1_memory:
        problem_id = entry.get('problem_id')
        if problem_id:
            l1_by_problem_id[problem_id] = entry

    def add_with_deps(result: Dict):
        entry = result['data']
        problem_id = entry.get('problem_id')

        if problem_id in seen_ids:
            return

        seen_ids.add(problem_id)
        expanded.append(result)

        # Add enabled problems
        for enabled_ref in entry.get('enables', []):
            # enabled_ref format: "problem_1", "problem_2", etc.
            enabled_id = int(enabled_ref.split('_')[1]) if '_' in enabled_ref else None
            if enabled_id and enabled_id in l1_by_problem_id:
                enabled_entry = l1_by_problem_id[enabled_id]
                add_with_deps({
                    'score': result['score'] * 0.9,  # Slightly lower score
                    'level': 'L1',
                    'source': 'dependency_enabled',
                    'data': enabled_entry,
                    'index': enabled_id
                })

        # Add enabling problems
        for enabling_ref in entry.get('enabled_by', []):
            enabling_id = int(enabling_ref.split('_')[1]) if '_' in enabling_ref else None
            if enabling_id and enabling_id in l1_by_problem_id:
                enabling_entry = l1_by_problem_id[enabling_id]
                add_with_deps({
                    'score': result['score'] * 0.9,
                    'level': 'L1',
                    'source': 'dependency_enabler',
                    'data': enabling_entry,
                    'index': enabling_id
                })

    # Add each top result with its dependencies
    for result in top_l1:
        add_with_deps(result)

    print(f"Expanded from {len(top_l1)} to {len(expanded)} L1 entries")
    print(f"  Direct matches: {len(top_l1)}")
    print(f"  Dependencies: {len(expanded) - len(top_l1)}")

    return expanded


# ============================================================================
# PHASE 4: Analyze Memories
# ============================================================================

def analyze_memories(expanded_l1: List[Dict], top_l2: List[Dict], top_l3: List[Dict]) -> Dict:
    """
    Analyze retrieved memories to extract actionable insights.
    """
    print(f"\n{'='*80}")
    print("PHASE 4: ANALYZE MEMORIES")
    print(f"{'='*80}")

    analysis = {
        'l1_similar_problems': [],
        'l1_common_files': set(),
        'l1_common_fixes': [],
        'l2_repair_sequences': [],
        'l2_hidden_problems': [],
        'l3_patterns': [],
        'l3_dependent_issues': set()
    }

    # L1 Analysis
    print("\nL1 Analysis:")
    for result in expanded_l1[:10]:  # Top 10 + dependencies
        entry = result['data']
        analysis['l1_similar_problems'].append({
            'problem': entry.get('problem', ''),
            'root_cause': entry.get('root_cause', ''),
            'how_fixed': entry.get('how_fixed', ''),
            'why_fix_works': entry.get('why_fix_works', ''),
            'score': result['score']
        })

        file_path = entry.get('file', '')
        if file_path and file_path != '<no specific file>':
            analysis['l1_common_files'].add(file_path)

        how_fixed = entry.get('how_fixed', '')
        if how_fixed:
            analysis['l1_common_fixes'].append(how_fixed)

    print(f"  Similar problems: {len(analysis['l1_similar_problems'])}")
    print(f"  Common files: {len(analysis['l1_common_files'])}")
    print(f"  Fix strategies: {len(analysis['l1_common_fixes'])}")

    # L2 Analysis
    print("\nL2 Analysis:")
    seen_trajectories = set()

    for result in top_l2[:10]:
        trajectory = result['data']['trajectory']
        trajectory_id = trajectory.get('issue_id')

        if trajectory_id in seen_trajectories:
            continue
        seen_trajectories.add(trajectory_id)

        repair_trajectory = trajectory.get('repair_trajectory', [])

        if repair_trajectory:
            analysis['l2_repair_sequences'].append({
                'issue_id': trajectory_id,
                'primary_problem': repair_trajectory[0],
                'hidden_problems': repair_trajectory[1:],
                'total_steps': len(repair_trajectory),
                'score': result['score']
            })

            # Extract hidden problems
            for hidden_step in repair_trajectory[1:]:
                for prob in hidden_step.get('problems', []):
                    analysis['l2_hidden_problems'].append({
                        'problem': prob.get('problem', ''),
                        'root_cause': prob.get('root_cause', ''),
                        'how_fixed': prob.get('how_fixed', ''),
                        'revealed_by': hidden_step.get('revealed_by', ''),
                        'validation_cmd': hidden_step.get('validation_cmd', '')
                    })

    print(f"  Repair sequences: {len(analysis['l2_repair_sequences'])}")
    print(f"  Anticipated hidden problems: {len(analysis['l2_hidden_problems'])}")

    # L3 Analysis
    print("\nL3 Analysis:")
    for result in top_l3[:10]:
        pattern = result['data']
        analysis['l3_patterns'].append({
            'pattern_name': pattern.get('pattern_name', ''),
            'failure_patterns': pattern.get('failure_patterns', []),
            'fix_approach': pattern.get('fix_approach', []),
            'score': result['score']
        })

    print(f"  Universal patterns: {len(analysis['l3_patterns'])}")

    return analysis


# ============================================================================
# PHASE 5: Create Problem List
# ============================================================================

def create_problem_list(ci_features: Dict, analysis: Dict) -> List[Dict]:
    """
    Create prioritized problem list from CI failure and memory analysis.
    """
    print(f"\n{'='*80}")
    print("PHASE 5: CREATE PROBLEM LIST")
    print(f"{'='*80}")

    problems = []

    # 1. Primary problem from CI failure
    problems.append({
        'id': 'P0',
        'type': 'primary',
        'priority': 1,
        'problem': ci_features['problem_description'],
        'context': {
            'similar_problems': analysis['l1_similar_problems'][:3],
            'similar_fixes': analysis['l1_common_fixes'][:3],
            'common_files': list(analysis['l1_common_files'])[:10]
        }
    })

    print(f"\n1. Primary problem: P0")
    print(f"   {ci_features['problem_description'][:100]}...")

    # 2. Anticipated hidden problems from L2 trajectories
    print(f"\n2. Anticipated hidden problems from L2:")
    for i, hidden in enumerate(analysis['l2_hidden_problems'][:5], 1):
        problems.append({
            'id': f'H{i}',
            'type': 'hidden_anticipated',
            'priority': 2,
            'problem': hidden['problem'],
            'root_cause': hidden['root_cause'],
            'how_fixed': hidden['how_fixed'],
            'revealed_by': hidden['revealed_by'],
            'validation_cmd': hidden['validation_cmd'],
            'context': {
                'repair_sequences': analysis['l2_repair_sequences'][:2]
            }
        })
        print(f"   H{i}: {hidden['problem'][:80]}...")

    # 3. Pattern-based problems from L3
    print(f"\n3. Pattern-based problems from L3:")
    for i, pattern in enumerate(analysis['l3_patterns'][:3], 1):
        # Extract first failure pattern and fix approach
        failure_pattern = pattern['failure_patterns'][0] if pattern['failure_patterns'] else ''
        fix_approach_list = pattern.get('fix_approach', [])

        # fix_approach can be list of strings or list of dicts
        if fix_approach_list:
            if isinstance(fix_approach_list[0], dict):
                fix_approach_text = fix_approach_list[0].get('approach', '')
            else:
                fix_approach_text = fix_approach_list[0]
        else:
            fix_approach_text = ''

        problems.append({
            'id': f'D{i}',
            'type': 'pattern_based',
            'priority': 3,
            'problem': f"{pattern['pattern_name']}: {failure_pattern}",
            'fix_approach': fix_approach_text,
            'context': {
                'pattern': pattern
            }
        })
        print(f"   D{i}: {pattern['pattern_name']}")

    print(f"\nTotal problems to solve: {len(problems)}")
    return problems


# ============================================================================
# PHASE 6: Sequential Repair (Placeholder for mini-swe-agent integration)
# ============================================================================

def call_mini_swe_agent(instance_id: str, problem: Dict) -> Optional[str]:
    """
    Call mini-swe-agent to solve one problem.

    TODO: Integrate with your actual mini-swe-agent implementation.
    """
    print(f"\n  Calling mini-swe-agent for {problem['id']}...")

    # Create problem statement with memory context
    problem_statement = f"""
Problem ID: {problem['id']} ({problem['type']})
Priority: {problem['priority']}

PROBLEM:
{problem.get('problem', '')}

"""

    if 'root_cause' in problem:
        problem_statement += f"""
ROOT CAUSE:
{problem['root_cause']}

"""

    if 'context' in problem:
        problem_statement += f"""
CONTEXT FROM MEMORY:
{json.dumps(problem['context'], indent=2)}

"""

    problem_statement += """
TASK: Fix this specific problem and generate a patch.
"""

    # TODO: Replace with actual mini-swe-agent call
    # For now, just save the prompt
    prompt_dir = PROJECT_ROOT / "data/prompts"
    prompt_dir.mkdir(exist_ok=True)

    prompt_file = prompt_dir / f"problem_{problem['id']}.txt"
    with open(prompt_file, 'w') as f:
        f.write(problem_statement)

    print(f"  Saved prompt to: {prompt_file}")

    # Placeholder: return None (no patch generated yet)
    return None


def sequential_repair(instance_id: str, problems: List[Dict]) -> List[Dict]:
    """
    Solve problems sequentially, one at a time.
    """
    print(f"\n{'='*80}")
    print("PHASE 6: SEQUENTIAL REPAIR")
    print(f"{'='*80}")

    patches = []

    for problem in problems:
        print(f"\n--- Problem {problem['id']} ({problem['type']}) ---")

        patch = call_mini_swe_agent(instance_id, problem)

        if patch:
            patches.append({
                'problem_id': problem['id'],
                'patch': patch
            })
            print(f"  ✓ Patch generated")
        else:
            print(f"  ⚠ No patch generated (placeholder)")

    print(f"\nTotal patches: {len(patches)}")
    return patches


# ============================================================================
# PHASE 7: Merge Patches
# ============================================================================

def merge_patches(patches: List[Dict]) -> str:
    """
    Merge all patches into final unified diff.

    TODO: Implement smart merge logic handling conflicts.
    """
    print(f"\n{'='*80}")
    print("PHASE 7: MERGE PATCHES")
    print(f"{'='*80}")

    if not patches:
        print("No patches to merge")
        return ""

    # Simple concatenation for now
    # TODO: Implement conflict resolution
    merged = "\n\n".join([p['patch'] for p in patches if p['patch']])

    print(f"Merged {len(patches)} patches")
    return merged


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/memory_guided_repair.py <instance_id>")
        print("Example: python scripts/memory_guided_repair.py 110")
        return 1

    instance_id = sys.argv[1]

    print(f"{'='*80}")
    print(f"MEMORY-GUIDED REPAIR: Issue #{instance_id}")
    print(f"{'='*80}")

    # PHASE 1: Load issue and memories
    print(f"\n{'='*80}")
    print("PHASE 1: LOAD ISSUE AND MEMORIES")
    print(f"{'='*80}")

    issue = load_issue_from_cibench(instance_id)
    print(f"Loaded issue: {issue.get('repo', '')} #{instance_id}")

    l1, l2, l3 = load_memories()
    print(f"Loaded memories: L1={len(l1)}, L2={len(l2)}, L3={len(l3)}")

    ci_features = extract_ci_features(issue)
    print(f"Extracted CI features from issue")

    # PHASE 2: Cosine similarity search
    search_query = ci_features['full_text']
    search_results = cosine_similarity_search(search_query, l1, l2, l3, top_k=10)

    # PHASE 3: Expand with dependencies
    expanded_l1 = expand_with_dependencies(search_results['top_l1'], l1)

    # PHASE 4: Analyze memories
    analysis = analyze_memories(expanded_l1, search_results['top_l2'], search_results['top_l3'])

    # PHASE 5: Create problem list
    problems = create_problem_list(ci_features, analysis)

    # PHASE 6: Sequential repair
    patches = sequential_repair(instance_id, problems)

    # PHASE 7: Merge patches
    final_diff = merge_patches(patches)

    # Save result
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    result = {
        'instance_id': instance_id,
        'model_patch': final_diff,
        'problems_identified': [p['id'] for p in problems],
        'patches_generated': len(patches),
        'memory_used': {
            'l1_entries': len(expanded_l1),
            'l2_trajectories': len(search_results['top_l2']),
            'l3_patterns': len(search_results['top_l3'])
        }
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    result_file = results_dir / f"memory_guided_{instance_id}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved result to: {result_file}")
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
