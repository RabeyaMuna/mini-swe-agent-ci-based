"""
STAIR Memory Retrieval - Main Implementation
Simple 6-stage pipeline with LLM-driven decision making

Uses existing project utilities:
- utilities.llm_invoker: Robust LLM calls with retry logic
- utilities.llm_chunking: Chunking for large inputs
- prompt_template.stair_retrieval: All prompts
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from prompt_template.stair_retrieval import (
    build_common_detection_prompt,
    build_filtering_prompt,
    build_clustering_prompt,
    build_final_generation_prompt,
    build_repair_plan_prompt
)
from utilities.llm_invoker import invoke_llm_with_retry, _load_json_flexible, STRICT_JSON_RULES


class STAIRRetrieval:
    """
    STAIR-inspired hierarchical memory retrieval.

    6 Stages:
    1. Cosine similarity retrieval (L1/L2/L3)
    2. LLM: Common problem detection
    3. LLM: Filtering + dependency analysis
    4. LLM: Clustering similar problems
    5. LLM: Final problem list generation
    6. LLM: Repair plan generation (optional)

    Supports:
    - Baseline mode (no memory)
    - Ablation study (control which levels: l1, l1+l2, l1+l2+l3)
    """

    def __init__(
        self,
        memory_dir: str,
        llm_client=None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        baseline_mode: bool = False,
        memory_levels: str = "l1+l2+l3"
    ):
        """
        Initialize retrieval system.

        Args:
            memory_dir: Path to L1/L2/L3 memory files
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            embedding_model: Sentence transformer for embeddings
            baseline_mode: If True, skip memory retrieval (for baseline comparison)
            memory_levels: Which levels to use - "l1", "l1+l2", or "l1+l2+l3"
                          For ablation studies to measure impact of each level
        """
        self.memory_dir = Path(memory_dir)
        self.llm = llm_client
        self.baseline_mode = baseline_mode
        self.enabled_levels = self._parse_memory_levels(memory_levels)
        self.embedding_model = embedding_model

        # Baseline mode: skip all loading
        if baseline_mode:
            self.l1_memory = []
            self.l2_memory = []
            self.l3_memory = []
            self.l1_embeddings = np.array([])
            self.l2_embeddings = np.array([])
            self.l3_embeddings = np.array([])
            self.encoder = None
            return

        # Load encoder/embeddings lazily. Initialization and query-building
        # should not initialize the sentence-transformers runtime.
        self.encoder = None

        # Load only enabled levels
        if 'l1' in self.enabled_levels:
            self.l1_memory = self._load_json('failure_memory.json')
            self.l1_embeddings = None
        else:
            self.l1_memory = []
            self.l1_embeddings = np.array([])

        if 'l2' in self.enabled_levels:
            self.l2_memory = self._load_json('repo_memory.json')
            self.l2_embeddings = None
        else:
            self.l2_memory = []
            self.l2_embeddings = np.array([])

        if 'l3' in self.enabled_levels:
            self.l3_memory = self._load_json('cross_memory.json')
            self.l3_embeddings = None
        else:
            self.l3_memory = []
            self.l3_embeddings = np.array([])


    def _parse_memory_levels(self, memory_levels: str) -> set:
        """
        Parse memory_levels string into set of enabled levels.

        Args:
            memory_levels: String like "l1", "l1+l2", or "l1+l2+l3"
                          Can also be a list: ['l1', 'l2']

        Returns:
            Set of enabled levels: {'l1'}, {'l1', 'l2'}, or {'l1', 'l2', 'l3'}

        Examples:
            "l1" -> {'l1'}
            "l1+l2" -> {'l1', 'l2'}
            "l1+l2+l3" -> {'l1', 'l2', 'l3'}
            ['l1', 'l2'] -> {'l1', 'l2'}
        """
        if isinstance(memory_levels, str):
            # Parse "l1+l2+l3" format
            levels = memory_levels.lower().replace(' ', '').split('+')
            return {level for level in levels if level in {'l1', 'l2', 'l3'}}
        elif isinstance(memory_levels, (list, set)):
            return {level for level in memory_levels if level in {'l1', 'l2', 'l3'}}
        else:
            # Default: all levels
            return {'l1', 'l2', 'l3'}


    def retrieve(self, query: Dict, top_k: int = 5) -> Dict[str, Any]:
        """
        Optimal step-by-step LLM-driven retrieval pipeline.

        Flow:
        1. Cosine search L1/L2/L3 (top-k retrieval)
        2. LLM filters relevant issues (NOT problems yet)
        3. LLM extracts ALL problems with full repair strategies:
           - CI failure problem (main)
           - Dependent problems (fix BEFORE)
           - Consecutive problems (fix AFTER)
        4. Detect repo/workflow common patterns:
           a. Flatten ALL L1/L2 problems (each keeps issue_id)
           b. Cluster flattened problems deterministically
           c. Group by cluster_id, count distinct issue_ids (frequency)
           d. LLM validates each group (is this a real pattern?)
        5. Merge all 4 types & cluster for deduplication
        6. LLM merges duplicates
        7. Reorder: CI failure -> dependent -> consecutive -> common

        Each problem includes complete repair strategy:
        - problem, root_cause, files, signals
        - repair_strategy (summary, actions, validation_cmd, pitfalls)
        """
        if self.baseline_mode:
            return {
                'problems': [],
                'metadata': {
                    'mode': 'baseline',
                    'enabled_levels': [],
                    'retrieved': {'l1': 0, 'l2': 0, 'l3': 0},
                    'ci_failure': None,
                    'dependent': 0,
                    'consecutive': 0,
                    'common_detected': 0,
                    'final': 0
                },
                'l1_matches': [],
                'l2_matches': [],
                'l3_matches': [],
                'ci_failure_problem': None,
                'dependent_problems': [],
                'consecutive_problems': [],
                'common_problems': []
            }

        # Stage 1: Cosine search
        print("[Memory] Stage 1: Starting cosine search...")
        retrieved = self._stage_1_cosine_search(query, top_k)
        print(f"[Memory] Stage 1: Retrieved L1={len(retrieved['l1'])}, L2={len(retrieved['l2'])}, L3={len(retrieved['l3'])}")

        if not self.llm:
            raise ValueError("LLM client required for stages 2+")

        # Stage 2: LLM filters relevant issues (not problems)
        print("[Memory] Stage 2: LLM filtering relevant issues...")
        filtered_issues = self._stage_2_llm_filter_relevant_issues(
            retrieved['l1'],
            retrieved['l2'],
            retrieved['l3'],
            query
        )
        print(f"[Memory] Stage 2: Filtered L1={len(filtered_issues['filtered_l1'])}, L2={len(filtered_issues['filtered_l2'])}, L3={len(filtered_issues['filtered_l3'])}")

        # Stage 3: LLM extracts problems with full repair strategies
        print("[Memory] Stage 3: LLM extracting problems & repair strategies...")
        problems_with_strategies = self._stage_3_llm_extract_problems_and_strategies(
            filtered_issues['filtered_l1'],
            filtered_issues['filtered_l2'],
            filtered_issues['filtered_l3'],
            query
        )
        ci_count = 1 if problems_with_strategies.get('ci_failure_problem') else 0
        dep_count = len(problems_with_strategies.get('dependent_problems', []))
        cons_count = len(problems_with_strategies.get('consecutive_problems', []))
        print(f"[Memory] Stage 3: Extracted CI={ci_count}, Dependent={dep_count}, Consecutive={cons_count}")

        # Stage 4: Common patterns via clustering
        print("[Memory] Stage 4: Detecting common patterns...")
        common_problems = self._stage_4_detect_repo_common_patterns(query)
        print(f"[Memory] Stage 4: Found {len(common_problems)} common patterns")

        # Stage 5: Combine all 4 types (just append, no merging yet)
        print("[Memory] Stage 5: Combining all problem types...")
        all_candidates = self._stage_5_merge_all_problem_types(
            ci_failure_problem=problems_with_strategies.get('ci_failure_problem'),
            dependent_problems=problems_with_strategies.get('dependent_problems', []),
            consecutive_problems=problems_with_strategies.get('consecutive_problems', []),
            common_patterns=common_problems
        )
        print(f"[Memory] Stage 5: Combined {len(all_candidates)} total candidates (CI + dependent + consecutive + common)")

        # Stage 6: Cluster & LLM deduplicate
        print("[Memory] Stage 6: Clustering for deduplication...")
        clusters = self._cluster_for_deduplication(all_candidates)
        print(f"[Memory] Stage 6: Created {len(clusters)} clusters, running LLM merge...")
        merged = self._stage_6_llm_merge_duplicates(clusters, query, common_problems)
        print(f"[Memory] Stage 6: After dedup: {len(merged.get('problems', []))} problems")

        # Stage 7: Reorder problems
        print("[Memory] Stage 7: Reordering problems by priority...")
        final_problems = self._stage_7_reorder_final_problems(
            merged.get('problems', []),
            query,
            ci_failure_problem=problems_with_strategies.get('ci_failure_problem'),
            dependent_problems=problems_with_strategies.get('dependent_problems', []),
            consecutive_problems=problems_with_strategies.get('consecutive_problems', [])
        )
        print(f"[Memory] Stage 7: Final ordered problems: {len(final_problems)}")

        # DEBUG: Show problem structure
        print("\n[Memory] FINAL PROBLEMS LIST:")
        for i, p in enumerate(final_problems, 1):
            print(f"  {i}. [{p.get('problem_type', 'unknown')}] {p.get('problem', 'N/A')[:80]}...")
            if p.get('repair_strategy'):
                rs = p['repair_strategy']
                print(f"     Strategy: {rs.get('summary', 'N/A')[:60]}...")
                if rs.get('actions'):
                    print(f"     Actions: {len(rs['actions'])} steps")
        print()

        print("[Memory] Retrieval complete!")
        # Return only problems - agent only needs this
        return {
            'problems': final_problems
        }


    def _stage_1_cosine_search(self, query: Dict, top_k: int) -> Dict[str, List]:
        """
        Stage 1: Cosine similarity search from enabled levels only.

        Returns top-k matches per level based on ablation settings.
        """
        # L1: Only retrieve if enabled
        if 'l1' in self.enabled_levels:
            if self.l1_embeddings is None:
                self.l1_embeddings = self._compute_embeddings(self.l1_memory, 'l1')
            l1_query = self._build_query(query, level='l1')
            l1_results = self._retrieve_topk(
                l1_query,
                self.l1_memory,
                self.l1_embeddings,
                top_k,
                filters={
                    'repo': query.get('repo'),
                    'workflow': query.get('workflow_name') or query.get('workflow_path')
                }
            )
        else:
            l1_results = []

        # L2: Only retrieve if enabled
        if 'l2' in self.enabled_levels:
            if self.l2_embeddings is None:
                self.l2_embeddings = self._compute_embeddings(self.l2_memory, 'l2')
            l2_query = self._build_query(query, level='l2')
            l2_results = self._retrieve_topk(
                l2_query,
                self.l2_memory,
                self.l2_embeddings,
                top_k,
                filters={'repo': query.get('repo')}
            )
        else:
            l2_results = []

        # L3: Only retrieve if enabled
        if 'l3' in self.enabled_levels:
            if self.l3_embeddings is None:
                self.l3_embeddings = self._compute_embeddings(self.l3_memory, 'l3')
            l3_query = self._build_query(query, level='l3')
            l3_results = self._retrieve_topk(
                l3_query,
                self.l3_memory,
                self.l3_embeddings,
                top_k,
                filters={}  # No filters for cross-repo
            )
        else:
            l3_results = []

        return {
            'l1': l1_results,
            'l2': l2_results,
            'l3': l3_results
        }


    def _stage_2_llm_filter_relevant_issues(
        self,
        l1_items: List[Dict],
        l2_items: List[Dict],
        l3_items: List[Dict],
        query: Dict
    ) -> Dict[str, Any]:
        """
        Stage 2: LLM filters relevant ISSUES (not problems yet).

        This stage ONLY selects which issues are relevant.
        Problem extraction happens in Stage 3.
        """
        compact_l1 = [self._compact_retrieved_item(item, "L1", idx) for idx, item in enumerate(l1_items)]
        compact_l2 = [self._compact_retrieved_item(item, "L2", idx) for idx, item in enumerate(l2_items)]
        compact_l3 = [self._compact_retrieved_item(item, "L3", idx) for idx, item in enumerate(l3_items)]

        prompt = f"""You are analyzing L1/L2/L3 memory matches for a CI failure.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Retrieved Memory Summaries:**
```json
{json.dumps({"l1": compact_l1, "l2": compact_l2, "l3": compact_l3}, indent=2)}
```

---

**Task: Select only memory items relevant to the current CI failure.**

Use:
- Similar error messages
- Same or compatible files/components
- Same failure type or validation tool
- Similar root cause or repair strategy

Return IDs from the summaries. Do not invent IDs.

Return JSON:
{{
  "selected_l1_ids": ["L1:0"],
  "selected_l2_ids": ["L2:0"],
  "selected_l3_ids": ["L3:0"],
  "relevance_notes": "brief reason"
}}

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            parse_json=True
        )

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        filtered_l1 = self._resolve_selected_items(l1_items, result.get('selected_l1_ids', []), "L1")
        filtered_l2 = self._resolve_selected_items(l2_items, result.get('selected_l2_ids', []), "L2")
        filtered_l3 = self._resolve_selected_items(l3_items, result.get('selected_l3_ids', []), "L3")

        return {
            'filtered_l1': filtered_l1,
            'filtered_l2': filtered_l2,
            'filtered_l3': filtered_l3,
            'relevance_notes': result.get('relevance_notes', '')
        }


    def _stage_3_llm_extract_problems_and_strategies(
        self,
        filtered_l1: List[Dict],
        filtered_l2: List[Dict],
        filtered_l3: List[Dict],
        query: Dict
    ) -> Dict[str, Any]:
        """
        Stage 3: LLM extracts ALL problems with full repair strategies.

        Extracts:
        1. CI failure problem (main problem matching current failure)
        2. Dependent problems (must fix BEFORE, from enabled[])
        3. Consecutive problems (may appear AFTER, from causal_chain)

        Each problem includes complete repair strategy:
        - problem, root_cause, files, signals
        - repair_strategy (summary, actions, validation_cmd, pitfalls)
        """
        compact = {
            "l1": [self._compact_retrieved_item(item, "L1", idx, include_dependencies=True)
                   for idx, item in enumerate(filtered_l1)],
            "l2": [self._compact_retrieved_item(item, "L2", idx, include_dependencies=True)
                   for idx, item in enumerate(filtered_l2)],
            "l3": [self._compact_retrieved_item(item, "L3", idx, include_dependencies=True)
                   for idx, item in enumerate(filtered_l3)],
        }

        prompt = f"""Analyze CI failure memory to extract problems and their repair strategies.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Relevant Past Fixes from Memory (L1/L2/L3):**
```json
{json.dumps(compact, indent=2)}
```

---

**Your Task: Extract Problem Patterns and Repair Strategies**

Analyze the memory data to find:

1. **CI Failure Problem (Main Problem):**
   - What is the MAIN problem causing the current CI failure?
   - Match error messages, failure signals, and files with problems in memory
   - Extract complete repair strategy from past fixes

2. **Dependent Problems (Fix BEFORE main problem):**

   Follow the **enabled[] chain backwards**:
   - If CI failure = problem 5, and problem 5 has enabled=[3], then problem 3 is DEPENDENT
   - If problem 3 has enabled=[1], then problem 1 is also DEPENDENT
   - Chain: problem 1 -> problem 3 -> problem 5 (CI failure)
   - Extract ALL problems in the chain that must be fixed BEFORE the CI failure

   Example:
   ```
   Problem 1: Install dependencies (enabled=[])
   Problem 3: Fix imports (enabled=[1])
   Problem 5: Type checking (enabled=[3])  ← CI FAILURE

   Result: Dependent problems = [1, 3] (must fix in order: 1 -> 3 -> 5)
   ```

3. **Consecutive Problems (May appear AFTER fixing main problem):**

  **Approach A: Follow enabled[] chain forward**
   - If CI failure = problem 2, find problems that have enabled=[2]
   - Those problems will appear AFTER fixing problem 2
   - Continue the chain: if problem 4 has enabled=[2], and problem 7 has enabled=[4], both are consecutive
   - Chain: CI (2) -> problem 4 -> problem 7

   Example:
   ```
   Problem 2: Type checking (enabled=[1])  ← CI FAILURE
   Problem 4: Test failure (enabled=[2])   ← Appears after fixing 2
   Problem 7: Doc formatting (enabled=[4]) ← Appears after fixing 4

   Result: Consecutive problems = [4, 7] (will appear in sequence after fixing CI)
   ```

  **Approach B: Pattern Analysis Across ALL Issues**
   - Look at problem sequences across multiple issues
   - Count: How many issues show "Problem A -> Problem B" pattern?
   - If ≥50% of issues show the same sequence -> it's a real consecutive pattern

   Example:
   ```
   Issue 105: [type_error, test_failure, doc_error]
   Issue 106: [type_error, test_failure]
   Issue 113: [type_error, test_failure]
   Issue 117: [type_error, doc_error]
   Issue 123: [type_error, test_failure]

   Pattern Analysis:
   - "type_error -> test_failure": appears in 4/5 issues (80%) OK CONSECUTIVE
   - "type_error -> doc_error": appears in 2/5 issues (40%) FAIL NOT COMMON
   ```

  **Use BOTH approaches and combine results**

**Summary - Follow the Full Chain:**
```
Dependent (backwards): D1 -> D2 -> CI FAILURE -> C1 -> C2 (forward): Consecutive

If enabled[] shows:
- D1 has enabled=[]
- D2 has enabled=[D1]
- CI has enabled=[D2]
- C1 has enabled=[CI]
- C2 has enabled=[C1]

Then extract:
- dependent_problems: [D1, D2]
- ci_failure_problem: CI
- consecutive_problems: [C1, C2]

PLUS add any consecutive problems found via pattern analysis!
```

4. **For EACH Problem, Extract:**
   - problem: Clear description
   - root_cause: Why it happens
   - failure_type: Category (type_checking, linting, test_failure, etc.)
   - files: Affected files
   - failure_signals: Error messages/patterns
   - repair_strategy:
     - summary: High-level approach
     - actions: Specific steps
     - validation_cmd: How to verify fix
     - pitfalls: What to avoid
   - source: Evidence from L1/L2/L3 (issue_ids)

**Return JSON:**
```json
{{
  "ci_failure_problem": {{
    "problem": "Type checking fails due to missing numpy types",
    "root_cause": "numpy.typing was removed but stubs not installed",
    "failure_type": "type_checking",
    "files": ["src/analysis.py", "src/utils.py"],
    "failure_signals": ["Cannot resolve type DTypeLike", "Module numpy.typing not found"],
    "repair_strategy": {{
      "summary": "Install numpy type stubs",
      "actions": [
        "Add numpy-stubs to dev dependencies",
        "Run pip install numpy-stubs",
        "Update import statements"
      ],
      "validation_cmd": "mypy src/",
      "pitfalls": [
        "Don't remove numpy.typing imports without installing stubs",
        "Check mypy cache if types still not found"
      ]
    }},
    "confidence": "HIGH",
    "source": {{
      "l1": ["issue_123"],
      "l2": ["issue_456"],
      "l3": []
    }}
  }},

  "dependent_problems": [
    {{
      "problem": "Missing dependency must be installed first",
      "root_cause": "...",
      "failure_type": "dependency",
      "files": ["..."],
      "failure_signals": ["..."],
      "repair_strategy": {{
        "summary": "...",
        "actions": ["..."],
        "validation_cmd": "...",
        "pitfalls": ["..."]
      }},
      "confidence": "MEDIUM",
      "source": {{"l1": ["issue_789"], "l2": [], "l3": []}}
    }}
  ],

  "consecutive_problems": [
    {{
      "problem": "After fixing types, tests may fail",
      "root_cause": "...",
      "failure_type": "test_failure",
      "files": ["..."],
      "failure_signals": ["..."],
      "repair_strategy": {{
        "summary": "...",
        "actions": ["..."],
        "validation_cmd": "pytest",
        "pitfalls": ["..."]
      }},
      "confidence": "MEDIUM",
      "source": {{"l1": ["issue_123"], "l2": [], "l3": []}}
    }}
  ]
}}
```

**Important:**
- If no CI failure problem found, return null: `"ci_failure_problem": null`
- If no dependent problems found, return empty array: `"dependent_problems": []`
- If no consecutive problems found, return empty array: `"consecutive_problems": []`
- Extract full repair strategies from L1/L2 memory
- Use enabled[] and causal_chain fields to identify dependencies

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            parse_json=True
        )

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}

        # Ensure all fields exist
        result.setdefault('ci_failure_problem', None)
        result.setdefault('dependent_problems', [])
        result.setdefault('consecutive_problems', [])

        return result


    def _stage_3_extract_consecutive(
        self,
        filtered_l1: List[Dict],
        filtered_l2: List[Dict],
        filtered_l3: List[Dict],
        query: Dict
    ) -> List[Dict]:
        """
        Stage 3: Extract dependent/consecutive problems from filtered L1/L2/L3.

        This is driven primarily by L1 enabled[] and L2 causal_chain fields.
        """
        compact = {
            "l1": [self._compact_retrieved_item(item, "L1", idx, include_dependencies=True) for idx, item in enumerate(filtered_l1)],
            "l2": [self._compact_retrieved_item(item, "L2", idx, include_dependencies=True) for idx, item in enumerate(filtered_l2)],
            "l3": [self._compact_retrieved_item(item, "L3", idx, include_dependencies=True) for idx, item in enumerate(filtered_l3)],
        }
        prompt = f"""Analyze filtered memory and extract dependent or consecutive CI problems.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Filtered Memory Summaries:**
```json
{json.dumps(compact, indent=2)}
```

---

**Task**

Extract only problems that may appear after or alongside the current failure:
- L1 enabled[] shows problems revealed after fixing another problem.
- L2 causal_chain explains problem sequences.
- L3 dependent_changes may describe downstream validators.

Return compact, actionable entries. Include source IDs when possible.

Return JSON:
{{
  "consecutive_problems": [
    {{
      "type": "dependent|consecutive",
      "problem": "...",
      "root_cause": "...",
      "failure_type": "...",
      "validation_cmd": "...",
      "files": ["..."],
      "fix_strategy": "...",
      "repair_strategies": ["..."],
      "signals": ["..."],
      "pitfalls": ["..."],
      "source": {{"l1": ["issue_id:problem_id"], "l2": ["issue_id:step"], "l3": ["pattern_id"]}}
    }}
  ]
}}

If no consecutive problems found, return empty array: {{"consecutive_problems": []}}

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            parse_json=True
        )

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        problems = result.get('consecutive_problems', [])
        for problem in problems:
            problem.setdefault('type', 'consecutive')
            problem.setdefault('source_kind', 'consecutive')
        return problems


    def _stage_4_detect_repo_common_patterns(self, query: Dict) -> List[Dict]:
        """
        Stage 4: Detect common repo/workflow patterns using flattened problems.

        Flow:
        1. Flatten all problems from L1/L2 memory (each keeps issue_id)
        2. Cluster flattened problems deterministically
        3. Group by cluster_id and count distinct issue_ids (frequency)
        4. Pass each group to LLM for validation
        """
        # Step 1: Flatten problems with issue_id
        flattened_problems: List[Dict] = []
        total_l1_issues = 0
        total_l2_issues = 0

        repo = query.get('repo')
        workflow = query.get('workflow_name') or query.get('workflow_path')
        print(f"[Memory] Stage 4: Searching for common patterns in repo={repo}, workflow={workflow}")

        if 'l1' in self.enabled_levels:
            l1_scope = [
                item for item in self.l1_memory
                if self._same_repo(item.get('repo'), repo)
                and self._same_workflow(item.get('workflow_name') or item.get('workflow'), workflow)
            ]
            total_l1_issues = len(self._unique_issue_keys(l1_scope))
            print(f"[Memory] Stage 4: L1 scope - {len(l1_scope)} entries, {total_l1_issues} unique issues")
            for item in l1_scope:
                flattened_problems.extend(self._extract_problem_candidates(item, "L1", source_kind="common_l1"))

        if 'l2' in self.enabled_levels:
            l2_scope = [item for item in self.l2_memory if self._same_repo(item.get('repo'), repo)]
            total_l2_issues = len(self._unique_issue_keys(l2_scope))
            for item in l2_scope:
                flattened_problems.extend(self._extract_problem_candidates(item, "L2", source_kind="common_l2"))

        if not flattened_problems:
            return []

        # Step 2: Cluster flattened problems (deterministic)
        clusters = self._cluster_problem_candidates_for_common(flattened_problems)
        print(f"[Memory] Stage 4: Created {len(clusters)} clusters from {len(flattened_problems)} problems")

        # Step 3: Calculate frequency and filter by coverage
        common_candidates = []
        for cluster in clusters:
            scope = cluster.get('scope', 'L2')
            total_issues = total_l1_issues if scope == 'L1' else total_l2_issues
            distinct_issue_count = cluster['distinct_issue_count']
            coverage = distinct_issue_count / total_issues if total_issues else 0.0

            # Threshold: Problem must appear in 50%+ of issues to be considered "common"
            # Each issue counted ONCE (even if problem appears multiple times in that issue)
            # 50% is more realistic for repos with 6-10 issues
            min_coverage = 0.50  # 50% of issues must have this problem

            # Absolute minimum: at least 3 issues (prevents single/double-issue false positives)
            min_issues = 3

            print(f"[Memory] Stage 4: Cluster - issues={distinct_issue_count}/{total_issues} ({coverage:.0%}), threshold={min_coverage:.0%} coverage AND {min_issues}+ issues - {'OK PASS' if distinct_issue_count >= min_issues and coverage >= min_coverage else 'FAIL SKIP'}")

            if distinct_issue_count >= min_issues and coverage >= min_coverage:
                cluster['total_scope_issues'] = total_issues
                cluster['coverage'] = round(coverage, 4)
                common_candidates.append(cluster)

        # Sort by FREQUENCY only (coverage, then issue count) - NOT relevance
        common_candidates.sort(
            key=lambda item: (item.get('coverage', 0), item.get('distinct_issue_count', 0)),
            reverse=True
        )

        # Step 4: LLM validates each group
        return self._llm_validate_common_patterns(common_candidates, query)


    def _stage_5_merge_all_problem_types(
        self,
        ci_failure_problem: Optional[Dict],
        dependent_problems: List[Dict],
        consecutive_problems: List[Dict],
        common_patterns: List[Dict]
    ) -> List[Dict]:
        """
        Stage 5: COMBINE all 4 problem types into one candidate list.

        NOTE: This is just appending/combining, NOT clustering or merging!
        Actual clustering/deduplication happens in Stage 6.

        Problem types:
        1. CI failure problem (main)
        2. Dependent problems (fix BEFORE)
        3. Consecutive problems (fix AFTER)
        4. Common patterns (recurring)

        Each type is tagged with problem_type and priority_group for later ordering.
        """
        all_candidates = []

        # 1. CI failure problem
        if ci_failure_problem:
            all_candidates.append({
               **ci_failure_problem,
                'problem_type': 'ci_failure',
                'priority_group': 1
            })

        # 2. Dependent problems
        for problem in dependent_problems:
            all_candidates.append({
               **problem,
                'problem_type': 'dependent',
                'priority_group': 2
            })

        # 3. Consecutive problems
        for problem in consecutive_problems:
            all_candidates.append({
               **problem,
                'problem_type': 'consecutive',
                'priority_group': 3
            })

        # 4. Common patterns
        for pattern in common_patterns:
            all_candidates.append({
                'problem': pattern.get('representative_problem', ''),
                'root_cause': pattern.get('representative_root_cause', ''),
                'failure_type': pattern.get('failure_type', ''),
                'files': pattern.get('files', []),
                'failure_signals': [],
                'repair_strategy': {
                    'summary': pattern.get('representative_fix_strategy', ''),
                    'actions': [],
                    'validation_cmd': pattern.get('validation_cmd', ''),
                    'pitfalls': []
                },
                'problem_type': 'common',
                'priority_group': 4,
                'frequency': pattern.get('distinct_issue_count', 0),
                'coverage': pattern.get('coverage', 0),
                'relevance': pattern.get('relevance', 'LOW'),
                'source': {
                    'common_pattern': True,
                    'examples': pattern.get('examples', [])
                }
            })

        return all_candidates


    def _cluster_for_deduplication(self, problems: List[Dict]) -> List[Dict]:
        """
        Cluster problems for deduplication.

        Groups similar problems together so LLM can merge duplicates.
        """
        if not problems:
            return []

        clusters = self._cluster_problem_candidates_for_common(problems, default_scope="dedup")
        final_clusters = []
        for idx, cluster in enumerate(clusters, 1):
            final_clusters.append({
                "cluster_id": f"D{idx}",
                "common_pattern": cluster.get('representative_problem', ''),
                "failure_type": cluster.get('failure_type', ''),
                "validation_tool": cluster.get('validation_tool', ''),
                "distinct_issue_count": cluster.get('distinct_issue_count', 0),
                "problem_occurrence_count": cluster.get('problem_occurrence_count', 0),
                "problems": cluster.get('problems', []),
            })
        return final_clusters


    def _stage_6_llm_merge_duplicates(
        self,
        clusters: List[Dict],
        query: Dict,
        common_problems: List[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 6: LLM merges duplicate problems.

        For each cluster, LLM decides:
        - Are these the same problem? -> Merge
        - Are these different problems? -> Keep separate

        Preserves all repair strategy information.
        """
        if not clusters:
            return {"problems": []}

        final_problems = []
        for chunk in self._chunk_list(clusters, max_items=8):
            prompt = f"""Merge duplicate problems and keep distinct ones.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Problem Clusters:**
```json
{json.dumps(self._compact_clusters_for_prompt(chunk), indent=2)}
```

---

**Task:**

For each cluster, decide:
1. Are these the SAME problem (worded differently)? -> Merge into ONE
2. Are these DIFFERENT problems? -> Keep SEPARATE

When merging:
- Combine the best information from all problems
- Keep ALL repair strategy details (actions, pitfalls, validation)
- Preserve problem_type (ci_failure, dependent, consecutive, common)
- Keep source evidence

When keeping separate:
- Preserve each problem with full repair strategy

**Return JSON:**
```json
{{
  "problems": [
    {{
      "problem": "Clear problem description",
      "root_cause": "Why it happened",
      "failure_type": "type_checking|linting|test_failure|...",
      "files": ["file.py"],
      "failure_signals": ["signal"],
      "repair_strategy": {{
        "summary": "High-level approach",
        "actions": ["specific action"],
        "validation_cmd": "command",
        "pitfalls": ["avoid this"]
      }},
      "problem_type": "ci_failure|dependent|consecutive|common",
      "priority_group": 1,
      "confidence": "HIGH|MEDIUM|LOW",
      "source": {{...}}
    }}
  ]
}}
```

{STRICT_JSON_RULES}
"""
            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                parse_json=True
            )
            # Response is already parsed JSON (parse_json=True)
            result = response if isinstance(response, dict) else {}
            final_problems.extend(result.get('problems', []))

        return {"problems": final_problems}


    def _stage_5_cluster_problem_candidates(self, problems: List[Dict]) -> List[Dict]:
        """
        Stage 5: Deterministically cluster candidate problems before the final
        LLM merge decision. This keeps prompts compact and prevents large inputs
        from dropping evidence.
        """
        if not problems:
            return []

        clusters = self._cluster_problem_candidates_for_common(problems, default_scope="retrieval")
        final_clusters = []
        for idx, cluster in enumerate(clusters, 1):
            final_clusters.append({
                "cluster_id": f"C{idx}",
                "common_pattern": cluster.get('representative_problem', ''),
                "failure_type": cluster.get('failure_type', ''),
                "validation_tool": cluster.get('validation_tool', ''),
                "distinct_issue_count": cluster.get('distinct_issue_count', 0),
                "problem_occurrence_count": cluster.get('problem_occurrence_count', 0),
                "coverage": cluster.get('coverage'),
                "examples": cluster.get('examples', []),
                "problems": cluster.get('problems', []),
            })
        return final_clusters


    def _stage_6_llm_comprehensive_decision(
        self,
        clusters: List[Dict],
        query: Dict,
        common_problems: List[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 6: LLM final decision.

        Each prompt receives compact clusters, not raw memory files. If many
        clusters exist they are processed in chunks and then combined.
        """
        if not clusters:
            return {"problems": []}

        final_problems = []
        for chunk in self._chunk_list(clusters, max_items=8):
            prompt = f"""Generate final CI repair problems from clustered memory evidence.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Repo/Workflow Common Patterns:**
```json
{json.dumps(self._compact_common_patterns(common_problems), indent=2)}
```

**Problem Clusters To Decide:**
```json
{json.dumps(self._compact_clusters_for_prompt(chunk), indent=2)}
```

---

**Task**
For each cluster:
1. Decide if the cluster represents one unique problem or multiple distinct problems.
2. Produce final problems with repair strategies ready to send to a repair agent one at a time.
3. Preserve source evidence: issue IDs, problem IDs, memory levels, and common-pattern frequency.

Prefer specific problems over broad categories. Common patterns are supporting evidence only; keep the current CI failure as the primary anchor.

Return JSON:
{{
  "problems": [
    {{
      "problem": "Clear problem description",
      "root_cause": "Why this problem happened",
      "failure_type": "type_checking|linting|test_failure|...",
      "files": ["file.py"],
      "failure_signals": ["signal"],
      "repair_strategy": {{
        "summary": "High-level repair approach",
        "actions": ["specific action"],
        "pitfalls": ["mistake to avoid"],
        "validation_cmd": "command"
      }},
      "confidence": "HIGH|MEDIUM|LOW",
      "priority": 1,
      "source": {{
        "l1": ["issue_id:problem_id"],
        "l2": ["issue_id:step"],
        "l3": ["pattern_id"],
        "common_pattern": true,
        "frequency": 0,
        "coverage": 0.0,
        "examples": [{{"issue_id": "75", "problem_ids": [1]}}]
      }}
    }}
  ]
}}

{STRICT_JSON_RULES}
"""
            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                parse_json=True
            )
            # Response is already parsed JSON (parse_json=True)
            result = response if isinstance(response, dict) else {}
            final_problems.extend(result.get('problems', []))

        return {"problems": final_problems}


    def _stage_7_reorder_final_problems(
        self,
        problems: List[Dict],
        query: Dict,
        ci_failure_problem: Optional[Dict] = None,
        dependent_problems: Optional[List[Dict]] = None,
        consecutive_problems: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Stage 7: Order problems for sequential repair.

        Order:
        1. CI failure problem (main) - FIRST
        2. Dependent problems (fix BEFORE) - priority_group=2
        3. Consecutive problems (fix AFTER) - priority_group=3
        4. Common patterns (by validation sequence) - priority_group=4
        """
        ordered = []

        # Use priority_group from problems if available
        def get_priority_group(problem: Dict) -> int:
            # Use problem_type to determine group
            problem_type = problem.get('problem_type', '')
            if problem_type == 'ci_failure':
                return 1
            elif problem_type == 'dependent':
                return 2
            elif problem_type == 'consecutive':
                return 3
            elif problem_type == 'common':
                return 4
            else:
                # Fallback to old logic
                if problem.get('priority_group'):
                    return problem.get('priority_group', 99)
                source = problem.get('source', {}) if isinstance(problem.get('source'), dict) else {}
                relation = self._normalize_text(problem.get('type') or problem.get('relation') or source.get('type'))
                if self._matches_current_failure(problem, query):
                    return 1
                elif 'dependent' in relation or source.get('dependent'):
                    return 2
                elif 'consecutive' in relation or source.get('consecutive'):
                    return 3
                else:
                    return 4

        validation_order = self._validation_sequence_order(query)

        def sort_key(problem: Dict) -> Tuple[int, int, int, int]:
            priority_group = get_priority_group(problem)
            validation_rank = self._problem_validation_rank(problem, validation_order)

            # Secondary priority
            priority = problem.get('priority', 999)
            priority_rank = priority if isinstance(priority, int) else 999

            original_index = problem.get('_original_index', 0)

            return (priority_group, validation_rank, priority_rank, original_index)

        for idx, problem in enumerate(problems):
            problem['_original_index'] = idx

        ordered = sorted(problems, key=sort_key)

        for idx, problem in enumerate(ordered, 1):
            problem['order'] = idx
            problem.pop('_original_index', None)

        return ordered


    def _compact_query(self, query: Dict) -> Dict:
        return {
            "repo": query.get("repo"),
            "workflow_name": query.get("workflow_name"),
            "workflow_path": query.get("workflow_path"),
            "error_context": query.get("error_context", [])[:8],
            "failure_signals": query.get("failure_signals", [])[:12],
            "error_types": query.get("error_types", [])[:8],
            "relevant_files": query.get("relevant_files", [])[:12],
            "failed_cmd": query.get("failed_cmd", [])[:5],
        }


    def _compact_retrieved_item(
        self,
        item: Dict,
        level: str,
        idx: int,
        include_dependencies: bool = False
    ) -> Dict:
        data = item.get('item', item)
        compact = {
            "id": f"{level}:{idx}",
            "score": round(float(item.get('score', 0.0)), 4) if isinstance(item, dict) else 0.0,
            "issue_id": data.get('issue_id') or data.get('source_issue_id'),
            "repo": data.get('repo') or data.get('source_repo'),
            "workflow": data.get('workflow_name') or data.get('workflow'),
        }

        if level == "L1":
            problems = []
            for problem in data.get('problems', [])[:8]:
                problems.append({
                    "problem_id": problem.get('problem_id'),
                    "failure_type": problem.get('failure_type'),
                    "problem": self._shorten(problem.get('problem')),
                    "root_cause": self._shorten(problem.get('root_cause')),
                    "validation_cmd": problem.get('verification_cmd') or problem.get('validation_cmd'),
                    "files": problem.get('files', [])[:8],
                    "fix_strategy": self._shorten(problem.get('fix_strategy')),
                    "enabled": problem.get('enabled', []) if include_dependencies else [],
                })
            compact["problems"] = problems
        elif level == "L2":
            strategies = []
            for strategy in data.get('repair_strategies', [])[:8]:
                strategies.append({
                    "step": strategy.get('step'),
                    "failure_type": strategy.get('failure_type'),
                    "summary": self._shorten(strategy.get('summary')),
                    "causal_chain": self._shorten(strategy.get('causal_chain'), 500 if include_dependencies else 220),
                    "validation_cmd": strategy.get('validation_cmd'),
                    "signals": strategy.get('signals', [])[:5],
                    "key_actions": strategy.get('key_actions', [])[:5],
                    "pitfalls": strategy.get('pitfalls', [])[:5],
                })
            compact["repair_strategies"] = strategies
        else:
            compact.update({
                "pattern_id": data.get('pattern_id'),
                "failure_type": data.get('failure_type'),
                "failure_pattern": self._shorten(data.get('failure_pattern')),
                "problem": self._shorten(data.get('problem')),
                "when_to_apply": self._shorten(data.get('when_to_apply')),
                "dependent_changes": data.get('dependent_changes', [])[:5] if include_dependencies else [],
            })
        return compact


    def _resolve_selected_items(self, items: List[Dict], selected_ids: List[str], level: str) -> List[Dict]:
        selected = set(str(item) for item in selected_ids)
        resolved = []
        for idx, item in enumerate(items):
            if f"{level}:{idx}" in selected:
                resolved.append(item)
        return resolved


    def _extract_problem_candidates(self, item: Dict, level: str, source_kind: str) -> List[Dict]:
        """
        Extract flattened problems with ONLY issue_id (no problem_id needed for clustering).

        Each candidate contains:
        - problem, root_cause, failure_type, files, fix_strategy, etc. (for clustering)
        - issue_id (for frequency counting)
        - repo, workflow (for context)

        problem_id is NOT needed because:
        - Clustering is problem-level, not problem_id-level
        - Frequency = distinct issue count, not distinct problem count
        """
        candidates = []
        issue_id = str(item.get('issue_id') or item.get('source_issue_id') or '')
        repo = item.get('repo') or item.get('source_repo') or ''
        workflow = item.get('workflow_name') or item.get('workflow') or ''

        if level == "L1":
            # Get issue-level signals from CI context
            ci_context = item.get('benchmark_ci_context', {}) or {}
            issue_signals = ci_context.get('overall_failure_reasons', []) or []

            for problem in item.get('problems', []) or []:
                candidate = self._base_candidate(problem, item, level, source_kind)
                candidate.update({
                    "failure_signals": issue_signals,  # Use issue-level signals
                    "enabled": problem.get('enabled', []),
                })
                candidates.append(candidate)
        elif level == "L2":
            for strategy in item.get('repair_strategies', []) or []:
                candidate = self._base_candidate(strategy, item, level, source_kind)
                candidate.update({
                    "problem": strategy.get('summary') or strategy.get('causal_chain') or strategy.get('intent', ''),
                    "root_cause": strategy.get('causal_chain') or strategy.get('reasoning') or strategy.get('rationale', ''),
                    "fix_strategy": strategy.get('intent') or strategy.get('rationale') or strategy.get('summary', ''),
                    "failure_signals": strategy.get('signals', []),
                    "files": self._extract_files_from_actions(strategy.get('key_actions', [])),
                })
                candidates.append(candidate)
        else:  # L3
            candidate = self._base_candidate(item, item, level, source_kind)
            candidate.update({
                "problem": item.get('problem') or item.get('failure_pattern', ''),
                "root_cause": item.get('reasoning', ''),
                "fix_strategy": (item.get('universal_fix') or {}).get('approach', ''),
                "failure_signals": item.get('signals', []),
            })
            candidates.append(candidate)

        for candidate in candidates:
            candidate.setdefault("issue_id", issue_id)
            candidate.setdefault("repo", repo)
            candidate.setdefault("workflow", workflow)
            candidate["validation_tool"] = self._validation_tool(candidate.get("validation_cmd"))
            candidate["file_area"] = self._file_area(candidate.get("files", []))
        return candidates


    def _base_candidate(self, problem: Dict, item: Dict, level: str, source_kind: str) -> Dict:
        """
        Build base candidate structure with repair_strategy from L1/L2 data.

        This creates the standard structure BEFORE passing to LLM.
        """
        validation_cmd = problem.get('verification_cmd') or problem.get('validation_cmd') or item.get('validation_cmd', '')

        # Extract files
        files = problem.get('affected_files', []) or problem.get('files', [])
        if not isinstance(files, list):
            files = []

        # Build repair_strategy from L1 fields
        how_fixed = problem.get('how_fixed', '')
        why_fix_works = problem.get('why_fix_works', '')

        # Extract actions from how_fixed and why_fix_works
        actions = []
        if how_fixed:
            # Split by sentences or common patterns
            # Simple approach: use how_fixed as main action
            actions.append(how_fixed)
        if why_fix_works:
            actions.append(f"Rationale: {why_fix_works}")

        repair_strategy = {
            "summary": how_fixed or problem.get('fix_strategy', '') or problem.get('intent', ''),
            "actions": actions if actions else ["Fix not specified"],
            "validation_cmd": validation_cmd,
            "pitfalls": []  # L1 doesn't have pitfalls, LLM can extract later if needed
        }

        return {
            "level": level,
            "source_kind": source_kind,
            "issue_id": str(item.get('issue_id') or item.get('source_issue_id') or ''),
            "repo": item.get('repo') or item.get('source_repo') or '',
            "workflow": item.get('workflow_name') or item.get('workflow') or '',
            "failure_type": problem.get('failure_type') or item.get('failure_type') or '',
            "problem": problem.get('problem') or problem.get('summary') or problem.get('failure_pattern') or '',
            "root_cause": problem.get('root_cause') or problem.get('causal_chain') or problem.get('reasoning') or '',
            "validation_cmd": validation_cmd,
            "files": files,
            "fix_strategy": how_fixed or problem.get('fix_strategy') or problem.get('intent') or problem.get('rationale') or '',
            "repair_strategy": repair_strategy,
        }


    def _cluster_problem_candidates_for_common(
        self,
        candidates: List[Dict],
        default_scope: Optional[str] = None
    ) -> List[Dict]:
        """
        Cluster flattened problems deterministically.

        Groups by:
        1. Bucket by scope + failure_type + validation_tool + repo
        2. Within bucket, cluster by similarity (problem + root_cause + files)
        3. Track examples by issue_id (each issue_id = 1 vote for frequency)

        Returns clusters with:
        - distinct_issue_count: how many different issues had this problem
        - examples_by_issue: grouped by issue_id (not problem_id)
        """
        # Bucket only by scope and repo (not by failure_type or validation_tool)
        # Let semantic similarity do the clustering, not pre-bucketing
        buckets: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for candidate in candidates:
            scope_key = default_scope or candidate.get('source_kind', '')
            bucket = (
                scope_key,
                self._normalize_text(candidate.get('repo')) if default_scope != "retrieval" else '',
            )
            buckets[bucket].append(candidate)

        all_clusters = []
        for bucket_candidates in buckets.values():
            clusters = []
            for candidate in bucket_candidates:
                matched = None
                best_sim = 0.0
                for cluster in clusters:
                    # Threshold 0.50 gives good clustering:
                    # - 51 flower problems -> 26 clusters
                    # - NumPy (11 problems, avg sim=0.56), RST (10 problems, avg sim=0.64)
                    # - Click (3 problems, avg sim=0.58), Pytest (2 problems, avg sim=0.84)
                    sim = self._candidate_similarity(candidate, cluster['representative'])
                    best_sim = max(best_sim, sim)
                    if sim >= 0.50:  # Balanced threshold for semantic clustering
                        matched = cluster
                        break

                # Debug: log similarity scores for first few candidates
                if len(clusters) <= 5 and best_sim > 0:
                    problem_short = candidate.get('problem', '')[:60]
                    print(f"[Memory] Stage 4: Similarity={best_sim:.2f} for '{problem_short}...'")

                if matched is None:
                    clusters.append({
                        "representative": candidate,
                        "problems": [],
                        "examples_by_issue": defaultdict(lambda: {
                            "issue_id": "",
                            "problems": [],  # problem descriptions from this issue
                            "files": [],
                        })
                    })
                    matched = clusters[-1]
                matched["problems"].append(candidate)
                issue_key = self._issue_key(candidate)
                issue_example = matched["examples_by_issue"][issue_key]
                issue_example["issue_id"] = candidate.get("issue_id", "")
                if candidate.get("problem"):
                    issue_example["problems"].append(self._shorten(candidate.get("problem"), 180))
                for file_path in candidate.get("files", [])[:8]:
                    if file_path not in issue_example["files"]:
                        issue_example["files"].append(file_path)
            all_clusters.extend(clusters)

        return [self._summarize_cluster(cluster, default_scope) for cluster in all_clusters]


    def _summarize_cluster(self, cluster: Dict, default_scope: Optional[str]) -> Dict:
        """
        Summarize cluster with frequency = distinct issue_id count.

        Key change: examples show issue_ids (not problem_ids) because
        frequency is measured by "how many different CI failures had this problem".
        """
        representative = cluster["representative"]
        examples = list(cluster["examples_by_issue"].values())
        examples.sort(key=lambda item: str(item.get("issue_id", "")))
        scope = "L1" if any(p.get("source_kind") == "common_l1" for p in cluster["problems"]) else "L2"
        if default_scope:
            scope = default_scope
        return {
            "scope": scope,
            "failure_type": representative.get("failure_type", ""),
            "validation_tool": representative.get("validation_tool", ""),
            "validation_cmd": representative.get("validation_cmd", ""),
            "representative_problem": representative.get("problem", ""),
            "representative_root_cause": representative.get("root_cause", ""),
            "representative_fix_strategy": representative.get("fix_strategy", ""),
            "files": self._merge_files([p.get("files", []) for p in cluster["problems"]]),
            "file_area": representative.get("file_area", ""),
            "distinct_issue_count": len(examples),  # ← Frequency: how many distinct issues
            "problem_occurrence_count": len(cluster["problems"]),
            "examples": examples[:8],  # ← Shows issue_ids with their problems
            "problems": cluster["problems"],
        }


    def _llm_validate_common_patterns(self, common_candidates: List[Dict], query: Dict) -> List[Dict]:
        """
        LLM validates each cluster to decide if it's a real recurring pattern.

        Input: Clusters with frequency counts (distinct_issue_count)
        Output: Validated common patterns with relevance to current failure

        Key: Frequency was calculated deterministically by counting distinct issue_ids.
        LLM decides if the pattern is real/relevant, NOT the frequency.
        """
        if not common_candidates:
            return []

        accepted = []
        for chunk in self._chunk_list(common_candidates, max_items=10):
            prompt = f"""Decide which recurring CI memory clusters are true repo common problem patterns.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Recurring Clusters With Deterministic Counts:**
```json
{json.dumps(self._compact_common_patterns(chunk), indent=2)}
```

**Understanding the data:**
- `distinct_issue_count`: How many different CI failures (issues) had this problem
- `problem_occurrence_count`: Total problem instances (one issue may have same problem multiple times)
- `coverage`: distinct_issue_count / total_issues in scope
- `examples`: List of issue_ids that had this problem

**Rules:**
- Frequency and coverage are already computed by code. Do not recalculate them.
- Accept clusters that represent a specific recurring problem pattern, not a vague category.
- Keep ALL recurring patterns regardless of current CI failure - selection is based on FREQUENCY, not relevance.
- Do NOT filter based on relevance to current failure - we want ALL common repo problems.
- Your job: validate if this is a REAL recurring pattern vs a vague/noise cluster.

Return JSON:
{{
  "common_problems": [
    {{
      "cluster_id": "stable short id",
      "common": true,
      "failure_type": "...",
      "validation_tool": "...",
      "distinct_issue_count": 3,
      "problem_occurrence_count": 5,
      "coverage": 0.3,
      "representative_problem": "...",
      "representative_root_cause": "...",
      "representative_fix_strategy": "...",
      "files": ["..."],
      "examples": [{{"issue_id": "75", "problems": ["import error", "..."]}}],
      "reasoning": "why this is a real recurring repo/workflow pattern (not just noise)"
    }}
  ]
}}

{STRICT_JSON_RULES}
"""
            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                parse_json=True
            )
        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        accepted.extend(result.get("common_problems", []))

        # Sort by FREQUENCY (coverage, issue count) - NOT relevance to current failure
        # Most common patterns first
        accepted.sort(
            key=lambda item: (
                item.get("coverage", 0),              # Primary: coverage %
                item.get("distinct_issue_count", 0),  # Secondary: absolute count
            ),
            reverse=True
        )
        return accepted


    def _common_patterns_to_problem_candidates(self, common_patterns: List[Dict]) -> List[Dict]:
        """
        Convert validated common patterns to problem candidates for final clustering.

        Note: Common patterns don't have issue_id because they represent
        aggregated patterns across multiple issues.
        """
        candidates = []
        for pattern in common_patterns:
            if not pattern.get("common", True):
                continue
            candidates.append({
                "level": "COMMON",
                "source_kind": "common_pattern",
                "issue_id": "",  # ← Common patterns are aggregated, no single issue_id
                "repo": "",
                "workflow": "",
                "failure_type": pattern.get("failure_type", ""),
                "problem": pattern.get("representative_problem", ""),
                "root_cause": pattern.get("representative_root_cause", ""),
                "validation_cmd": pattern.get("validation_cmd", ""),
                "validation_tool": pattern.get("validation_tool", ""),
                "files": pattern.get("files", []),
                "file_area": self._file_area(pattern.get("files", [])),
                "fix_strategy": pattern.get("representative_fix_strategy", ""),
                "repair_strategy": pattern.get("representative_fix_strategy", ""),
                "signals": [],
                "pitfalls": [],
                "common_pattern": True,
                "frequency": pattern.get("distinct_issue_count", 0),
                "coverage": pattern.get("coverage", 0),
                "examples": pattern.get("examples", []),
            })
        return candidates


    def _compact_common_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        Compact common patterns for LLM prompt.

        Examples show issue_ids (not problem_ids) because frequency is measured
        by "how many distinct issues had this problem".
        """
        compact = []
        for pattern in patterns:
            compact.append({
                "cluster_id": pattern.get("cluster_id"),
                "scope": pattern.get("scope"),
                "common": pattern.get("common", True),
                "relevance": pattern.get("relevance"),
                "failure_type": pattern.get("failure_type"),
                "validation_tool": pattern.get("validation_tool"),
                "distinct_issue_count": pattern.get("distinct_issue_count"),  # ← distinct CI failures
                "problem_occurrence_count": pattern.get("problem_occurrence_count"),  # ← total problem instances
                "coverage": pattern.get("coverage"),
                "common_score": pattern.get("common_score"),
                "representative_problem": self._shorten(pattern.get("representative_problem"), 260),
                "representative_root_cause": self._shorten(pattern.get("representative_root_cause"), 260),
                "representative_fix_strategy": self._shorten(pattern.get("representative_fix_strategy"), 260),
                "files": pattern.get("files", [])[:8],
                "examples": pattern.get("examples", [])[:5],  # ← [{issue_id, problems, files}]
            })
        return compact


    def _compact_clusters_for_prompt(self, clusters: List[Dict]) -> List[Dict]:
        compact = []
        for cluster in clusters:
            compact.append({
                "cluster_id": cluster.get("cluster_id"),
                "common_pattern": self._shorten(cluster.get("common_pattern"), 260),
                "failure_type": cluster.get("failure_type"),
                "validation_tool": cluster.get("validation_tool"),
                "distinct_issue_count": cluster.get("distinct_issue_count"),
                "problem_occurrence_count": cluster.get("problem_occurrence_count"),
                "coverage": cluster.get("coverage"),
                "examples": cluster.get("examples", [])[:5],
                "problems": [
                    {
                        "level": problem.get("level"),
                        "source_kind": problem.get("source_kind"),
                        "source_ref": problem.get("source_ref"),
                        "problem": self._shorten(problem.get("problem"), 220),
                        "root_cause": self._shorten(problem.get("root_cause"), 220),
                        "failure_type": problem.get("failure_type"),
                        "validation_cmd": problem.get("validation_cmd"),
                        "files": problem.get("files", [])[:6],
                        "fix_strategy": self._shorten(problem.get("fix_strategy"), 220),
                        "repair_strategy": self._shorten(problem.get("repair_strategy"), 220),
                        "common_pattern": problem.get("common_pattern", False),
                        "frequency": problem.get("frequency"),
                        "coverage": problem.get("coverage"),
                    }
                    for problem in cluster.get("problems", [])[:8]
                ],
            })
        return compact


    def _candidate_similarity(self, left: Dict, right: Dict) -> float:
        """
        Calculate semantic similarity using ALL available fields dynamically.
        No hard rejects - everything contributes to the score.

        Fields used:
        - problem: main description
        - root_cause: why it happened
        - failure_signals: error messages
        - failure_type: category (semantic similarity, not exact match)
        - validation_cmd: what command failed (semantic similarity)
        - fix_strategy/repair_strategy: how to fix
        - files: which files involved
        """
        # Core semantic fields (most important)
        problem_sim = self._text_similarity(left.get('problem', ''), right.get('problem', ''))
        root_sim = self._text_similarity(left.get('root_cause', ''), right.get('root_cause', ''))

        # Failure type (semantic similarity, not exact match)
        failure_sim = self._text_similarity(
            left.get('failure_type', ''),
            right.get('failure_type', '')
        )

        # Fix strategy (how to repair it)
        left_fix = left.get('fix_strategy', '') or left.get('repair_strategy', {}).get('summary', '')
        right_fix = right.get('fix_strategy', '') or right.get('repair_strategy', {}).get('summary', '')
        fix_sim = self._text_similarity(left_fix, right_fix)

        # File similarity (smart: exact match, directory match, file type match)
        left_files = left.get('files', [])
        right_files = right.get('files', [])
        if left_files and right_files:
            file_sim = self._file_similarity(left_files, right_files)
        else:
            file_sim = 0.0

        # Weighted combination focusing on actual data fields
        # Note: verification_cmd ignored - it's often the same (./dev/test.sh)
        return (
            0.40 * problem_sim           # Main description (increased weight)
            + 0.30 * root_sim             # Root cause (increased weight)
            + 0.15 * failure_sim          # Failure type (semantic)
            + 0.10 * fix_sim              # Fix strategy
            + 0.05 * file_sim             # File overlap
        )


    def _file_similarity(self, left_files: List[str], right_files: List[str]) -> float:
        """
        Smart file similarity considering:
        1. Exact file match (1.0)
        2. Same directory + same extension (0.7)
        3. Same directory (0.5)
        4. Same extension (0.3)
        5. No match (0.0)
        """
        if not left_files or not right_files:
            return 0.0

        def file_features(filepath):
            """Extract: directory, filename, extension"""
            parts = filepath.rsplit('/', 1)
            directory = parts[0] if len(parts) > 1 else ''
            filename = parts[-1]
            ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
            return directory, filename, ext

        # Calculate max similarity for each left file with any right file
        similarities = []
        for left in left_files:
            left_dir, left_name, left_ext = file_features(left)
            max_sim = 0.0

            for right in right_files:
                right_dir, right_name, right_ext = file_features(right)

                if left == right:
                    # Exact match
                    sim = 1.0
                elif left_dir == right_dir and left_ext == right_ext:
                    # Same directory + same extension (e.g., same module, different files)
                    sim = 0.7
                elif left_dir == right_dir:
                    # Same directory (different extensions)
                    sim = 0.5
                elif left_ext == right_ext and left_ext in ('py', 'ts', 'tsx', 'js', 'yml', 'yaml', 'toml', 'json', 'md', 'rst'):
                    # Same extension (same file type, different dirs)
                    sim = 0.3
                else:
                    sim = 0.0

                max_sim = max(max_sim, sim)

            similarities.append(max_sim)

        # Average similarity across all files
        return sum(similarities) / len(similarities)


    def _common_score(self, cluster: Dict, query: Dict) -> float:
        coverage = float(cluster.get('coverage', 0.0))
        relevance = self._text_similarity(
            " ".join(query.get("error_context", []) + query.get("failure_signals", [])),
            " ".join([
                str(cluster.get("representative_problem", "")),
                str(cluster.get("representative_root_cause", "")),
                str(cluster.get("representative_fix_strategy", "")),
            ])
        )
        specificity = 1.0 if cluster.get("scope") == "L1" else 0.75
        return round((0.55 * coverage) + (0.30 * relevance) + (0.15 * specificity), 4)


    def _validation_sequence_order(self, query: Dict) -> Dict[str, int]:
        order: Dict[str, int] = {}
        validation_sequence = query.get("validation_sequence", []) or []
        for idx, step in enumerate(validation_sequence):
            if not isinstance(step, dict):
                continue
            command = step.get("validation_cmd") or step.get("command") or ""
            tool = self._validation_tool(command)
            normalized_command = self._normalize_text(command)
            if tool and tool not in order:
                order[tool] = idx
            if normalized_command and normalized_command not in order:
                order[normalized_command] = idx

        for idx, command in enumerate(query.get("failed_cmd", []) or []):
            tool = self._validation_tool(command)
            normalized_command = self._normalize_text(command)
            fallback_rank = len(order) + idx
            if tool and tool not in order:
                order[tool] = fallback_rank
            if normalized_command and normalized_command not in order:
                order[normalized_command] = fallback_rank
        return order


    def _problem_validation_rank(self, problem: Dict, validation_order: Dict[str, int]) -> int:
        if not validation_order:
            return 999

        commands = []
        direct_cmd = problem.get("validation_cmd")
        if direct_cmd:
            commands.append(direct_cmd)
        repair_strategy = problem.get("repair_strategy")
        if isinstance(repair_strategy, dict) and repair_strategy.get("validation_cmd"):
            commands.append(repair_strategy.get("validation_cmd"))

        ranks = []
        for command in commands:
            tool = self._validation_tool(command)
            normalized_command = self._normalize_text(command)
            if tool in validation_order:
                ranks.append(validation_order[tool])
            if normalized_command in validation_order:
                ranks.append(validation_order[normalized_command])
        return min(ranks) if ranks else 999


    def _matches_current_failure(self, problem: Dict, query: Dict) -> bool:
        source = problem.get('source', {}) if isinstance(problem.get('source'), dict) else {}
        if source.get('current_failure'):
            return True

        query_text = " ".join(
            [str(value) for value in query.get("error_context", [])]
            + [str(value) for value in query.get("failure_signals", [])]
            + [json.dumps(value) for value in query.get("error_types", [])]
        )
        problem_text = " ".join([
            str(problem.get("problem", "")),
            str(problem.get("root_cause", "")),
            " ".join(str(value) for value in problem.get("failure_signals", []) or []),
        ])
        text_match = self._text_similarity(query_text, problem_text) >= 0.35

        query_tools = {self._validation_tool(command) for command in query.get("failed_cmd", []) or []}
        query_tools.discard('')
        problem_cmd = (
            problem.get("validation_cmd")
            or (problem.get("repair_strategy") or {}).get("validation_cmd")
            if isinstance(problem.get("repair_strategy"), dict)
            else problem.get("validation_cmd")
        )
        problem_tool = self._validation_tool(problem_cmd)
        tool_match = not query_tools or not problem_tool or problem_tool in query_tools

        query_files = self._query_file_set(query)
        problem_files = set(problem.get("files", []) or [])
        file_match = not query_files or not problem_files or bool(query_files & problem_files)

        return text_match and tool_match and file_match


    def _query_file_set(self, query: Dict) -> set:
        files = set()
        for item in query.get("relevant_files", []) or []:
            if isinstance(item, dict):
                file_path = item.get("file") or item.get("path")
                if file_path:
                    files.add(file_path)
            elif item:
                files.add(str(item))
        return files


    def _text_similarity(self, left: Any, right: Any) -> float:
        left_norm = self._normalize_text(left)
        right_norm = self._normalize_text(right)
        if not left_norm or not right_norm:
            return 0.0
        sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
        left_tokens = set(left_norm.split())
        right_tokens = set(right_norm.split())
        token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0
        return max(sequence_score, token_score)


    def _normalize_text(self, value: Any) -> str:
        text = str(value or '').lower()
        text = re.sub(r'`[^`]+`', ' ', text)
        text = re.sub(r'\b[\w./-]+\.(py|rst|toml|yml|yaml|md|json|js|ts)\b', ' ', text)
        text = re.sub(r'\b\d+\b', ' ', text)
        text = re.sub(r'[^a-z0-9_+\[\] -]+', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()


    def _validation_tool(self, validation_cmd: Any) -> str:
        """
        Extract validation tool name from command dynamically.
        No hardcoded tool list - just parse the command naturally.
        """
        command = self._normalize_text(validation_cmd)
        if not command:
            return ''

        # Extract first significant word (skip ./dev/, ./scripts/, etc)
        parts = command.replace('./', '').replace('_', ' ').split()
        if not parts:
            return ''

        # First word is usually the tool name
        tool = parts[0]

        # Common patterns: "python -m mypy" -> "mypy", "npm run test" -> "test"
        if tool in ('python', 'python3', 'node', 'npm', 'npx', 'bash', 'sh'):
            # Skip interpreter, get actual tool
            if len(parts) > 2 and parts[1] in ('-m', 'run', '-c', 'exec'):
                return parts[2] if len(parts) > 2 else tool
            elif len(parts) > 1:
                return parts[1]

        return tool


    def _file_area(self, files: List[str]) -> str:
        if not files:
            return ''
        first = str(files[0])
        parts = [part for part in first.split('/') if part]
        if len(parts) >= 3:
            return '/'.join(parts[:3])
        return '/'.join(parts)


    def _merge_files(self, file_lists: List[List[str]]) -> List[str]:
        merged = []
        for files in file_lists:
            for file_path in files or []:
                if file_path and file_path not in merged:
                    merged.append(file_path)
        return merged[:20]


    def _extract_files_from_actions(self, actions: List[Any]) -> List[str]:
        files = []
        for action in actions or []:
            text = json.dumps(action) if isinstance(action, dict) else str(action)
            for match in re.findall(r'[\w./-]+\.(?:py|rst|toml|yml|yaml|md|json|js|ts)', text):
                if match not in files:
                    files.append(match)
        return files[:20]


    def _same_repo(self, left: Any, right: Any) -> bool:
        left_norm = self._normalize_repo(left)
        right_norm = self._normalize_repo(right)
        return bool(left_norm and right_norm and (left_norm == right_norm or left_norm.endswith('/' + right_norm) or right_norm.endswith('/' + left_norm)))


    def _normalize_repo(self, repo: Any) -> str:
        repo_text = str(repo or '').strip().lower()
        repo_text = repo_text.replace('ci-repair/', '')
        return repo_text


    def _same_workflow(self, left: Any, right: Any) -> bool:
        left_text = str(left or '').strip()
        right_text = str(right or '').strip()
        if not right_text:
            return True
        return left_text == right_text or Path(left_text).name == Path(right_text).name


    def _unique_issue_keys(self, items: List[Dict]) -> set:
        return {self._issue_key(item) for item in items}


    def _issue_key(self, item: Dict) -> Tuple[str, str, str]:
        return (
            str(item.get('repo') or item.get('source_repo') or ''),
            str(item.get('workflow_name') or item.get('workflow') or ''),
            str(item.get('issue_id') or item.get('source_issue_id') or ''),
        )


    def _chunk_list(self, items: List[Dict], max_items: int) -> List[List[Dict]]:
        return [items[index:index + max_items] for index in range(0, len(items), max_items)]


    def _shorten(self, value: Any, limit: int = 240) -> Any:
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        text = str(value or '')
        return text if len(text) <= limit else text[:limit - 3] + '...'


    def _stage_5_merge_decision(self, clusters: List[Dict]) -> List[Dict]:
        """
        Stage 5: LLM decides if cluster members should be merged.

        For each cluster, LLM decides:
        - Are these the SAME problem? -> Merge into one
        - Are these DIFFERENT problems? -> Keep separate

        Returns: List of merged/separate problems
        """
        merged_problems = []

        for cluster in clusters:
            cluster_problems = cluster.get('problems', [])

            if len(cluster_problems) == 1:
                # Single problem - no merge needed
                merged_problems.append(cluster_problems[0])
                continue

            # Ask LLM: should these be merged?
            prompt = f"""
You are analyzing a cluster of similar problems to decide if they are the SAME problem or DIFFERENT problems.

Cluster Problems:
{json.dumps(cluster_problems, indent=2)}

Decide:
1. Are these the SAME problem (just worded differently)?
2. Or are these DIFFERENT problems (that happen to be similar)?

If SAME: Merge into one problem
If DIFFERENT: Keep as separate problems

Return JSON:
{{
    "should_merge": true/false,
    "reasoning": "why they are same/different",
    "merged_problem": {{...}} if merged,
    "separate_problems": [...] if not merged
}}

{STRICT_JSON_RULES}
"""

            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt
            )

            # Response is already parsed JSON (parse_json=True)
            decision = response if isinstance(response, dict) else {}

            if decision.get('should_merge'):
                merged_problems.append(decision.get('merged_problem'))
            else:
                merged_problems.extend(decision.get('separate_problems', cluster_problems))

        return merged_problems


    def _stage_6_reorder_problems(
        self,
        problems: List[Dict],
        main_problem: Dict,
        dependencies: List[Dict]
    ) -> List[Dict]:
        """
        Stage 6: Reorder problems - CI failure first, then dependencies.

        Order:
        1. Main CI failure problem (first)
        2. Dependent problems (in dependency order)
        3. Consecutive/other problems
        """
        if not problems:
            return []

        reordered = []

        # 1. Main CI failure problem first
        if main_problem:
            reordered.append(main_problem)

        # 2. Build dependency order
        dep_graph = {}
        for dep in dependencies:
            from_id = dep.get('from')
            to_id = dep.get('to')
            if from_id and to_id:
                dep_graph.setdefault(from_id, []).append(to_id)

        # 3. Add problems in dependency order (problems that depend on nothing first)
        added_ids = {self._get_problem_id(main_problem)} if main_problem else set()

        # Add problems with no dependencies first
        for problem in problems:
            prob_id = self._get_problem_id(problem)
            if prob_id not in added_ids and prob_id not in dep_graph:
                reordered.append(problem)
                added_ids.add(prob_id)

        # Add remaining problems
        for problem in problems:
            prob_id = self._get_problem_id(problem)
            if prob_id not in added_ids:
                reordered.append(problem)
                added_ids.add(prob_id)

        return reordered

    def _get_problem_id(self, problem: Dict) -> str:
        """Get unique ID for a problem."""
        return problem.get('problem_id') or problem.get('problem', '')[:50]

    def _estimate_prompt_tokens(
        self,
        l1_matches: List[Dict],
        l2_matches: List[Dict],
        l3_matches: List[Dict],
        query: Dict,
        common_problems: List[Dict]
    ) -> int:
        """
        Estimate total tokens needed for comprehensive prompt.

        Rough estimation: 1 token ≈ 4 characters
        """
        # Convert to JSON string to get approximate size
        total_chars = 0

        total_chars += len(json.dumps(l1_matches))
        total_chars += len(json.dumps(l2_matches))
        total_chars += len(json.dumps(l3_matches))
        total_chars += len(json.dumps(query))
        total_chars += len(json.dumps(common_problems))

        # Add prompt template overhead (~2000 chars)
        total_chars += 2000

        # Convert chars to tokens (rough: 4 chars = 1 token)
        estimated_tokens = total_chars // 4

        return estimated_tokens

    def _get_model_token_limit(self) -> int:
        """
        Get model's input context window from model config.
        Falls back to 100k if model not found.
        """
        try:
            from utilities.model_token_config import get_model_config

            # Get model name from LLM instance
            model_name = getattr(self.llm, 'model_name', None)
            if model_name:
                config = get_model_config(model_name)
                return config['input_context_window']
        except Exception:
            pass  # Fall back to default

        # Conservative default
        return 100_000

    def _comprehensive_analysis_single_call(
        self,
        l1_matches: List[Dict],
        l2_matches: List[Dict],
        l3_matches: List[Dict],
        query: Dict
    ) -> Dict[str, Any]:
        """
        Single comprehensive LLM call - analyze L1/L2 data structures.

        Analyzes rich L1/L2 structure to extract consecutive/dependent problems.
        Used when prompt size < 50% of model limit.

        Note: Common problems are handled separately in Stage 4, not here.
        """
        prompt = f"""
You are analyzing CI failure memory to find problem relationships from L1/L2 data.

=== CURRENT CI FAILURE ===
{json.dumps(query, indent=2)}

=== L1 MATCHES (Same repo + workflow) ===
Structure: Each match has problems[] array with:
- problem_id, problem, root_cause, fix_strategy, files
- enabled[] field: which problems are revealed after fixing this one

{json.dumps(l1_matches, indent=2)}

=== L2 MATCHES (Same repo patterns) ===
Structure: Each match has repair_strategies[] with:
- causal_chain: Shows problem dependencies and sequences
- enabled field: Sequential dependencies
- key_actions: Step-by-step repair

{json.dumps(l2_matches, indent=2)}

=== L3 MATCHES (Universal patterns) ===
{json.dumps(l3_matches, indent=2)}

=== YOUR TASK ===

**ANALYZE L1/L2 DATA STRUCTURE:**

1. **Extract from L1 problems[] array:**
   - Look at each problem in the problems[] array
   - Identify which problem matches the current CI failure
   - Check enabled[] field to find consecutive/dependent problems

2. **Extract from L2 causal_chain:**
   - Parse causal_chain to understand problem sequences
   - Example: "Problem A -> enables Problem B" means B depends on A
   - Example: "Problem A and B appear together" means consecutive

3. **Identify problem relationships:**
   - **Main problem**: The current CI failure
   - **Dependent problems**: From enabled[] field - must fix these FIRST before main
   - **Consecutive problems**: From causal_chain - problems that appear together in sequence

4. **If no consecutive/dependent found**: Return empty arrays []

5. **For EACH problem, extract complete repair info:**
   - problem: From L1/L2 data
   - root_cause: From L1/L2 data
   - files: From L1/L2 data
   - fix_strategy: From L1/L2 data
   - fix_actions: From L2 key_actions or L1 fix_strategy
   - pitfalls: Extract from L2 rationale/reasoning
   - signals: From L2 signals field
   - validation_cmd: From L1 verification_cmd or L2 validation_cmd
   - confidence: HIGH if from L1/L2, MEDIUM if inferred
   - source: {{l1: [...], l2: [...], l3: [...]}}

6. **Cluster similar problems for deduplication:**
   - Group problems that might be same issue worded differently
   - Will be passed to merge decision next

Return JSON:
{{
    "main_problem": {{
        "problem": "...",
        "root_cause": "...",
        "files": [...],
        "fix_strategy": "...",
        "fix_actions": [...],
        "pitfalls": [...],
        "signals": [...],
        "validation_cmd": "...",
        "confidence": "HIGH/MEDIUM/LOW",
        "source": {{"l1": [...], "l2": [...], "l3": [...]}}
    }},
    "dependent_problems": [{{...}}],  // From enabled[] field
    "consecutive_problems": [{{...}}],  // From causal_chain
    "dependencies": [
        {{"from": "dep_problem_id", "to": "main_problem_id", "type": "requires"}}
    ],
    "clusters": [
        {{
            "cluster_id": "C1",
            "problems": [{{...}}, {{...}}],
            "similarity_reason": "..."
        }}
    ]
}}

**IMPORTANT:**
- Extract info from L1/L2 data structure, don't generate from scratch
- If no consecutive problems found in data, return []
- If no dependent problems found in data, return []
- Use enabled[] and causal_chain fields to find relationships

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt
        )

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        return result

    def _comprehensive_analysis_chunked(
        self,
        l1_matches: List[Dict],
        l2_matches: List[Dict],
        l3_matches: List[Dict],
        query: Dict,
        common_problems: List[Dict]
    ) -> Dict[str, Any]:
        """
        Chunked LLM analysis - analyze each level separately, then combine.

        Used when prompt size >= 50% of model limit.
        Safer but might miss cross-level connections.
        """
        # Analyze each level separately
        l1_problems = self._analyze_level_chunk(l1_matches, query, common_problems, "L1")
        l2_problems = self._analyze_level_chunk(l2_matches, query, common_problems, "L2")
        l3_problems = self._analyze_level_chunk(l3_matches, query, common_problems, "L3")

        # Combine all problems
        all_problems = l1_problems + l2_problems + l3_problems

        # Final LLM call: deduplicate and organize
        prompt = f"""
You have analyzed CI failures from 3 memory levels separately.
Now combine and deduplicate the results.

=== CURRENT CI FAILURE ===
{json.dumps(query, indent=2)}

=== PROBLEMS FROM L1 ===
{json.dumps(l1_problems, indent=2)}

=== PROBLEMS FROM L2 ===
{json.dumps(l2_problems, indent=2)}

=== PROBLEMS FROM L3 ===
{json.dumps(l3_problems, indent=2)}

TASK:
1. Cluster and merge duplicate problems
2. Identify: main problem, dependent problems, consecutive problems
3. Extract dependencies

Return JSON with deduplicated results.

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            parse_json=True
        )

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        return result

    def _analyze_level_chunk(
        self,
        matches: List[Dict],
        query: Dict,
        common_problems: List[Dict],
        level_name: str
    ) -> List[Dict]:
        """Analyze a single level (L1, L2, or L3) separately."""
        if not matches:
            return []

        prompt = f"""
Analyze {level_name} memory matches for the current CI failure.

Current CI Failure:
{json.dumps(query, indent=2)}

{level_name} Matches:
{json.dumps(matches, indent=2)}

Common Problems:
{json.dumps(common_problems, indent=2)}

Extract relevant problems with complete repair plans.

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt
        )

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        return result.get('problems', [])

    def _merge_decision_for_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """
        LLM decides if cluster members should be merged.

        For each cluster, LLM decides: SAME problem or DIFFERENT problems?
        """
        merged_problems = []

        for cluster in clusters:
            cluster_problems = cluster.get('problems', [])

            if len(cluster_problems) == 1:
                # Single problem - no merge needed
                merged_problems.append(cluster_problems[0])
                continue

            # Ask LLM: should these be merged?
            prompt = f"""
Cluster of similar problems - decide if SAME (merge) or DIFFERENT (keep separate):

{json.dumps(cluster_problems, indent=2)}

Similarity reason: {cluster.get('similarity_reason', 'Similar based on content')}

**Your Task:**
Analyze these problems and return a list. If they're the SAME problem (just worded differently),
merge them into ONE entry. If they're DIFFERENT problems, return them separately.

Return JSON:
{{
    "problems": [
        {{
            "problem": "Clear description (clearest from variants if merged)",
            "root_cause": "Complete root cause explanation",
            "failure_type": "Type from problems",
            "issue_type": "Issue type if available",
            "reasoning": "Why you merged or kept separate",
            "files": ["All affected files"],
            "failure_signals": ["All error messages/signals"],
            "fix_strategy": {{
                "summary": "Best summary from all variants",
                "actions": ["All repair actions"],
                "validation_cmd": "Validation command",
                "pitfalls": ["All pitfalls mentioned"]
            }},
            "confidence": "Highest confidence level",
            "source": {{
                "l1": ["All L1 sources"],
                "l2": ["All L2 sources"],
                "l3": ["All L3 sources"]
            }},
            "merged_from": ["id1", "id2"]  // IDs of merged problems, or [] if not merged
        }},
        // Repeat for each problem (merged or separate)
    ]
}}


{STRICT_JSON_RULES}
"""

            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt
            )

            # Response is already parsed JSON (parse_json=True)
            decision = response if isinstance(response, dict) else {}

            # Just extend with the problems list (whether merged or separate)
            merged_problems.extend(decision.get('problems', cluster_problems))

        return merged_problems

    def _reorder_problems(
        self,
        problems: List[Dict],
        main_problem: Dict,
        dependencies: List[Dict]
    ) -> List[Dict]:
        """
        Reorder problems: Main CI failure first, then dependencies.
        """
        if not problems:
            return []

        reordered = []

        # 1. Main problem first
        if main_problem:
            reordered.append(main_problem)
            main_id = self._get_problem_id(main_problem)
        else:
            main_id = None

        # 2. Add remaining problems
        for problem in problems:
            prob_id = self._get_problem_id(problem)
            if prob_id != main_id:
                reordered.append(problem)

        return reordered


    # Helper methods

    def _load_json(self, filename: str) -> List[Dict]:
        """Load JSON memory file."""
        file_path = self.memory_dir / filename
        if not file_path.exists():
            return []

        with open(file_path, 'r') as f:
            return json.load(f)


    def _get_encoder(self):
        """Load the embedding model only when retrieval actually needs it."""
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model)
        return self.encoder


    def _compute_embeddings(self, items: List[Dict], level: str) -> np.ndarray:
        """Compute embeddings for memory items."""
        if not items:
            return np.array([])

        texts = []
        for item in items:
            if level == 'l1':
                text = f"{item.get('repo', '')} {item.get('workflow', '')} "
                text += ' '.join([p.get('problem', '') for p in item.get('problems', [])])
            elif level == 'l2':
                text = f"{item.get('repo', '')} "
                text += ' '.join(item.get('failure_identify', []))
            else:  # l3
                text = f"{item.get('failure_pattern', '')} {item.get('problem', '')} {item.get('reasoning', '')}"

            texts.append(text)

        return self._get_encoder().encode(texts)


    def _build_query(self, query: Dict, level: str) -> str:
        """
        Build query string for specific level.

        L1: repo + workflow + error details
        L2: repo + error types
        L3: semantic only (no repo/file names)
        """
        if level == 'l1':
            # L1: Same repo + workflow specific
            parts = [
                query.get('repo', ''),
                query.get('workflow_name', ''),
                ' '.join(query.get('error_context', [])),
                ' '.join(query.get('failure_signals', []))
            ]
            return ' '.join(parts)

        elif level == 'l2':
            # L2: Same repo, any workflow
            parts = [
                query.get('repo', ''),
            ]
            # Add error type categories
            for et in query.get('error_types', []):
                parts.append(et.get('category', ''))
                parts.append(et.get('subcategory', ''))

            parts.extend(query.get('error_context', []))
            return ' '.join(parts)

        else:  # l3
            # L3: Semantic only (abstract repo/file details)
            parts = []

            # Error categories
            for et in query.get('error_types', []):
                parts.append(et.get('category', ''))
                parts.append(et.get('subcategory', ''))

            # Failed tools
            for rf in query.get('relevant_files', []):
                if rf.get('failed_tool'):
                    parts.append(rf.get('failed_tool'))

            # Error context (abstracted)
            for error in query.get('error_context', []):
                # Replace specific file names with <FILE>
                error = re.sub(r'\b[\w/]+\.py\b', '<FILE>', error)
                error = re.sub(r'\b[\w/]+\.js\b', '<FILE>', error)
                # Replace common package names with <PKG>
                error = re.sub(r'\b(numpy|pandas|pytest|requests|click|flask)\b', '<PKG>', error)
                parts.append(error)

            return ' '.join(parts)


    def _retrieve_topk(
        self,
        query: str,
        items: List[Dict],
        embeddings: np.ndarray,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """Retrieve top-k items using cosine similarity."""
        if embeddings is None or len(items) == 0 or len(embeddings) == 0:
            return []

        # Apply filters
        filtered_items = []
        filtered_embeddings = []

        for i, item in enumerate(items):
            passes = True
            for key, value in filters.items():
                if not value:
                    continue
                if key == 'repo':
                    if not self._same_repo(item.get('repo') or item.get('source_repo'), value):
                        passes = False
                        break
                    continue
                if key == 'workflow':
                    item_workflow = item.get('workflow_name') or item.get('workflow')
                    if not self._same_workflow(item_workflow, value):
                        passes = False
                        break
                    continue
                if item.get(key) != value:
                    passes = False
                    break
            if passes:
                filtered_items.append(item)
                filtered_embeddings.append(embeddings[i])

        if not filtered_items:
            return []

        # Compute similarity
        query_emb = self._get_encoder().encode([query])[0]
        filtered_embeddings = np.array(filtered_embeddings)

        similarities = np.dot(filtered_embeddings, query_emb) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {'item': filtered_items[idx], 'score': float(similarities[idx])}
            for idx in top_indices
        ]
