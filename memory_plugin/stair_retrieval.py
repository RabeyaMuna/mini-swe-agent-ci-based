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
from typing import Any

import numpy as np

from utilities.llm_invoker import (
    STRICT_JSON_RULES,
    invoke_llm_with_retry,
)


class DecompositionGenerationError(RuntimeError):
    """Raised when valid CI analysis cannot be decomposed into valid problems."""


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

    @staticmethod
    def _is_valid_problem(problem: dict) -> bool:
        """
        Validate that a problem has required content for agent to work with.

        REQUIRED fields (MUST be present and non-empty):
        - problem: Non-empty problem description
        - root_cause OR failure_signals: At least one must be present

        OPTIONAL fields (can be empty/null):
        - repair_strategy: Can be null (agent can work without it)
        - files: Can be empty (some problems are general)
        - error_context: Can be empty

        Args:
            problem: Problem dictionary to validate

        Returns:
            True if problem has required content, False otherwise
        """
        if not isinstance(problem, dict):
            return False

        # 1. Check problem description (REQUIRED)
        problem_desc = str(problem.get("problem", "")).strip()
        # Filter out empty strings, placeholders, and single characters
        if not problem_desc or len(problem_desc) <= 1 or problem_desc in ["", "N/A", "Unknown", "|"]:
            return False

        # 2. Check root cause OR failure signals (at least one REQUIRED)
        root_cause = str(problem.get("root_cause", "")).strip()
        failure_signals = problem.get("failure_signals", [])

        has_root_cause = root_cause and root_cause not in ["", "N/A", "Unknown"]
        has_failure_signals = isinstance(failure_signals, list) and len(failure_signals) > 0

        if not (has_root_cause or has_failure_signals):
            return False

        # 3. Files and repair_strategy are OPTIONAL
        # Agent can work without files (for general problems)
        # Agent can work without repair_strategy (will create one)

        return True

    def __init__(
        self,
        memory_dir: str,
        llm_client=None,
        embedding_model: str = "all-MiniLM-L6-v2",
        baseline_mode: bool = False,
        memory_levels: str = "l1+l2+l3",
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

        # Storage for dependency problems (extracted in STAGE 4)
        self._dependency_problems = []

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
        if "l1" in self.enabled_levels:
            self.l1_memory = self._load_json("failure_memory.json")
            self.l1_embeddings = None
        else:
            self.l1_memory = []
            self.l1_embeddings = np.array([])

        if "l2" in self.enabled_levels:
            self.l2_memory = self._load_json("repo_memory.json")
            self.l2_embeddings = None
        else:
            self.l2_memory = []
            self.l2_embeddings = np.array([])

        if "l3" in self.enabled_levels:
            self.l3_memory = self._load_json("cross_memory.json")
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
            levels = memory_levels.lower().replace(" ", "").split("+")
            return {level for level in levels if level in {"l1", "l2", "l3"}}
        elif isinstance(memory_levels, (list, set)):
            return {level for level in memory_levels if level in {"l1", "l2", "l3"}}
        else:
            # Default: all levels
            return {"l1", "l2", "l3"}

    def retrieve(
        self,
        problems: list[dict],
        top_k: int = 5,
        query: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        STAGE 1-9: Memory retrieval and problem enrichment.

        Args:
            problems: Decomposed CI problems (from decompose_ci_failure)
            top_k: Number of similar memory entries to retrieve per level
            query: Optional metadata dict with repo, workflow_path, workflow_name
                   If not provided, will extract from first problem's query

        Returns:
            Dict with enriched problems and memory data
        """
        if not self.llm:
            raise ValueError("LLM client required for all stages")

        # Baseline mode check (should not reach here if memory_plugin handles it)
        if self.baseline_mode or len(self.enabled_levels) == 0:
            print("[Memory] WARNING: baseline_mode in retrieve() - should be handled by memory_plugin")
            return {"problems": problems}

        # Use provided query metadata OR extract from first problem
        if query is None:
            query = {}
            if problems and len(problems) > 0:
                first_problem = problems[0]
                # Extract repo/workflow from problem's query
                problem_query = first_problem.get("query", {})
                if problem_query:
                    l1_query = problem_query.get("l1", {})
                    query = {
                        "repo": l1_query.get("repo", ""),
                        "workflow_name": l1_query.get("workflow_name", ""),
                        "workflow_path": l1_query.get("workflow_path", ""),
                    }

        # Use passed problems as CI problems
        ci_problems = problems
        print(f"[Memory] Starting memory retrieval for {len(ci_problems)} CI problems")

        # ============================================================
        # STAGE 1-4: Process EACH CI problem individually (FULLY MERGED)
        # ============================================================
        # For each CI problem:
        #   1. Retrieve L1/L2/L3 matches
        #   2. Filter relevant matches
        #   3. Enrich with repair strategies
        #   4. Extract dependencies
        #   5. Append all to flat problems list
        # ============================================================
        print(f"[Memory] STAGE 1-4: Processing each CI problem (retrieve → filter → enrich → dependencies)...")

        problems_list = []  # Flat list: [CI1, dep1, dep2, CI2, dep3, ...]
        total_enriched = 0
        total_dependencies = 0

        # Process EACH CI problem individually
        for idx, ci_prob in enumerate(ci_problems, 1):
            ci_prob_desc = ci_prob.get("problem", "")
            print(f"[Memory]   Processing CI problem {idx}/{len(ci_problems)}: {ci_prob_desc}...")

            # Validate CI problem has required content
            if not self._is_valid_problem(ci_prob):
                print(f"[Memory]     ⚠️  SKIP: CI problem {idx} is empty or incomplete - missing required fields")
                print(f"[Memory]          problem: {bool(ci_prob.get('problem'))}, root_cause: {bool(ci_prob.get('root_cause'))}, files: {len(ci_prob.get('files', []))}")
                continue

            # STAGE 1: Retrieve L1/L2/L3 matches for THIS CI problem
            matches = self._stage_1_per_problem_retrieval([ci_prob], query, top_k)

            if not matches or len(matches) == 0:
                # No retrieval results - add CI problem without enrichment
                print(f"[Memory]     ✗ No memory retrieval results")
                problems_list.append({
                    **ci_prob,
                    "problem_type": "ci_failure",
                    "repair_strategy": None,
                    "source": "ci_failure_no_retrieval"
                })
                total_enriched += 1
                continue

            # STAGE 2: Filter matches for THIS CI problem
            filtered = self._stage_2_per_problem_filtering(matches, query)

            if not filtered or len(filtered) == 0:
                # Filtering removed all matches
                print(f"[Memory]     ✗ All matches filtered out - no relevant memory")
                problems_list.append({
                    **ci_prob,
                    "problem_type": "ci_failure",
                    "repair_strategy": None,
                    "source": "ci_failure_no_relevant_matches"
                })
                total_enriched += 1
                continue

            # Get the filtered match set (contains ci_problem + filtered L1/L2/L3)
            match_set = filtered[0]

            # STAGE 3: Enrich THIS CI problem (with filtered matches)
            enriched_list = self._stage_3_enrich_ci_problems([match_set], query)

            if enriched_list:
                enriched = enriched_list[0]
                problems_list.append(enriched)
                total_enriched += 1
                print(f"[Memory]     ✓ Enriched with repair strategy")
            else:
                # Filtered matches didn't enrich - add without repair strategy
                enriched = {
                    **ci_prob,
                    "problem_type": "ci_failure",
                    "repair_strategy": None,
                    "source": "ci_failure_no_enrichment"
                }
                problems_list.append(enriched)
                total_enriched += 1
                print(f"[Memory]     ✗ Enrichment failed - added without repair strategy")

            # STAGE 4: Extract dependencies for THIS CI problem
            related = self._stage_4_extract_related_problems([match_set], [enriched])

            # Append dependencies (flat)
            dep_count = len(related)

            if related:
                problems_list.extend(related)
                total_dependencies += dep_count
                print(f"[Memory]     ✓ Found {dep_count} dependency problem(s)")
            else:
                print(f"[Memory]     ✗ No dependency problems")

        print(f"[Memory] STAGE 1-4: Completed {total_enriched} CI problems, {total_dependencies} dependencies")

        # ============================================================
        # STAGE 5: Detect common patterns
        # ============================================================
        print("[Memory] STAGE 4: Detecting common patterns...")
        common_problems = self._stage_5_detect_common_patterns(query)
        print(f"[Memory] STAGE 5: Found {len(common_problems)} common patterns")

        # Append common problems to the flat list
        problems_list.extend(common_problems)

        # ============================================================
        # STAGE 5.5: Deduplicate across all problem sources
        # ============================================================
        print("[Memory] STAGE 5.5: Deduplicating problems...")

        all_problems = problems_list  # Already combined: CI + dependencies + common
        # Filter out any None values that might have been returned by LLM stages
        all_problems = [p for p in all_problems if p is not None and isinstance(p, dict)]
        print(f"[Memory] STAGE 5.5: Before dedup: {len(all_problems)} problems")

        # Cluster similar problems for deduplication
        clusters = self._cluster_for_deduplication(all_problems)
        print(f"[Memory] STAGE 5.5: Created {len(clusters)} clusters")

        # LLM processes ONE CLUSTER AT A TIME to merge or separate problems
        deduped_result = self._stage_6_llm_merge_duplicates(clusters, query, common_problems)
        deduped_problems = deduped_result.get("problems", all_problems)
        print(f"[Memory] STAGE 5.5: After dedup: {len(deduped_problems)} problems")

        # ============================================================
        # STAGE 6: Reorder by dependencies and priority
        # ============================================================
        print("[Memory] STAGE 6: Reordering problems by dependencies and priority...")
        final_problems = self._stage_6_final_reorder(deduped_problems)
        print(f"[Memory] STAGE 6: Final ordered: {len(final_problems)} problems")

        # ============================================================
        # NO FILTERING: Keep all problems after deduplication and reordering
        # ============================================================
        print("[Memory] FINAL STAGE: Keeping all problems (no filtering)")
        # Filter out None values that might have been returned by LLM stages
        valid_problems = [p for p in final_problems if p is not None and isinstance(p, dict)]
        print(f"[Memory] Kept {len(valid_problems)} problems (removed {len(final_problems) - len(valid_problems)} None/invalid types)")

        # ============================================================
        # SAFETY FALLBACK: Always return at least CI problems
        # ============================================================
        # If LLM stages returned nothing, fall back to original CI problems
        if len(valid_problems) == 0 and len(ci_problems) > 0:
            print("[Memory]  WARNING: No problems after LLM processing stages!")
            print("[Memory] SAFETY FALLBACK: Returning original CI problems from decomposition")
            valid_problems = ci_problems  # Return original CI problems as-is
            print(f"[Memory] SAFETY FALLBACK: Restored {len(valid_problems)} CI problem(s)")

        # DEBUG: Show final list
        print("\n[Memory] FINAL PROBLEMS LIST:")
        for i, p in enumerate(valid_problems, 1):
            ptype = p.get('problem_type', 'unknown')
            has_repair = "✓" if p.get("repair_strategy") else "✗"
            print(f"  {i}. [{ptype}] [repair:{has_repair}] {p.get('problem', 'N/A')[:70]}")
        print()

        print("[Memory] Retrieval complete!\n")
        return {"problems": valid_problems}

    # ============================================================
    # NEW PIPELINE STAGES (0-8)
    # ============================================================

    # STAGE 0 (Decomposition) has been moved to memory_plugin.py
    # Memory plugin handles caching and decomposition
    # This class only handles STAGE 1-9 (memory retrieval)

    def _stage_1_per_problem_retrieval(
        self, ci_problems: list[dict], query: dict, top_k: int
    ) -> list[dict]:
        """
        STAGE 1: For EACH CI problem, do separate L1/L2/L3 retrieval.

        Uses structured queries from problem.query.l1/l2/l3 if available.
        Falls back to building queries from problem data if not.

        Returns:
            [
                {
                    "ci_problem": {...},
                    "l1_matches": [...],
                    "l2_matches": [...],
                    "l3_matches": [...]
                },
                ...
            ]
        """
        per_problem_matches = []

        for idx, ci_prob in enumerate(ci_problems):
            print(f"[Memory]   Retrieving for problem {idx+1}: {ci_prob.get('problem', 'N/A')[:50]}...")

            # Check if problem has structured queries
            if "query" in ci_prob and isinstance(ci_prob["query"], dict):
                # Use pre-built structured queries
                retrieved = self._stage_1_cosine_search_with_structured_queries(ci_prob, query, top_k)
            else:
                # Fallback: Build query from problem data (backward compatibility)
                problem_query = {
                    **query,
                    "problem_description": ci_prob.get("problem", ""),
                    "problem_files": ci_prob.get("files", []),
                    "problem_signals": ci_prob.get("failure_signals", []),
                    "problem_type": ci_prob.get("failure_type", ""),
                }
                retrieved = self._stage_1_cosine_search_original(problem_query, top_k)

            per_problem_matches.append({
                "ci_problem": ci_prob,
                "l1_matches": retrieved.get("l1", []),
                "l2_matches": retrieved.get("l2", []),
                "l3_matches": retrieved.get("l3", []),
            })

            print(f"[Memory]     Retrieved: L1={len(retrieved.get('l1', []))}, L2={len(retrieved.get('l2', []))}, L3={len(retrieved.get('l3', []))}")

        return per_problem_matches

    def _stage_2_per_problem_filtering(
        self, per_problem_matches: list[dict], query: dict
    ) -> list[dict]:
        """
        STAGE 2: Filter relevant matches for EACH CI problem.

        For each problem, LLM filters its L1/L2/L3 matches.
        """
        filtered = []

        for idx, match_set in enumerate(per_problem_matches):
            ci_prob = match_set["ci_problem"]
            print(f"[Memory]   Filtering problem {idx+1}: {ci_prob.get('problem', 'N/A')[:50]}...")

            # LLM filter for this specific problem
            filtered_result = self._stage_2_llm_filter_relevant_issues(
                match_set["l1_matches"],
                match_set["l2_matches"],
                match_set["l3_matches"],
                query,
            )

            filtered.append({
                "ci_problem": ci_prob,
                "l1_filtered": filtered_result.get("filtered_l1", []),
                "l2_filtered": filtered_result.get("filtered_l2", []),
                "l3_filtered": filtered_result.get("filtered_l3", []),
            })

            print(f"[Memory]     Filtered: L1={len(filtered_result.get('filtered_l1', []))}, L2={len(filtered_result.get('filtered_l2', []))}, L3={len(filtered_result.get('filtered_l3', []))}")

        return filtered

    def _stage_3_enrich_ci_problems(
        self, filtered_matches: list[dict], query: dict
    ) -> list[dict]:
        """
        STAGE 3: Enrich each CI problem with repair strategy from memory.

        For each CI problem:
        - Match with its L1/L2/L3 data
        - If match found → Extract repair strategy
        - If no match → Keep as-is (repair_strategy = null)
        """
        enriched = []

        for idx, match_set in enumerate(filtered_matches):
            ci_prob = match_set["ci_problem"]
            print(f"[Memory]   Enriching problem {idx+1}: {ci_prob.get('problem', 'N/A')[:50]}...")

            compact = {
                "l1": [
                    self._compact_retrieved_item(item, "L1", i, include_dependencies=True)
                    for i, item in enumerate(match_set["l1_filtered"])
                ],
                "l2": [
                    self._compact_retrieved_item(item, "L2", i, include_dependencies=True)
                    for i, item in enumerate(match_set["l2_filtered"])
                ],
                "l3": [
                    self._compact_retrieved_item(item, "L3", i, include_dependencies=True)
                    for i, item in enumerate(match_set["l3_filtered"])
                ],
            }

            prompt = f"""Match CI problem with memory and extract repair strategy.

**CI Problem:**
```json
{json.dumps(ci_prob, indent=2)}
```

**Memory Matches (L1/L2/L3):**
```json
{json.dumps(compact, indent=2)}
```

**Task:**
1. Check if this CI problem has similar fixes in memory (L1/L2/L3)
2. If match found → Extract repair strategy using **best available data**:

**Strategy Selection (adaptive):**
- **If L2 available**: Use L2.key_actions as actions, L2.summary as summary, L2.pitfalls as pitfalls
  - L2 `key_actions` are already detailed step-by-step - copy verbatim!
- **Else if L3 available**: Use L3.universal_fix.steps as actions, L3.approach as summary
- **Else if only L1 available**: Convert L1.fix_strategy narrative into structured action steps
  - Parse the narrative and extract concrete steps
  - Example: "Added helper function that checks..." → ["Add helper function", "Check condition", ...]

3. If NO match in any level → Return problem as-is (repair_strategy = null)

**IMPORTANT:**
- Check what levels actually have data for this problem
- Use the most structured data available
- DO NOT simplify detailed actions to generic "Analyze and fix"

**Return JSON:**
```json
{{
  "problem": "CI problem description",
  "root_cause": "Root cause (from CI or enhanced from memory)",
  "files": ["file1.py"],
  "failure_type": "type_checking",
  "failure_signals": ["error 1", "error 2"],
  "verification_cmd": "./test.sh",
  "problem_type": "ci_failure",
  "repair_strategy": {{
    "summary": "High-level repair approach",
    "actions": ["step 1", "step 2", "step 3"],
    "pitfalls": ["avoid this", "watch that"]
  }} // OR null if no match
}}
```

{STRICT_JSON_RULES}
"""

            response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)
            enriched_prob = response if isinstance(response, dict) else ci_prob

            # Ensure problem_type is set
            if "problem_type" not in enriched_prob:
                enriched_prob["problem_type"] = "ci_failure"

            enriched.append(enriched_prob)

            has_repair = "✓" if enriched_prob.get("repair_strategy") else "✗"
            print(f"[Memory]     Repair strategy: {has_repair}")

        return enriched

    def _stage_4_extract_related_problems(
        self, filtered_matches: list[dict], enriched_ci: list[dict]
    ) -> list[dict]:
        """
        STAGE 4: Extract dependent or related problems from the repair sequence.

        Extract problems that are dependent on or related to the CI problem - typically
        problems that must be fixed together or came BEFORE in the same repair sequence.

        IMPORTANT: Process EACH CI problem individually with ITS OWN L1/L2/L3 matches.
        This preserves the connection between CI problems and their dependent/related problems.

        Returns:
            List of dependent or related problems:
            [
                {"problem_type": "related", ...}
            ]
        """
        all_related_problems = []
        skipped_count = 0

        # Process EACH CI problem individually
        for match_set in filtered_matches:
            ci_prob = match_set.get("ci_problem", {})
            ci_prob_desc = ci_prob.get("problem", "")[:50]

            # Get THIS problem's L1/L2/L3 matches (not mixed with others!)
            l1_matches = match_set.get("l1_filtered", [])
            l2_matches = match_set.get("l2_filtered", [])
            l3_matches = match_set.get("l3_filtered", [])

            # Skip if no matches for this CI problem
            if not (l1_matches or l2_matches or l3_matches):
                skipped_count += 1
                print(f"[Memory] STAGE 4: Skipping '{ci_prob_desc}...' (no similarity matches)")
                continue

            # Compact THIS problem's matches
            compact = {
                "l1": [self._compact_retrieved_item(item, "L1", i, include_dependencies=True) for i, item in enumerate(l1_matches)],
                "l2": [self._compact_retrieved_item(item, "L2", i, include_dependencies=True) for i, item in enumerate(l2_matches)],
                "l3": [self._compact_retrieved_item(item, "L3", i, include_dependencies=True) for i, item in enumerate(l3_matches)],
            }

            # Extract BOTH dependency and consecutive problems in ONE prompt
            prompt = f"""Extract related problems for this CI problem by analyzing enabled[] chains and relationships.

**THIS CI Problem:**
```json
{json.dumps(ci_prob, indent=2)}
```

**Memory Data for THIS Problem (L1/L2/L3 matches):**
```json
{json.dumps(compact, indent=2)}
```

**Task:**
Extract **dependency problems** using **L1 and L2 data only** (L3 is for repair strategies, not dependencies):

**Dependency Problems** (fix BEFORE this CI problem):
   - **Primary source: L1 data** (repo+workflow specific)
   - **Check repair_sequence_index**: Problems with LOWER index numbers are dependencies
   - **Check problem_type**: If CI problem has repair_sequence_index=5, problems with index 1-4 are dependencies
   - **Extract ALL problems from matched L1 entry**: If L1 entry has 7 problems and problem #3 matches CI, return problems #1-2 as dependencies

   - **SPECIAL: Configuration problems** (same repo, CI/setup related):
   - **Config files**: `pyproject.toml`, `setup.py`, `requirements.txt`, `pytest.ini`, `mypy.ini`, `tox.ini`, `setup.cfg`
   - **If L1/L2 from SAME repo** mentions config problems related to **CI verification, installation, or setup** → **include them**
   - **Why**: These config problems are repo-specific and cause repeated CI failures
   - **Example**: "Updated pytest config in pyproject.toml" (CI verification) → include
   - **Example**: "Fixed package version in requirements.txt" (installation) → include
   - **Skip**: Non-CI configs like editor settings, formatting configs unrelated to CI

   - **Secondary source: L2 data** (repo-level patterns)
   - **Check step numbers**: Earlier steps (step 1, 2) are dependencies for later steps
   - Example: L1 matched entry shows problems with repair_sequence_index [1, 2, 3] → extract 1, 2 as dependencies
   - **IGNORE L3 for dependencies** (L3 is universal, lacks repo-specific chains)

**How to Identify Dependency Relationships:**
- **L1**: Look at `repair_sequence_index` field
  * If matched problem has index=3, problems with index=1,2 are dependencies
  * **Extract ALL problems from the SAME L1 entry** with lower indices
- **L2**: Look at `step` field in repair_strategies
  * If matched strategy is step=3, strategies with step=1,2 are dependencies
  * Extract earlier steps as dependency problems
- **Example L1 entry has 5 problems with sequence indices [1,2,3,4,5]:**
  * Index 1: "Install pip" (dependency - comes before matched problem)
  * Index 2: "Activate venv" (dependency - comes before matched problem)
  * Index 3: "Import error" (MATCHED!)
  * Index 4: "Type checking" (skip - not a dependency)
  * Index 5: "Linting" (skip - not a dependency)

  → Extract problems at index 1,2 as dependencies

**IMPORTANT - Building Repair Strategies (Adaptive):**

After extracting dependency problems from **L1/L2 data**, build repair strategies using **best available source**:

Priority order for repair strategies:

1. **If L2 data exists for this problem**:
   - Use L2.key_actions as actions (already detailed step-by-step)
   - Use L2.summary as summary
   - Use L2.pitfalls as pitfalls

2. **Else if L1 data exists for this problem**:
   - Parse L1.fix_strategy narrative into structured action steps
   - Extract concrete steps from the narrative text
   - Build summary from L1.fix_strategy

3. **Else use L3 data as fallback** (universal patterns):
   - Use L3.universal_fix.steps as actions
   - Use L3.approach as summary
   - NOTE: L3 provides generic guidance when L1/L2 lack specific strategies

**Remember**: L1/L2 are for finding problems (enabled chains), L3 is for repair strategies only.

**DO NOT default to generic actions!** Extract specific steps from whatever data is available.

**Return JSON:**
```json
{{
  "problems": [
    {{
      "problem": "Problem that must be fixed BEFORE",
      "root_cause": "Why needed first",
      "files": [...],
      "failure_type": "...",
      "failure_signals": [...],
      "verification_cmd": "...",
      "problem_type": "dependency",
      "source_ci_problem": "{ci_prob_desc}",
      "repair_strategy": {{
        "summary": "...",
        "actions": [...],
        "pitfalls": [...]
      }}
    }}
  ]
}}
```

Return empty array if no dependency problems found.
Use `problem_type: "dependency"` for all returned problems.

{STRICT_JSON_RULES}
"""

            try:
                response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)
                result = response if isinstance(response, dict) else {"problems": response} if isinstance(response, list) else {}

                # Extract all problems (both dependency and consecutive)
                problems = result.get("problems", [])

                # Validate and add only non-empty problems
                valid_problems = []
                invalid_count = 0
                for p in problems:
                    if p is not None and isinstance(p, dict) and self._is_valid_problem(p):
                        valid_problems.append(p)
                    else:
                        invalid_count += 1

                if invalid_count > 0:
                    print(f"[Memory] STAGE 4: Filtered {invalid_count} invalid related problem(s) for '{ci_prob_desc}...'")

                all_related_problems.extend(valid_problems)

                # Log count
                dep_count = len(valid_problems)
                if dep_count > 0:
                    print(f"[Memory] STAGE 4: Found {dep_count} dependency problem(s) for '{ci_prob_desc}...'")

            except Exception as e:
                print(f"[Memory] STAGE 4: Error extracting dependency problems for '{ci_prob_desc}...': {e}")
                continue

        if skipped_count > 0:
            print(f"[Memory] STAGE 4: Skipped {skipped_count} CI problem(s) without matches")

        # Total count
        print(f"[Memory] STAGE 4: Total {len(all_related_problems)} dependency problem(s) extracted")

        # Return all related problems (problem_type field distinguishes them)
        return all_related_problems


    def _stage_5_detect_common_patterns(self, query: dict) -> list[dict]:
        """
        STAGE 6: Detect common patterns.

        Uses existing _stage_4_detect_repo_common_patterns.
        """
        return self._stage_4_detect_repo_common_patterns(query)

    def _stage_6_organize_all_problems(
        self, all_problems: list[dict], query: dict
    ) -> list[dict]:
        """
        STAGE 7: LLM organize ALL problems (standardize structure).

        Takes all problems and ensures each has:
        - problem, root_cause, files, failure_signals, failure_type
        - verification_cmd
        - repair_strategy (with summary, actions, pitfalls)

        Fills gaps:
        - If has fix_strategy but no actions → LLM builds actions
        - If has actions but no summary → LLM builds summary
        - If missing pitfalls → Adds empty list
        """
        prompt = f"""Organize and standardize all problems.

**All Problems (CI + dependencies + consecutive + common):**
```json
{json.dumps(all_problems, indent=2)}
```

**Task:**
For EACH problem, standardize to complete structure:

1. **problem**: Clear description (keep existing or improve)
2. **root_cause**: Why it happens (keep existing or infer)
3. **files**: Affected files (keep existing)
4. **failure_signals**: Error messages (keep existing)
5. **failure_type**: Category (keep existing)
6. **verification_cmd**: How to verify (keep existing or infer)
7. **problem_type**: ci_failure, dependency, consecutive, or common (keep existing)
8. **repair_strategy**:
   - If has fix_strategy text but no actions → Build actions from text
   - If has actions but no summary → Build summary from actions
   - If missing pitfalls → Use empty list
   - Ensure structure: {{summary, actions[], pitfalls[]}}

**Return JSON:**
```json
{{
  "problems": [
    {{
      "problem": "...",
      "root_cause": "...",
      "files": [...],
      "failure_signals": [...],
      "failure_type": "...",
      "verification_cmd": "...",
      "problem_type": "ci_failure|dependency|consecutive|common",
      "repair_strategy": {{
        "summary": "High-level approach",
        "actions": ["step 1", "step 2", "step 3"],
        "pitfalls": ["avoid this"] // or []
      }} // can be null for CI problems without matches
    }}
  ]
}}
```

Keep ALL problems, just standardize their structure.

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)
        result = response if isinstance(response, dict) else {}
        problems = result.get("problems", all_problems)
        # Filter out None values from LLM response
        return [p for p in problems if p is not None and isinstance(p, dict)]

    def _stage_7_analyze_dependencies(
        self, problems: list[dict], query: dict
    ) -> list[dict]:
        """
        STAGE 8: LLM analyzes problem dependencies and creates repair order.

        Analyzes:
        - Which problems must be fixed BEFORE others
        - Config changes before code changes
        - Build changes before runtime changes
        - Type checking before tests

        Returns problems in dependency order (topologically sorted).
        """
        if len(problems) <= 1:
            return problems

        prompt = f"""Analyze problem dependencies and create optimal repair order.

**All Problems:**
```json
{json.dumps([
    {
        "id": i,
        "problem": p.get("problem", "")[:100],
        "root_cause": p.get("root_cause", "")[:100],
        "files": p.get("files", [])[:3],
        "failure_type": p.get("failure_type", ""),
        "problem_type": p.get("problem_type", ""),
        "repair_strategy": {
            "summary": (p.get("repair_strategy") or {}).get("summary", "")[:80] if isinstance(p.get("repair_strategy"), dict) else ""
        } if p.get("repair_strategy") else None
    }
    for i, p in enumerate(problems, 1)
], indent=2)}
```

**Task:**
Analyze the ACTUAL dependencies between these specific problems.

**Instructions:**
1. Read each problem's description, root cause, files, and repair strategy
2. Identify which problems **actually depend on others** based on:
   - Does Problem B use files/changes from Problem A?
   - Does Problem B's root cause mention Problem A?
   - Do Problem B's repair actions require Problem A to be fixed first?
   - Is Problem A a config/setup that Problem B needs?

3. Build a dependency graph dynamically:
   - If Problem B needs Problem A → A must come before B
   - If problems are independent → keep original order
   - If there's a chain A→B→C → order is A, B, C

**DO NOT use hardcoded rules!** Analyze the ACTUAL problems:
- Don't assume all config changes come first
- Don't assume all type checking comes before tests
- Look at the specific problems and their relationships

**Example Analysis Process:**

Look at Problem 1: "Update mdformat-beautysh dependency"
- Files: pyproject.toml
- Type: dependency

Look at Problem 2: "Fix RST files to comply with mdformat 1.0.0"
- Root cause mentions: "mdformat-beautysh 1.0.0 enforces stricter rules"
- **Depends on Problem 1!** (needs the upgrade)

Look at Problem 3: "Fix exit_code_test reading RST titles"
- Files: exit_code_test.py
- Root cause mentions: "RST format changed from underline-only to overline+underline"
- **Depends on Problem 2!** (reads the changed RST files)

Result: 1 → 2 → 3

**Return JSON:**
```json
{{
  "dependency_analysis": "Brief explanation of dependency relationships found",
  "ordered_problem_ids": [1, 3, 2, 5, 4, ...]  // IDs in dependency order
}}
```

**Rules:**
- Put CI failures first UNLESS they depend on other problems
- Put config/dependency problems early
- Respect causal chains (A enables B, B enables C → order: A, B, C)
- If no dependency, keep original order

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)
        result = response if isinstance(response, dict) else {}

        ordered_ids = result.get("ordered_problem_ids", list(range(1, len(problems) + 1)))

        # Reorder problems based on LLM's dependency analysis
        id_to_problem = {i+1: p for i, p in enumerate(problems)}
        reordered = []
        for problem_id in ordered_ids:
            if problem_id in id_to_problem:
                reordered.append(id_to_problem[problem_id])

        # Add any problems that weren't in the ordered list (safety)
        reordered_ids = set(ordered_ids)
        for i, p in enumerate(problems, 1):
            if i not in reordered_ids:
                reordered.append(p)

        # Log dependency analysis
        if result.get("dependency_analysis"):
            print(f"[Memory]   Analysis: {result['dependency_analysis'][:100]}...")

        return reordered

    def _stage_6_final_reorder(self, problems: list[dict]) -> list[dict]:
        """
        STAGE 9: Final reordering with strict priority enforcement.

        Priority order (highest to lowest):
        1. ci_failure - Problems from current CI logs (MUST BE FIRST)
        2. dependent - Prerequisites needed before ci_failure fixes
        3. consecutive - Problems that arise after fixes
        4. common - General patterns from memory
        5. unknown - Unclassified problems

        Within each group, preserve Stage 8 dependency order.
        """
        if not problems:
            return []

        # Group problems by type
        ci_failures = []
        dependents = []
        consecutives = []
        commons = []
        unknowns = []

        for prob in problems:
            problem_type = prob.get("problem_type", "").lower()
            if problem_type == "ci_failure":
                ci_failures.append(prob)
            elif problem_type in ["dependent", "dependency"]:
                dependents.append(prob)
            elif problem_type == "consecutive":
                consecutives.append(prob)
            elif problem_type == "common":
                commons.append(prob)
            else:
                unknowns.append(prob)

        # Combine in priority order
        final_order = ci_failures + dependents + consecutives + commons + unknowns

        print(f"[Memory] STAGE 9: Reordered by priority - ci_failure({len(ci_failures)}), dependent({len(dependents)}), consecutive({len(consecutives)}), common({len(commons)}), unknown({len(unknowns)})")

        return final_order

    # ============================================================
    # ORIGINAL METHODS (used by new pipeline)
    # ============================================================

    def _stage_1_cosine_search_with_structured_queries(
        self, problem: dict, query: dict, top_k: int
    ) -> dict[str, list]:
        """
        Stage 1: Cosine similarity search using structured queries from problem.

        Uses problem.query.l1/l2/l3 structured queries instead of building them.
        """
        structured_queries = problem.get("query", {})

        # L1: Use structured query if available
        if "l1" in self.enabled_levels:
            if self.l1_embeddings is None:
                self.l1_embeddings = self._compute_embeddings(self.l1_memory, "l1")

            l1_query_obj = structured_queries.get("l1", {})
            l1_query_str = self._build_search_string_from_structured_query(l1_query_obj, "l1")

            # Build filters: use workflow_path only (not workflow_name from YAML)
            # If workflow_path is missing, skip workflow filter to be more lenient
            workflow_filter = l1_query_obj.get("workflow_path") or query.get("workflow_path")

            l1_results = self._retrieve_topk(
                l1_query_str,
                self.l1_memory,
                self.l1_embeddings,
                top_k,
                filters={
                    "repo": l1_query_obj.get("repo") or query.get("repo"),
                    "workflow": workflow_filter or "",  # Empty string = skip filter
                },
            )
        else:
            l1_results = []

        # L2: Use structured query if available
        if "l2" in self.enabled_levels:
            if self.l2_embeddings is None:
                self.l2_embeddings = self._compute_embeddings(self.l2_memory, "l2")

            l2_query_obj = structured_queries.get("l2", {})
            l2_query_str = self._build_search_string_from_structured_query(l2_query_obj, "l2")

            l2_results = self._retrieve_topk(
                l2_query_str,
                self.l2_memory,
                self.l2_embeddings,
                top_k,
                filters={"repo": l2_query_obj.get("repo") or query.get("repo")},
            )
        else:
            l2_results = []

        # L3: Use structured query if available
        if "l3" in self.enabled_levels:
            if self.l3_embeddings is None:
                self.l3_embeddings = self._compute_embeddings(self.l3_memory, "l3")

            l3_query_obj = structured_queries.get("l3", {})
            l3_query_str = self._build_search_string_from_structured_query(l3_query_obj, "l3")

            l3_results = self._retrieve_topk(
                l3_query_str,
                self.l3_memory,
                self.l3_embeddings,
                top_k,
                filters={},  # No filters for cross-repo
            )
        else:
            l3_results = []

        return {"l1": l1_results, "l2": l2_results, "l3": l3_results}

    def _build_search_string_from_structured_query(self, query_obj: dict, level: str) -> str:
        """
        Convert structured query object to search string.

        Args:
            query_obj: Structured query with fields like problem, root_cause, files, etc.
            level: "l1", "l2", or "l3"

        Returns:
            Space-separated search string
        """
        if not query_obj:
            return ""

        parts = []

        # Add level-specific fields
        if level == "l1":
            parts.append(query_obj.get("repo", ""))
            # Use workflow_path filename for semantic matching (more reliable than YAML name)
            workflow_path = query_obj.get("workflow_path", "")
            if workflow_path:
                from pathlib import Path
                parts.append(Path(workflow_path).name)  # e.g., "run-tests.yml"
            else:
                # Fallback to workflow_name if path not available
                parts.append(query_obj.get("workflow_name", ""))
        elif level == "l2":
            parts.append(query_obj.get("repo", ""))

        # Common fields across all levels
        parts.append(query_obj.get("problem", ""))
        parts.append(query_obj.get("root_cause", ""))

        # Add files (only for L1/L2)
        if level in ["l1", "l2"]:
            parts.extend(query_obj.get("files", []))

        # Add failure types (array of 2 strings)
        parts.extend(query_obj.get("failure_types", []))

        # Add failure signals (array)
        parts.extend(query_obj.get("failure_signals", []))

        # Filter out empty strings and join
        return " ".join(str(p) for p in parts if p)

    def _stage_1_cosine_search_original(self, query: dict, top_k: int) -> dict[str, list]:
        """
        Stage 1: Cosine similarity search from enabled levels only.

        Returns top-k matches per level based on ablation settings.
        """
        # L1: Only retrieve if enabled
        if "l1" in self.enabled_levels:
            if self.l1_embeddings is None:
                self.l1_embeddings = self._compute_embeddings(self.l1_memory, "l1")
            l1_query = self._build_query(query, level="l1")
            # Use workflow_path only (not workflow_name from YAML)
            # If workflow_path is missing, skip workflow filter
            workflow_filter = query.get("workflow_path") or ""
            l1_results = self._retrieve_topk(
                l1_query,
                self.l1_memory,
                self.l1_embeddings,
                top_k,
                filters={
                    "repo": query.get("repo"),
                    "workflow": workflow_filter,  # Empty string = skip filter
                },
            )
        else:
            l1_results = []

        # L2: Only retrieve if enabled
        if "l2" in self.enabled_levels:
            if self.l2_embeddings is None:
                self.l2_embeddings = self._compute_embeddings(self.l2_memory, "l2")
            l2_query = self._build_query(query, level="l2")
            l2_results = self._retrieve_topk(
                l2_query,
                self.l2_memory,
                self.l2_embeddings,
                top_k,
                filters={"repo": query.get("repo")},
            )
        else:
            l2_results = []

        # L3: Only retrieve if enabled
        if "l3" in self.enabled_levels:
            if self.l3_embeddings is None:
                self.l3_embeddings = self._compute_embeddings(self.l3_memory, "l3")
            l3_query = self._build_query(query, level="l3")
            l3_results = self._retrieve_topk(
                l3_query,
                self.l3_memory,
                self.l3_embeddings,
                top_k,
                filters={},  # No filters for cross-repo
            )
        else:
            l3_results = []

        return {"l1": l1_results, "l2": l2_results, "l3": l3_results}

    def _stage_2_llm_filter_relevant_issues(
        self,
        l1_items: list[dict],
        l2_items: list[dict],
        l3_items: list[dict],
        query: dict,
    ) -> dict[str, Any]:
        """
        Stage 2: LLM filters relevant ISSUES (not problems yet).

        This stage ONLY selects which issues are relevant.
        Problem extraction happens in Stage 3.
        """
        compact_l1 = [
            self._compact_retrieved_item(item, "L1", idx)
            for idx, item in enumerate(l1_items)
        ]
        compact_l2 = [
            self._compact_retrieved_item(item, "L2", idx)
            for idx, item in enumerate(l2_items)
        ]
        compact_l3 = [
            self._compact_retrieved_item(item, "L3", idx)
            for idx, item in enumerate(l3_items)
        ]

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

IMPORTANT - Use DIFFERENT matching criteria for each level:

**L1 (File-level problems)** - STRICT matching:
- Exact or similar error messages
- Same or compatible files/paths
- Same validation tool

**L2 (Repo-specific strategies)** - MODERATE matching:
- Same failure type
- Same repo or similar technology stack
- Compatible validation approach

**L3 (Universal patterns)** - BROAD matching:
- Same failure TYPE (e.g., dependency_upgrade, type_checking, formatting)
- Similar failure PATTERN (e.g., "config change requires code adaptation")
- Apply L3 if the pattern/approach could work, even if files/repo differ
- L3 patterns are UNIVERSAL - they apply across repos/files

Return IDs from the summaries. Do not invent IDs.

Return JSON:
{{
  "selected_l1_ids": ["L1:0"],
  "selected_l2_ids": ["L2:0"],
  "selected_l3_ids": ["L3:0", "L3:1"],
  "relevance_notes": "brief reason"
}}

    {STRICT_JSON_RULES}
    """

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        filtered_l1 = self._resolve_selected_items(
            l1_items, result.get("selected_l1_ids", []), "L1"
        )
        filtered_l2 = self._resolve_selected_items(
            l2_items, result.get("selected_l2_ids", []), "L2"
        )
        filtered_l3 = self._resolve_selected_items(
            l3_items, result.get("selected_l3_ids", []), "L3"
        )

        return {
            "filtered_l1": filtered_l1,
            "filtered_l2": filtered_l2,
            "filtered_l3": filtered_l3,
            "relevance_notes": result.get("relevance_notes", ""),
        }

    def _stage_3_llm_extract_problems_and_strategies(
        self,
        filtered_l1: list[dict],
        filtered_l2: list[dict],
        filtered_l3: list[dict],
        query: dict,
    ) -> dict[str, Any]:
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
            "l1": [
                self._compact_retrieved_item(item, "L1", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l1)
            ],
            "l2": [
                self._compact_retrieved_item(item, "L2", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l2)
            ],
            "l3": [
                self._compact_retrieved_item(item, "L3", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l3)
            ],
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

**Understanding Memory Data Structures:**

- **L1** (repo+workflow specific): Each problem has `how_fixed` (description) and `why_fix_works` (rationale)
  - These are narrative descriptions, not step-by-step actions
  - Organize them into actionable steps when extracting

- **L2** (repo-level patterns): Has `key_actions` (list of steps) and `intent` (summary)
  - Already structured as steps
  - May include `pitfalls` list

- **L3** (universal patterns): Has `universal_fix.approach` and `universal_fix.steps`
  - Generic patterns applicable across repos
  - Structure is similar to L2

**Your Task: Extract Problem Patterns and Repair Strategies**

For each problem you extract, build a COMPLETE repair_strategy:

1. **CI Failure Problem (Main Problem):**
   - What is the MAIN problem causing the current CI failure?
   - Match error messages, failure signals, and files with problems in memory
   - Build repair_strategy from available data:
     * If L1: Convert `how_fixed` + `why_fix_works` into organized action steps
     * If L2: Use `key_actions` directly as actions, `intent` as summary
     * If L3: Use `universal_fix.steps` as actions, `approach` as summary

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
  "problems": [
    {{
      "problem": "Type checking fails due to missing numpy types",
      "root_cause": "numpy.typing was removed but stubs not installed",
      "failure_type": "type_checking",
      "problem_type": "ci_failure",
      "files": ["src/analysis.py", "src/utils.py"],
      "failure_signals": ["Cannot resolve type DTypeLike", "Module numpy.typing not found"],
      "verification_cmd": "mypy src/",
      "repair_strategy": {{
        "summary": "Install numpy type stubs to resolve missing type definitions. This fix works because numpy-stubs provides the type definitions that were removed from the main numpy package.",
        "actions": [
          "Add numpy-stubs to dev dependencies in pyproject.toml or requirements-dev.txt",
          "Run pip install numpy-stubs to install the package",
          "Verify mypy can now resolve the types by running mypy src/",
          "If errors persist, clear mypy cache with mypy --clear-cache"
        ],
        "pitfalls": [
          "Don't remove numpy.typing imports without installing stubs first",
          "Check mypy cache if types still not found after install"
        ]
      }},
      "enabled": []
    }},
    {{
      "problem": "After fixing types, tests may fail due to changed behavior",
      "root_cause": "Type fixes may expose runtime type mismatches in test data",
      "failure_type": "test_failure",
      "problem_type": "consecutive",
      "files": ["tests/test_analysis.py"],
      "failure_signals": [],
      "verification_cmd": "pytest tests/",
      "repair_strategy": {{
        "summary": "Update test fixtures to match the corrected types. Tests fail because they were written for the old (incorrect) type handling.",
        "actions": [
          "Run pytest to identify failing tests",
          "Review test data and fixtures for type mismatches",
          "Update test fixtures to use correct types",
          "Re-run pytest to verify all tests pass"
        ],
        "pitfalls": []
      }},
      "enabled": [0]
    }},
    // ... more problems
  ]
}}
```

**Important:**
- **problem_type**: "ci_failure" (main), "dependent" (fix before), "consecutive" (appears after), or "common" (general pattern)
- **enabled**: List of problem indices this depends on (e.g., [0] means depends on problem 0)
- All problems in ONE flat list with dependency info for reordering
- Create organized action steps even if source only has L1 fix_strategy text
- Keep failure_signals and pitfalls empty if not available
- Use verification_cmd (not validation_cmd)
- Extract full repair strategies from L1/L2/L3 memory based on their data structures

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}

        # Return just the problems list (LLM returns flat list, not categorized)
        problems = result.get("problems", [])

        return {"problems": problems}

    def _stage_3_match_ci_and_extract_consecutive(
        self,
        ci_problems: list[dict],
        filtered_l1: list[dict],
        filtered_l2: list[dict],
        filtered_l3: list[dict],
        query: dict,
    ) -> dict[str, Any]:
        """
        Stage 3: Match CI problems with memory and extract consecutive problems.

        Process:
        1. For each CI problem, search memory for matching problems
        2. If match found, add repair strategy from memory
        3. Extract consecutive problems (appear after fixing CI problems)

        Returns:
            {
                "problems": [
                    {CI problem 1 with/without repair strategy},
                    {CI problem 2 with/without repair strategy},
                    ...
                    {consecutive problem 1 with repair strategy},
                    {consecutive problem 2 with repair strategy},
                    ...
                ]
            }
        """
        compact = {
            "l1": [
                self._compact_retrieved_item(item, "L1", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l1)
            ],
            "l2": [
                self._compact_retrieved_item(item, "L2", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l2)
            ],
            "l3": [
                self._compact_retrieved_item(item, "L3", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l3)
            ],
        }

        prompt = f"""Match current CI problems with memory and extract consecutive problems.

**Current CI Problems:**
```json
{json.dumps(ci_problems, indent=2)}
```

**Current CI Failure Context:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Relevant Past Fixes from Memory (L1/L2/L3):**
```json
{json.dumps(compact, indent=2)}
```

---

**Your Task:**

1. **Match CI Problems with Memory:**
   - For each CI problem, find similar problems in memory (match by file, error type, failure signals)
   - If match found, extract repair strategy from memory:
     * L1: Convert `how_fixed` + `why_fix_works` into action steps
     * L2: Use `key_actions` as actions, `intent` as summary
     * L3: Use `universal_fix.steps` as actions, `approach` as summary
   - If no match found, set repair_strategy = null

2. **Extract Consecutive Problems:**
   - Find problems that typically appear AFTER fixing these CI problems
   - Use enabled[] chains and causal relationships from memory
   - Each consecutive problem must have complete repair strategy

**Return JSON Structure:**
```json
{{
  "problems": [
    // CI problems FIRST (in order received)
    {{
      "problem": "CI problem description",
      "root_cause": "Why it happens (from memory if found, or 'Unknown')",
      "failure_type": "type_checking|linting|test_failure|build|...",
      "problem_type": "ci_failure",
      "files": ["file1.py", "file2.py"],
      "failure_signals": ["error message 1", "error message 2"],
      "verification_cmd": "command to verify fix",
      "repair_strategy": {{
        "summary": "High-level approach from memory",
        "actions": ["step 1", "step 2", "step 3"],
        "pitfalls": ["avoid this", "watch for that"]
      }} // OR null if no match in memory
    }},
    // Consecutive problems AFTER
    {{
      "problem": "Consecutive problem description",
      "root_cause": "Why it appears after CI fix",
      "failure_type": "...",
      "problem_type": "consecutive",
      "files": [...],
      "failure_signals": [...],
      "verification_cmd": "...",
      "repair_strategy": {{...}} // Must have strategy from memory
    }}
  ]
}}
```

{STRICT_JSON_RULES}
"""

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)
        result = response if isinstance(response, dict) else {}
        problems = result.get("problems", [])

        return {"problems": problems}

    def _stage_3_extract_consecutive(
        self,
        filtered_l1: list[dict],
        filtered_l2: list[dict],
        filtered_l3: list[dict],
        query: dict,
    ) -> list[dict]:
        """
        Stage 3: Extract dependent/consecutive problems from filtered L1/L2/L3.

        This is driven primarily by L1 enabled[] and L2 causal_chain fields.
        """
        compact = {
            "l1": [
                self._compact_retrieved_item(item, "L1", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l1)
            ],
            "l2": [
                self._compact_retrieved_item(item, "L2", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l2)
            ],
            "l3": [
                self._compact_retrieved_item(item, "L3", idx, include_dependencies=True)
                for idx, item in enumerate(filtered_l3)
            ],
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

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        problems = result.get("consecutive_problems", [])
        for problem in problems:
            problem.setdefault("type", "consecutive")
            problem.setdefault("source_kind", "consecutive")
        return problems

    def _stage_4_detect_repo_common_patterns(self, query: dict) -> list[dict]:
        """
        Stage 4: Detect common repo/workflow patterns using flattened problems.

        Flow:
        1. Flatten all problems from L1/L2 memory (each keeps issue_id)
        2. Cluster flattened problems deterministically
        3. Group by cluster_id and count distinct issue_ids (frequency)
        4. Pass each group to LLM for validation
        """
        # Step 1: Flatten problems with issue_id
        flattened_problems: list[dict] = []
        total_l1_issues = 0
        total_l2_issues = 0

        repo = query.get("repo")
        # Use workflow_path (reliable) instead of workflow_name (YAML name)
        workflow = query.get("workflow_path") or query.get("workflow_name") or ""
        print(
            f"[Memory] Stage 4: Searching for common patterns in repo={repo}, workflow_path={workflow}"
        )

        if "l1" in self.enabled_levels:
            l1_scope = [
                item
                for item in self.l1_memory
                if self._same_repo(item.get("repo"), repo)
                and self._same_workflow(
                    item.get("workflow_path") or item.get("workflow_name"), workflow
                )
            ]
            total_l1_issues = len(self._unique_issue_keys(l1_scope))
            print(
                f"[Memory] Stage 4: L1 scope - {len(l1_scope)} entries, {total_l1_issues} unique issues"
            )
            for item in l1_scope:
                flattened_problems.extend(
                    self._extract_problem_candidates(
                        item, "L1", source_kind="common_l1"
                    )
                )

        if "l2" in self.enabled_levels:
            # L2 is repo+workflow specific (same as L1), but contains repair strategies instead of problems
            l2_scope = [
                item
                for item in self.l2_memory
                if self._same_repo(item.get("repo"), repo)
                and self._same_workflow(
                    item.get("workflow_path") or item.get("workflow_name"), workflow
                )
            ]
            total_l2_issues = len(self._unique_issue_keys(l2_scope))
            print(
                f"[Memory] Stage 4: L2 scope - {len(l2_scope)} entries, {total_l2_issues} unique issues"
            )
            for item in l2_scope:
                flattened_problems.extend(
                    self._extract_problem_candidates(
                        item, "L2", source_kind="common_l2"
                    )
                )

        if not flattened_problems:
            print("[Memory] Stage 4: No problems found in L1/L2 for this repo+workflow")
            return []

        # Early exit: Need at least 3 unique issues to establish commonality
        total_unique_issues = max(total_l1_issues, total_l2_issues)
        if total_unique_issues < 3:
            print(f"[Memory] Stage 4: Only {total_unique_issues} unique issue(s) found - need at least 3 to establish commonality. Skipping common pattern detection.")
            return []

        print(f"[Memory] Stage 4: Found {total_unique_issues} unique issues - proceeding with common pattern detection")

        # Step 2: Cluster flattened problems (deterministic)
        clusters = self._cluster_problem_candidates_for_common(flattened_problems)
        print(
            f"[Memory] Stage 4: Created {len(clusters)} clusters from {len(flattened_problems)} problems"
        )

        # Step 3: Calculate frequency and filter by coverage
        common_candidates = []
        for cluster in clusters:
            scope = cluster.get("scope", "L2")
            total_issues = total_l1_issues if scope == "L1" else total_l2_issues
            distinct_issue_count = cluster["distinct_issue_count"]
            coverage = distinct_issue_count / total_issues if total_issues else 0.0

            # Threshold: Problem must appear in 50%+ of issues to be considered "common"
            # Each issue counted ONCE (even if problem appears multiple times in that issue)
            # 50% is more realistic for repos with 6-10 issues
            min_coverage = 0.50  # 50% of issues must have this problem

            # Absolute minimum: at least 3 issues (prevents single/double-issue false positives)
            min_issues = 3

            print(
                f"[Memory] Stage 4: Cluster - issues={distinct_issue_count}/{total_issues} ({coverage:.0%}), threshold={min_coverage:.0%} coverage AND {min_issues}+ issues - {'OK PASS' if distinct_issue_count >= min_issues and coverage >= min_coverage else 'FAIL SKIP'}"
            )

            if distinct_issue_count >= min_issues and coverage >= min_coverage:
                cluster["total_scope_issues"] = total_issues
                cluster["coverage"] = round(coverage, 4)
                common_candidates.append(cluster)

        # Sort by FREQUENCY only (coverage, then issue count) - NOT relevance
        common_candidates.sort(
            key=lambda item: (
                item.get("coverage", 0),
                item.get("distinct_issue_count", 0),
            ),
            reverse=True,
        )

        # Step 4: LLM validates each group
        return self._llm_validate_common_patterns(common_candidates, query)

    def _stage_5_merge_all_problem_types(
        self,
        ci_failure_problem: dict | None,
        dependent_problems: list[dict],
        consecutive_problems: list[dict],
        common_patterns: list[dict],
    ) -> list[dict]:
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
            all_candidates.append(
                {
                    **ci_failure_problem,
                    "problem_type": "ci_failure",
                    "priority_group": 1,
                }
            )

        # 2. Dependent problems
        for problem in dependent_problems:
            all_candidates.append(
                {**problem, "problem_type": "dependent", "priority_group": 2}
            )

        # 3. Consecutive problems
        for problem in consecutive_problems:
            all_candidates.append(
                {**problem, "problem_type": "consecutive", "priority_group": 3}
            )

        # 4. Common patterns
        for pattern in common_patterns:
            all_candidates.append(
                {
                    "problem": pattern.get("representative_problem", ""),
                    "root_cause": pattern.get("representative_root_cause", ""),
                    "failure_type": pattern.get("failure_type", ""),
                    "files": pattern.get("files", []),
                    "failure_signals": [],
                    "repair_strategy": {
                        "summary": pattern.get("representative_fix_strategy", ""),
                        "actions": [],
                        "validation_cmd": pattern.get("validation_cmd", ""),
                        "pitfalls": [],
                    },
                    "problem_type": "common",
                    "priority_group": 4,
                    "frequency": pattern.get("distinct_issue_count", 0),
                    "coverage": pattern.get("coverage", 0),
                    "relevance": pattern.get("relevance", "LOW"),
                    "source": {
                        "common_pattern": True,
                        "examples": pattern.get("examples", []),
                    },
                }
            )

        return all_candidates

    def _cluster_for_deduplication(self, problems: list[dict]) -> list[dict]:
        """
        Cluster problems for deduplication.

        Groups similar problems together so LLM can merge duplicates.
        """
        if not problems:
            return []

        clusters = self._cluster_problem_candidates_for_common(
            problems, default_scope="dedup"
        )
        final_clusters = []
        for idx, cluster in enumerate(clusters, 1):
            final_clusters.append(
                {
                    "cluster_id": f"D{idx}",
                    "common_pattern": cluster.get("representative_problem", ""),
                    "failure_type": cluster.get("failure_type", ""),
                    "validation_tool": cluster.get("validation_tool", ""),
                    "distinct_issue_count": cluster.get("distinct_issue_count", 0),
                    "problem_occurrence_count": cluster.get(
                        "problem_occurrence_count", 0
                    ),
                    "problems": cluster.get("problems", []),
                }
            )
        return final_clusters

    def _stage_6_llm_merge_duplicates(
        self, clusters: list[dict], query: dict, common_problems: list[dict]
    ) -> dict[str, Any]:
        """
        Stage 6: LLM merges duplicate problems ONE CLUSTER AT A TIME.

        For each cluster:
        - If problems in cluster are the SAME → merge into ONE problem
        - If problems in cluster are DIFFERENT → keep them separate

        This ensures we don't lose problems due to LLM errors.
        """
        if not clusters:
            return {"problems": []}

        final_problems = []
        print(f"[Memory] STAGE 6.5: Processing {len(clusters)} clusters one at a time")

        for cluster_idx, cluster in enumerate(clusters, 1):
            cluster_problems = cluster.get("problems", [])
            if not cluster_problems:
                continue

            # If cluster has only 1 problem, no need to merge
            if len(cluster_problems) == 1:
                print(f"[Memory] STAGE 6.5: Cluster {cluster_idx} has 1 problem - keeping as-is")
                final_problems.append(cluster_problems[0])
                continue

            print(f"[Memory] STAGE 6.5: Cluster {cluster_idx} has {len(cluster_problems)} problems - asking LLM to merge/separate")

            # Prepare compact problem descriptions for this cluster
            compact_problems = []
            for prob in cluster_problems:
                compact_problems.append({
                    "problem": prob.get("problem", "")[:200],
                    "root_cause": prob.get("root_cause", "")[:200],
                    "files": prob.get("files", [])[:5],
                    "failure_type": prob.get("failure_type", ""),
                    "failure_signals": prob.get("failure_signals", [])[:3],
                })

            prompt = f"""Analyze these {len(cluster_problems)} problems from ONE cluster to decide if they should be merged or kept separate.

**Current CI Failure:**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Problems in This Cluster:**
```json
{json.dumps(compact_problems, indent=2)}
```

---

**Merging Criteria:**

Compare the problems and check if they have:
1. **Same problem description** (or variants describing the same issue)
2. **Same root cause** (why the problem happens)
3. **Same fix strategy** (how to solve it)

**Decision Rules:**

✓ **MERGE into ONE** if all 3 criteria match:
- Problem variants are the same (e.g., "mypy fails on DTypeLike" = "DTypeLike type error")
- Root cause is the same (e.g., "NumPy 2.0 removed DTypeLike")
- Fix strategy is the same (e.g., "Update type annotations")
- When merging: Combine ALL files, failure_signals from all problems into one

✗ **KEEP SEPARATE** if any criteria differ:
- Different problems (e.g., "mypy error" vs "pylint error")
- Different root causes (e.g., "missing import" vs "wrong type")
- Different fix strategies (e.g., "upgrade dependency" vs "fix code")
- Return each as a separate problem with its own files

**Example:**

Problem 1: "mypy fails on numpy DTypeLike in typing.py", files: ["src/typing.py"]
Problem 2: "DTypeLike type annotation error in mypy", files: ["src/utils.py", "src/models.py"]

Analysis:
- Problem: ✓ Same (both about DTypeLike type error)
- Root cause: ✓ Same (NumPy 2.0 removed DTypeLike)
- Fix strategy: ✓ Same (update type annotations)

→ **MERGE** into one problem with files: ["src/typing.py", "src/utils.py", "src/models.py"]

**Return JSON:**
```json
{{
  "problems": [
    {{
      "problem": "Clear description (merge variants if same issue)",
      "root_cause": "Why it happened",
      "failure_type": "type_checking|linting|test_failure|...",
      "problem_type": "ci_failure|dependent|consecutive|common",
      "verification_cmd": "command to verify fix",
      "files": ["all", "files", "from", "merged", "problems"],
      "failure_signals": ["all", "signals", "combined"],
      "repair_strategy": {{
        "summary": "High-level approach",
        "actions": ["specific action"],
        "pitfalls": ["avoid this"],
        "validation_cmd": "command"
      }}
    }}
  ]
}}
```

**IMPORTANT:**
- Return at least 1 problem
- If all problems are the SAME → return 1 merged problem
- If problems are DIFFERENT → return multiple separate problems
- When merging, combine ALL files and failure_signals

{STRICT_JSON_RULES}
"""
            response = invoke_llm_with_retry(
                llm=self.llm, prompt=prompt, parse_json=True
            )
            result = response if isinstance(response, dict) else {}
            cluster_result_problems = result.get("problems", [])

            # Fallback: If LLM returns empty, keep the first problem from cluster
            if not cluster_result_problems:
                print(f"[Memory] STAGE 6.5: Cluster {cluster_idx} - LLM returned 0 problems, keeping first as fallback")
                final_problems.append(cluster_problems[0])
            else:
                print(f"[Memory] STAGE 6.5: Cluster {cluster_idx} - LLM returned {len(cluster_result_problems)} problem(s)")
                final_problems.extend(cluster_result_problems)

        print(f"[Memory] STAGE 6.5: Final dedup result: {len(final_problems)} problems")
        return {"problems": final_problems}

    def _stage_5_cluster_problem_candidates(self, problems: list[dict]) -> list[dict]:
        """
        Stage 5: Deterministically cluster candidate problems before the final
        LLM merge decision. This keeps prompts compact and prevents large inputs
        from dropping evidence.
        """
        if not problems:
            return []

        clusters = self._cluster_problem_candidates_for_common(
            problems, default_scope="retrieval"
        )
        final_clusters = []
        for idx, cluster in enumerate(clusters, 1):
            final_clusters.append(
                {
                    "cluster_id": f"C{idx}",
                    "common_pattern": cluster.get("representative_problem", ""),
                    "failure_type": cluster.get("failure_type", ""),
                    "validation_tool": cluster.get("validation_tool", ""),
                    "distinct_issue_count": cluster.get("distinct_issue_count", 0),
                    "problem_occurrence_count": cluster.get(
                        "problem_occurrence_count", 0
                    ),
                    "coverage": cluster.get("coverage"),
                    "examples": cluster.get("examples", []),
                    "problems": cluster.get("problems", []),
                }
            )
        return final_clusters

    def _stage_6_llm_comprehensive_decision(
        self, clusters: list[dict], query: dict, common_problems: list[dict]
    ) -> dict[str, Any]:
        """
        Stage 6: LLM final decision.

        Each prompt receives compact clusters, not raw memory files. If many
        clusters exist they are processed in chunks and then combined.
        """
        if not clusters:
            return {"problems": []}

        final_problems = []
        # CRITICAL: Use size-aware chunking to handle variable-size issues
        for chunk in self._chunk_by_size(clusters, max_tokens=30000):
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
                llm=self.llm, prompt=prompt, parse_json=True
            )
            # Response is already parsed JSON (parse_json=True)
            result = response if isinstance(response, dict) else {}
            final_problems.extend(result.get("problems", []))

        return {"problems": final_problems}

    def _stage_7_reorder_by_dependencies(self, problems: list[dict]) -> list[dict]:
        """
        Stage 7: Order problems by CI verification order.

        Order:
        1. CI failure problem (main issue) - FIRST
        2. All other problems (dependent, consecutive, common) - ordered by CI verification sequence
           - Uses failure_type to determine CI pipeline order
           - E.g., linting → type_checking → tests → build → deploy

        This ensures problems are shown in the order they would fail in CI.
        """
        if not problems:
            return []

        # Separate CI failure from others
        ci_failure = []
        other_problems = []

        for p in problems:
            ptype = p.get("problem_type", "common")
            if ptype == "ci_failure":
                ci_failure.append(p)
            else:
                other_problems.append(p)

        # CI verification order (typical CI pipeline sequence)
        VERIFICATION_ORDER = {
            "linting": 1,
            "formatting": 2,
            "type_checking": 3,
            "unit_test": 4,
            "test_failure": 4,
            "integration_test": 5,
            "build": 6,
            "deployment": 7,
            "dependency": 0,  # Dependencies come first
            "import_error": 0,
            "unknown": 99,
        }

        def verification_rank(p: dict) -> tuple:
            failure_type = p.get("failure_type", "unknown")
            # Get rank from verification order
            rank = VERIFICATION_ORDER.get(failure_type, 99)

            # Secondary: dependency chain (enabled[])
            enabled = p.get("enabled", [])
            dependency_depth = max(enabled) + 1 if enabled else 0

            return (rank, dependency_depth)

        # Sort other problems by CI verification order
        other_problems = sorted(other_problems, key=verification_rank)

        # Final order: CI failure FIRST, then others by verification sequence
        ordered = ci_failure + other_problems

        return ordered

    def _stage_7_reorder_final_problems(
        self,
        problems: list[dict],
        query: dict,
        ci_failure_problem: dict | None = None,
        dependent_problems: list[dict] | None = None,
        consecutive_problems: list[dict] | None = None,
    ) -> list[dict]:
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
        def get_priority_group(problem: dict) -> int:
            # Defensive: handle None problem
            if problem is None or not isinstance(problem, dict):
                return 999
            # Use problem_type to determine group
            problem_type = problem.get("problem_type", "")
            if problem_type == "ci_failure":
                return 1
            elif problem_type == "dependent":
                return 2
            elif problem_type == "consecutive":
                return 3
            elif problem_type == "common":
                return 4
            else:
                # Fallback to old logic
                if problem.get("priority_group"):
                    return problem.get("priority_group", 99)
                source = (
                    problem.get("source", {})
                    if isinstance(problem.get("source"), dict)
                    else {}
                )
                relation = self._normalize_text(
                    problem.get("type") or problem.get("relation") or source.get("type")
                )
                if self._matches_current_failure(problem, query):
                    return 1
                elif "dependent" in relation or source.get("dependent"):
                    return 2
                elif "consecutive" in relation or source.get("consecutive"):
                    return 3
                else:
                    return 4

        validation_order = self._validation_sequence_order(query)

        def sort_key(problem: dict) -> tuple[int, int, int, int]:
            priority_group = get_priority_group(problem)
            validation_rank = self._problem_validation_rank(problem, validation_order)

            # Secondary priority
            priority = problem.get("priority", 999)
            priority_rank = priority if isinstance(priority, int) else 999

            original_index = problem.get("_original_index", 0)

            return (priority_group, validation_rank, priority_rank, original_index)

        for idx, problem in enumerate(problems):
            problem["_original_index"] = idx

        ordered = sorted(problems, key=sort_key)

        for idx, problem in enumerate(ordered, 1):
            problem["order"] = idx
            problem.pop("_original_index", None)

        return ordered

    def _compact_query(self, query: dict | None) -> dict:
        """
        Compact query for LLM prompts - include only key metadata.

        Full logs are NOT needed for cluster validation - just context to understand
        what we're looking for. The clusters themselves contain the repair strategies.
        """
        # Defensive: handle None query
        if query is None or not isinstance(query, dict):
            query = {}

        return {
            "repo": query.get("repo"),
            "workflow_name": query.get("workflow_name"),
            "workflow_path": query.get("workflow_path"),
            # Only include summaries, not full logs - use _shorten for consistency
            "error_context": [self._shorten(str(e), 200) for e in query.get("error_context", [])[:5]],
            "failure_signals": [self._shorten(str(s), 200) for s in query.get("failure_signals", [])[:5]],
            "error_types": query.get("error_types", [])[:10],
            "relevant_files": [str(f)[:100] for f in query.get("relevant_files", [])[:15]],
            "failed_job": [self._shorten(str(j), 300) for j in query.get("failed_job", [])[:3]],
        }

    def _compact_retrieved_item(
        self, item: dict | None, level: str, idx: int, include_dependencies: bool = False
    ) -> dict:
        # Defensive: handle None or non-dict items
        if item is None or not isinstance(item, dict):
            item = {}
        data = item.get("item", item)
        compact = {
            "id": f"{level}:{idx}",
            "score": round(float(item.get("score", 0.0)), 4)
            if isinstance(item, dict)
            else 0.0,
            "issue_id": data.get("issue_id") or data.get("source_issue_id"),
            "repo": data.get("repo") or data.get("source_repo"),
            "workflow": data.get("workflow_name") or data.get("workflow"),
        }

        if level == "L1":
            problems = []
            for problem in data.get("problems", [])[:8]:
                prob_data = {
                    "problem_id": problem.get("problem_id"),
                    "failure_type": problem.get("failure_type"),
                    "problem": self._shorten(problem.get("problem")),
                    "root_cause": self._shorten(problem.get("root_cause")),
                    "validation_cmd": problem.get("verification_cmd")
                    or problem.get("validation_cmd"),
                    "files": problem.get("affected_files", [])[:8] or problem.get("files", [])[:8],
                    "fix_strategy": self._shorten(problem.get("how_fixed") or problem.get("fix_strategy")),
                }

                # Include dependency/sequence information when requested
                if include_dependencies:
                    prob_data.update({
                        "problem_type": problem.get("problem_type"),  # "primary", "cascading"
                        "is_cascading": problem.get("is_cascading", False),
                        "repair_sequence_index": problem.get("repair_sequence_index"),  # Order of repair
                        "dependency_type": problem.get("dependency_type", ""),
                    })

                problems.append(prob_data)
            compact["problems"] = problems
        elif level == "L2":
            strategies = []
            for strategy in data.get("repair_strategies", [])[:8]:
                strategies.append(
                    {
                        "step": strategy.get("step"),
                        "failure_type": strategy.get("failure_type"),
                        "summary": self._shorten(strategy.get("summary")),
                        "causal_chain": self._shorten(
                            strategy.get("causal_chain"),
                            500 if include_dependencies else 220,
                        ),
                        "validation_cmd": strategy.get("validation_cmd"),
                        "signals": strategy.get("signals", [])[:5],
                        "key_actions": strategy.get("key_actions", [])[:5],
                        "pitfalls": strategy.get("pitfalls", [])[:5],
                    }
                )
            compact["repair_strategies"] = strategies
        else:
            compact.update(
                {
                    "pattern_id": data.get("pattern_id"),
                    "failure_type": data.get("failure_type"),
                    "failure_pattern": self._shorten(data.get("failure_pattern")),
                    "problem": self._shorten(data.get("problem")),
                    "when_to_apply": self._shorten(data.get("when_to_apply")),
                    "dependent_changes": data.get("dependent_changes", [])[:5]
                    if include_dependencies
                    else [],
                }
            )
        return compact

    def _resolve_selected_items(
        self, items: list[dict], selected_ids: list[str], level: str
    ) -> list[dict]:
        selected = {str(item) for item in selected_ids}
        resolved = []
        for idx, item in enumerate(items):
            if f"{level}:{idx}" in selected:
                resolved.append(item)
        return resolved

    def _extract_problem_candidates(
        self, item: dict, level: str, source_kind: str
    ) -> list[dict]:
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
        issue_id = str(item.get("issue_id") or item.get("source_issue_id") or "")
        repo = item.get("repo") or item.get("source_repo") or ""
        workflow = item.get("workflow_path") or item.get("workflow_name") or ""

        if level == "L1":
            # Get issue-level signals from CI context
            ci_context = item.get("benchmark_ci_context", {}) or {}
            issue_signals = ci_context.get("overall_failure_reasons", []) or []

            for problem in item.get("problems", []) or []:
                candidate = self._base_candidate(problem, item, level, source_kind)
                candidate.update(
                    {
                        "failure_signals": issue_signals,  # Use issue-level signals
                        "enabled": problem.get("enabled", []),
                    }
                )
                candidates.append(candidate)
        elif level == "L2":
            for strategy in item.get("repair_strategies", []) or []:
                candidate = self._base_candidate(strategy, item, level, source_kind)
                candidate.update(
                    {
                        "problem": strategy.get("summary")
                        or strategy.get("causal_chain")
                        or strategy.get("intent", ""),
                        "root_cause": strategy.get("causal_chain")
                        or strategy.get("reasoning")
                        or strategy.get("rationale", ""),
                        "fix_strategy": strategy.get("intent")
                        or strategy.get("rationale")
                        or strategy.get("summary", ""),
                        "failure_signals": strategy.get("signals", []),
                        "files": self._extract_files_from_actions(
                            strategy.get("key_actions", [])
                        ),
                    }
                )
                candidates.append(candidate)
        else:  # L3
            if item.get("no_forward_problems") or item.get("no_decomposed_problems"):
                return []
            candidate = self._base_candidate(item, item, level, source_kind)
            candidate.update(
                {
                    "problem": item.get("problem") or item.get("failure_pattern", ""),
                    "root_cause": item.get("reasoning", ""),
                    "fix_strategy": (
                        (item.get("universal_fix") or {}).get("approach", "")
                        if isinstance(item.get("universal_fix"), dict)
                        else ""
                    ),
                    "failure_signals": item.get("signals", []),
                }
            )
            candidates.append(candidate)

        for candidate in candidates:
            candidate.setdefault("issue_id", issue_id)
            candidate.setdefault("repo", repo)
            candidate.setdefault("workflow", workflow)
            candidate["validation_tool"] = self._validation_tool(
                candidate.get("validation_cmd")
            )
            candidate["file_area"] = self._file_area(candidate.get("files", []))
        return candidates

    def _base_candidate(
        self, problem: dict, item: dict, level: str, source_kind: str
    ) -> dict:
        """
        Build base candidate structure with repair_strategy from L1/L2/L3 data.

        Level-specific handling:
        - L1: Build from how_fixed + why_fix_works (descriptive, not step-by-step)
        - L2: Use key_actions (structured steps)
        - L3: Use universal_fix.steps (structured steps)
        """
        validation_cmd = (
            problem.get("verification_cmd")
            or problem.get("validation_cmd")
            or item.get("validation_cmd", "")
        )

        # Extract files
        files = problem.get("affected_files", []) or problem.get("files", [])
        if not isinstance(files, list):
            files = []

        # Build repair_strategy based on level
        repair_strategy = self._build_repair_strategy(problem, level, validation_cmd)

        return {
            "level": level,
            "source_kind": source_kind,
            "issue_id": str(item.get("issue_id") or item.get("source_issue_id") or ""),
            "repo": item.get("repo") or item.get("source_repo") or "",
            "workflow": item.get("workflow_path") or item.get("workflow_name") or "",
            "failure_type": problem.get("failure_type")
            or item.get("failure_type")
            or "",
            "problem": problem.get("problem")
            or problem.get("summary")
            or problem.get("failure_pattern")
            or "",
            "root_cause": problem.get("root_cause")
            or problem.get("causal_chain")
            or problem.get("reasoning")
            or "",
            "validation_cmd": validation_cmd,
            "files": files,
            "fix_strategy": problem.get("how_fixed")
            or problem.get("fix_strategy")
            or problem.get("intent")
            or "",
            "repair_strategy": repair_strategy,
        }

    def _build_repair_strategy(
        self, problem: dict, level: str, validation_cmd: str
    ) -> dict:
        """
        Build repair_strategy structure based on data level.

        L1: how_fixed + why_fix_works (narrative form)
        L2: key_actions (structured steps)
        L3: universal_fix.steps (structured steps)
        """
        if level == "L1":
            # L1: Build from narrative fields
            how_fixed = problem.get("how_fixed", "")
            why_fix_works = problem.get("why_fix_works", "")

            # Organize as structured actions
            actions = []
            if how_fixed:
                actions.append(how_fixed)
            if why_fix_works:
                actions.append(f"Why this works: {why_fix_works}")

            return {
                "summary": how_fixed or "Fix based on L1 data",
                "actions": actions
                if actions
                else ["Analyze problem and fix accordingly"],
                "validation_cmd": validation_cmd or "",
                "pitfalls": [],
            }

        elif level == "L2":
            # L2: Use structured key_actions
            intent = problem.get("intent") or problem.get("summary", "")
            key_actions = problem.get("key_actions", [])
            if not isinstance(key_actions, list):
                key_actions = [str(key_actions)] if key_actions else []

            return {
                "summary": intent or "Fix based on L2 repair strategy",
                "actions": key_actions
                if key_actions
                else [intent]
                if intent
                else ["Follow L2 strategy"],
                "validation_cmd": validation_cmd or "",
                "pitfalls": problem.get("pitfalls", []) or [],
            }

        else:  # L3
            # L3: Use universal_fix structure
            universal_fix = problem.get("universal_fix", {})
            approach = (
                universal_fix.get("approach", "")
                if isinstance(universal_fix, dict)
                else ""
            )
            steps = (
                universal_fix.get("steps", [])
                if isinstance(universal_fix, dict)
                else []
            )
            if not isinstance(steps, list):
                steps = [str(steps)] if steps else []

            return {
                "summary": approach
                or problem.get("problem", "")
                or "Universal fix pattern",
                "actions": steps
                if steps
                else [approach]
                if approach
                else ["Apply universal pattern"],
                "validation_cmd": validation_cmd or "",
                "pitfalls": [],
            }

    def _cluster_problem_candidates_for_common(
        self, candidates: list[dict], default_scope: str | None = None
    ) -> list[dict]:
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
        # Filter out None candidates
        candidates = [c for c in candidates if c is not None and isinstance(c, dict)]
        if not candidates:
            return []

        # Bucket only by scope and repo (not by failure_type or validation_tool)
        # Let semantic similarity do the clustering, not pre-bucketing
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for candidate in candidates:
            scope_key = default_scope or candidate.get("source_kind", "")
            bucket = (
                scope_key,
                self._normalize_text(candidate.get("repo"))
                if default_scope != "retrieval"
                else "",
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
                    sim = self._candidate_similarity(
                        candidate, cluster["representative"]
                    )
                    best_sim = max(best_sim, sim)
                    if sim >= 0.30:  # Lower threshold to group variations of same problem
                        matched = cluster
                        break

                # Debug: log similarity scores for first few candidates
                if len(clusters) <= 5 and best_sim > 0:
                    problem_short = candidate.get("problem", "")[:60]
                    print(
                        f"[Memory] Stage 4: Similarity={best_sim:.2f} for '{problem_short}...'"
                    )

                if matched is None:
                    clusters.append(
                        {
                            "representative": candidate,
                            "problems": [],
                            "examples_by_issue": defaultdict(
                                lambda: {
                                    "issue_id": "",
                                    "problems": [],  # problem descriptions from this issue
                                    "files": [],
                                }
                            ),
                        }
                    )
                    matched = clusters[-1]
                matched["problems"].append(candidate)
                issue_key = self._issue_key(candidate)
                issue_example = matched["examples_by_issue"][issue_key]
                issue_example["issue_id"] = candidate.get("issue_id", "")
                if candidate.get("problem"):
                    issue_example["problems"].append(
                        self._shorten(candidate.get("problem"), 180)
                    )
                for file_path in candidate.get("files", [])[:8]:
                    if file_path not in issue_example["files"]:
                        issue_example["files"].append(file_path)
            all_clusters.extend(clusters)

        return [
            self._summarize_cluster(cluster, default_scope) for cluster in all_clusters
        ]

    def _summarize_cluster(self, cluster: dict, default_scope: str | None) -> dict:
        """
        Summarize cluster with frequency = distinct issue_id count.

        Key change: examples show issue_ids (not problem_ids) because
        frequency is measured by "how many different CI failures had this problem".
        """
        representative = cluster["representative"]
        examples = list(cluster["examples_by_issue"].values())
        examples.sort(key=lambda item: str(item.get("issue_id", "")))
        scope = (
            "L1"
            if any(p.get("source_kind") == "common_l1" for p in cluster["problems"])
            else "L2"
        )
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
            "files": self._merge_files(
                [p.get("files", []) for p in cluster["problems"]]
            ),
            "file_area": representative.get("file_area", ""),
            "distinct_issue_count": len(
                examples
            ),  # ← Frequency: how many distinct issues
            "problem_occurrence_count": len(cluster["problems"]),
            "examples": examples[:8],  # ← Shows issue_ids with their problems
            "problems": cluster["problems"],
        }

    def _llm_validate_common_patterns(
        self, common_candidates: list[dict], query: dict
    ) -> list[dict]:
        """
        LLM validates each cluster to decide if it's a real recurring pattern.

        Input: Clusters with frequency counts (distinct_issue_count)
        Output: Validated common patterns with relevance to current failure

        Key: Frequency was calculated deterministically by counting distinct issue_ids.
        LLM decides if the pattern is real/relevant, NOT the frequency.

        OPTIMIZATION: Single-problem clusters are auto-accepted without LLM call.
        """
        if not common_candidates:
            return []

        # OPTIMIZATION: Filter out single-occurrence clusters (not "common")
        # Common patterns must appear in at least 2 different issues
        single_occurrence_clusters = []
        multi_occurrence_clusters = []

        for candidate in common_candidates:
            # Check how many distinct issues had this problem
            distinct_count = candidate.get("distinct_issue_count", 0)
            if distinct_count <= 1:
                # Only appeared in 1 issue - NOT common, skip it
                single_occurrence_clusters.append(candidate)
            else:
                # Appeared in 2+ issues - potential common pattern
                multi_occurrence_clusters.append(candidate)

        print(f"[Memory] Common pattern filtering: {len(single_occurrence_clusters)} single-occurrence (skipped - not common), {len(multi_occurrence_clusters)} multi-occurrence (validate as common)")

        # Skip single-occurrence clusters entirely (they're not "common")
        # Only validate multi-occurrence clusters with LLM
        # CRITICAL: Use size-aware chunking instead of fixed item count
        accepted = []

        # Smart chunking: fit as many clusters as possible within token limit
        chunks = self._chunk_by_size(multi_occurrence_clusters, max_tokens=30000)

        print(f"[Memory] Stage 5: Split {len(multi_occurrence_clusters)} clusters into {len(chunks)} size-aware chunks")

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_size = sum(self._estimate_tokens(c) for c in chunk)
            print(f"[Memory] Stage 5 DEBUG: Validating chunk {chunk_idx}/{len(chunks)} with {len(chunk)} clusters (~{chunk_size:,} tokens)")
            for i, cluster in enumerate(chunk, 1):
                print(f"[Memory] Stage 5 DEBUG:   Cluster {i}: {cluster.get('distinct_issue_count')} issues ({cluster.get('coverage', 0):.0%}), problem='{cluster.get('representative_problem', '')[:60]}...'")

            prompt = f"""Extract the most repetitive problem and repair from each cluster.

**Current CI Failure (for context only):**
```json
{json.dumps(self._compact_query(query), indent=2)}
```

**Common Pattern Clusters (already validated by ≥50% coverage):**
```json
{json.dumps(self._compact_common_patterns(chunk), indent=2)}
```

**Your Task:**
For EACH cluster, analyze the `problems` field (list of all problem instances in that cluster):

1. **Count problem descriptions**: Which exact description appears most often?
2. **Count root causes**: Which exact root_cause appears most often?
3. **Count repair strategies**: Which exact repair approach appears most often?
4. **Merge files**: Combine all files from all problems

**Select the MOST REPETITIVE (highest count) as the representative for that cluster.**

**Example - Cluster with 24 problem instances:**

Problem descriptions:
- "Missing type annotation for Optional[Any] variables" → 15 occurrences ✓ MOST COMMON
- "Type error in Optional handling" → 8 occurrences
- "mypy arg-type error" → 1 occurrence

Repair strategies:
- "Add None guard before accessing" → 12 occurrences ✓ MOST COMMON
- "Add type annotation" → 10 occurrences
- "Use isinstance check" → 2 occurrences

**Return for this cluster:**
- problem: "Missing type annotation for Optional[Any] variables" (appeared 15/24 times)
- repair_strategy summary: "Add None guard before accessing" (appeared 12/24 times)

**Output: Return ONE representative problem per cluster (the most repetitive within that cluster).**

Return JSON with the MOST REPETITIVE problem/repair from each cluster:
{{
  "common_problems": [
    {{
      "cluster_id": "short descriptive id",
      "common": true,
      "failure_type": "most common failure_type from cluster",
      "validation_tool": "most common validation_tool from cluster",
      "distinct_issue_count": <from input>,
      "problem_occurrence_count": <from input>,
      "coverage": <from input>,
      "problem": "THE MOST REPETITIVE problem description from cluster",
      "root_cause": "THE MOST REPETITIVE root_cause from cluster",
      "files": ["merged files from all problems in cluster"],
      "failure_signals": ["merged signals from all problems in cluster"],
      "repair_strategy": {{
        "summary": "THE MOST REPETITIVE repair summary from cluster",
        "actions": ["THE MOST REPETITIVE repair actions from cluster"],
        "validation_cmd": "THE MOST REPETITIVE validation_cmd from cluster",
        "pitfalls": ["THE MOST REPETITIVE pitfalls from cluster"]
      }},
      "examples": <from input>,
      "repetition_analysis": "Which problem/repair was most common in this cluster (e.g., 'Missing type annotation appeared 15/24 times, Add None guard appeared 12/24 times')"
    }}
  ]
}}

**IMPORTANT:**
- Return ALL input clusters (if 2 clusters → return 2 problems)
- Each output problem = the MOST REPETITIVE problem/repair from that cluster
- Each cluster already has ≥50% coverage (validated), so include all
- Do NOT reject clusters - just extract the most common problem/repair from each

{STRICT_JSON_RULES}
"""
            response = invoke_llm_with_retry(
                llm=self.llm, prompt=prompt, parse_json=True
            )
        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}

        # DEBUG: Log what LLM returned
        common_found = result.get("common_problems", [])
        print(f"[Memory] Stage 5 DEBUG: LLM returned {len(common_found)} common patterns")
        if not common_found:
            print(f"[Memory] Stage 5 DEBUG: Response type={type(response)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            if isinstance(result, dict) and result:
                print(f"[Memory] Stage 5 DEBUG: Response preview: {str(result)[:300]}")

        accepted.extend(common_found)

        # Sort by FREQUENCY (coverage, issue count) - NOT relevance to current failure
        # Most common patterns first
        accepted.sort(
            key=lambda item: (
                item.get("coverage", 0),  # Primary: coverage %
                item.get("distinct_issue_count", 0),  # Secondary: absolute count
            ),
            reverse=True,
        )
        return accepted

    def _common_patterns_to_problem_candidates(
        self, common_patterns: list[dict]
    ) -> list[dict]:
        """
        Convert validated common patterns to problem candidates for final clustering.

        LLM now returns complete repair_strategy structure, so we use it directly.
        """
        candidates = []
        for pattern in common_patterns:
            if not pattern.get("common", True):
                continue

            # Get repair_strategy from LLM (already a dict with summary, actions, etc.)
            # or build from legacy fields if LLM didn't provide it
            repair_strategy = pattern.get("repair_strategy")
            if not isinstance(repair_strategy, dict):
                # Fallback: build from legacy fields
                repair_strategy = {
                    "summary": pattern.get("representative_fix_strategy", ""),
                    "actions": [pattern.get("representative_fix_strategy", "")]
                    if pattern.get("representative_fix_strategy")
                    else [],
                    "validation_cmd": pattern.get("validation_cmd", ""),
                    "pitfalls": [],
                }

            candidates.append(
                {
                    "level": "COMMON",
                    "source_kind": "common_pattern",
                    "issue_id": "",  # Common patterns are aggregated
                    "repo": "",
                    "workflow": "",
                    "failure_type": pattern.get("failure_type", ""),
                    "problem_type": "common",
                    "problem": pattern.get("problem")
                    or pattern.get("representative_problem", ""),
                    "root_cause": pattern.get("root_cause")
                    or pattern.get("representative_root_cause", ""),
                    "verification_cmd": repair_strategy.get("validation_cmd", ""),
                    "files": pattern.get("files", []),
                    "file_area": self._file_area(pattern.get("files", [])),
                    "failure_signals": pattern.get("failure_signals", []),
                    "repair_strategy": repair_strategy,  # ← Use complete dict structure
                    "enabled": [],  # Common patterns have no dependencies
                }
            )
        return candidates

    def _compact_common_patterns(self, patterns: list[dict]) -> list[dict]:
        """
        Compact common patterns for LLM prompt.

        Include sample repair_strategy from the clustered problems so LLM can extract it.
        """
        compact = []
        for pattern in patterns:
            # Get a sample repair_strategy from the actual clustered problems
            sample_repair = None
            for prob in pattern.get("problems", [])[:3]:
                if prob.get("repair_strategy") and isinstance(
                    prob["repair_strategy"], dict
                ):
                    sample_repair = prob["repair_strategy"]
                    break

            # If no structured repair_strategy, build from legacy fields
            if not sample_repair:
                sample_repair = {
                    "summary": pattern.get("representative_fix_strategy", ""),
                    "actions": [pattern.get("representative_fix_strategy", "")]
                    if pattern.get("representative_fix_strategy")
                    else [],
                    "validation_cmd": pattern.get("validation_cmd", ""),
                    "pitfalls": [],
                }

            compact.append(
                {
                    "cluster_id": pattern.get("cluster_id"),
                    "scope": pattern.get("scope"),
                    "failure_type": pattern.get("failure_type"),
                    "validation_tool": pattern.get("validation_tool"),
                    "distinct_issue_count": pattern.get("distinct_issue_count"),
                    "problem_occurrence_count": pattern.get("problem_occurrence_count"),
                    "coverage": pattern.get("coverage"),
                    "representative_problem": self._shorten(
                        pattern.get("representative_problem"), 200
                    ),
                    "representative_root_cause": self._shorten(
                        pattern.get("representative_root_cause"), 200
                    ),
                    "files": pattern.get("files", [])[:8],
                    "sample_repair_strategy": sample_repair,  # ← Include actual repair strategy
                    "examples": pattern.get("examples", [])[:5],
                }
            )
        return compact

    def _compact_clusters_for_prompt(self, clusters: list[dict]) -> list[dict]:
        compact = []
        for cluster in clusters:
            compact.append(
                {
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
                            "fix_strategy": self._shorten(
                                problem.get("fix_strategy"), 220
                            ),
                            "repair_strategy": self._shorten(
                                problem.get("repair_strategy"), 220
                            ),
                            "common_pattern": problem.get("common_pattern", False),
                            "frequency": problem.get("frequency"),
                            "coverage": problem.get("coverage"),
                        }
                        for problem in cluster.get("problems", [])[:8]
                    ],
                }
            )
        return compact

    def _candidate_similarity(self, left: dict, right: dict) -> float:
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
        # Defensive: handle None candidates
        if left is None or not isinstance(left, dict):
            left = {}
        if right is None or not isinstance(right, dict):
            right = {}

        # Core semantic fields (most important)
        problem_sim = self._text_similarity(
            left.get("problem", ""), right.get("problem", "")
        )
        root_sim = self._text_similarity(
            left.get("root_cause", ""), right.get("root_cause", "")
        )

        # Failure type (semantic similarity, not exact match)
        failure_sim = self._text_similarity(
            left.get("failure_type", ""), right.get("failure_type", "")
        )

        # Fix strategy (summary + actions)
        # Defensive: repair_strategy might be None, so check before accessing
        left_repair = left.get("repair_strategy") or {}
        right_repair = right.get("repair_strategy") or {}
        left_fix_summary = left.get("fix_strategy", "") or left_repair.get("summary", "") if isinstance(left_repair, dict) else ""
        right_fix_summary = right.get("fix_strategy", "") or right_repair.get("summary", "") if isinstance(right_repair, dict) else ""

        # Also compare repair actions if available
        left_actions = left.get("repair_actions", []) or (left_repair.get("actions", []) if isinstance(left_repair, dict) else [])
        right_actions = right.get("repair_actions", []) or (right_repair.get("actions", []) if isinstance(right_repair, dict) else [])

        # Convert actions to strings if they're dicts
        left_actions_str = []
        for action in (left_actions[:3] if left_actions else []):
            if isinstance(action, dict):
                left_actions_str.append(action.get("description", "") or action.get("action", "") or str(action))
            else:
                left_actions_str.append(str(action))

        right_actions_str = []
        for action in (right_actions[:3] if right_actions else []):
            if isinstance(action, dict):
                right_actions_str.append(action.get("description", "") or action.get("action", "") or str(action))
            else:
                right_actions_str.append(str(action))

        left_fix_text = left_fix_summary + " " + " ".join(left_actions_str)  # First 3 actions
        right_fix_text = right_fix_summary + " " + " ".join(right_actions_str)

        fix_sim = self._text_similarity(left_fix_text, right_fix_text)

        # File similarity (smart: exact match, directory match, file type match)
        left_files = left.get("files", [])
        right_files = right.get("files", [])
        if left_files and right_files:
            file_sim = self._file_similarity(left_files, right_files)
        else:
            file_sim = 0.0

        # Weighted combination focusing on actual data fields
        # Files get higher weight - same file + same failure type often = same problem
        # Even if described differently
        return (
            0.30 * problem_sim  # Main description
            + 0.20 * root_sim  # Root cause
            + 0.20 * file_sim  # File overlap (INCREASED - key indicator!)
            + 0.15 * failure_sim  # Failure type
            + 0.10 * fix_sim  # Fix strategy
            + 0.05 * self._file_type_bonus(left_files, right_files)  # Same test file bonus
        )

    def _file_similarity(self, left_files: list[str], right_files: list[str]) -> float:
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
            parts = filepath.rsplit("/", 1)
            directory = parts[0] if len(parts) > 1 else ""
            filename = parts[-1]
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            return directory, filename, ext

        # Calculate max similarity for each left file with any right file
        similarities = []
        for left in left_files:
            left_dir, _left_name, left_ext = file_features(left)
            max_sim = 0.0

            for right in right_files:
                right_dir, _right_name, right_ext = file_features(right)

                if left == right:
                    # Exact match
                    sim = 1.0
                elif left_dir == right_dir and left_ext == right_ext:
                    # Same directory + same extension (e.g., same module, different files)
                    sim = 0.7
                elif left_dir == right_dir:
                    # Same directory (different extensions)
                    sim = 0.5
                elif left_ext == right_ext and left_ext in (
                    "py",
                    "ts",
                    "tsx",
                    "js",
                    "yml",
                    "yaml",
                    "toml",
                    "json",
                    "md",
                    "rst",
                ):
                    # Same extension (same file type, different dirs)
                    sim = 0.3
                else:
                    sim = 0.0

                max_sim = max(max_sim, sim)

            similarities.append(max_sim)

        # Average similarity across all files
        return sum(similarities) / len(similarities)

    def _file_type_bonus(self, left_files: list[str], right_files: list[str]) -> float:
        """
        Bonus for same file type patterns (especially test files).

        If both have test files or both have same file pattern, boost similarity.
        This helps cluster variations of "test X fails" problems.
        """
        if not left_files or not right_files:
            return 0.0

        left_has_test = any("test" in f.lower() for f in left_files)
        right_has_test = any("test" in f.lower() for f in right_files)

        # Both are test file problems
        if left_has_test and right_has_test:
            # Check if same test file
            left_test_files = {f for f in left_files if "test" in f.lower()}
            right_test_files = {f for f in right_files if "test" in f.lower()}

            # Exact same test file
            if left_test_files & right_test_files:
                return 1.0

            # Same test directory or similar test name
            left_test_names = {f.split("/")[-1] for f in left_test_files}
            right_test_names = {f.split("/")[-1] for f in right_test_files}
            if left_test_names & right_test_names:
                return 0.8

            # Just both test files (generic bonus)
            return 0.3

        return 0.0

    def _common_score(self, cluster: dict, query: dict) -> float:
        coverage = float(cluster.get("coverage", 0.0))
        relevance = self._text_similarity(
            " ".join(query.get("error_context", []) + query.get("failure_signals", [])),
            " ".join(
                [
                    str(cluster.get("representative_problem", "")),
                    str(cluster.get("representative_root_cause", "")),
                    str(cluster.get("representative_fix_strategy", "")),
                ]
            ),
        )
        specificity = 1.0 if cluster.get("scope") == "L1" else 0.75
        return round((0.55 * coverage) + (0.30 * relevance) + (0.15 * specificity), 4)

    def _validation_sequence_order(self, query: dict) -> dict[str, int]:
        order: dict[str, int] = {}
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

    def _problem_validation_rank(
        self, problem: dict, validation_order: dict[str, int]
    ) -> int:
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

    def _matches_current_failure(self, problem: dict, query: dict) -> bool:
        source = (
            problem.get("source", {}) if isinstance(problem.get("source"), dict) else {}
        )
        if source.get("current_failure"):
            return True

        query_text = " ".join(
            [str(value) for value in query.get("error_context", [])]
            + [str(value) for value in query.get("failure_signals", [])]
            + [json.dumps(value) for value in query.get("error_types", [])]
        )
        problem_text = " ".join(
            [
                str(problem.get("problem", "")),
                str(problem.get("root_cause", "")),
                " ".join(
                    str(value) for value in problem.get("failure_signals", []) or []
                ),
            ]
        )
        text_match = self._text_similarity(query_text, problem_text) >= 0.35

        query_tools = {
            self._validation_tool(command)
            for command in query.get("failed_cmd", []) or []
        }
        query_tools.discard("")
        repair_strat = problem.get("repair_strategy")
        problem_cmd = (
            problem.get("validation_cmd")
            or (repair_strat.get("validation_cmd") if isinstance(repair_strat, dict) else None)
            if isinstance(problem.get("repair_strategy"), dict)
            else problem.get("validation_cmd")
        )
        problem_tool = self._validation_tool(problem_cmd)
        tool_match = not query_tools or not problem_tool or problem_tool in query_tools

        query_files = self._query_file_set(query)
        problem_files = set(problem.get("files", []) or [])
        file_match = (
            not query_files or not problem_files or bool(query_files & problem_files)
        )

        return text_match and tool_match and file_match

    def _query_file_set(self, query: dict) -> set:
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
        token_score = (
            len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if left_tokens | right_tokens
            else 0.0
        )
        return max(sequence_score, token_score)

    def _normalize_text(self, value: Any) -> str:
        text = str(value or "").lower()
        text = re.sub(r"`[^`]+`", " ", text)
        text = re.sub(r"\b[\w./-]+\.(py|rst|toml|yml|yaml|md|json|js|ts)\b", " ", text)
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"[^a-z0-9_+\[\] -]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _validation_tool(self, validation_cmd: Any) -> str:
        """
        Extract validation tool name from command dynamically.
        No hardcoded tool list - just parse the command naturally.
        """
        command = self._normalize_text(validation_cmd)
        if not command:
            return ""

        # Extract first significant word (skip ./dev/, ./scripts/, etc)
        parts = command.replace("./", "").replace("_", " ").split()
        if not parts:
            return ""

        # First word is usually the tool name
        tool = parts[0]

        # Common patterns: "python -m mypy" -> "mypy", "npm run test" -> "test"
        if tool in ("python", "python3", "node", "npm", "npx", "bash", "sh"):
            # Skip interpreter, get actual tool
            if len(parts) > 2 and parts[1] in ("-m", "run", "-c", "exec"):
                return parts[2] if len(parts) > 2 else tool
            elif len(parts) > 1:
                return parts[1]

        return tool

    def _file_area(self, files: list[str]) -> str:
        if not files:
            return ""
        first = str(files[0])
        parts = [part for part in first.split("/") if part]
        if len(parts) >= 3:
            return "/".join(parts[:3])
        return "/".join(parts)

    def _merge_files(self, file_lists: list[list[str]]) -> list[str]:
        merged = []
        for files in file_lists:
            for file_path in files or []:
                if file_path and file_path not in merged:
                    merged.append(file_path)
        return merged[:20]

    def _extract_files_from_actions(self, actions: list[Any]) -> list[str]:
        files = []
        for action in actions or []:
            text = json.dumps(action) if isinstance(action, dict) else str(action)
            for match in re.findall(
                r"[\w./-]+\.(?:py|rst|toml|yml|yaml|md|json|js|ts)", text
            ):
                if match not in files:
                    files.append(match)
        return files[:20]

    def _same_repo(self, left: Any, right: Any) -> bool:
        left_norm = self._normalize_repo(left)
        right_norm = self._normalize_repo(right)
        return bool(
            left_norm
            and right_norm
            and (
                left_norm == right_norm
                or left_norm.endswith("/" + right_norm)
                or right_norm.endswith("/" + left_norm)
            )
        )

    def _normalize_repo(self, repo: Any) -> str:
        repo_text = str(repo or "").strip().lower()
        repo_text = repo_text.replace("ci-repair/", "")
        return repo_text

    def _same_workflow(self, left: Any, right: Any) -> bool:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not right_text:
            return True
        return left_text == right_text or Path(left_text).name == Path(right_text).name

    def _unique_issue_keys(self, items: list[dict]) -> set:
        return {self._issue_key(item) for item in items}

    def _issue_key(self, item: dict) -> tuple[str, str, str]:
        return (
            str(item.get("repo") or item.get("source_repo") or ""),
            str(item.get("workflow_path") or item.get("workflow_name") or ""),
            str(item.get("issue_id") or item.get("source_issue_id") or ""),
        )

    def _chunk_list(self, items: list[dict], max_items: int) -> list[list[dict]]:
        """Simple chunking by item count (legacy - use _chunk_by_size for better results)"""
        return [
            items[index : index + max_items]
            for index in range(0, len(items), max_items)
        ]

    def _estimate_tokens(self, obj: any) -> int:
        """Estimate token count for an object (rough: 1 token ≈ 4 chars)"""
        if obj is None:
            return 0
        import json
        try:
            text = json.dumps(obj) if isinstance(obj, (dict, list)) else str(obj)
            return len(text) // 4  # Rough estimate: 4 chars per token
        except:
            return len(str(obj)) // 4

    def _chunk_by_size(self, items: list[dict], max_tokens: int = 30000) -> list[list[dict]]:
        """
        Smart chunking: group items until token limit reached.

        Args:
            items: List of items to chunk
            max_tokens: Max tokens per chunk (default 30K, leaving room for prompt overhead)

        Returns:
            List of chunks, each under the token limit
        """
        if not items:
            return []

        chunks = []
        current_chunk = []
        current_size = 0

        for item in items:
            item_size = self._estimate_tokens(item)

            # If single item exceeds limit, put it in its own chunk (will be handled separately)
            if item_size > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                chunks.append([item])  # Single large item in its own chunk
                continue

            # If adding this item would exceed limit, start new chunk
            if current_size + item_size > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [item]
                current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size

        # Add remaining items
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _shorten(self, value: Any, limit: int = 240) -> Any:
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        text = str(value or "")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _stage_5_merge_decision(self, clusters: list[dict]) -> list[dict]:
        """
        Stage 5: LLM decides if cluster members should be merged.

        For each cluster, LLM decides:
        - Are these the SAME problem? -> Merge into one
        - Are these DIFFERENT problems? -> Keep separate

        Returns: List of merged/separate problems
        """
        merged_problems = []

        for cluster in clusters:
            cluster_problems = cluster.get("problems", [])

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

            response = invoke_llm_with_retry(llm=self.llm, prompt=prompt)

            # Response is already parsed JSON (parse_json=True)
            decision = response if isinstance(response, dict) else {}

            if decision.get("should_merge"):
                merged_problems.append(decision.get("merged_problem"))
            else:
                merged_problems.extend(
                    decision.get("separate_problems", cluster_problems)
                )

        return merged_problems

    def _stage_6_reorder_problems(
        self, problems: list[dict], main_problem: dict, dependencies: list[dict]
    ) -> list[dict]:
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
            from_id = dep.get("from")
            to_id = dep.get("to")
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

    def _get_problem_id(self, problem: dict) -> str:
        """Get unique ID for a problem."""
        return problem.get("problem_id") or problem.get("problem", "")[:50]

    def _estimate_prompt_tokens(
        self,
        l1_matches: list[dict],
        l2_matches: list[dict],
        l3_matches: list[dict],
        query: dict,
        common_problems: list[dict],
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
            model_name = getattr(self.llm, "model_name", None)
            if model_name:
                config = get_model_config(model_name)
                return config["input_context_window"]
        except Exception:  # noqa: BLE001
            return 100_000

        # Conservative default
        return 100_000

    def _comprehensive_analysis_single_call(
        self,
        l1_matches: list[dict],
        l2_matches: list[dict],
        l3_matches: list[dict],
        query: dict,
    ) -> dict[str, Any]:
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

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt)

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        return result

    def _comprehensive_analysis_chunked(
        self,
        l1_matches: list[dict],
        l2_matches: list[dict],
        l3_matches: list[dict],
        query: dict,
        common_problems: list[dict],
    ) -> dict[str, Any]:
        """
        Chunked LLM analysis - analyze each level separately, then combine.

        Used when prompt size >= 50% of model limit.
        Safer but might miss cross-level connections.
        """
        # Analyze each level separately
        l1_problems = self._analyze_level_chunk(
            l1_matches, query, common_problems, "L1"
        )
        l2_problems = self._analyze_level_chunk(
            l2_matches, query, common_problems, "L2"
        )
        l3_problems = self._analyze_level_chunk(
            l3_matches, query, common_problems, "L3"
        )

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

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt, parse_json=True)

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        return result

    def _analyze_level_chunk(
        self,
        matches: list[dict],
        query: dict,
        common_problems: list[dict],
        level_name: str,
    ) -> list[dict]:
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

        response = invoke_llm_with_retry(llm=self.llm, prompt=prompt)

        # Response is already parsed JSON (parse_json=True)
        result = response if isinstance(response, dict) else {}
        return result.get("problems", [])

    def _merge_decision_for_clusters(self, clusters: list[dict]) -> list[dict]:
        """
        LLM decides if cluster members should be merged.

        For each cluster, LLM decides: SAME problem or DIFFERENT problems?
        """
        merged_problems = []

        for cluster in clusters:
            cluster_problems = cluster.get("problems", [])

            if len(cluster_problems) == 1:
                # Single problem - no merge needed
                merged_problems.append(cluster_problems[0])
                continue

            # Ask LLM: should these be merged?
            prompt = f"""
Cluster of similar problems - decide if SAME (merge) or DIFFERENT (keep separate):

{json.dumps(cluster_problems, indent=2)}

Similarity reason: {cluster.get("similarity_reason", "Similar based on content")}

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

            response = invoke_llm_with_retry(llm=self.llm, prompt=prompt)

            # Response is already parsed JSON (parse_json=True)
            decision = response if isinstance(response, dict) else {}

            # Just extend with the problems list (whether merged or separate)
            merged_problems.extend(decision.get("problems", cluster_problems))

        return merged_problems

    def _reorder_problems(
        self, problems: list[dict], main_problem: dict, dependencies: list[dict]
    ) -> list[dict]:
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

    def _load_json(self, filename: str) -> list[dict]:
        """Load JSON memory file."""
        file_path = self.memory_dir / filename
        if not file_path.exists():
            return []

        with open(file_path, "r") as f:
            return json.load(f)

    def _get_encoder(self):
        """Load the embedding model only when retrieval actually needs it."""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.embedding_model)
            except (ImportError, ModuleNotFoundError) as e:
                print(f"[Memory] WARNING: Could not load sentence-transformers: {e}")
                print(f"[Memory] This usually means PyTorch/transformers version mismatch")
                print(f"[Memory] Falling back to baseline mode (no memory retrieval)")
                # Set encoder to a dummy value to prevent retrying
                self.encoder = "unavailable"
                # Switch to baseline mode
                self.baseline_mode = True
                self.l1_memory = []
                self.l2_memory = []
                self.l3_memory = []
                return None

        # If encoder loading failed, return None
        if self.encoder == "unavailable":
            return None

        return self.encoder

    def _compute_embeddings(self, items: list[dict], level: str) -> np.ndarray:
        """Compute embeddings for memory items."""
        if not items:
            return np.array([])

        # Get encoder - will return None if loading failed
        encoder = self._get_encoder()
        if encoder is None:
            print(f"[Memory] Encoder unavailable, returning empty embeddings for {level}")
            return np.array([])

        texts = []
        for item in items:
            if level == "l1":
                text = f"{item.get('repo', '')} {item.get('workflow', '')} "
                text += " ".join(
                    [p.get("problem", "") for p in item.get("problems", [])]
                )
            elif level == "l2":
                text = f"{item.get('repo', '')} "
                text += " ".join(item.get("failure_identify", []))
            else:  # l3
                text = f"{item.get('failure_pattern', '')} {item.get('problem', '')} {item.get('reasoning', '')}"

            texts.append(text)

        return encoder.encode(texts)

    def _build_query(self, query: dict, level: str) -> str:
        """
        Build query string for specific level.

        L1: repo + workflow + error details
        L2: repo + error types
        L3: semantic only (no repo/file names)
        """
        if level == "l1":
            # L1: Same repo + workflow specific
            parts = [
                query.get("repo", ""),
                query.get("workflow_name", ""),
                " ".join(query.get("error_context", [])),
                " ".join(query.get("failure_signals", [])),
            ]
            return " ".join(parts)

        elif level == "l2":
            # L2: Same repo, any workflow
            parts = [
                query.get("repo", ""),
            ]
            # Add error type categories
            for et in query.get("error_types", []):
                parts.append(et.get("category", ""))
                parts.append(et.get("subcategory", ""))

            parts.extend(query.get("error_context", []))
            return " ".join(parts)

        else:  # l3
            # L3: Semantic only (abstract repo/file details)
            parts = []

            # Error categories
            for et in query.get("error_types", []):
                parts.append(et.get("category", ""))
                parts.append(et.get("subcategory", ""))

            # Failed tools
            for rf in query.get("relevant_files", []):
                if rf.get("failed_tool"):
                    parts.append(rf.get("failed_tool"))

            # Error context (abstracted)
            for error in query.get("error_context", []):
                # Replace specific file names with <FILE>
                error = re.sub(r"\b[\w/]+\.py\b", "<FILE>", error)
                error = re.sub(r"\b[\w/]+\.js\b", "<FILE>", error)
                # Replace common package names with <PKG>
                error = re.sub(
                    r"\b(numpy|pandas|pytest|requests|click|flask)\b", "<PKG>", error
                )
                parts.append(error)

            return " ".join(parts)

    def _retrieve_topk(
        self,
        query: str,
        items: list[dict],
        embeddings: np.ndarray,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[dict]:
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
                if key == "repo":
                    if not self._same_repo(
                        item.get("repo") or item.get("source_repo"), value
                    ):
                        passes = False
                        break
                    continue
                if key == "workflow":
                    item_workflow = item.get("workflow_path") or item.get("workflow_name")
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
            {"item": filtered_items[idx], "score": float(similarities[idx])}
            for idx in top_indices
        ]
