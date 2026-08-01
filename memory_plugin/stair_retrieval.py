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
from pathlib import Path
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
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

        # Load embedder
        self.encoder = SentenceTransformer(embedding_model)

        # Load only enabled levels
        if 'l1' in self.enabled_levels:
            self.l1_memory = self._load_json('failure_memory.json')
            self.l1_embeddings = self._compute_embeddings(self.l1_memory, 'l1')
        else:
            self.l1_memory = []
            self.l1_embeddings = np.array([])

        if 'l2' in self.enabled_levels:
            self.l2_memory = self._load_json('repo_memory.json')
            self.l2_embeddings = self._compute_embeddings(self.l2_memory, 'l2')
        else:
            self.l2_memory = []
            self.l2_embeddings = np.array([])

        if 'l3' in self.enabled_levels:
            self.l3_memory = self._load_json('cross_memory.json')
            self.l3_embeddings = self._compute_embeddings(self.l3_memory, 'l3')
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
            "l1" → {'l1'}
            "l1+l2" → {'l1', 'l2'}
            "l1+l2+l3" → {'l1', 'l2', 'l3'}
            ['l1', 'l2'] → {'l1', 'l2'}
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


    def retrieve(self, ci_failure: Dict, top_k: int = 5, generate_plans: bool = False) -> Dict[str, Any]:
        """
        Main retrieval pipeline.

        Args:
            ci_failure: Current CI failure dict with:
                - repo: str
                - workflow: str
                - problem_statement: str
                - error_signals: List[str]
                - config_signals: List[str] (optional)
                - failure_type: str (optional)
            top_k: Number of items to retrieve per level
            generate_plans: If True, generate repair plans (Stage 6)

        Returns:
            Dict with:
                - problems: List of structured problems
                - repair_plans: List of repair plans (if generate_plans=True)
                - metadata: Pipeline statistics
        """

        # Baseline mode: return empty results
        if self.baseline_mode:
            return {
                'problems': [],
                'metadata': {
                    'mode': 'baseline',
                    'enabled_levels': [],
                    'retrieved': {'l1': 0, 'l2': 0, 'l3': 0},
                    'common_detected': 0,
                    'filtered': 0,
                    'clusters': 0,
                    'final': 0
                },
                'common_analysis': {'common_problems': [], 'config_problems': []},
                'consecutive_sequences': [],
                'dependencies': []
            }

        # Stage 1: Cosine similarity retrieval (only from enabled levels)
        retrieved = self._stage_1_retrieval(ci_failure, top_k)

        if not self.llm:
            raise ValueError("LLM client required for stages 2-5")

        # Stage 2: LLM detects common problems
        common_analysis = self._stage_2_common_detection(
            retrieved['l1'],
            retrieved['l2'],
            retrieved['l3']
        )

        # Stage 3: LLM filters and analyzes dependencies
        filtered = self._stage_3_filtering(
            retrieved['l1'],
            retrieved['l2'],
            retrieved['l3'],
            ci_failure,
            common_analysis['common_problems']
        )

        # Stage 4: LLM clusters similar problems
        clusters = self._stage_4_clustering(filtered['problems'])

        # Stage 5: LLM generates final structured problems
        final_problems = self._stage_5_final_generation(
            clusters,
            filtered['problems']
        )

        result = {
            'problems': final_problems,
            'metadata': {
                'mode': 'memory',
                'enabled_levels': sorted(list(self.enabled_levels)),
                'retrieved': {
                    'l1': len(retrieved['l1']),
                    'l2': len(retrieved['l2']),
                    'l3': len(retrieved['l3'])
                },
                'common_detected': len(common_analysis['common_problems']),
                'filtered': len(filtered['problems']),
                'clusters': len(clusters),
                'final': len(final_problems)
            },
            'common_analysis': common_analysis,
            'consecutive_sequences': filtered.get('consecutive_sequences', []),
            'dependencies': filtered.get('dependencies', [])
        }

        # Stage 6: Optional repair plan generation
        if generate_plans and final_problems:
            repair_plans = self._stage_6_repair_plans(final_problems)
            result['repair_plans'] = repair_plans

        return result


    def _stage_1_retrieval(self, ci_failure: Dict, top_k: int) -> Dict[str, List]:
        """
        Stage 1: Cosine similarity retrieval from enabled levels only.

        If l1 disabled: returns empty l1 list
        If l2 disabled: returns empty l2 list
        If l3 disabled: returns empty l3 list
        """

        # L1: Only retrieve if enabled
        if 'l1' in self.enabled_levels:
            l1_query = self._build_query(ci_failure, level='l1')
            l1_results = self._retrieve_topk(
                l1_query,
                self.l1_memory,
                self.l1_embeddings,
                top_k,
                filters={'repo': ci_failure.get('repo'), 'workflow': ci_failure.get('workflow')}
            )
        else:
            l1_results = []

        # L2: Only retrieve if enabled
        if 'l2' in self.enabled_levels:
            l2_query = self._build_query(ci_failure, level='l2')
            l2_results = self._retrieve_topk(
                l2_query,
                self.l2_memory,
                self.l2_embeddings,
                top_k,
                filters={'repo': ci_failure.get('repo')}
            )
        else:
            l2_results = []

        # L3: Only retrieve if enabled
        if 'l3' in self.enabled_levels:
            l3_query = self._build_query(ci_failure, level='l3')
            l3_results = self._retrieve_topk(
                l3_query,
                self.l3_memory,
                self.l3_embeddings,
                top_k,
                filters={}
            )
        else:
            l3_results = []

        return {
            'l1': l1_results,
            'l2': l2_results,
            'l3': l3_results
        }


    def _stage_2_common_detection(
        self,
        l1_items: List[Dict],
        l2_items: List[Dict],
        l3_items: List[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 2: LLM detects common problems across retrieved data.
        """

        prompt = build_common_detection_prompt(l1_items, l2_items, l3_items)
        prompt += f"\n\n{STRICT_JSON_RULES}"

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            max_retries=3,
            timeout=60
        )

        result = _load_json_flexible(response)

        return {
            'common_problems': result.get('common_problems', []),
            'config_problems': result.get('config_problems', [])
        }


    def _stage_3_filtering(
        self,
        l1_items: List[Dict],
        l2_items: List[Dict],
        l3_items: List[Dict],
        ci_failure: Dict,
        common_problems: List[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 3: LLM filters relevant problems and analyzes dependencies.
        """

        prompt = build_filtering_prompt(
            l1_items,
            l2_items,
            l3_items,
            ci_failure,
            common_problems
        )
        prompt += f"\n\n{STRICT_JSON_RULES}"

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            max_retries=3,
            timeout=90
        )

        result = _load_json_flexible(response)

        return {
            'problems': result.get('problems', []),
            'consecutive_sequences': result.get('consecutive_sequences', []),
            'dependencies': result.get('dependencies', [])
        }


    def _stage_4_clustering(self, problems: List[Dict]) -> List[Dict]:
        """
        Stage 4: LLM clusters similar problems.
        """

        if len(problems) <= 1:
            return [{'cluster_id': 'cluster_0', 'problem_ids': [problems[0]['problem_id']], 'should_merge': False}] if problems else []

        prompt = build_clustering_prompt(problems)
        prompt += f"\n\n{STRICT_JSON_RULES}"

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            max_retries=3,
            timeout=60
        )

        result = _load_json_flexible(response)

        return result.get('clusters', [])


    def _stage_5_final_generation(
        self,
        clusters: List[Dict],
        problems: List[Dict]
    ) -> List[Dict]:
        """
        Stage 5: LLM generates final structured problem list.
        """

        prompt = build_final_generation_prompt(clusters, problems)
        prompt += f"\n\n{STRICT_JSON_RULES}"

        response = invoke_llm_with_retry(
            llm=self.llm,
            prompt=prompt,
            max_retries=3,
            timeout=90
        )

        result = _load_json_flexible(response)

        return result.get('final_problems', [])


    def _stage_6_repair_plans(self, problems: List[Dict]) -> List[Dict]:
        """
        Stage 6: LLM generates repair plans for each problem.
        """

        repair_plans = []

        for problem in problems:
            prompt = build_repair_plan_prompt(problem, STRICT_JSON_RULES)

            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                max_retries=3,
                timeout=60
            )

            plan = _load_json_flexible(response)
            repair_plans.append(plan)

        return repair_plans


    # Helper methods

    def _load_json(self, filename: str) -> List[Dict]:
        """Load JSON memory file."""
        file_path = self.memory_dir / filename
        if not file_path.exists():
            return []

        with open(file_path, 'r') as f:
            return json.load(f)


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

        return self.encoder.encode(texts)


    def _build_query(self, ci_failure: Dict, level: str) -> str:
        """Build query for specific level."""
        if level == 'l1':
            return f"{ci_failure.get('repo', '')} {ci_failure.get('workflow', '')} {ci_failure.get('problem_statement', '')} {' '.join(ci_failure.get('error_signals', []))}"
        elif level == 'l2':
            return f"{ci_failure.get('repo', '')} {ci_failure.get('failure_type', '')} {ci_failure.get('problem_statement', '')}"
        else:  # l3
            # Abstract repo-specific details
            problem = ci_failure.get('problem_statement', '')
            problem = re.sub(r'\b[\w/]+\.py\b', '<FILE>', problem)
            problem = re.sub(r'\b(requests|click|numpy|pandas|pytest)\b', '<PKG>', problem)
            return f"{ci_failure.get('failure_type', '')} {problem}"


    def _retrieve_topk(
        self,
        query: str,
        items: List[Dict],
        embeddings: np.ndarray,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """Retrieve top-k items using cosine similarity."""
        if len(items) == 0 or len(embeddings) == 0:
            return []

        # Apply filters
        filtered_items = []
        filtered_embeddings = []

        for i, item in enumerate(items):
            passes = True
            for key, value in filters.items():
                if value and item.get(key) != value:
                    passes = False
                    break
            if passes:
                filtered_items.append(item)
                filtered_embeddings.append(embeddings[i])

        if not filtered_items:
            return []

        # Compute similarity
        query_emb = self.encoder.encode([query])[0]
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
