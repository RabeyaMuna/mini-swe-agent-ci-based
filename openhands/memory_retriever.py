"""
Memory retrieval system for OpenHands
Retrieves L1/L2/L3 memory and formats for prompt injection
"""

import json
from pathlib import Path
from typing import Any, Optional


class MemoryRetriever:
    """Retrieve L1/L2/L3 memory for CI-Bench issues"""

    def __init__(self, memory_root: str, layers: Optional[list[str]] = None):
        """
        Args:
            memory_root: Path to data/trs/ directory
            layers: List of layers to use, e.g. ["L1", "L2", "L3"]
        """
        self.memory_root = Path(memory_root)
        self.layers = layers or ['L1', 'L2', 'L3']

        # Load memory files
        self.memory_data = {}
        self._load_memory_files()

    def _load_memory_files(self):
        """Load all memory JSON files"""
        memory_files = {
            'L1': self.memory_root / 'failure_memory.json',
            'L2': self.memory_root / 'repo_memory.json',
            'L3': self.memory_root / 'cross_memory.json',
        }

        for layer, path in memory_files.items():
            if layer in self.layers:
                if path.exists():
                    with open(path, 'r') as f:
                        self.memory_data[layer] = json.load(f)
                    print(f' Loaded {layer} memory from {path}')
                else:
                    print(f'  {path} not found, {layer} memory disabled')
                    self.memory_data[layer] = {}

    def retrieve(
        self, instance_id: str, problem_statement: str, repo: str, top_k: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve relevant memory for an issue

        Args:
            instance_id: Issue instance ID
            problem_statement: Problem description
            repo: Repository name
            top_k: Number of memory items to retrieve per layer

        Returns:
            Dict with keys L1, L2, L3 containing relevant memory
        """
        memory_context = {}

        # L1: Similar CI failures from same repo
        if 'L1' in self.layers and self.memory_data.get('L1'):
            memory_context['L1'] = self._retrieve_l1(repo, problem_statement, top_k)

        # L2: Repository patterns
        if 'L2' in self.layers and self.memory_data.get('L2'):
            memory_context['L2'] = self._retrieve_l2(repo, top_k)

        # L3: Cross-repo principles
        if 'L3' in self.layers and self.memory_data.get('L3'):
            memory_context['L3'] = self._retrieve_l3(problem_statement, top_k)

        return memory_context

    def _retrieve_l1(self, repo: str, problem: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve L1: Similar failures from same repo"""
        # TODO: Implement similarity search
        # For now, return first top_k from same repo
        l1_data = self.memory_data.get('L1', [])

        # Handle both list and dict formats
        if isinstance(l1_data, dict):
            l1_data = list(l1_data.values())

        # Filter by repo
        repo_failures = []
        for item in l1_data:
            if isinstance(item, dict) and item.get('repo') == repo:
                repo_failures.append(item)

        return repo_failures[:top_k]

    def _retrieve_l2(self, repo: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve L2: Repository patterns"""
        l2_data = self.memory_data.get('L2', [])

        # Handle both list and dict formats
        if isinstance(l2_data, dict):
            l2_data = list(l2_data.values())

        # Filter by repo
        repo_patterns = []
        for item in l2_data:
            if isinstance(item, dict) and item.get('repo') == repo:
                repo_patterns.append(item)

        return repo_patterns[:top_k]

    def _retrieve_l3(self, problem: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve L3: Universal principles"""
        l3_data = self.memory_data.get('L3', [])

        # Handle both list and dict formats
        if isinstance(l3_data, dict):
            l3_data = list(l3_data.values())

        # TODO: Implement similarity search
        # For now, return first top_k
        principles = []
        for item in l3_data:
            if isinstance(item, dict):
                principles.append(item)

        return principles[:top_k]

    def format_for_prompt(self, memory_context: dict[str, list[dict[str, Any]]]) -> str:
        """
        Format memory context as Repair Plan for injection into OpenHands prompt

        Args:
            memory_context: Retrieved memory with L1/L2/L3

        Returns:
            Formatted string as "## Repair Plan (if available)"
        """
        if not memory_context:
            return ''

        repair_plan_items = []

        # L1: Extract fixes from similar failures
        if memory_context.get('L1'):
            for i, mem in enumerate(memory_context['L1'], 1):
                # Actual field names from memory files
                problem = mem.get('problem', mem.get('issue_type', ''))
                fix = mem.get('fixes', mem.get('solution', ''))

                if problem and fix:
                    repair_plan_items.append(
                        {
                            'type': 'L1',
                            'source': f'Similar issue: {problem}',
                            'action': fix,
                        }
                    )

        # L2: Extract actionable patterns
        if memory_context.get('L2'):
            for mem in memory_context['L2']:
                pattern = mem.get('pattern', mem.get('description', ''))
                if pattern:
                    repair_plan_items.append(
                        {
                            'type': 'L2',
                            'source': 'Repository pattern',
                            'action': pattern,
                        }
                    )

        # L3: Extract principles as guidance
        if memory_context.get('L3'):
            for mem in memory_context['L3']:
                principle = mem.get('principle', mem.get('description', ''))
                if principle:
                    repair_plan_items.append(
                        {'type': 'L3', 'source': 'Best practice', 'action': principle}
                    )

        # Format repair plan
        if repair_plan_items:
            repair_plan = '## Repair Plan (if available)\n\n'
            repair_plan += (
                'Based on previous experiences, consider these approaches:\n\n'
            )

            # Group by type for better organization
            l1_items = [item for item in repair_plan_items if item['type'] == 'L1']
            l2_items = [item for item in repair_plan_items if item['type'] == 'L2']
            l3_items = [item for item in repair_plan_items if item['type'] == 'L3']

            step = 1

            # L1: Similar fixes
            if l1_items:
                repair_plan += '**From Similar Past Failures:**\n'
                for item in l1_items[:3]:  # Top 3
                    repair_plan += f'{step}. {item["action"]}\n'
                    repair_plan += f'   ({item["source"]})\n'
                    step += 1
                repair_plan += '\n'

            # L2: Repository patterns
            if l2_items:
                repair_plan += '**Repository-Specific Patterns:**\n'
                for item in l2_items[:2]:  # Top 2
                    repair_plan += f'{step}. {item["action"]}\n'
                    step += 1
                repair_plan += '\n'

            # L3: General guidance
            if l3_items:
                repair_plan += '**General Debugging Strategies:**\n'
                for item in l3_items[:2]:  # Top 2
                    repair_plan += f'{step}. {item["action"]}\n'
                    step += 1

            return repair_plan
        else:
            return ''


if __name__ == '__main__':
    # Test memory retriever
    retriever = MemoryRetriever(memory_root='../data/trs', layers=['L1', 'L2', 'L3'])

    # Test retrieval
    memory_context = retriever.retrieve(
        instance_id='test__repo__sha',
        problem_statement='CI test failing',
        repo='owner/repo',
        top_k=3,
    )

    # Format for prompt
    prompt_section = retriever.format_for_prompt(memory_context)
    print('\nFormatted memory for prompt:')
    print(prompt_section)
