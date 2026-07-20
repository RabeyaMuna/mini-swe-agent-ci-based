"""
Data loader for CI-Bench evaluation
Handles caching of CI logs and workflow data
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASET = 'ci-benchmark-user/ci-repair-bench'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'


def _load_shared_script(module_name: str) -> Any:
    """Load a reusable helper from the project-level scripts directory."""
    module_path = SHARED_SCRIPTS_ROOT / f'{module_name}.py'
    if not module_path.exists():
        raise FileNotFoundError(
            f'Missing shared analyzer script: {module_path}. '
            'Copy the CI analyzer scripts into the project-level scripts/ directory.'
        )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load shared analyzer script: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CIBenchDataLoader:
    """Load and cache CI-Bench data"""

    def __init__(self, data_root: str = '../data'):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.log_details_path = self.data_root / 'log_details.json'
        self.workflow_cache_path = self.data_root / 'workflow_validation_cache.json'

    @staticmethod
    def _instance_id(issue: dict[str, Any]) -> str:
        """Return a stable issue id across supported dataset schemas."""
        return str(issue.get('instance_id') or issue['id'])

    @staticmethod
    def _repo(issue: dict[str, Any]) -> str:
        """Return owner/repo across supported dataset schemas."""
        if issue.get('repo'):
            return str(issue['repo'])
        return f'{issue["repo_owner"]}/{issue["repo_name"]}'

    @staticmethod
    def _normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
        """Normalize benchmark rows from local eval files or Hugging Face."""
        normalized = dict(issue)
        normalized['instance_id'] = CIBenchDataLoader._instance_id(normalized)
        normalized['repo'] = CIBenchDataLoader._repo(normalized)

        if 'workflow_path' not in normalized and normalized.get('workflow_filename'):
            workflow_filename = str(normalized['workflow_filename'])
            if workflow_filename.startswith('.github/'):
                normalized['workflow_path'] = workflow_filename
            else:
                normalized['workflow_path'] = f'.github/workflows/{workflow_filename}'

        return normalized

    @staticmethod
    def _find_cache_record(
        issue: dict[str, Any],
        cache: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find a cache record by instance id, benchmark id, or sha_fail."""
        instance_id = CIBenchDataLoader._instance_id(issue)
        sha_fail = issue['sha_fail']
        benchmark_id = str(issue.get('id', '')).strip()

        if isinstance(cache, dict):
            return cache.get(instance_id) or cache.get(sha_fail) or {}

        if isinstance(cache, list):
            for item in cache:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get('id', '')).strip()
                if item.get('sha_fail') == sha_fail or (
                    benchmark_id and item_id == benchmark_id
                ):
                    return item

        return {}

    @staticmethod
    def _format_processed_ci_failure(log_detail: dict[str, Any]) -> str:
        """Format processed CI log analysis without including raw logs."""
        if not log_detail:
            return ''

        lines = []

        error_context = log_detail.get('error_context')
        if isinstance(error_context, list) and error_context:
            lines.append('Failure context:')
            for item in error_context[:3]:
                lines.append(f'- {item}')

        error_types = log_detail.get('error_types')
        if isinstance(error_types, list) and error_types:
            lines.append('\nError types:')
            for item in error_types[:5]:
                if not isinstance(item, dict):
                    continue
                category = item.get('category', '')
                subcategory = item.get('subcategory', '')
                evidence = item.get('evidence', '')
                label = ' / '.join(part for part in (category, subcategory) if part)
                if label:
                    lines.append(f'- {label}')
                if evidence:
                    lines.append(f'  evidence: {evidence}')

        relevant_files = log_detail.get('relevant_files')
        if isinstance(relevant_files, list) and relevant_files:
            lines.append('\nRelevant files from CI failure analysis:')
            for item in relevant_files[:10]:
                if not isinstance(item, dict):
                    continue
                file_path = item.get('file') or item.get('path')
                line_number = item.get('line_number')
                issue_type = item.get('issue_type', '')
                failed_tool = item.get('failed_tool', '')
                reason = item.get('reason', '')
                location = f'{file_path}:{line_number}' if line_number else file_path
                if location:
                    lines.append(f'- {location} ({issue_type or failed_tool})'.rstrip())
                if reason:
                    lines.append(f'  reason: {reason}')

        failed_job = log_detail.get('failed_job')
        if isinstance(failed_job, list) and failed_job:
            lines.append('\nFailed CI jobs/steps:')
            for item in failed_job[:5]:
                if not isinstance(item, dict):
                    continue
                job = item.get('job', '')
                step = item.get('step', '')
                command = item.get('command', '')
                parts = [part for part in (job, step, command) if part]
                if parts:
                    lines.append(f'- {" | ".join(parts)}')

        return '\n'.join(lines)

    @staticmethod
    def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load a JSON array or JSONL file."""
        raw_text = path.read_text(encoding='utf-8').strip()
        if not raw_text:
            return []

        if raw_text.startswith('['):
            data = json.loads(raw_text)
            return [item for item in data if isinstance(item, dict)]

        records = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, list):
                records.extend(row for row in item if isinstance(row, dict))
        return records

    @staticmethod
    def _load_hf_rows(
        dataset_name: str = DEFAULT_HF_DATASET,
        split: str = 'train',
    ) -> list[dict[str, Any]]:
        """Load CI-Bench rows from Hugging Face."""
        from datasets import load_dataset  # type: ignore

        split_candidates = [split, 'test', 'train', 'validation']
        last_error: Exception | None = None
        for split_name in dict.fromkeys(split_candidates):
            try:
                dataset = load_dataset(dataset_name, split=split_name)
                return [dict(row) for row in dataset]
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f'Could not load Hugging Face dataset {dataset_name}: {last_error}'
        )

    @staticmethod
    def _index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Index records by id / instance_id / sha_fail."""
        index = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ('id', 'instance_id', 'sha_fail'):
                value = record.get(key)
                if value is not None and str(value):
                    index[str(value)] = record
        return index

    def load_eval_issues(
        self,
        eval_issues_path: str,
        hf_dataset: str = DEFAULT_HF_DATASET,
        split: str = 'train',
    ) -> list[dict[str, Any]]:
        """Load full benchmark rows from Hugging Face filtered by eval issues.

        ``eval_issues_path`` should contain the subset to run, usually
        ``eval_issues.json``. Rows are matched against the Hugging Face dataset
        by ``id`` / ``instance_id`` / ``sha_fail``. If Hugging Face is not
        available, the local rows are used as a fallback.
        """
        eval_path = Path(eval_issues_path)
        eval_issues = self._load_json_or_jsonl(eval_path)
        print(f' Loaded {len(eval_issues)} eval issue ids from {eval_issues_path}')

        try:
            hf_rows = self._load_hf_rows(hf_dataset, split)
            hf_index = self._index_records(hf_rows)
            print(f' Loaded {len(hf_rows)} benchmark rows from {hf_dataset}')
        except Exception as exc:
            print(f'  Could not load Hugging Face dataset: {exc}')
            hf_index = {}

        issues = []
        for eval_issue in eval_issues:
            match = None
            for key in ('id', 'instance_id', 'sha_fail'):
                value = eval_issue.get(key)
                if value is not None:
                    match = hf_index.get(str(value))
                    if match:
                        break
            merged = {**match, **eval_issue} if match else eval_issue
            issues.append(self._normalize_issue(merged))

        print(f' Prepared {len(issues)} filtered benchmark issues')
        return issues

    def ensure_data_files(
        self,
        eval_issues: list[dict[str, Any]],
        analyzer_llm: Any = None,
        analyzer_model: str = '',
    ) -> tuple:
        """
        Ensure log_details.json and workflow_cache.json exist
        Load if exists, generate if not
        """
        # Load or generate log_details
        if self.log_details_path.exists():
            print(f' Loading cached CI logs from {self.log_details_path}')
            with open(self.log_details_path, 'r') as f:
                log_details = json.load(f)
        else:
            print(f'  {self.log_details_path} not found, starting empty cache')
            log_details = {}

        log_details, logs_changed = self._ensure_ci_failure_records(
            eval_issues,
            log_details,
            analyzer_llm=analyzer_llm,
            analyzer_model=analyzer_model,
        )
        if logs_changed:
            with open(self.log_details_path, 'w') as f:
                json.dump(log_details, f, indent=2)
            print(f' Updated CI failure cache at {self.log_details_path}')

        # Load or generate workflow_cache
        if self.workflow_cache_path.exists():
            print(f' Loading cached workflow data from {self.workflow_cache_path}')
            with open(self.workflow_cache_path, 'r') as f:
                workflow_cache = json.load(f)
        else:
            print(f'  {self.workflow_cache_path} not found, starting empty cache')
            workflow_cache = {}

        workflow_cache, workflow_changed = self._ensure_workflow_records(
            eval_issues,
            workflow_cache,
            analyzer_llm=analyzer_llm,
        )
        if workflow_changed:
            with open(self.workflow_cache_path, 'w') as f:
                json.dump(workflow_cache, f, indent=2)
            print(f' Updated workflow validation cache at {self.workflow_cache_path}')

        return log_details, workflow_cache

    def _ensure_ci_failure_records(
        self,
        eval_issues: list[dict[str, Any]],
        log_details: dict[str, Any] | list[dict[str, Any]],
        analyzer_llm: Any = None,
        analyzer_model: str = '',
    ) -> tuple[dict[str, Any] | list[dict[str, Any]], bool]:
        """Generate missing CI failure entries with the shared CI log analyzer."""
        changed = False
        if isinstance(log_details, list):
            records = log_details
        else:
            records = list(log_details.values())

        for issue in eval_issues:
            if self._find_cache_record(issue, log_details):
                continue
            records.append(
                self._run_ci_log_analyzer(
                    issue,
                    analyzer_llm=analyzer_llm,
                    analyzer_model=analyzer_model,
                )
            )
            changed = True

        if isinstance(log_details, list):
            return records, changed

        return {
            str(
                record.get('id') or record.get('instance_id') or record.get('sha_fail')
            ): record
            for record in records
            if isinstance(record, dict)
        }, changed

    def _ensure_workflow_records(
        self,
        eval_issues: list[dict[str, Any]],
        workflow_cache: dict[str, Any] | list[dict[str, Any]],
        analyzer_llm: Any = None,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]], bool]:
        """Generate missing workflow entries with the shared workflow analyzer."""
        changed = False
        if isinstance(workflow_cache, list):
            records = workflow_cache
        else:
            records = list(workflow_cache.values())

        for issue in eval_issues:
            if self._find_cache_record(issue, workflow_cache):
                continue
            records.append(
                self._run_workflow_analyzer(issue, analyzer_llm=analyzer_llm)
            )
            changed = True

        if isinstance(workflow_cache, list):
            return records, changed

        return {
            str(
                record.get('id') or record.get('instance_id') or record.get('sha_fail')
            ): record
            for record in records
            if isinstance(record, dict)
        }, changed

    def _run_ci_log_analyzer(
        self,
        issue: dict[str, Any],
        analyzer_llm: Any = None,
        analyzer_model: str = '',
    ) -> dict[str, Any]:
        """Generate processed CI failure details through scripts/ci_log_analyzer.py."""
        try:
            module = _load_shared_script('ci_log_analyzer')
            analyzer = module.CILogAnalyzer(
                logs=issue.get('logs') or '',
                sha_fail=issue['sha_fail'],
                workflow=issue.get('workflow') or '',
                workflow_path=issue.get('workflow_path') or '',
                llm=analyzer_llm,
                model_name=analyzer_model,
                task_id=self._instance_id(issue),
            )
            result = analyzer.run()
        except Exception as exc:
            raise RuntimeError(
                'Missing CI failure cache entry and shared CI log analyzer could '
                f'not generate it for issue {self._instance_id(issue)}. '
                'Provide cached log_details.json or pass a configured analyzer LLM.'
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                f'CI log analyzer returned non-dict result for {self._instance_id(issue)}'
            )
        result.setdefault('id', self._instance_id(issue))
        result.setdefault('sha_fail', issue['sha_fail'])
        return result

    def _run_workflow_analyzer(
        self,
        issue: dict[str, Any],
        analyzer_llm: Any = None,
    ) -> dict[str, Any]:
        """Generate workflow validation through scripts/ci_workflow_aware_retrieval.py."""
        workflow = issue.get('workflow') or ''
        workflow_content = (
            workflow if isinstance(workflow, str) else json.dumps(workflow)
        )
        try:
            module = _load_shared_script('ci_workflow_aware_retrieval')
            result = module.analyze_workflow_from_benchmark(
                workflow_content=workflow_content,
                workflow_path=issue.get('workflow_path') or '',
                repo_path=None,
                llm=analyzer_llm,
                issue_id=self._instance_id(issue),
                sha_fail=issue['sha_fail'],
            )
        except Exception as exc:
            raise RuntimeError(
                'Missing workflow validation cache entry and shared workflow '
                f'analyzer could not generate it for issue {self._instance_id(issue)}. '
                'Provide cached workflow_validation_cache.json or pass a configured '
                'analyzer LLM.'
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                f'Workflow analyzer returned non-dict result for {self._instance_id(issue)}'
            )
        result.setdefault('id', self._instance_id(issue))
        result.setdefault('sha_fail', issue['sha_fail'])
        return result

    @staticmethod
    def _find_workflow_validation(
        issue: dict[str, Any],
        workflow_cache: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find workflow validation data for an issue.

        The cache may be stored either as a dict keyed by instance id or as a
        list of records keyed by benchmark id / sha_fail.
        """
        return CIBenchDataLoader._find_cache_record(issue, workflow_cache)

    def get_issue_data(
        self,
        issue: dict[str, Any],
        log_details: dict[str, Any] | list[dict[str, Any]],
        workflow_cache: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Combine all data for a single issue"""
        instance_id = self._instance_id(issue)
        log_detail = self._find_cache_record(issue, log_details)
        workflow_validation = self._find_workflow_validation(issue, workflow_cache)
        ci_failure_summary = self._format_processed_ci_failure(log_detail)

        return {
            'instance_id': instance_id,
            'id': issue.get('id'),
            'repo': self._repo(issue),
            'sha_fail': issue['sha_fail'],
            'base_sha': issue.get('base_sha', 'main'),
            'problem_statement': issue.get('problem_statement')
            or issue.get('problem')
            or ci_failure_summary
            or '',
            'ci_failure': log_detail,
            'ci_failure_summary': ci_failure_summary,
            'ci_logs': '',
            'error_summary': log_detail.get('error_summary', ''),
            'workflow': workflow_validation,
            'workflow_path': workflow_validation.get(
                'workflow_path', issue.get('workflow_path', '')
            ),
            'validation_command': workflow_validation.get(
                'validation_command', issue.get('validation_command', 'pytest')
            ),
        }


if __name__ == '__main__':
    # Test data loader
    loader = CIBenchDataLoader(data_root='../data')

    # Load issues
    issues = loader.load_eval_issues('../data/trs/eval_set.jsonl')

    # Ensure data files exist
    log_details, workflow_cache = loader.ensure_data_files(issues)

    # Get data for first issue
    if issues:
        issue_data = loader.get_issue_data(issues[0], log_details, workflow_cache)
        print('\nSample issue data:')
        print(json.dumps(issue_data, indent=2))
