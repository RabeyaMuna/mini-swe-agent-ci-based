"""
Automatic fault localization from CI failure details.

Extracts:
- File paths from error messages
- Line numbers from stack traces
- Error types
- Failed tests
"""

import re
from typing import Any


def extract_faulty_files(problem_description: str) -> list[dict[str, Any]]:
    """
    Extract file paths and line numbers from CI failure.

    Returns list of:
    {
        'file_path': 'path/to/file.py',
        'line_number': 42,
        'error': 'TypeError: ...',
        'context': 'surrounding error text'
    }
    """
    locations = []

    # Pattern 1: Python stack trace format
    # File "path/to/file.py", line 42, in function_name
    pattern1 = r'File "([^"]+)", line (\d+)'
    for match in re.finditer(pattern1, problem_description):
        file_path = match.group(1)
        line_num = int(match.group(2))

        # Get surrounding context
        start = max(0, match.start() - 100)
        end = min(len(problem_description), match.end() + 200)
        context = problem_description[start:end].strip()

        locations.append(
            {
                'file_path': file_path,
                'line_number': line_num,
                'error': _extract_error_message(problem_description, match.end()),
                'context': context,
            }
        )

    # Pattern 2: pytest/unittest error format
    # path/to/test_file.py::test_function FAILED
    pattern2 = r'([a-zA-Z0-9_./\-]+\.py)::([\w_]+)\s+FAILED'
    for match in re.finditer(pattern2, problem_description):
        file_path = match.group(1)
        test_name = match.group(2)

        start = max(0, match.start() - 50)
        end = min(len(problem_description), match.end() + 150)
        context = problem_description[start:end].strip()

        locations.append(
            {
                'file_path': file_path,
                'line_number': None,
                'error': f'Test {test_name} failed',
                'context': context,
            }
        )

    # Pattern 3: mypy/flake8/type checker errors
    # path/to/file.py:42: error: Missing return type
    pattern3 = r'([a-zA-Z0-9_./\-]+\.py):(\d+):\s*(error|warning):\s*([^\n]+)'
    for match in re.finditer(pattern3, problem_description):
        file_path = match.group(1)
        line_num = int(match.group(2))
        error_msg = match.group(4).strip()

        locations.append(
            {
                'file_path': file_path,
                'line_number': line_num,
                'error': error_msg,
                'context': match.group(0),
            }
        )

    # Pattern 4: Generic file mentions in errors
    # Mentioned in: "error in app.py"
    if not locations:
        pattern4 = r'\b([a-zA-Z0-9_./\-]+\.py)\b'
        seen_files = set()
        for match in re.finditer(pattern4, problem_description):
            file_path = match.group(1)
            if file_path not in seen_files:
                seen_files.add(file_path)
                locations.append(
                    {
                        'file_path': file_path,
                        'line_number': None,
                        'error': 'File mentioned in error',
                        'context': '',
                    }
                )

    # Deduplicate by file_path (keep first occurrence)
    seen = set()
    unique_locations = []
    for loc in locations:
        fp = loc['file_path']
        if fp not in seen:
            seen.add(fp)
            unique_locations.append(loc)

    return unique_locations


def _extract_error_message(text: str, start_pos: int) -> str:
    """Extract error message after a stack trace line."""
    # Look for common error patterns after the file/line reference
    remaining = text[start_pos : start_pos + 300]

    # Find first line that looks like an error
    for line in remaining.split('\n'):
        line = line.strip()
        if any(err in line for err in ['Error:', 'Exception:', 'Failed:', 'FAILED']):
            return line[:200]

    return ''


def build_fault_localization_summary(problem_description: str) -> str:
    """
    Build a summary of fault localization results.

    Returns formatted string showing files to fix.
    """
    locations = extract_faulty_files(problem_description)

    if not locations:
        return 'No specific files identified in error. Manual investigation needed.'

    lines = ['Fault Localization Results:', '']

    for i, loc in enumerate(locations[:5], 1):  # Top 5 locations
        file_path = loc['file_path']
        line_num = loc.get('line_number')
        error = loc.get('error', '')

        if line_num:
            lines.append(f'{i}. {file_path}:{line_num}')
        else:
            lines.append(f'{i}. {file_path}')

        if error:
            lines.append(f'   Error: {error[:100]}')

        lines.append('')

    return '\n'.join(lines)
