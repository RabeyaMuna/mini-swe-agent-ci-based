#!/usr/bin/env python3
"""Add debug logging to track down where synthesis is failing"""

import sys

# Read the file
file_path = 'src/minisweagent/run/benchmarks/utils/ci_memory_system.py'
with open(file_path) as f:
    lines = f.readlines()

# Find line 596 (parsed = _parse_json(raw))
found = False
for i, line in enumerate(lines):
    if 'parsed = _parse_json(raw)' in line and 'def _run_single_llm_synthesis' in ''.join(lines[max(0, i-10):i]):
        # Add debug code after this line
        indent = '    '
        debug_code = f'''
{indent}# TEMPORARY DEBUG - Added by add_debug.py
{indent}import sys
{indent}print(f"\\n{{'='*80}}", file=sys.stderr)
{indent}print(f"[DEBUG] LLM SYNTHESIS AT LINE {i+1}", file=sys.stderr)
{indent}print(f"{{'='*80}}", file=sys.stderr)
{indent}print(f"Raw LLM output length: {{len(raw) if raw else 0}}", file=sys.stderr)
{indent}if raw:
{indent}    print(f"First 500 chars:\\n{{raw[:500]}}", file=sys.stderr)
{indent}    print(f"Last 200 chars:\\n{{raw[-200:]}}", file=sys.stderr)
{indent}print(f"Parsed result: {{parsed is not None}}", file=sys.stderr)
{indent}if parsed:
{indent}    print(f"Parsed keys: {{list(parsed.keys())}}", file=sys.stderr)
{indent}    print(f"Has primary_files: {{'primary_files' in parsed}}", file=sys.stderr)
{indent}    print(f"Has additional_files: {{'additional_files' in parsed or ('full_scope' in parsed and 'additional_files' in parsed.get('full_scope', {{}}))}}", file=sys.stderr)
{indent}else:
{indent}    print(f"[ERROR] PARSING FAILED! Returning deterministic fallback.", file=sys.stderr)
{indent}print(f"{{'='*80}}\\n", file=sys.stderr)
'''
        lines.insert(i+1, debug_code)
        found = True
        print(f"OK Added debug code after line {i+1}")
        break

if not found:
    print("ERROR Could not find the target line")
    sys.exit(1)

# Write back
with open(file_path, 'w') as f:
    f.writelines(lines)

print(f"OK Debug code added to {file_path}")
print("\nNow run your test and check stderr for debug output:")
print("  python -m minisweagent.run.benchmarks.cibench ... 2>&1 | grep -A20 'LLM SYNTHESIS'")
