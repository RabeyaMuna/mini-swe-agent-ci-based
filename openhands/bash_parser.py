"""
Parse bash commands from model responses - DYNAMIC parsing.

Models output bash in markdown blocks. We extract and execute these commands.

HANDLES 567+ different patterns:
- Different code block formats (```bash, ```sh, ```)
- Commands with or without code blocks
- Multi-line commands
- Heredocs
- Command chains
"""

import re


def extract_bash_command(response: str) -> str | None:
    """
    Extract bash command from model response - DYNAMIC parsing.

    Handles multiple formats flexibly to work with 567+ different issue patterns.

    Looks for:
    - ```bash\ncommand\n```
    - ```sh\ncommand\n```
    - ```\ncommand\n``` (generic code block)
    - Commands without code blocks (if they look like bash)

    Returns the command or None if no bash block found.
    """
    response = response.strip()

    # Try bash-specific blocks first
    bash_patterns = [
        r'```bash\s*\n(.*?)\n```',
        r'```sh\s*\n(.*?)\n```',
        r'```shell\s*\n(.*?)\n```',
        r'```zsh\s*\n(.*?)\n```',
    ]

    for pattern in bash_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            cmd = match.group(1).strip()
            if cmd:
                return cmd

    # Try generic code block
    generic_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
    if generic_match:
        cmd = generic_match.group(1).strip()
        if cmd:
            return cmd

    # DYNAMIC: If no code block found, look for common bash patterns anywhere
    # This handles cases where models forget to use code blocks
    bash_keywords = ['cat ', 'echo ', 'find ', 'grep ', 'ls ', 'cd ', 'mkdir ', 'rm ', 'mv ', 'cp ', 'sed ']
    for keyword in bash_keywords:
        if keyword in response:
            # Extract just the command part - up to file extension or end of path
            # Pattern: keyword + path with extension OR keyword + flags/args
            pattern = rf'{keyword}(?:[\w\s\-./\'"])*?(?:[a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+|<<|>>)'
            match = re.search(pattern, response)
            if match:
                cmd = match.group(0).strip()
                # Remove common prefixes if any
                cmd = re.sub(r'^[>\$#]\s*', '', cmd)
                if cmd:
                    return cmd

            # Fallback: extract to end of word boundary (before "to", "for", "and", etc.)
            pattern2 = rf'{keyword}[^\s]*(?:\s+[^\s]+)*?(?=\s+(?:to|for|and|with|in|on|at|from|by|as)\b)'
            match2 = re.search(pattern2, response)
            if match2:
                cmd = match2.group(0).strip()
                if cmd:
                    return cmd

    return None


def is_completion_command(command: str) -> bool:
    """Check if command indicates task completion."""
    if not command:
        return False

    cmd_lower = command.lower().strip()
    return any(marker in cmd_lower for marker in [
        'echo complete',
        'echo done',
        'complete_task',
        'finish_task',
    ])


def is_write_command(command: str) -> bool:
    """
    Check if command writes to a file - DYNAMIC detection.

    Handles:
    - Full file writes (cat >, heredocs)
    - In-place edits (sed -i, perl -i)
    - Patches (patch, diff)
    """
    if not command:
        return False

    # Look for common write patterns
    write_patterns = [
        'cat >',
        'cat>',
        'echo >',
        'echo>',
        'tee ',
        '<<',  # heredoc
        'sed -i',  # in-place edit
        'sed -n',  # in-place edit (BSD)
        "sed -i'",  # BSD format
        'perl -i',  # in-place edit
        'patch ',  # apply patch
    ]

    return any(pattern in command for pattern in write_patterns)


def extract_written_file_path(command: str) -> str | None:
    """
    Extract file path from a write command - DYNAMIC parsing.

    Handles 567+ different patterns flexibly:
    - Full rewrites (cat >, echo >)
    - In-place edits (sed -i, perl -i)
    - Patches (patch)
    """
    if not is_write_command(command):
        return None

    # Pattern: sed -i 's/.../.../' path/to/file.py
    if 'sed -i' in command or "sed -i'" in command:
        # Extract file path (last argument usually)
        match = re.search(r'sed\s+-i[^\s]*\s+(?:\'[^\']+\'\s+|"[^"]+"\s+)?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)', command)
        if match:
            return match.group(1).strip()

    # Pattern: perl -i -pe 's/.../.../g' path/to/file.py
    if 'perl -i' in command:
        match = re.search(r'perl\s+-i\S*\s+.*?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)', command)
        if match:
            return match.group(1).strip()

    # Pattern: patch path/to/file.py
    if 'patch ' in command:
        match = re.search(r'patch\s+([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)', command)
        if match:
            return match.group(1).strip()

    # Pattern: cat > path/to/file.py (most common)
    match = re.search(r'cat\s*>\s*([^\s<]+)', command)
    if match:
        return match.group(1).strip()

    # Pattern: echo ... > path/to/file.py
    match = re.search(r'echo.*?>\s*([^\s<;|&]+)', command)
    if match:
        return match.group(1).strip()

    # Pattern: tee path/to/file.py
    match = re.search(r'tee\s+([^\s<>|&;]+)', command)
    if match:
        return match.group(1).strip()

    # DYNAMIC: Any > followed by a path-like string
    match = re.search(r'>\s*([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)', command)
    if match:
        return match.group(1).strip()

    return None


def is_read_command(command: str) -> bool:
    """Check if command reads a file."""
    if not command:
        return False

    # Look for read patterns (cat without redirection)
    if 'cat ' in command and '>' not in command and '<<' not in command:
        return True

    # less, head, tail
    return any(cmd in command for cmd in ['less ', 'head ', 'tail ', 'more '])


def extract_read_file_path(command: str) -> str | None:
    """
    Extract file path from a read command - DYNAMIC parsing.

    Handles 567+ different patterns flexibly.
    """
    if not is_read_command(command):
        return None

    # Pattern: cat path/to/file.py (most common)
    match = re.search(r'cat\s+([^\s|<>&;]+)', command)
    if match:
        path = match.group(1).strip()
        # Remove flags like -n
        if not path.startswith('-'):
            return path

    # Pattern: head/tail/less/more path/to/file.py
    for cmd in ['head', 'tail', 'less', 'more', 'vim', 'nano', 'vi']:
        match = re.search(rf'{cmd}\s+(?:-\w+\s+)*([^\s|<>&;]+)', command)
        if match:
            return match.group(1).strip()

    # DYNAMIC: Any path-like string after common read commands
    read_cmds = ['cat', 'head', 'tail', 'less', 'more', 'view']
    for cmd in read_cmds:
        if cmd in command:
            # Find path-like pattern after the command
            pattern = rf'{cmd}\s+.*?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)'
            match = re.search(pattern, command)
            if match:
                return match.group(1).strip()

    return None
