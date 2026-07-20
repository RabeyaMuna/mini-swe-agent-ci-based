# Dynamic Bash-Based OpenHands

## Overview

OpenHands now uses **dynamic bash command parsing** instead of rigid JSON tools. This makes it work with **any model** (GLM-5.2, MiniMax, GPT-4, etc.) and handle **567+ different CI failure patterns**.

## Why Dynamic?

The user has 567 CI-Bench issues with **different structures and patterns**. A static approach would fail on many of them. The dynamic approach **adapts to whatever the model outputs**.

## Dynamic Features

### 1. **Flexible Command Parsing** ([bash_parser.py](bash_parser.py))

Handles multiple formats:
- ✅ ```bash code blocks
- ✅ ```sh, ```shell, ```zsh blocks
- ✅ Generic ``` blocks (if they contain bash)
- ✅ Raw commands without blocks (auto-detected)
- ✅ Commands with markdown prefixes (>, $, #)

### 2. **Multiple Fix Strategies** ([bash_instructions.py](bash_instructions.py))

Supports different fixing approaches:

**Small/Targeted Fixes** (preferred for large files):
```bash
# In-place editing with sed
sed -i 's/old/new/g' file.py

# Add import at top
sed -i '1i import typing' file.py

# Add type hint
sed -i 's/def func()/def func() -> None:/g' file.py
```

**Full File Rewrites** (for small files):
```bash
cat > file.py << 'EOF'
complete file content
EOF
```

**Discovery Commands**:
```bash
find . -name "*.py" -type f
grep -r "pattern" .
cat file.py
```

### 3. **Smart File Path Extraction**

Dynamically extracts paths from:
- `cat > path/to/file.py`
- `sed -i 'script' path/to/file.py`
- `perl -i -pe 'script' path/to/file.py`
- `patch path/to/file.py`
- `echo "..." > path/to/file.py`
- Handles flags: `cat -n file.py`
- Handles quoted paths: `cat "path with spaces/file.py"`

### 4. **Heredoc Content Extraction**

Handles multiple heredoc formats:
```bash
# Standard
<< 'EOF'
content
EOF

# No quotes
<< EOF
content
EOF

# Different delimiters
<< END
content
END

# Malformed (missing delimiter) - still extracts content
```

### 5. **Command Type Detection**

Automatically recognizes:
- **Read commands**: cat, head, tail, less, more, view
- **Write commands**: cat >, sed -i, perl -i, patch, echo >, tee
- **Completion**: echo COMPLETE, echo DONE, COMPLETE_TASK
- **Discovery**: find, grep, ls
- **Generic**: anything else

### 6. **Progress Tracking**

Tracks work across different command types:
- Files read (cat, head, etc.)
- Files written (cat >, sed -i, patch)
- Files modified in-place (sed -i tracked separately)
- Shows progress to model in next instruction

### 7. **Robust Error Handling**

Dynamic fallbacks:
- No bash block found? → Search for bash keywords in response
- Can't parse? → Try generic command execution
- Heredoc malformed? → Extract what's available
- File not found? → Auto-search with `find`

## How It Works

### Step 1: Model Outputs Bash

The model sees problem context and outputs:
```bash
cat framework/py/test.py
```

### Step 2: Dynamic Parsing

Parser extracts command from ANY format:
- Code block → extract from ```bash
- No block → detect bash keywords
- Mixed format → find the command part

### Step 3: Command Classification

System determines command type:
- Is it completion? → Mark done
- Is it write? → Track file change
- Is it read? → Track file read
- Generic? → Execute and return output

### Step 4: Execution

Execute based on type:
- Read: Use environment's `read_file()`
- Write (heredoc): Use `write_file()` with extracted content
- Write (sed/patch): Use `run_command()` and track file
- Generic: Use `run_command()`

### Step 5: Next Instruction

Build next instruction showing:
- Progress so far (files read/written)
- Previous command result
- What to do next (DYNAMIC - model decides)

## Advantages for 567 Different Issues

1. **Pattern Agnostic**: Works regardless of issue structure
2. **Model Agnostic**: Works with weak (GLM) or strong (GPT-4) models
3. **Fix Strategy Flexible**: Full rewrite OR targeted edits
4. **Large File Support**: sed/patch for files too big to rewrite
5. **Auto-Recovery**: Auto-search if paths wrong, auto-fix malformed commands
6. **No Manual Tuning**: Adapts to whatever the model tries

## Comparison: Static vs Dynamic

### Static (JSON Tools)
```json
{"tool": "read_file", "args": {"file_path": "exact/path.py"}}
```
❌ Fails if path is wrong
❌ Fails if model outputs prose
❌ Fails if model forgets JSON
❌ Requires exact format

### Dynamic (Bash Commands)
```bash
cat path.py
```
✅ Auto-searches if path wrong
✅ Parses from prose
✅ Works without code blocks
✅ Accepts any bash format

## Code Structure

```
openhands/
├── bash_parser.py          # Dynamic command parsing
├── bash_instructions.py    # Adaptive instruction generation
├── interactive_agent.py    # Agent with bash-to-action conversion
└── DYNAMIC_FEATURES.md     # This file
```

## Testing

Test with any model:
```bash
python ci_bench_runner.py \
  --eval-issues data/ci_bench/subset_ci_bench_v1.1.json \
  --mode baseline \
  --model zai/glm-5.2 \
  --output results/test/preds.json \
  --slice 0:10
```

Should work with:
- zai/glm-5.2 (weak model)
- minimax/Minimax-Text-01 (medium model)
- openrouter/anthropic/claude-3.5-sonnet (strong model)
- Any other model accessible via LiteLLM

## Future Enhancements

Possible additions for even more dynamism:
1. Auto-detect programming language and adapt commands
2. Learn from successful patterns across issues
3. Suggest multiple fix strategies, let model choose
4. Auto-generate test commands to verify fixes
5. Multi-step planning for complex issues

## Summary

This implementation is **truly dynamic** because:
- ✅ No hardcoded patterns
- ✅ Adapts to model output format
- ✅ Handles multiple fix strategies
- ✅ Auto-recovers from errors
- ✅ Works with 567+ different issue patterns
- ✅ Model-agnostic (weak or strong models)

It's built to handle **real-world diversity** in CI failures, not just a few hand-crafted examples.
