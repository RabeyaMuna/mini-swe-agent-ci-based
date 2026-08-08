from pathlib import Path
import re
p=Path("README.md")
s=p.read_text(encoding="utf-8")
# Fix agents list block
pattern=r"(This repository runs CI repair evaluations for two agents:\n)(?:.|\n)*?\n\n"
replacement=(
    "\\1\n"
    "- `mini-swe-agent`\n"
    "- `codex` (OpenAI Codex CLI–based repair agent)\n\n"
)
s=re.sub(pattern, replacement, s, flags=re.M)
# Remove stray phrase everywhere except codex bullet
fixed=[]
for ln in s.splitlines():
    if "`codex` (OpenAI Codex CLI–based repair agent)" in ln:
        fixed.append(ln)
    else:
        fixed.append(ln.replace(" (OpenAI Codex CLI–based repair agent)", ""))
# Remove any residual openhands lines
fixed=[ln for ln in fixed if "openhands" not in ln]
# Collapse duplicate spaces after dash
fixed=[re.sub(r"^-\s+","- ",ln) if ln.lstrip().startswith("-") else ln for ln in fixed]
p.write_text("\n".join(fixed)+"\n", encoding="utf-8")
print("fixed README.md")
