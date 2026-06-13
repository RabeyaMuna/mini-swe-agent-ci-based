from __future__ import annotations

from typing import List, Dict, Any
import json
import os
import re
import tempfile
import time

# ── Optional third-party dependencies ─────────────────────────────────────────
try:
    import demjson3  # type: ignore
    _DEMJSON3_AVAILABLE = True
except ImportError:
    demjson3 = None  # type: ignore
    _DEMJSON3_AVAILABLE = False

try:
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_core.messages import HumanMessage  # type: ignore
except ImportError:
    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


# ── JSON Cleaning Utility ──────────────────────────────────────────────────────
def _clean_malformed_json(content: str) -> str:
    """
    Clean common LLM JSON errors:
    - Markdown fences
    - Trailing commas
    - Missing commas
    - Extra text before/after JSON
    """
    # Remove markdown fences
    content = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)

    # Fix trailing commas before } or ]
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Fix missing commas between string values
    content = re.sub(r'"\s+"', '", "', content)

    # Fix double commas
    content = re.sub(r',\s*,', ',', content)

    # Fix missing commas between } and {
    content = re.sub(r'}\s*{', '}, {', content)
    content = re.sub(r'}\s*\[', '}, [', content)
    content = re.sub(r']\s*{', '], {', content)

    # Extract JSON from surrounding text (find first { or [ to last } or ])
    content = content.strip()
    start_brace = content.find('{')
    start_bracket = content.find('[')

    # Determine which comes first
    if start_brace == -1 and start_bracket == -1:
        return content
    elif start_brace == -1:
        start = start_bracket
        end_char = ']'
    elif start_bracket == -1:
        start = start_brace
        end_char = '}'
    else:
        start = min(start_brace, start_bracket)
        end_char = '}' if start == start_brace else ']'

    # Find matching closing brace/bracket
    end = content.rfind(end_char)
    if end > start:
        content = content[start:end+1]

    return content.strip()


# ── Fallbacks for missing 'utilities.*' modules ────────────────────────────────

# utilities.constant.ERROR_KEYWORDS
ERROR_KEYWORDS: List[str] = [
    "error", "Error", "ERROR",
    "errors", "Errors", "ERRORS",
    "failed", "Failed", "FAILED",
    "failure", "Failure", "FAILURE",
    "exception", "Exception", "EXCEPTION",
    "traceback", "Traceback",
    "fatal", "Fatal", "FATAL",
    "critical", "Critical", "CRITICAL",
    "AssertionError", "ImportError", "ModuleNotFoundError",
    "SyntaxError", "TypeError", "ValueError", "RuntimeError",
    "no module named", "not found", "permission denied",
    "syntax error", "timeout", "timed out",
    "connection refused",
]


def load_config() -> Dict[str, Any]:
    """Fallback for utilities.load_config.load_config()."""
    out_folder = os.environ.get(
        "MEMCI_OUT_FOLDER",
        os.path.join(tempfile.gettempdir(), "memci_out"),
    )
    return {
        "exception_dir": os.path.join(out_folder, "exceptions"),
        "out_folder":    out_folder,
    }


def chunk_log_by_tokens(
    text: str,
    max_tokens: int = 70000,
    overlap: int = 200,
    model: str = "",
) -> List[str]:
    """Fallback for utilities.chunking_logic.chunk_log_by_tokens()."""
    chars_per_chunk = max_tokens * 4          # ~1 token ≈ 4 chars
    overlap_chars   = overlap * 4
    if len(text) <= chars_per_chunk:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks or [text]


def get_chunk_threshold_simple(model_name: str = "") -> int:
    """Fallback for utilities.model_token_limits.get_chunk_threshold_simple()."""
    return 70_000

class CILogAnalyzerLLM:
    def __init__(
        self,
        repo_path: str,
        ci_log: List[Dict[str, Any]],
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: ChatOpenAI,
        model_name: str,
        task_id: str
    ):
        self.config = load_config()
        self.repo_path = repo_path
        self.ci_log = ci_log
        self.sha_fail = sha_fail
        self.workflow = workflow
        self.workflow_path = workflow_path
        self.task_id = task_id

        self.llm = llm
        self.model_name = model_name
        self._encoder = self._get_encoder()

        self.error_details: List[Dict[str, Any]] = []

    def ci_log_analysis(self) -> List[Dict[str, Any]]:
        """
        Analyze CI logs using LLM
        """
        print("Running Tool: LLM-based CI Log Analysis")
        results: List[Dict[str, Any]] = []
        THRESHOLD = get_chunk_threshold_simple(self.model_name)
        chunk_tracker = []
        
        for step in self.ci_log:
            step_name = step.get("step_name", "unknown_step")
            log = step.get("log", "")
            print(f"\nProcessing Step: {step_name}")

            try:
                log_text = log if isinstance(log, str) else "\n".join(log)
                total_tokens = self._estimate_tokens(log_text)


                print(f"Token count for '{step_name}': {total_tokens}")

                if total_tokens > THRESHOLD:
                    raw_chunks = chunk_log_by_tokens(log_text, max_tokens=70000, overlap=200, model=self.model_name)
                    print(f"Chunking activated: {len(raw_chunks)} chunks created for step '{step_name}'")
                    
                    chunk_tracker.append((step_name, len(raw_chunks)))
                    
                    if len(raw_chunks) > 6:
                        chunks = self._filter_chunks(raw_chunks)
                    else:
                        chunks = raw_chunks
                else:
                    chunks = [log_text]
                    print(f"No chunking needed for '{step_name}'")
                    
                self._save_chunk_tracker(chunk_tracker)

                step_chunks = []

                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i + 1}/{len(chunks)}...")

                    prompt = f"""
You are CI Log Analyzer. Analyze the following CI log chunk and extract structured information while staying strictly faithful to the text.

You may be dealing with ANY CI system and ANY tools.

DO NOT guess. DO NOT invent missing context. DO NOT give advice.

────────────────────────────────
1) SUMMARY (SHORT, HIGH-SIGNAL)

Write a concise 2–5 sentence summary describing:
- The purpose of this CI log chunk
- Whether failures or errors occur
- The nature of those failures (if present)
- Whether execution appears to continue or stop

Rules:
- Omit repetitive and low-signal output
- Mention repeated failures only once
- Stay strictly faithful to the log

────────────────────────────────
2) RELEVANT FILES (ALL FILE PATHS MENTIONED)

Extract ALL file paths that appear anywhere in the chunk.
For each file:
- Normalize to repo-relative paths when possible
- Explain WHY the file appears using evidence from the log
- State whether it is directly related to a failure, a warning, or normal execution

Rules:
- Deduplicate files
- If the same file appears in multiple contexts, merge the reasons
- Do NOT invent relevance

────────────────────────────────
3) RELEVANT FAILURES (NORMALIZED, NATURAL LANGUAGE — CRITICAL)

For each DISTINCT failure observed in this chunk, write ONE concise natural-language description.

Each failure description MUST:
- Describe WHAT failed (trial / test / command / step / job, if identifiable)
- State the EXACT error type and message as shown (e.g., ValueError)
- State WHERE it occurred (file and line if shown; normalize paths)
- Explain HOW the failure is evidenced in the log (warnings, traceback mention, failure message)
- Mention repetition if the same failure occurs multiple times

STRICT RULES:
- Do NOT copy raw stack traces
- Do NOT include long multiline blocks
- Do NOT repeat identical failures multiple times
- Do NOT speculate about root cause beyond what the log states
- Keep each failure description under 500 characters

If the chunk contains NO failures, return an empty array [].

────────────────────────────────
PATH NORMALIZATION

Remove common CI workspace prefixes when possible, such as:
- /home/runner/work/<repo>/<repo>/
- /workspace/
- /__w/<repo>/<repo>/
- /opt/.../<repo>/

If unsure, keep the original path.

────────────────────────────────
OUTPUT FORMAT (STRICT)

Return ONLY valid JSON with EXACTLY this structure.
Do NOT add extra keys.
Do NOT use ``` or ```json.

{{
  "step_name": "{step_name}",
  "summary": "Detailed, faithful narrative summary of everything shown in this chunk.",
  "relevant_files": [
    {{
      "file": "repo/relative/path.ext",
      "reason": "Evidence-based explanation of why this file appears in the log and whether it relates to a failure."
    }}
  ],
  "relevant_failures": [
    "Natural-language, evidence-based description of a distinct failure."
  ]
}}

Rules:
- If no files exist, return: "relevant_files": []
- If no failures exist, return: "relevant_failures": []
- Output plain JSON only — no text before or after.

────────────────────────────────
CI LOG CHUNK
{chunk}
"""



                    response = self.llm.invoke([HumanMessage(content=prompt)]).content
                    content = self.load_json_maybe_fenced(response)
                    if not content or not content.strip():
                        continue

                    try:
                        cleaned_json = json.loads(content)
                    except json.JSONDecodeError:
                        if demjson3 is not None:
                            try:
                                cleaned_json = demjson3.decode(content)
                            except Exception as dec_err:
                                print(f"[WARN] demjson3 failed for chunk {i + 1}: {dec_err}")
                                continue
                        else:
                            print(f"[WARN] json.loads failed for chunk {i + 1} and demjson3 not installed; skipping.")
                            continue
                        
                    # Decide whether to skip this chunk
                    no_failures = not cleaned_json.get("relevant_failures")   # [] or missing -> True
                    no_files = not cleaned_json.get("relevant_files")         # [] or missing -> True
                    empty_summary = not (cleaned_json.get("summary") or "").strip()

                    if no_failures and no_files and empty_summary:
                        print(f"Skipping chunk {i + 1}/{len(chunks)}: no failure evidence found.")
                        continue

                    cleaned_json.update(
                        self._build_chunk_metadata(
                            step_name=step_name,
                            chunk=chunk,
                            chunk_index=i,
                            total_chunks=len(chunks),
                        )
                    )
                    step_chunks.append(cleaned_json)

                results.append({
                    "step_name": step_name,
                    "chunks": step_chunks,
                    "step_document": self._build_step_document(step_name, step_chunks),
                })

            except Exception as e:
                print(f"[ERROR] Processing step '{step_name}': {str(e)}")
                results.append({
                    "step_name": step_name,
                    "chunks": [],
                    "error": str(e)
                })

        return results
   
    # ------------------------------------------------------------------
    def generate_log_summary(self, all_step_outputs) ->  List[Dict[str, Any]]:
        """
        Generate a structured final error summary from error details,
        workflow tools, and validation checks.
        """
        print(" Running Tool: _generate_summary")
        log_details = []
        for step in all_step_outputs:
            step_name = step.get("step_name", "UNKNOWN_STEP")
            chunks = step.get("chunks", [])
            
            step_payload = {
                "step_name": step_name,
                "chunks": chunks,
            }
            step_payload_json = json.dumps(step_payload, indent=2, ensure_ascii=False)
            prompt = f"""
You are a CI log analyzer.

You receive pre-processed CI log information for a FAILED CI RUN, for a SINGLE CI STEP.

## Input Details
--- PRE-PROCESSED Log information for Given STEP DATA (JSON) ---
{step_payload_json}

Using ONLY this information for THIS STEP (all its chunks), produce a structured summary
of the CI failure for this step using the following STRICT JSON schema
(**do not add or remove top-level keys**)

{{
"step_name": "{step_name}",
"sha_fail": "{self.sha_fail}",
"log_content": "<Explain overall details of the CI log in natural language>",
"error_context": [
    "English explanation(s) of the root cause(s) visible of CI workflow failed supported by log evidence. Provide detail and evident reasons of the given step failure."
],
"relevant_files": [
    {{
    "file": "path/to/file.py",
    "line_number": 123,
    "reason": "Short explanation of why this file is tied to the failure."
    }},
],
"error_types": [
    {{
    "category": "High-level category, e.g. 'Test Failure', 'Runtime Error', 'Dependency Error', 'Configuration Error', 'Code Formatting', 'Type Checking'",
    "subcategory": "More specific description, e.g. 'Runtime Error – ValueError in Optuna objective', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
    "evidence": 'Short quote or paraphrase from THIS CHUNK that justifies this classification.'
    }}
]
}}

### Rules (IMPORTANT)

- "step_name": the CI step name (use the input step_name).
- "sha_fail": the failing commit SHA (given).
- "log_content":
  - A concise but informative natural-language description of what happened in this CI STEP.
  - You may integrate information across ALL chunks for this step.

- "error_context":
  - A list of English explanations of the root cause(s) of the CI failure related to THIS STEP.
  - Each entry must be supported by evidence in the summaries and/or relevant_failures.
  - If nothing meaningful appears, use an empty list [].

- "relevant_files":
  - Consider all chunk-level data ("relevant_files", "relevant_failures", and summaries),
    but INCLUDE a file ONLY if:
      * it is clearly linked to a failing test, assertion error, runtime exception,
        dependency error, configuration error, or critical warning in THIS STEP, OR
      * the log explicitly states that the failure occurs in that file.
  - It is OK to discard files that appear in chunk-level "relevant_files" if they were only
    mentioned in setup/installation and are not clearly tied to the failure.
  - Deduplicate by "file" path. If the same file appears with different reasons, merge
    them into one concise, evidence-based "reason".
  - "line_number":
      * Use the failing line number if it is clearly shown in the logs,
      * otherwise null.
  - If no file clearly meets these conditions, return "relevant_files": [].
  - provide evidence-based "reasoning" explaining how this file is tied to the CI failure.

- "error_types":
  - Describe the kinds of errors visible in this STEP:
      * Test failures,
      * Runtime exceptions,
      * Dependency / environment issues,
      * Configuration / CI YAML problems,
      * Code formatting or linting issues,
      * Type checking errors, etc.
  - Each entry MUST include:
      * "category"  (broad bucket),
      * "subcategory" (more specific),
      * "evidence"   (short quote or paraphrase from summaries / relevant_failures).

### Global Rules
1. Use ONLY the information from the STEP JSON shown below.
2. Use null for any unknown scalar values (e.g., line_number if not visible).
3. Do NOT add extra top-level keys.
4. Return STRICT JSON ONLY — no markdown, no comments, no natural language outside JSON.

### Output Rules (STRICT)
- Output MUST be a single raw JSON object.
- Do NOT wrap the JSON in triple backticks.
- Do NOT include ```json or any other marker/fence.
- Do NOT add any text before or after the JSON.
"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)]).content
                content = self.load_json_maybe_fenced(response)

                if not content or not content.strip():
                    raise ValueError("LLM returned an empty response for generate_log_summary")

                try:
                    summary = json.loads(content)
                except json.JSONDecodeError:
                    # Try cleaning malformed JSON first
                    try:
                        cleaned = _clean_malformed_json(content)
                        summary = json.loads(cleaned)
                    except json.JSONDecodeError:
                        # Fall back to demjson3
                        if demjson3 is not None:
                            try:
                                cleaned = _clean_malformed_json(content)
                                summary = demjson3.decode(cleaned)
                            except Exception as dec_err:
                                raise ValueError(f"JSON parse failed: {dec_err} | raw: {content[:200]}")
                        else:
                            raise ValueError(f"JSON parse failed (demjson3 not installed) | raw: {content[:200]}")

                log_details.append(summary)
            except Exception as e:
                # If one chunk fails, log and continue
                self._log_error(
                    method="generate_log_summary",
                    error=e,
                    step=f"{step_name}",
                )

        return log_details

    # ------------------------------------------------------------------
    def full_content_summary(
        self, log_details: List[Dict[str, Any]], workflow_details: Any
    ) -> Dict[str, Any]:
        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read step-level CI job log summaries and workflow details, then produce a **single, structured, evidence-based JSON summary** that explains why the CI run failed and clearly classifies the errors by category and subcategory.

---

## INPUTS

1. CI Log Details (from step analysis, list of objects):

Each element in `log_details` corresponds to ONE CI step and typically includes:
- "step_name": name of the CI step.
- "sha_fail": the failing commit SHA.
- "log_content": natural-language description of what happened in this step.
- "error_context": list of step-level explanations of root causes in this step.
- "relevant_files": list of files tied to the failure in this step, each with:
  - "file"
  - "line_number" (may be null)
  - "issue_type": short failure classification for this file (may be null)
  - "failed_cmd": the command that triggered the failure for this file (may be null)
  - "failed_tool": the tool that reported the failure for this file (may be null)
  - "reason"
- "error_types": list of error classifications for this step, each with:
  - "category"
  - "subcategory"
  - "evidence"

Full step-level details:
{json.dumps(log_details, indent=2, ensure_ascii=False)}

2. Workflow Details:

Parsed CI workflow (e.g., GitHub Actions YAML) including jobs, steps, and commands.
Use this to map failing steps to their jobs and commands when possible.

Full workflow details:
{json.dumps(workflow_details, indent=2, ensure_ascii=False)}

---

## OUTPUT FORMAT (strict JSON only)

Return a SINGLE aggregated summary for the entire failed run using this exact structure:

{{
  "error_context": [
    "Plain-English explanation(s) of the root cause(s), supported by log evidence. Mention all the steps involved in the failure and how and why it failed."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "issue_type": "Short failure classification for this file, e.g. 'Test Failure', 'Import Error', 'Type Error', 'Dependency Error', 'Lint Error'. Use null if not clearly tied to a failure.",
      "failed_cmd": "The exact command that triggered the failure for this file, e.g. 'pytest tests/', 'python -m mypy src/'. Use null if not identifiable.",
      "failed_tool": "The tool that reported or caused the failure for this file, e.g. 'pytest', 'mypy', 'flake8', 'pip'. Use null if not identifiable.",
      "reason": "Short evidence-based explanation of why this file is tied to the failure."
    }},
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Code Formatting', 'Dependency Error', 'Test Failure', 'Runtime Error', 'Type Checking', 'Configuration Error'",
      "subcategory": "More specific type under that category, e.g. 'Unused Import', 'Line Length Exceeded', 'ImportError: No module named X', 'AssertionError', 'Missing dependency', 'Mypy type mismatch'",
      "evidence": "Brief quote or paraphrase from logs that proves this classification."
    }}
  ],
  "failed_job": [
    {{
      "job": "Job name or ID",
      "step": "Step name that failed",
      "command": "Exact command or action that caused the failure"
    }}
  ]
}}

---

## INSTRUCTIONS

1. **Aggregate across ALL steps (do not treat them independently).**
   - Read every element in `log_details`.
   - Combine their information into ONE global summary for the entire CI run.

2. **Error Context (global explanation).**
   - Use English sentences summarizing the main root cause(s) of the failure.
   - Explicitly reference the step names involved and has failed in that steps, e.g.:
     - "In step 'Install dependencies', pip failed due to missing package X..."
     - "In step 'Run tests', pytest reported failing tests because..."
   - If multiple steps contribute to the failure, mention each clearly.
   - Base your explanations on `log_content`, `error_context`, and `error_types` from the steps.

3. **Relevant Files (deduplicated across steps).**
   - Consider all `relevant_files` from all steps.
   - Deduplicate by `"file"` path: each file should appear at most once in the final list.
   - If the same file appears in multiple steps, merge the reasons into a single concise,
     evidence-based `"reason"` that reflects all relevant contexts.
   - Use `"line_number":` the most specific failing line if available; otherwise `null`.
   - For each file, analyze the failure evidence together with the matching workflow job/step/command to infer:
     - `"failed_cmd"` from the exact workflow `run` command or `uses` action tied to the failing step.
     - `"failed_tool"` from the reporting tool or command invoked in that step (for example `pytest`, `mypy`, `ruff`, `flake8`, `pip`, `npm`, `go test`).
     - `"issue_type"` from the file-specific failure mode described in the logs (for example test failure, lint error, import error, type error, dependency error).
   - Prefer file-specific values from the step analysis when present. Otherwise, infer them from the workflow step that produced the file-level failure. Use `null` only when the value cannot be supported by the logs or workflow.
   - The `"reason"` for each file must explain why that file is tied to the failure and, when possible, mention the step/tool/command that connects the file to the failing workflow step.
   - Include only files that are clearly tied to failures, errors, or critical warnings.
   - If no such file exists, return `"relevant_files": []`.

4. **Error Types (all distinct types found).**
   - Look at all `error_types` from all steps.
   - Aggregate them into a list of distinct (category, subcategory) pairs.
   - For each distinct pair, keep one entry with:
     - `"category"` and `"subcategory"` exactly once.
     - `"evidence"` summarizing or quoting representative log evidence (you may combine evidence from multiple steps briefly).

5. **Failed Job (use workflow + step info).**
   - Use both `log_details` and `workflow_details` to identify which job(s) and step(s) failed.
   - Match step names from `log_details` to steps in the workflow (by their `"name"` field) to infer:
     - `"job"`: the job display name or ID (from the workflow, if available; otherwise null).
     - `"step"`: the failing step name from `log_details`.
     - `"command"`:
       - The value of `"run"` if present, or `"uses"` if it is an action reference.
       - If no command can be found, use `null`.
   - If multiple jobs/steps clearly fail, include multiple entries in `"failed_job"`.
   - Reuse this same step-to-workflow mapping when populating each file's `"failed_cmd"` and `"failed_tool"` so the file-level metadata stays consistent with the recorded failed job/step.

6. **Output Rules**
   - Return **only valid JSON** — no markdown, commentary, or code fences.
   - Do not hallucinate; use `null` for unknown values (e.g., line_number, job, command).
   - Merge duplicates carefully (same file paths, same category/subcategory pairs).
   - Ensure every item is concise, evidence-based, and traceable to the logs and/or workflow.
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            content = self.load_json_maybe_fenced(response)

            if not content or not content.strip():
                raise ValueError("LLM returned an empty response for full_content_summary")

            try:
                summary = json.loads(content)
            except json.JSONDecodeError:
                # Try cleaning malformed JSON first
                try:
                    cleaned = _clean_malformed_json(content)
                    summary = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Fall back to demjson3
                    if demjson3 is not None:
                        try:
                            cleaned = _clean_malformed_json(content)
                            summary = demjson3.decode(cleaned)
                        except Exception as dec_err:
                            raise ValueError(f"JSON parse failed: {dec_err} | raw: {content[:200]}")
                    else:
                        raise ValueError(f"JSON parse failed (demjson3 not installed) | raw: {content[:200]}")
            print(" Completed: _generate_summary")
            
            summary["sha_fail"] = self.sha_fail
            summary["id"] = self.task_id
            
            return summary

        except Exception as e:
            error_dir = os.path.join(
                self.config["exception_dir"], "interrupted_error_log"
            )
            os.makedirs(error_dir, exist_ok=True)

            error_data = {
                "sha_fail": self.sha_fail,
                "error": str(e),
                "tool": "ErrorContextExtractionAgent.run",
            }

            error_file = os.path.join(error_dir, f"{self.sha_fail}_error.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=4)

            return {"error": f"Failed to generate summary: {e}"}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        print(f"Fully Autonomous Execution for Commit: {self.sha_fail}")
        selected_logs = self.ci_log_analysis()
        log_details = self.generate_log_summary(selected_logs)
        if not log_details:
            return {
                "error": "No step summaries produced — all CI log steps failed to parse.",
                "sha_fail": self.sha_fail,
                "id": self.task_id,
            }
        generated_summary = self.full_content_summary(log_details, workflow_details=self.workflow)
        generated_summary["chunk_summaries"] = self._flatten_chunk_summaries(selected_logs)
        generated_summary["step_documents"] = [
            {
                "step_name": str(step.get("step_name") or "unknown_step"),
                "document": str(step.get("step_document") or "").strip(),
            }
            for step in selected_logs
            if str(step.get("step_document") or "").strip()
        ]
        generated_summary["overall_ci_summary"] = self._build_overall_ci_summary(
            generated_summary,
            log_details,
        )
        generated_summary["analysis_document"] = self._build_analysis_document(
            generated_summary,
            selected_logs,
            log_details,
        )
        return generated_summary

    # ------------------------------------------------------------------
    def _get_encoder(self):
        """Safely get a tiktoken encoder for the model."""
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for a given text using the cached encoder."""
        if text is None:
            return 0
        if self._encoder is None:
            return len(text) // 4  # rough estimate: ~1 token per 4 chars
        return len(self._encoder.encode(text))

    def _log_error(self, method: str, error: Exception, step: str = ""):
        base_dir = os.path.join(
            self.config["exception_dir"], "interrupted_error_log"
        )
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"{self.sha_fail}_{method}_{int(time.time())}_error.json"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "commit": self.sha_fail,
                    "method": method,
                    "step": step,
                    "error": str(error),
                },
                f,
                indent=2,
            )
        print(f"[ERROR LOGGED] {file_path}")

    def _filter_chunks(self, raw_chunks: List[str]) -> List[str]:
        """
        Keep:
        - All chunks in the first (n-6) that contain any ERROR_KEYWORDS (by word)
        - Always keep the last 6 chunks (serial order preserved)
        """
        n_chunks = len(raw_chunks)

        cutoff = n_chunks - 4
        filtered_chunks: List[str] = []

        # 1) Check the first (n-6) chunks and keep only those with error keywords
        for idx, chunk in enumerate(raw_chunks[:cutoff]):
            for line_no, line in enumerate(chunk.splitlines(), start=1):
                hits = self.is_line_error(line, ERROR_KEYWORDS)
                if hits:
                    # optional debug:
                    # print(f"[FILTER] chunk {idx}, line {line_no}: hits={hits}")
                    filtered_chunks.append(chunk)
                    break  # done with this chunk

        # 2) Append the last 4 chunks unconditionally (preserve serial order)
        filtered_chunks.extend(raw_chunks[cutoff:])
        
        if len(filtered_chunks)> 20:
            print(
            f"After filtering, too many chunks ({len(filtered_chunks)}). Truncating to last 12 "
            f"(checked first {cutoff}, always kept last 4)"
            )
            filtered_chunks = filtered_chunks[-12:]
            

        print(
            f"Filtered from {n_chunks} -> {len(filtered_chunks)} chunks "
            f"(checked first {cutoff}, always kept last 4)"
        )

        return filtered_chunks
      
    def _save_chunk_tracker(self, chunk_tracker: List[tuple]):
        debug_dir = os.path.join(self.config["out_folder"], "chunk_tracking")
        os.makedirs(debug_dir, exist_ok=True)

        file_path = os.path.join(debug_dir, "chunk_tracker.json")

        # Load existing data if file exists
        existing: List[Dict[str, Any]] = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except Exception:
                existing = []

        # Append this run
        existing.append({"sha_fail": self.sha_fail, "chunks": chunk_tracker})

        # Write back pretty JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        print(f"[CHUNK TRACKER SAVED] {file_path}")
        
        
    def is_line_error(self, line: str, keywords) -> list[str]:
        """
        Return list of keywords that appear as separate words or exact phrases.
        - Single-word keywords: must match a whole word in the line (case-sensitive).
        - Multi-word keywords: matched as exact substrings (case-sensitive).
        - Special rule: if we see 'error' or 'errors' but the previous word is 'no'
        (e.g. 'no error', 'No errors'), we IGNORE that match.
        """
        hits = []

        # Very simple tokenization: split on whitespace, strip basic punctuation
        raw_tokens = line.split()
        tokens = [tok.strip("[]():,") for tok in raw_tokens]

        # 1) Phrase keywords -> use substring
        for kw in keywords:
            if " " in kw:
                if kw in line:  # exact phrase, case-sensitive
                    hits.append(kw)

        # 2) Single-word keywords -> must match full token
        # We also apply the "no error" rule here.
        for idx, tok in enumerate(tokens):
            for kw in keywords:
                if " " in kw:
                    continue  # phrases already handled

                if tok == kw:
                    # Special: ignore "error"/"errors" if previous token is 'no' (any case)
                    if kw in ("error", "errors") and idx > 0:
                        prev_tok = tokens[idx - 1]
                        if prev_tok.lower() == "no":
                            # e.g. "no error", "No errors" -> not a real error
                            continue

                    hits.append(kw)

        # Remove duplicates while preserving order
        seen = set()
        unique_hits = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                unique_hits.append(h)

        return unique_hits


    def load_json_maybe_fenced(self, text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            # drop opening fence line (``` or ```json)
            lines = lines[1:] if lines else lines
            # drop closing fence line if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _build_chunk_metadata(
        self,
        *,
        step_name: str,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
    ) -> Dict[str, Any]:
        token_count = self._estimate_tokens(chunk)
        token_keywords = self._extract_chunk_keywords(chunk)
        code_context = self._extract_code_context(chunk)
        return {
            "chunk_index": chunk_index + 1,
            "chunk_total": total_chunks,
            "chunk_token_count": token_count,
            "token_keywords": token_keywords,
            "code_context": code_context,
            "chunk_document": self._build_chunk_document(
                step_name=step_name,
                chunk_index=chunk_index + 1,
                total_chunks=total_chunks,
                token_count=token_count,
                token_keywords=token_keywords,
                code_context=code_context,
            ),
        }

    def _extract_chunk_keywords(self, chunk: str, limit: int = 12) -> List[str]:
        candidates: List[str] = []
        for match in re.finditer(r"[A-Za-z0-9_./:-]{3,}", chunk):
            token = match.group(0).strip()
            lower = token.lower()
            if lower.startswith(("http://", "https://")):
                continue
            if token.isdigit():
                continue
            if "/" in token or "." in token or "_" in token or "::" in token:
                candidates.append(token)
                continue
            if any(ch.isupper() for ch in token[1:]):
                candidates.append(token)
                continue
            if lower in {"error", "errors", "failed", "failure", "traceback", "exception", "warning"}:
                candidates.append(token)

        seen = set()
        ordered: List[str] = []
        for token in candidates:
            if token not in seen:
                seen.add(token)
                ordered.append(token)
            if len(ordered) >= limit:
                break
        return ordered

    def _extract_code_context(self, chunk: str, limit: int = 5) -> List[str]:
        lines: List[str] = []
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if len(line) > 160:
                line = line[:157] + "..."
            if self._looks_like_code_or_trace(line):
                lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _looks_like_code_or_trace(self, line: str) -> bool:
        code_markers = (
            "Traceback",
            "File \"",
            " line ",
            "Error:",
            "Exception",
            "AssertionError",
            "FAILED",
            "E   ",
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "raise ",
            "pytest",
            "mypy",
            "ruff",
            "flake8",
        )
        if any(marker in line for marker in code_markers):
            return True
        return bool(re.search(r"[A-Za-z0-9_./-]+\.(py|js|ts|tsx|jsx|java|go|rb|php|yml|yaml|json|toml|ini|cfg)(:\d+)?", line))

    def _build_chunk_document(
        self,
        *,
        step_name: str,
        chunk_index: int,
        total_chunks: int,
        token_count: int,
        token_keywords: List[str],
        code_context: List[str],
    ) -> str:
        parts = [
            f"step={step_name}",
            f"chunk={chunk_index}/{total_chunks}",
            f"tokens={token_count}",
        ]
        if token_keywords:
            parts.append("keywords=" + ", ".join(token_keywords[:8]))
        if code_context:
            parts.append("code=" + " | ".join(code_context[:3]))
        return " ; ".join(parts)

    def _build_step_document(self, step_name: str, step_chunks: List[Dict[str, Any]]) -> str:
        lines = [f"step: {step_name}"]
        for chunk in step_chunks:
            summary = str(chunk.get("summary") or "").strip()
            failures = chunk.get("relevant_failures") or []
            files = [str(item.get("file") or "").strip() for item in (chunk.get("relevant_files") or []) if isinstance(item, dict)]
            doc = str(chunk.get("chunk_document") or "").strip()
            if doc:
                lines.append(doc)
            if summary:
                lines.append(f"summary: {summary}")
            if failures:
                lines.append("failures: " + " | ".join(str(item).strip() for item in failures[:3] if str(item).strip()))
            if files:
                lines.append("files: " + ", ".join(path for path in files[:6] if path))
        return "\n".join(lines).strip()

    def _flatten_chunk_summaries(self, selected_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for step in selected_logs:
            step_name = str(step.get("step_name") or "unknown_step")
            for chunk in step.get("chunks", []) or []:
                if not isinstance(chunk, dict):
                    continue
                flattened.append({
                    "step_name": step_name,
                    "chunk_index": chunk.get("chunk_index"),
                    "chunk_total": chunk.get("chunk_total"),
                    "chunk_token_count": chunk.get("chunk_token_count"),
                    "token_keywords": chunk.get("token_keywords") or [],
                    "summary": chunk.get("summary") or "",
                    "relevant_failures": chunk.get("relevant_failures") or [],
                    "relevant_files": chunk.get("relevant_files") or [],
                    "code_context": chunk.get("code_context") or [],
                    "chunk_document": chunk.get("chunk_document") or "",
                })
        return flattened

    def _build_analysis_document(
        self,
        generated_summary: Dict[str, Any],
        selected_logs: List[Dict[str, Any]],
        log_details: List[Dict[str, Any]],
    ) -> str:
        lines = []
        overall_ci_summary = str(generated_summary.get("overall_ci_summary") or "").strip()
        if overall_ci_summary:
            lines.append(f"overall_ci_summary: {overall_ci_summary}")
        error_context = generated_summary.get("error_context") or []
        if error_context:
            lines.append("overall_error_context:")
            lines.extend(f"- {str(item).strip()}" for item in error_context[:6] if str(item).strip())
        error_types = generated_summary.get("error_types") or []
        if error_types:
            lines.append("error_types:")
            for item in error_types[:10]:
                if not isinstance(item, dict):
                    continue
                category = str(item.get("category") or "").strip()
                subcategory = str(item.get("subcategory") or "").strip()
                evidence = str(item.get("evidence") or "").strip()
                row = " | ".join(x for x in [category, subcategory, evidence] if x)
                if row:
                    lines.append(f"- {row}")
        failed_jobs = generated_summary.get("failed_job") or generated_summary.get("failed_jobs") or []
        if failed_jobs:
            lines.append("failed_jobs:")
            for item in failed_jobs[:10]:
                if not isinstance(item, dict):
                    continue
                job = str(item.get("job") or "").strip()
                step = str(item.get("step") or "").strip()
                command = str(item.get("command") or "").strip()
                row = " | ".join(x for x in [job, step, command] if x)
                if row:
                    lines.append(f"- {row}")
        relevant_files = generated_summary.get("relevant_files") or []
        if relevant_files:
            lines.append("relevant_files:")
            for item in relevant_files[:12]:
                if not isinstance(item, dict):
                    continue
                file_path = str(item.get("file") or "").strip()
                line_number = str(item.get("line_number") or "").strip()
                reason = str(item.get("reason") or "").strip()
                row = " | ".join(x for x in [file_path, line_number, reason] if x)
                if row:
                    lines.append(f"- {row}")
        for step in log_details:
            step_name = str(step.get("step_name") or "unknown_step")
            log_content = str(step.get("log_content") or "").strip()
            if log_content:
                lines.append(f"step_log_content[{step_name}]: {log_content}")
        for step in selected_logs:
            step_document = str(step.get("step_document") or "").strip()
            if step_document:
                lines.append(step_document)
        return "\n".join(lines).strip()

    def _build_overall_ci_summary(
        self,
        generated_summary: Dict[str, Any],
        log_details: List[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        for step in log_details:
            step_name = str(step.get("step_name") or "unknown_step").strip()
            log_content = str(step.get("log_content") or "").strip()
            if step_name and log_content:
                parts.append(f"{step_name}: {log_content}")
        if parts:
            return " | ".join(parts)
        error_context = [
            str(item).strip()
            for item in (generated_summary.get("error_context") or [])
            if str(item).strip()
        ]
        return " | ".join(error_context)


# ─────────────────────────────────────────────────────────────────────────────
# CILogAnalyzer — public adapter used by ci_context.py
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_logs(logs: Any) -> List[Dict[str, Any]]:
    """
    Normalise raw CI logs to the ``[{step_name, log}]`` format that
    ``CILogAnalyzerLLM`` expects.
    """
    if isinstance(logs, list):
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(logs):
            if isinstance(item, dict):
                step_name = str(
                    item.get("step_name") or item.get("name") or f"step_{i + 1}"
                )
                log_text = str(
                    item.get("log") or item.get("text") or item.get("output") or ""
                )
            else:
                step_name = f"step_{i + 1}"
                log_text  = str(item)
            out.append({"step_name": step_name, "log": log_text})
        return out

    # Plain string — treat as a single step
    return [{"step_name": "ci_log", "log": str(logs or "")}]


def _make_langchain_llm(llm: Any) -> Any:
    """
    Return a LangChain-compatible LLM object from *llm*.

    Supports:
      • Already LangChain (has ``invoke([HumanMessage(...)])``): returned as-is.
      • Callable / litellm-style (``llm(prompt) -> str``): wrapped in a thin
        shim so ``CILogAnalyzerLLM`` can call ``.invoke([HumanMessage(…)])``.
    """
    if llm is None:
        raise ValueError("llm must not be None for CILogAnalyzerLLM")

    # Test whether it already speaks LangChain
    try:
        from langchain_core.messages import HumanMessage as _HM  # type: ignore
        if callable(getattr(llm, "invoke", None)):
            return llm
    except ImportError:
        pass

    # Wrap a plain callable
    class _LLMShim:
        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def invoke(self, messages: Any) -> Any:
            # Extract text from a list of messages or pass through a plain string
            if isinstance(messages, list):
                content = " ".join(
                    getattr(m, "content", str(m)) for m in messages
                )
            else:
                content = str(messages)
            result = self._fn(content)
            # Return an object with a .content attribute
            class _Resp:
                def __init__(self, c: str) -> None:
                    self.content = c
            return _Resp(str(result) if result is not None else "")

    return _LLMShim(llm)


class CILogAnalyzer:
    """
    Public adapter for ``CILogAnalyzerLLM``, used by ``ci_context.py``.

    Accepts the interface that ``ci_context._run_log_analysis()`` uses:

        analyzer = CILogAnalyzer(
            logs=logs,           # str | list[{step_name, log}]
            sha_fail=sha_fail,
            workflow=workflow,
            workflow_path=workflow_path,
            llm=llm,             # LangChain ChatModel | callable | None
            model_name=model,
            task_id=task_id,
        )
        result = analyzer.run()  # -> same schema as CILogAnalyzerLLM.run()

    Raises ``RuntimeError`` when ``llm`` is ``None`` so that the caller's
    fallback (``_minimal_log_analysis``) is triggered automatically.
    """

    def __init__(
        self,
        *,
        logs: Any,
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: Any,
        model_name: str,
        task_id: str,
    ) -> None:
        if llm is None:
            raise RuntimeError(
                "CILogAnalyzer requires an LLM; llm=None -> using fallback analysis."
            )

        ci_log = _normalize_logs(logs)
        lc_llm = _make_langchain_llm(llm)

        self._inner = CILogAnalyzerLLM(
            repo_path=".",
            ci_log=ci_log,
            sha_fail=sha_fail,
            workflow=workflow,
            workflow_path=workflow_path,
            llm=lc_llm,
            model_name=model_name,
            task_id=task_id,
        )

    def run(self) -> Dict[str, Any]:
        return self._inner.run()
