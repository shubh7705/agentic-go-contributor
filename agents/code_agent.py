"""
agents/code_agent.py — Code generation agent.

Responsibilities:
  - Read the implementation plan and relevant file contents
  - Generate minimal, style-preserving diffs for each file to modify
  - Output complete modified file contents
  - Follow Go conventions (gofmt style, package naming, error handling patterns)

Output keys written: proposed_changes, logs
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from graph.state import AgentState
from services.llm import chat_completion, build_messages
from tools.file_editor import extract_code_blocks, extract_diff_blocks, apply_proposed_changes
from tools.git import read_file

logger = logging.getLogger(__name__)

# Rough token budget: keep each file under this many characters (~4 chars/token)
# to stay comfortably inside free-tier limits (38k tokens total budget).
_MAX_FILE_CHARS = 6_000   # ≈1500 tokens per file
_MAX_TOTAL_CHARS = 20_000  # hard cap for the entire file-context section


def _truncate_file_for_context(path: str, content: str, plan: str) -> str:
    """Return a token-budget-friendly excerpt of *content*.

    Strategy:
    1. If the file is short (≤ 200 lines) return it whole.
    2. Otherwise extract: import block + the function/method most
       relevant to the plan (±75 lines around the first match).
    3. Always append an ellipsis comment so the LLM knows the file
       was trimmed.
    """
    lines = content.splitlines()
    if len(lines) <= 200:
        return content

    # Build a search pattern from function-like keywords in the plan.
    # Pull bare identifiers that look like Go function names (CamelCase).
    identifiers = re.findall(r'\b([A-Z][A-Za-z0-9]+)\b', plan)
    # Also grab lowercase func names from the plan
    identifiers += re.findall(r'\bfunc\s+([a-zA-Z0-9]+)', plan)
    identifiers = list(dict.fromkeys(identifiers))  # deduplicate, preserve order

    # Find the import block (lines starting with "import")
    import_start, import_end = 0, 0
    in_import = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ('):
            import_start = i
            in_import = True
        elif in_import and stripped == ')':
            import_end = i + 1
            break
        elif stripped.startswith('import "') and not in_import:
            import_start = i
            import_end = i + 1

    import_block = lines[import_start:import_end]

    # Find the best matching function window.
    best_center = None
    for ident in identifiers:
        pattern = re.compile(rf'func.*{re.escape(ident)}')
        for i, line in enumerate(lines):
            if pattern.search(line):
                best_center = i
                break
        if best_center is not None:
            break

    if best_center is None:
        # No specific function found — return first 150 lines + import block
        excerpt = lines[:max(import_end, 20)] + ['', '// ... (truncated for token budget) ...', ''] + lines[import_end:import_end + 120]
    else:
        window_start = max(import_end, best_center - 75)
        window_end = min(len(lines), best_center + 75)
        excerpt = import_block + ['', '// ... (truncated for token budget) ...', ''] + lines[window_start:window_end]

    result = '\n'.join(excerpt)
    # Final safety clamp
    if len(result) > _MAX_FILE_CHARS:
        result = result[:_MAX_FILE_CHARS] + '\n// ... (further truncated)'
    return result

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Staff Go Engineer implementing a bug fix / feature based on an implementation plan.

Rules:
1. Follow idiomatic Go conventions (gofmt formatting, named returns sparingly, error wrapping).
2. Produce MINIMAL diffs — only change what is necessary.
3. Preserve all existing comments, copyright headers, and code style.
4. For each file you modify or create, output the COMPLETE new file content.
5. Format each file as:

   // File: path/to/file.go
   ```go
   <complete file content>
   ```

6. After all files, output a unified diff block:

   ```diff
   <unified diff>
   ```

7. Do NOT add unnecessary imports or change unrelated code.
8. Ensure all new exported symbols have godoc comments.
9. Handle errors explicitly — no panic() unless absolutely justified.
10. If you create test files, use the _test suffix and table-driven tests.
"""


def _build_user_prompt(
    issue_title: str,
    issue_body: str,
    implementation_plan: str,
    file_contexts: list[tuple[str, str]],
) -> str:
    # Apply per-file truncation and a hard total cap
    trimmed_sections: list[str] = []
    total_chars = 0
    for path, content in file_contexts:
        excerpt = _truncate_file_for_context(path, content, implementation_plan)
        section = f"### {path}\n```go\n{excerpt}\n```"
        if total_chars + len(section) > _MAX_TOTAL_CHARS:
            trimmed_sections.append(f"### {path}\n(omitted — total context budget reached)")
        else:
            trimmed_sections.append(section)
            total_chars += len(section)

    file_section = "\n\n".join(trimmed_sections)
    return f"""\
## Issue
**Title:** {issue_title}

**Body (truncated):**
{issue_body[:800]}

## Implementation Plan
{implementation_plan[:3000]}

## Current File Contents (excerpts)
{file_section}

Implement the changes according to the plan.
Output each modified file completely using:
// File: <filename>
```go
<complete file content>
```
Then output a unified diff block.
"""


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_code_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Code Agent.

    Generates code changes based on the implementation plan and writes them
    to the repository files.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates (``proposed_changes``).
    """
    logs: list[str] = list(state.get("logs", []))
    logs.append("[CodeAgent] Generating code changes …")

    repo_path = Path(state["repo_path"])
    issue_title: str = state.get("issue_title", "")
    issue_body: str = state.get("issue_body", "")
    plan: str = state.get("implementation_plan", "")
    retrieved_files: list[str] = state.get("retrieved_files", [])

    if not plan:
        logs.append("[CodeAgent] ERROR: no implementation plan in state")
        return {"logs": logs}

    # ── 1. Read file contents ─────────────────────────────────────────────────
    file_contexts: list[tuple[str, str]] = []
    for rel_path in retrieved_files[:8]:
        try:
            content = read_file(repo_path, rel_path)
            file_contexts.append((rel_path, content))
        except Exception as exc:
            logger.warning("CodeAgent: could not read %s: %s", rel_path, exc)

    logs.append(f"[CodeAgent] Context: {len(file_contexts)} files loaded")

    # ── 2. Ask LLM to generate changes ───────────────────────────────────────
    logger.info("CodeAgent: requesting code generation from LLM")
    user_prompt = _build_user_prompt(issue_title, issue_body, plan, file_contexts)
    estimated_tokens = len(user_prompt) // 4
    logs.append(f"[CodeAgent] Estimated prompt tokens: ~{estimated_tokens}")
    try:
        messages = build_messages(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
        )
        llm_response = chat_completion(messages, temperature=0.05, max_tokens=4096)
    except Exception as exc:
        logger.error("CodeAgent: LLM call failed: %s", exc)
        logs.append(f"[CodeAgent] ERROR: LLM call failed: {exc}")
        return {"logs": logs}

    # ── 3. Parse proposed changes ─────────────────────────────────────────────
    proposed_changes: dict[str, str] = extract_code_blocks(llm_response)
    diff_text = extract_diff_blocks(llm_response)

    if not proposed_changes:
        # Fallback: try parsing any fenced go blocks with the file context
        logger.warning("CodeAgent: no file blocks parsed from LLM output — checking diff")
        if diff_text:
            logs.append("[CodeAgent] WARNING: only diff found, no complete file contents")
        else:
            logs.append("[CodeAgent] ERROR: could not extract any code changes from LLM output")
            logs.append(f"[CodeAgent] LLM output (first 500 chars):\n{llm_response[:500]}")
            return {"logs": logs}

    logs.append(
        f"[CodeAgent] Parsed {len(proposed_changes)} file(s): "
        + ", ".join(proposed_changes.keys())
    )

    # ── 4. Apply changes to the repository ───────────────────────────────────
    try:
        diffs = apply_proposed_changes(repo_path, proposed_changes)
        for file_path, diff in diffs.items():
            if diff.startswith("ERROR"):
                logs.append(f"[CodeAgent] WRITE ERROR {file_path}: {diff}")
            else:
                added = diff.count("\n+") - diff.count("\n+++")
                removed = diff.count("\n-") - diff.count("\n---")
                logs.append(f"[CodeAgent] Modified {file_path} (+{added}/-{removed} lines)")
    except Exception as exc:
        logger.error("CodeAgent: failed to apply changes: %s", exc)
        logs.append(f"[CodeAgent] ERROR applying changes: {exc}")

    logger.info("CodeAgent: %d files modified", len(proposed_changes))

    return {
        "proposed_changes": proposed_changes,
        "logs": logs,
    }
