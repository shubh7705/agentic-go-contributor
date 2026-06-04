"""
agents/repair_agent.py — Validation failure repair agent.

Responsibilities:
  - Receive failing test/lint output
  - Read the current proposed changes
  - Ask the LLM to produce a corrected patch
  - Apply the corrected patch
  - Track repair iteration count

Maximum 3 repair iterations (settings.max_repair_iterations).

Output keys written: proposed_changes, logs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.settings import settings
from graph.state import AgentState
from services.llm import chat_completion, build_messages
from tools.file_editor import extract_code_blocks, apply_proposed_changes
from tools.git import read_file
from agents.code_agent import _SYSTEM_PROMPT as _CODE_SYSTEM_PROMPT, _build_user_prompt as _build_code_prompt, _truncate_file_for_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior Go engineer performing a repair pass on a failed code change.

You will receive:
1. The original implementation plan
2. The current proposed file changes (that caused failures)
3. Test/lint failure output

Your task:
- Diagnose the root cause of the test/lint failures
- Produce corrected file contents that fix the failures
- Minimise changes — only touch what is broken
- Follow Go conventions: gofmt style, explicit error handling, no panics

For each file you modify, output the COMPLETE corrected file:

// File: path/to/file.go
```go
<complete corrected content>
```

Then explain what you changed and why in a ## Repair Notes section.
"""


def _build_user_prompt(
    implementation_plan: str,
    current_changes: dict[str, str],
    test_failures: str,
    lint_failures: str,
    iteration: int,
) -> str:
    changes_section = "\n\n".join(
        f"### {path}\n```go\n{content[:2000]}\n```" + ("\n// ...(truncated)" if len(content) > 2000 else "")
        for path, content in current_changes.items()
    )

    return f"""\
## Repair Iteration #{iteration}

## Original Implementation Plan
{implementation_plan[:2000]}

## Current Proposed Changes (failing)
{changes_section}

## Test Failures
```
{test_failures[:3000]}
```

## Lint Issues
```
{lint_failures[:1000]}
```

Diagnose and fix the failures. Output corrected files.
"""


# ---------------------------------------------------------------------------
# Repair iteration counter helper
# ---------------------------------------------------------------------------

def count_repair_iterations(logs: list[str]) -> int:
    """Count how many repair iterations have been logged."""
    return sum(1 for line in logs if "[RepairAgent] Repair iteration" in line)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_repair_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Repair Agent.

    Attempts to fix failing tests/lint by re-generating the affected files.
    Tracks iteration count; the workflow will stop after max_repair_iterations.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates (``proposed_changes``).
    """
    logs: list[str] = list(state.get("logs", []))
    iteration = count_repair_iterations(logs) + 1
    logs.append(f"[RepairAgent] Repair iteration {iteration}/{settings.max_repair_iterations}")

    repo_path = Path(state["repo_path"])
    plan: str = state.get("implementation_plan", "")
    current_changes: dict[str, str] = dict(state.get("proposed_changes", {}))
    test_failures: str = state.get("test_results", "")
    lint_failures: str = state.get("lint_results", "")

    if not current_changes:
        # Code Agent produced no changes (likely a parser failure).
        # Fall back to re-running the code generation prompt from scratch.
        logs.append(
            "[RepairAgent] No proposed changes found — re-running code generation (parser fallback)"
        )
        retrieved_files: list[str] = state.get("retrieved_files", [])
        file_contexts = []
        for rel_path in retrieved_files[:8]:
            try:
                raw = read_file(repo_path, rel_path)
                # Use smart truncation to stay within token budget
                excerpt = _truncate_file_for_context(rel_path, raw, plan)
                file_contexts.append((rel_path, excerpt))
            except Exception as exc:
                logger.warning("RepairAgent fallback: could not read %s: %s", rel_path, exc)

        issue_title: str = state.get("issue_title", "")
        issue_body: str = state.get("issue_body", "")
        try:
            messages = build_messages(
                system=_CODE_SYSTEM_PROMPT,
                user=_build_code_prompt(issue_title, issue_body, plan, file_contexts),
            )
            llm_response = chat_completion(messages, temperature=0.05, max_tokens=4096)
        except Exception as exc:
            logger.error("RepairAgent fallback: LLM call failed: %s", exc)
            logs.append(f"[RepairAgent] ERROR: fallback LLM call failed: {exc}")
            return {"logs": logs}

        current_changes = extract_code_blocks(llm_response)
        if not current_changes:
            logs.append(
                "[RepairAgent] ERROR: fallback code generation also produced no parseable blocks"
            )
            logs.append(f"[RepairAgent] LLM output (first 500 chars):\n{llm_response[:500]}")
            return {"logs": logs}

        logs.append(
            f"[RepairAgent] Fallback generated {len(current_changes)} file(s): "
            + ", ".join(current_changes.keys())
        )
        try:
            diffs = apply_proposed_changes(repo_path, current_changes)
            for file_path, diff in diffs.items():
                if diff.startswith("ERROR"):
                    logs.append(f"[RepairAgent] WRITE ERROR {file_path}: {diff}")
                else:
                    added = diff.count("\n+") - diff.count("\n+++")
                    removed = diff.count("\n-") - diff.count("\n---")
                    logs.append(f"[RepairAgent] Applied {file_path} (+{added}/-{removed})")
        except Exception as exc:
            logs.append(f"[RepairAgent] ERROR applying fallback changes: {exc}")
        return {"proposed_changes": current_changes, "logs": logs}

    # ── Ask LLM for corrected files ───────────────────────────────────────────
    logger.info("RepairAgent: requesting corrected patch (iteration %d)", iteration)
    try:
        messages = build_messages(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(
                plan, current_changes, test_failures, lint_failures, iteration
            ),
        )
        llm_response = chat_completion(messages, temperature=0.05, max_tokens=4096)
    except Exception as exc:
        logger.error("RepairAgent: LLM call failed: %s", exc)
        logs.append(f"[RepairAgent] ERROR: LLM call failed: {exc}")
        return {"logs": logs}

    # ── Parse corrected files ─────────────────────────────────────────────────
    corrected = extract_code_blocks(llm_response)

    if not corrected:
        logs.append("[RepairAgent] WARNING: LLM produced no parseable code blocks")
        return {"logs": logs}

    logs.append(
        f"[RepairAgent] Corrected files: {', '.join(corrected.keys())}"
    )

    # Merge with existing changes (corrected takes precedence)
    merged_changes = {**current_changes, **corrected}

    # ── Apply to repository ───────────────────────────────────────────────────
    try:
        diffs = apply_proposed_changes(repo_path, corrected)
        for file_path, diff in diffs.items():
            if diff.startswith("ERROR"):
                logs.append(f"[RepairAgent] WRITE ERROR {file_path}: {diff}")
            else:
                added = diff.count("\n+") - diff.count("\n+++")
                removed = diff.count("\n-") - diff.count("\n---")
                logs.append(f"[RepairAgent] Repaired {file_path} (+{added}/-{removed})")
    except Exception as exc:
        logger.error("RepairAgent: failed to apply corrected changes: %s", exc)
        logs.append(f"[RepairAgent] ERROR applying corrections: {exc}")

    # ── Extract repair notes ──────────────────────────────────────────────────
    if "## Repair Notes" in llm_response:
        notes_start = llm_response.index("## Repair Notes")
        notes = llm_response[notes_start:notes_start + 1000]
        logs.append(f"[RepairAgent] Notes:\n{notes}")

    logger.info("RepairAgent: iteration %d complete", iteration)
    return {
        "proposed_changes": merged_changes,
        "logs": logs,
    }
