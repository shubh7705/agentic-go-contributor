"""
agents/pr_agent.py — Pull Request generation agent.

Responsibilities:
  - Summarise everything done by previous agents
  - Generate a conventional-commit PR title
  - Generate a rich PR description with:
    ## Summary
    ## Changes
    ## Testing
    ## Notes

Output keys written: pr_title, pr_body, logs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from graph.state import AgentState
from services.llm import chat_completion, build_messages
from tools.file_editor import generate_diff
from tools.git import read_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior engineer writing a GitHub Pull Request.

Generate a PR title and description for the provided change.

PR Title rules:
- Follow Conventional Commits: <type>(<scope>): <short description>
- Types: fix, feat, refactor, test, docs, chore, perf
- Scope: the Go package name most affected (e.g., binding, render, router)
- ≤72 characters
- lowercase, imperative mood

PR Description must use this exact structure:

## Summary
[2–4 sentence plain-English explanation of what was changed and why]

## Changes
[Bullet list of specific code changes, grouped by file]

## Testing
[How the change was validated — test commands, test cases added]

## Notes
[Any breaking changes, limitations, or follow-up work needed]

Return a JSON object:
{
  "title": "<conventional commit title>",
  "body": "<full markdown body>"
}
"""


def _build_user_prompt(
    issue_title: str,
    issue_body: str,
    implementation_plan: str,
    proposed_changes: dict[str, str],
    test_results: str,
    validation_passed: bool,
    repo_map: dict,
) -> str:
    # Build a compact change summary
    files_changed = list(proposed_changes.keys())
    change_summary = "\n".join(f"- {f}" for f in files_changed)

    # Detect dominant package from repo map
    package_counts: dict[str, int] = {}
    for fp in files_changed:
        meta = repo_map.get(fp, {})
        pkg = meta.get("package", "")
        if pkg:
            package_counts[pkg] = package_counts.get(pkg, 0) + 1
    dominant_pkg = max(package_counts, key=package_counts.get) if package_counts else "core"

    return f"""\
## GitHub Issue
**Title:** {issue_title}
**Body (truncated):**
{issue_body[:1000]}

## Implementation Plan
{implementation_plan[:2000]}

## Files Modified
{change_summary}

## Dominant Package
{dominant_pkg}

## Validation
Tests passed: {validation_passed}
{test_results[:500]}

Generate the PR title and body as JSON.
"""


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_pr_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: PR Agent.

    Generates a conventional-commit PR title and a structured markdown
    PR description based on everything the previous agents produced.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates (``pr_title``, ``pr_body``).
    """
    logs: list[str] = list(state.get("logs", []))
    logs.append("[PRAgent] Generating PR title and description …")

    issue_title: str = state.get("issue_title", "")
    issue_body: str = state.get("issue_body", "")
    plan: str = state.get("implementation_plan", "")
    proposed_changes: dict[str, str] = state.get("proposed_changes", {})
    test_results: str = state.get("test_results", "")
    lint_results: str = state.get("lint_results", "")
    validation_passed: bool = state.get("validation_passed", False)
    repo_map: dict = state.get("repository_map", {})

    # ── LLM call ─────────────────────────────────────────────────────────────
    try:
        messages = build_messages(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(
                issue_title,
                issue_body,
                plan,
                proposed_changes,
                test_results,
                validation_passed,
                repo_map,
            ),
        )
        raw = chat_completion(messages, json_mode=True, temperature=0.2)
        result = json.loads(raw)
        pr_title: str = result.get("title", "fix: resolve issue")
        pr_body: str = result.get("body", "")
    except Exception as exc:
        logger.error("PRAgent: LLM call failed: %s", exc)
        logs.append(f"[PRAgent] ERROR: {exc}")
        # Fallback minimal PR
        pr_title = f"fix: {issue_title[:60]}"
        pr_body = _fallback_pr_body(issue_title, proposed_changes, validation_passed)

    # ── Append validation status to body ─────────────────────────────────────
    status_badge = "✅ All tests passing" if validation_passed else "⚠️ Tests may not pass — see validation notes"
    pr_body = pr_body.rstrip() + f"\n\n---\n_{status_badge}_\n"

    logs.append(f"[PRAgent] Title: {pr_title}")
    logs.append(f"[PRAgent] Body ({len(pr_body)} chars) generated")
    logger.info("PRAgent: complete. Title=%r", pr_title)

    return {
        "pr_title": pr_title,
        "pr_body": pr_body,
        "logs": logs,
    }


def _fallback_pr_body(
    issue_title: str,
    proposed_changes: dict[str, str],
    validation_passed: bool,
) -> str:
    files = "\n".join(f"- `{f}`" for f in proposed_changes)
    return f"""\
## Summary
This PR addresses the issue: {issue_title}

## Changes
{files or "_(no files detected)_"}

## Testing
Validation passed: {validation_passed}

## Notes
This PR was generated automatically by the Agentic Go Contributor.
"""
