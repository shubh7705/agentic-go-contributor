"""
agents/planner_agent.py — Implementation planning agent.

Responsibilities:
  Analyse:
    - Issue title and body
    - Repository map
    - Retrieved relevant files (with contents)

  Produce a structured Implementation Plan:
    1. Root Cause
    2. Files to Modify
    3. Tests to Update
    4. Risks
    5. Validation Strategy

  The planner does NOT write any code.

Output keys written: implementation_plan, logs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from graph.state import AgentState
from services.llm import chat_completion, build_messages
from tools.git import read_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Staff Software Engineer specialising in Go. 
Your task is to produce a detailed implementation plan for solving a GitHub issue.

You will be given:
- The issue title and body
- A repository map (packages, functions, types)
- The contents of the most relevant files

Your plan must be in this exact format (use markdown headers):

## 1. Root Cause
[Explain the technical root cause of the issue in 2-4 sentences]

## 2. Files to Modify
[List each file that needs to be changed, with a brief reason for each]
- `path/to/file.go` — reason

## 3. New Files to Create
[List any new files needed, or "None"]

## 4. Tests to Update
[List existing tests to update and new test cases to add]
- `path/to/file_test.go` — what to change

## 5. Implementation Steps
[Ordered, numbered implementation steps — be specific about what code to change]

## 6. Risks
[List potential risks or breaking changes]

## 7. Validation Strategy
[How to verify the fix works: specific test commands, edge cases to check]

IMPORTANT:
- Do NOT write any code. This is planning only.
- Be specific about function names, struct fields, package names.
- Reference the actual file paths and line-level details from the provided context.
"""


def _build_user_prompt(
    issue_title: str,
    issue_body: str,
    repo_map_summary: str,
    file_contexts: list[tuple[str, str]],
) -> str:
    file_section = ""
    for path, content in file_contexts:
        # Truncate very large files
        truncated = content if len(content) <= 3000 else content[:3000] + "\n... [truncated]"
        file_section += f"\n\n### {path}\n```go\n{truncated}\n```"

    return f"""\
## Issue
**Title:** {issue_title}

**Body:**
{issue_body[:3000]}

## Repository Map (summary)
```
{repo_map_summary}
```

## Relevant Files
{file_section}

Now produce the implementation plan.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_repo_map_summary(repo_map: dict[str, dict]) -> str:
    """Create a compact text representation of the repository map."""
    lines: list[str] = []
    for path, meta in sorted(repo_map.items()):
        pkg = meta.get("package", "?")
        funcs = meta.get("functions", [])
        methods = [m.get("name", "") if isinstance(m, dict) else str(m) for m in meta.get("methods", [])]
        structs = meta.get("structs", [])
        interfaces = meta.get("interfaces", [])

        parts = [f"pkg:{pkg}"]
        if funcs:
            parts.append(f"funcs:[{', '.join(funcs[:5])}{'...' if len(funcs) > 5 else ''}]")
        if methods:
            parts.append(f"methods:[{', '.join(methods[:5])}]")
        if structs:
            parts.append(f"structs:[{', '.join(structs)}]")
        if interfaces:
            parts.append(f"ifaces:[{', '.join(interfaces)}]")

        lines.append(f"{path}: {' | '.join(parts)}")
    return "\n".join(lines[:80])  # cap at 80 files to stay within context


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_planner_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Planner Agent.

    Reads the issue, repo map, and retrieved file contents, then asks
    the LLM to produce a structured implementation plan.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates (``implementation_plan``).
    """
    logs: list[str] = list(state.get("logs", []))
    logs.append("[PlannerAgent] Building implementation plan …")

    repo_path = Path(state["repo_path"])
    issue_title: str = state.get("issue_title", "")
    issue_body: str = state.get("issue_body", "")
    repo_map: dict = state.get("repository_map", {})
    retrieved_files: list[str] = state.get("retrieved_files", [])

    # ── 1. Read file contents ─────────────────────────────────────────────────
    file_contexts: list[tuple[str, str]] = []
    for rel_path in retrieved_files[:12]:  # cap at 12 files for context length
        try:
            content = read_file(repo_path, rel_path)
            file_contexts.append((rel_path, content))
        except Exception as exc:
            logger.warning("PlannerAgent: could not read %s: %s", rel_path, exc)

    logs.append(f"[PlannerAgent] Loaded {len(file_contexts)} file(s) for context")

    # ── 2. Build repository map summary ──────────────────────────────────────
    repo_summary = _build_repo_map_summary(repo_map)

    # ── 3. Call LLM ──────────────────────────────────────────────────────────
    logger.info("PlannerAgent: requesting implementation plan from LLM")
    try:
        messages = build_messages(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(issue_title, issue_body, repo_summary, file_contexts),
        )
        plan = chat_completion(messages, temperature=0.05, max_tokens=4096)
    except Exception as exc:
        logger.error("PlannerAgent: LLM call failed: %s", exc)
        logs.append(f"[PlannerAgent] ERROR: {exc}")
        return {"logs": logs}

    logs.append("[PlannerAgent] Implementation plan generated")
    logger.info("PlannerAgent: plan ready (%d chars)", len(plan))

    # Log the first 500 chars of the plan for quick inspection
    logs.append(f"[PlannerAgent] Plan preview:\n{plan[:500]}…")

    return {
        "implementation_plan": plan,
        "logs": logs,
    }
