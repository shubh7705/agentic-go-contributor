"""
agents/issue_agent.py — GitHub Issue parsing agent.

Responsibilities:
  1. Fetch the GitHub issue via the GitHub API
  2. Extract title, body, labels, and comments
  3. Ask the LLM to extract structured acceptance criteria
  4. Store everything in AgentState

Output keys written: issue_title, issue_body, logs
"""

from __future__ import annotations

import logging
from typing import Any

from tools.github import fetch_issue, IssueData
from services.llm import chat_completion, build_messages
from graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior software engineer analysing a GitHub issue.
Your goal is to extract a clean, structured summary that will guide an AI coding agent.

Extract and return a JSON object with these keys:
- "summary":              A 2–3 sentence plain-English summary of the issue.
- "root_problem":         The core technical problem in one sentence.
- "acceptance_criteria":  A list of strings, one per acceptance criterion.
- "affected_areas":       A list of likely affected code areas / packages.
- "keywords":             A list of important technical keywords for search.
"""


def _build_user_prompt(issue: IssueData) -> str:
    comments_section = ""
    if issue.comments:
        comments_section = "\n\n## Top Comments\n" + "\n---\n".join(issue.comments[:5])

    return f"""\
## Issue #{issue.number}: {issue.title}

**Labels:** {", ".join(issue.labels) or "none"}
**State:** {issue.state}
**URL:** {issue.url}

## Body
{issue.body or "(no body)"}
{comments_section}

Return valid JSON only. No markdown fences.
"""


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_issue_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Issue Agent.

    Reads ``state["issue_url"]`` and populates ``issue_title``, ``issue_body``,
    and appends structured analysis to ``logs``.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates.
    """
    issue_url: str = state["issue_url"]
    logs: list[str] = list(state.get("logs", []))

    logs.append(f"[IssueAgent] Fetching issue: {issue_url}")
    logger.info("IssueAgent: fetching %s", issue_url)

    # ── 1. Fetch from GitHub ─────────────────────────────────────────────────
    try:
        issue_data = fetch_issue(issue_url)
    except Exception as exc:
        logger.error("IssueAgent: fetch failed: %s", exc)
        logs.append(f"[IssueAgent] ERROR: {exc}")
        return {"logs": logs}

    logs.append(f"[IssueAgent] Fetched issue #{issue_data.number}: {issue_data.title}")

    # ── 2. LLM structured extraction ─────────────────────────────────────────
    try:
        messages = build_messages(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(issue_data),
        )
        analysis_raw = chat_completion(messages, json_mode=True)
        import json
        analysis: dict[str, Any] = json.loads(analysis_raw)
    except Exception as exc:
        logger.warning("IssueAgent: LLM analysis failed (%s) — using raw body", exc)
        analysis = {
            "summary": issue_data.body[:300],
            "root_problem": issue_data.title,
            "acceptance_criteria": [],
            "affected_areas": [],
            "keywords": [],
        }

    logs.append(f"[IssueAgent] Root problem: {analysis.get('root_problem', '')}")
    if analysis.get("acceptance_criteria"):
        criteria_str = "\n  ".join(analysis["acceptance_criteria"])
        logs.append(f"[IssueAgent] Acceptance criteria:\n  {criteria_str}")

    # ── 3. Enrich issue body with analysis ───────────────────────────────────
    enriched_body = (
        f"{issue_data.body}\n\n"
        f"---\n"
        f"**Root Problem:** {analysis.get('root_problem', '')}\n\n"
        f"**Acceptance Criteria:**\n"
        + "\n".join(f"- {c}" for c in analysis.get("acceptance_criteria", []))
    )

    # Store keyword extraction in logs for Retriever Agent to use later
    keywords = analysis.get("keywords", [])
    if keywords:
        logs.append(f"[IssueAgent] Keywords: {', '.join(keywords)}")

    # Save keywords and affected areas in the body for downstream agents
    state_update: dict[str, Any] = {
        "issue_title": issue_data.title,
        "issue_body": enriched_body,
        "logs": logs,
    }

    # Stash extra analysis metadata in logs as a JSON marker (picked up by retriever)
    import json
    logs.append(f"[IssueAgent] ANALYSIS_JSON:{json.dumps(analysis)}")

    logger.info("IssueAgent: complete. Title=%r", issue_data.title)
    return state_update
