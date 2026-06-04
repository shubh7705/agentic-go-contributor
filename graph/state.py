"""
graph/state.py — LangGraph shared state definition.

AgentState is the single TypedDict that flows through every node in the graph.
All agents read from and write to this shared object.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state for the Agentic Go Contributor LangGraph workflow.

    Agents are only required to return the keys they modify.
    All fields are Optional by convention (total=False).
    """

    # ── Input ────────────────────────────────────────────────────────────────
    issue_url: str
    """Full GitHub issue URL, e.g. https://github.com/gin-gonic/gin/issues/1234"""

    # ── Issue Agent output ───────────────────────────────────────────────────
    issue_title: str
    """Issue title extracted from GitHub."""

    issue_body: str
    """Issue body, enriched with LLM-extracted acceptance criteria."""

    # ── Repository Agent output ───────────────────────────────────────────────
    repo_path: str
    """
    Absolute path to the cloned repository on disk.
    May also temporarily hold the repository URL before cloning.
    """

    repository_map: dict[str, Any]
    """
    Parsed repository structure:
    {
        "binding/json.go": {
            "package": "binding",
            "functions": ["decodeJSON"],
            "methods": [{"receiver": "jsonBinding", "name": "Bind"}],
            "structs": ["jsonBinding"],
            "interfaces": ["Binding"],
            "imports": ["encoding/json"],
        }
    }
    """

    # ── Retriever Agent output ───────────────────────────────────────────────
    retrieved_files: list[str]
    """
    Ranked list of relevant file paths (relative to repo root),
    ordered by hybrid retrieval score.
    """

    # ── Planner Agent output ─────────────────────────────────────────────────
    implementation_plan: str
    """
    Structured markdown implementation plan:
    1. Root Cause
    2. Files to Modify
    3. Tests to Update
    4. Risks
    5. Validation Strategy
    """

    # ── Code Agent output ────────────────────────────────────────────────────
    proposed_changes: dict[str, str]
    """
    Mapping of {relative_file_path: complete_new_file_content}.
    Written to disk and tracked across repair iterations.
    """

    # ── Validation Agent output ──────────────────────────────────────────────
    test_results: str
    """Human-readable go test output / summary."""

    lint_results: str
    """Human-readable golangci-lint output / summary."""

    validation_passed: bool
    """True if both tests and lint passed (or lint was unavailable)."""

    # ── PR Agent output ───────────────────────────────────────────────────────
    pr_title: str
    """Conventional-commit style PR title."""

    pr_body: str
    """Full markdown PR description."""

    # ── Audit trail ──────────────────────────────────────────────────────────
    logs: list[str]
    """
    Append-only list of human-readable log lines.
    Each agent prefixes its lines with [AgentName].
    """
