"""
tests/test_agents.py — Unit tests for agent nodes.

All external dependencies (LLM, GitHub, git, filesystem) are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Issue Agent tests
# ---------------------------------------------------------------------------

class TestIssueAgent:
    @patch("agents.issue_agent.fetch_issue")
    @patch("agents.issue_agent.chat_completion")
    def test_run_issue_agent_success(self, mock_llm, mock_fetch):
        from agents.issue_agent import run_issue_agent
        from tools.github import IssueData

        mock_fetch.return_value = IssueData(
            number=42,
            title="Fix JSON validation",
            body="JSON binding fails when field is missing.",
            labels=["bug"],
            state="open",
            url="https://github.com/gin-gonic/gin/issues/42",
            repo_full_name="gin-gonic/gin",
            repo_url="https://github.com/gin-gonic/gin.git",
            comments=[],
        )
        mock_llm.return_value = json.dumps({
            "summary": "JSON binding breaks.",
            "root_problem": "Missing field validation.",
            "acceptance_criteria": ["Return 400 on missing field"],
            "affected_areas": ["binding"],
            "keywords": ["json", "binding", "validation"],
        })

        state = {
            "issue_url": "https://github.com/gin-gonic/gin/issues/42",
            "logs": [],
        }
        result = run_issue_agent(state)

        assert result["issue_title"] == "Fix JSON validation"
        assert "enriched" or "JSON binding" in result["issue_body"]
        assert isinstance(result["logs"], list)
        assert len(result["logs"]) > 0

    @patch("agents.issue_agent.fetch_issue")
    def test_run_issue_agent_fetch_failure(self, mock_fetch):
        from agents.issue_agent import run_issue_agent

        mock_fetch.side_effect = RuntimeError("GitHub rate limit")

        state = {
            "issue_url": "https://github.com/gin-gonic/gin/issues/42",
            "logs": [],
        }
        result = run_issue_agent(state)
        assert any("ERROR" in log for log in result["logs"])


# ---------------------------------------------------------------------------
# Validation Agent tests
# ---------------------------------------------------------------------------

class TestValidationAgent:
    @patch("agents.validation_agent.validate")
    def test_validation_passed(self, mock_validate):
        from agents.validation_agent import run_validation_agent
        from tools.test_runner import ValidationResult, TestResult, LintResult

        mock_validate.return_value = ValidationResult(
            tests=TestResult(passed=True, stdout="ok", stderr="", exit_code=0),
            lint=LintResult(passed=True, stdout="", stderr="", exit_code=0),
        )

        state = {
            "repo_path": "/fake/repo",
            "logs": [],
        }
        result = run_validation_agent(state)
        assert result["validation_passed"] is True

    @patch("agents.validation_agent.validate")
    def test_validation_failed(self, mock_validate):
        from agents.validation_agent import run_validation_agent
        from tools.test_runner import ValidationResult, TestResult, LintResult

        mock_validate.return_value = ValidationResult(
            tests=TestResult(
                passed=False,
                stdout="--- FAIL: TestFoo (0.01s)",
                stderr="./foo.go:10:5: undefined: Bar",
                exit_code=1,
                failed_tests=["TestFoo"],
            ),
            lint=LintResult(passed=True, stdout="", stderr="", exit_code=0),
        )

        state = {
            "repo_path": "/fake/repo",
            "logs": [],
        }
        result = run_validation_agent(state)
        assert result["validation_passed"] is False
        assert "TestFoo" in result["test_results"]


# ---------------------------------------------------------------------------
# Repair Agent tests
# ---------------------------------------------------------------------------

class TestRepairAgent:
    @patch("agents.repair_agent.chat_completion")
    @patch("agents.repair_agent.apply_proposed_changes")
    def test_repair_extracts_corrected_files(self, mock_apply, mock_llm):
        from agents.repair_agent import run_repair_agent

        mock_llm.return_value = textwrap_dedent_helper("""\
            // File: binding/json.go
            ```go
            package binding

            func fixed() {}
            ```

            ## Repair Notes
            Fixed missing import.
        """)
        mock_apply.return_value = {"binding/json.go": "+func fixed() {}"}

        state = {
            "repo_path": "/fake/repo",
            "implementation_plan": "Fix JSON binding",
            "proposed_changes": {"binding/json.go": "package binding\n"},
            "test_results": "FAIL: TestJSON",
            "lint_results": "",
            "logs": [],
        }
        result = run_repair_agent(state)
        assert "binding/json.go" in result["proposed_changes"]

    def test_repair_counts_iterations(self):
        from agents.repair_agent import count_repair_iterations

        logs = [
            "[RepairAgent] Repair iteration 1/3",
            "[RepairAgent] Repair iteration 2/3",
        ]
        assert count_repair_iterations(logs) == 2


# ---------------------------------------------------------------------------
# PR Agent tests
# ---------------------------------------------------------------------------

class TestPRAgent:
    @patch("agents.pr_agent.chat_completion")
    def test_pr_agent_generates_title_and_body(self, mock_llm):
        from agents.pr_agent import run_pr_agent

        mock_llm.return_value = json.dumps({
            "title": "fix(binding): improve JSON validation error handling",
            "body": "## Summary\nFixed the issue.\n\n## Changes\n- binding/json.go\n\n## Testing\ngo test passed.\n\n## Notes\nNone.",
        })

        state = {
            "issue_title": "Fix JSON validation",
            "issue_body": "JSON binding fails.",
            "implementation_plan": "1. Fix decodeJSON",
            "proposed_changes": {"binding/json.go": "package binding\n"},
            "test_results": "PASS",
            "lint_results": "",
            "validation_passed": True,
            "repository_map": {},
            "logs": [],
        }
        result = run_pr_agent(state)
        assert result["pr_title"] == "fix(binding): improve JSON validation error handling"
        assert "## Summary" in result["pr_body"]


# ---------------------------------------------------------------------------
# Workflow routing tests
# ---------------------------------------------------------------------------

class TestWorkflowRouting:
    def test_route_passes_when_validation_passed(self):
        from graph.workflow import _route_after_validation

        state = {"validation_passed": True, "logs": []}
        assert _route_after_validation(state) == "pr_agent"

    def test_route_repairs_when_failed_and_below_max(self):
        from graph.workflow import _route_after_validation

        state = {"validation_passed": False, "logs": []}
        assert _route_after_validation(state) == "repair_agent"

    def test_route_pr_when_max_iterations_reached(self):
        from graph.workflow import _route_after_validation
        from config.settings import settings

        logs = [
            f"[RepairAgent] Repair iteration {i+1}/{settings.max_repair_iterations}"
            for i in range(settings.max_repair_iterations)
        ]
        state = {"validation_passed": False, "logs": logs}
        assert _route_after_validation(state) == "pr_agent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def textwrap_dedent_helper(text: str) -> str:
    import textwrap
    return textwrap.dedent(text)
