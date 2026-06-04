"""
agents/validation_agent.py — Test and lint validation agent.

Responsibilities:
  1. Run `go test ./...` in the cloned repository
  2. Run `golangci-lint run` if available
  3. Capture stdout, stderr, exit codes
  4. Determine pass/fail and store structured results in state

Output keys written: test_results, lint_results, validation_passed, logs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from graph.state import AgentState
from tools.test_runner import validate, ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_validation_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Validation Agent.

    Runs go test + golangci-lint and records results in state.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates (``test_results``, ``lint_results``, ``validation_passed``).
    """
    logs: list[str] = list(state.get("logs", []))
    logs.append("[ValidationAgent] Running validation …")

    repo_path = Path(state["repo_path"])

    # ── Run validation ────────────────────────────────────────────────────────
    try:
        result: ValidationResult = validate(
            repo_path,
            run_lint_check=True,
            test_timeout=300,
            lint_timeout=120,
        )
    except Exception as exc:
        logger.error("ValidationAgent: unexpected error: %s", exc)
        logs.append(f"[ValidationAgent] ERROR: {exc}")
        return {
            "test_results": f"ERROR: {exc}",
            "lint_results": "",
            "validation_passed": False,
            "logs": logs,
        }

    # ── Build human-readable summaries ────────────────────────────────────────
    test_summary_parts = [
        f"Exit code: {result.tests.exit_code}",
        f"Status: {'PASSED' if result.tests.passed else 'FAILED'}",
    ]
    if result.tests.failed_tests:
        test_summary_parts.append(f"Failed tests: {', '.join(result.tests.failed_tests)}")
    if result.tests.error_summary:
        test_summary_parts.append(f"Errors:\n{result.tests.error_summary[:2000]}")
    if result.tests.stdout:
        test_summary_parts.append(f"Output (last 1000 chars):\n{result.tests.stdout[-1000:]}")

    test_results_str = "\n".join(test_summary_parts)

    lint_results_str = ""
    if result.lint:
        lint_parts = [
            f"Available: {result.lint.available}",
            f"Exit code: {result.lint.exit_code}",
            f"Status: {'PASSED' if result.lint.passed else 'FAILED'}",
        ]
        if result.lint.issues:
            lint_parts.append(f"Issues ({len(result.lint.issues)}):")
            lint_parts.extend(f"  {issue}" for issue in result.lint.issues[:30])
        lint_results_str = "\n".join(lint_parts)
    else:
        lint_results_str = "Lint not run"

    # ── Log ───────────────────────────────────────────────────────────────────
    overall = result.summary()
    logs.append(f"[ValidationAgent] Results:\n{overall}")
    logger.info(
        "ValidationAgent: tests=%s lint=%s",
        "PASS" if result.tests.passed else "FAIL",
        "PASS" if (result.lint and result.lint.passed) else "SKIP/FAIL",
    )

    return {
        "test_results": test_results_str,
        "lint_results": lint_results_str,
        "validation_passed": result.passed,
        "logs": logs,
    }
