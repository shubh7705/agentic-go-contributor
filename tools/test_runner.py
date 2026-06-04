"""
tools/test_runner.py — Go test and lint execution.

Runs:
  - go test ./...
  - golangci-lint run (if available)

Captures stdout, stderr, exit codes, and structured failure summaries.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Outcome of a `go test` run."""

    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    failed_tests: list[str] = field(default_factory=list)
    error_summary: str = ""


@dataclass
class LintResult:
    """Outcome of a `golangci-lint` run."""

    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    issues: list[str] = field(default_factory=list)
    available: bool = True   # False if golangci-lint is not installed


@dataclass
class ValidationResult:
    """Combined result of tests + lint."""

    tests: TestResult
    lint: Optional[LintResult]

    @property
    def passed(self) -> bool:
        lint_ok = self.lint.passed if self.lint else True
        return self.tests.passed and lint_ok

    def summary(self) -> str:
        lines = []
        status = "✅ PASSED" if self.tests.passed else "❌ FAILED"
        lines.append(f"Tests: {status} (exit code {self.tests.exit_code})")
        if self.tests.failed_tests:
            lines.append("Failed tests:")
            for t in self.tests.failed_tests:
                lines.append(f"  - {t}")
        if self.lint:
            lint_status = "✅ PASSED" if self.lint.passed else "❌ FAILED"
            lines.append(f"Lint:  {lint_status} (exit code {self.lint.exit_code})")
            if self.lint.issues:
                lines.append("Lint issues:")
                for issue in self.lint.issues[:20]:
                    lines.append(f"  - {issue}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_tests(
    repo_path: Path,
    *,
    timeout: int = 300,
    packages: str = "./...",
    extra_flags: Optional[list[str]] = None,
) -> TestResult:
    """
    Run `go test` in the repository.

    Args:
        repo_path:    Repository root.
        timeout:      Seconds before killing the process (default: 5 min).
        packages:     Package pattern (default: ``./...``).
        extra_flags:  Additional flags to pass to ``go test``.

    Returns:
        TestResult with pass/fail status and captured output.
    """
    cmd = ["go", "test", "-v", "-count=1", packages]
    if extra_flags:
        cmd.extend(extra_flags)

    logger.info("Running: %s (cwd=%s)", " ".join(cmd), repo_path)

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        logger.error("go test timed out after %ds", timeout)
        return TestResult(
            passed=False,
            stdout="",
            stderr=f"go test timed out after {timeout}s",
            exit_code=-1,
            error_summary="Test execution timed out",
        )
    except FileNotFoundError:
        return TestResult(
            passed=False,
            stdout="",
            stderr="'go' binary not found on PATH",
            exit_code=-1,
            error_summary="Go is not installed or not on PATH",
        )

    failed = _extract_failed_tests(result.stdout + result.stderr)
    error_summary = _extract_error_summary(result.stderr + result.stdout)

    return TestResult(
        passed=result.returncode == 0,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.returncode,
        failed_tests=failed,
        error_summary=error_summary,
    )


def run_lint(
    repo_path: Path,
    *,
    timeout: int = 120,
) -> LintResult:
    """
    Run `golangci-lint run` in the repository.

    If golangci-lint is not installed, returns a result with ``available=False``.

    Args:
        repo_path: Repository root.
        timeout:   Seconds before killing the process (default: 2 min).

    Returns:
        LintResult with pass/fail status and issue list.
    """
    if not shutil.which("golangci-lint"):
        logger.warning("golangci-lint not found on PATH — skipping lint")
        return LintResult(
            passed=True,
            stdout="",
            stderr="",
            exit_code=0,
            available=False,
        )

    cmd = ["golangci-lint", "run", "--out-format", "line-number", "./..."]
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), repo_path)

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        logger.warning("golangci-lint timed out after %ds", timeout)
        return LintResult(
            passed=False,
            stdout="",
            stderr=f"golangci-lint timed out after {timeout}s",
            exit_code=-1,
        )

    issues = [line for line in result.stdout.splitlines() if line.strip()]
    return LintResult(
        passed=result.returncode == 0,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.returncode,
        issues=issues,
    )


def validate(
    repo_path: Path,
    *,
    run_lint_check: bool = True,
    test_timeout: int = 300,
    lint_timeout: int = 120,
) -> ValidationResult:
    """
    Run full validation: tests + (optional) lint.

    Args:
        repo_path:       Repository root.
        run_lint_check:  Whether to run golangci-lint (default: True).
        test_timeout:    Timeout for ``go test`` in seconds.
        lint_timeout:    Timeout for ``golangci-lint`` in seconds.

    Returns:
        ValidationResult combining both outcomes.
    """
    test_result = run_tests(repo_path, timeout=test_timeout)
    lint_result = run_lint(repo_path, timeout=lint_timeout) if run_lint_check else None
    return ValidationResult(tests=test_result, lint=lint_result)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FAIL_RE = re.compile(r"^--- FAIL:\s+(\S+)", re.MULTILINE)
_COMPILE_ERROR_RE = re.compile(
    r"^(.*\.go:\d+:\d+:.*)", re.MULTILINE
)


def _extract_failed_tests(output: str) -> list[str]:
    return _FAIL_RE.findall(output)


def _extract_error_summary(output: str, max_lines: int = 30) -> str:
    """Extract the most relevant error lines from test / compiler output."""
    errors = _COMPILE_ERROR_RE.findall(output)
    if errors:
        return "\n".join(errors[:max_lines])
    # Fallback: last N lines of stderr
    lines = [l for l in output.splitlines() if l.strip()]
    return "\n".join(lines[-max_lines:])
