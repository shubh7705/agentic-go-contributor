"""
tools/search.py — Ripgrep-powered symbol and text search for Go repositories.

All retrieval is performed via the `rg` (ripgrep) binary for maximum speed.

Functions:
  search_symbol()  — Find Go symbol definitions (func, type, interface, struct)
  search_text()    — Full-text search across .go files
  search_package() — Find all files belonging to a Go package

Scoring:
  Returns normalised scores in [0.0, 1.0] suitable for hybrid retrieval.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SearchMatch:
    """A single ripgrep match."""

    file_path: str       # Relative path within the repository
    line_number: int
    line_content: str
    score: float = 1.0   # Relevance score (normalised)


@dataclass
class SearchResults:
    """Aggregated results from a ripgrep query."""

    query: str
    matches: list[SearchMatch] = field(default_factory=list)
    file_scores: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def matched_files(self) -> list[str]:
        return list(self.file_scores.keys())


# ---------------------------------------------------------------------------
# Ripgrep runner
# ---------------------------------------------------------------------------

def _find_rg() -> str:
    """Locate the ripgrep binary, raising RuntimeError if not found."""
    # Common locations on Windows and Unix
    candidates = ["rg", "rg.exe"]
    for candidate in candidates:
        result = subprocess.run(
            ["where" if sys.platform == "win32" else "which", candidate],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
    raise RuntimeError(
        "ripgrep (rg) is not installed or not on PATH. "
        "Install it from https://github.com/BurntSushi/ripgrep/releases"
    )


_RG_BIN: Optional[str] = None


def _rg_bin() -> str:
    global _RG_BIN
    if _RG_BIN is None:
        _RG_BIN = _find_rg()
    return _RG_BIN


def _run_rg(
    args: list[str],
    cwd: Path,
    *,
    timeout: int = 30,
) -> tuple[str, str, int]:
    """
    Run ripgrep and return (stdout, stderr, returncode).

    Always uses --no-heading and --line-number for structured output.
    """
    cmd = [_rg_bin(), "--no-heading", "--line-number", "--with-filename"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        logger.warning("ripgrep timed out after %ds", timeout)
        return "", "timeout", 1
    except FileNotFoundError as exc:
        raise RuntimeError(f"ripgrep binary not found: {exc}") from exc


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<content>.*)$")


def _parse_rg_output(
    stdout: str,
    repo_path: Path,
) -> list[SearchMatch]:
    """Parse ripgrep output lines into SearchMatch objects."""
    matches: list[SearchMatch] = []
    for raw_line in stdout.splitlines():
        m = _LINE_RE.match(raw_line)
        if not m:
            continue
        abs_file = Path(m.group("file"))
        try:
            rel_file = abs_file.relative_to(repo_path).as_posix()
        except ValueError:
            rel_file = m.group("file")
        matches.append(
            SearchMatch(
                file_path=rel_file,
                line_number=int(m.group("line")),
                line_content=m.group("content"),
            )
        )
    return matches


def _normalise_scores(matches: list[SearchMatch]) -> dict[str, float]:
    """
    Compute a normalised relevance score per file.

    Score = number_of_matches_in_file / max_matches_in_any_file.
    """
    counts: dict[str, int] = {}
    for m in matches:
        counts[m.file_path] = counts.get(m.file_path, 0) + 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {fp: count / max_count for fp, count in counts.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_symbol(
    repo_path: Path,
    symbol: str,
    *,
    kinds: Optional[list[str]] = None,
) -> SearchResults:
    """
    Search for Go symbol definitions (functions, types, interfaces, structs).

    Args:
        repo_path: Repository root directory.
        symbol:    Symbol name to search for.
        kinds:     Optional list of kinds: 'func', 'type', 'interface', 'struct'.
                   If None, searches all kinds.

    Returns:
        SearchResults with matched files and scores.

    Example:
        search_symbol(repo, "ValidateJSON", kinds=["func"])
    """
    _kinds = kinds or ["func", "type", "interface", "struct"]
    kind_patterns = {
        "func":      rf"func\s+(\(\w[^)]*\)\s+)?{re.escape(symbol)}\b",
        "type":      rf"type\s+{re.escape(symbol)}\b",
        "interface": rf"type\s+{re.escape(symbol)}\s+interface",
        "struct":    rf"type\s+{re.escape(symbol)}\s+struct",
    }

    all_matches: list[SearchMatch] = []
    for kind in _kinds:
        pattern = kind_patterns.get(kind)
        if not pattern:
            continue
        stdout, stderr, rc = _run_rg(
            ["--type", "go", "--regexp", pattern],
            cwd=repo_path,
        )
        if rc not in (0, 1):  # 1 = no matches (not an error)
            logger.warning("rg error for symbol %s/%s: %s", symbol, kind, stderr)
        all_matches.extend(_parse_rg_output(stdout, repo_path))

    scores = _normalise_scores(all_matches)
    return SearchResults(query=symbol, matches=all_matches, file_scores=scores)


def search_text(
    repo_path: Path,
    query: str,
    *,
    case_insensitive: bool = True,
    go_files_only: bool = True,
    max_results: int = 200,
) -> SearchResults:
    """
    Full-text search across repository files.

    Args:
        repo_path:        Repository root directory.
        query:            Text string to search for.
        case_insensitive: Case-insensitive search (default: True).
        go_files_only:    Restrict to .go files (default: True).
        max_results:      Cap on number of returned lines.

    Returns:
        SearchResults with matched files and scores.
    """
    args: list[str] = ["--fixed-strings"]
    if case_insensitive:
        args.append("--ignore-case")
    if go_files_only:
        args += ["--type", "go"]
    args += ["--max-count", str(max_results), query]

    stdout, stderr, rc = _run_rg(args, cwd=repo_path)
    if rc not in (0, 1):
        logger.warning("rg error for text '%s': %s", query, stderr)

    matches = _parse_rg_output(stdout, repo_path)
    scores = _normalise_scores(matches)
    return SearchResults(query=query, matches=matches, file_scores=scores)


def search_package(
    repo_path: Path,
    package_name: str,
) -> SearchResults:
    """
    Find all Go files that belong to a specific package.

    Args:
        repo_path:    Repository root directory.
        package_name: Go package name (e.g., "binding", "render").

    Returns:
        SearchResults with files declaring that package.
    """
    pattern = rf"^package\s+{re.escape(package_name)}\b"
    args = ["--type", "go", "--regexp", pattern]
    stdout, stderr, rc = _run_rg(args, cwd=repo_path)
    if rc not in (0, 1):
        logger.warning("rg error for package %s: %s", package_name, stderr)

    matches = _parse_rg_output(stdout, repo_path)
    scores = _normalise_scores(matches)
    return SearchResults(query=package_name, matches=matches, file_scores=scores)


def build_rg_scores(
    repo_path: Path,
    query_terms: list[str],
    *,
    go_files_only: bool = True,
) -> dict[str, float]:
    """
    Run ripgrep for multiple query terms and aggregate normalised file scores.

    Useful for building the ``rg_scores`` argument to ``VectorStore.search()``.

    Args:
        repo_path:    Repository root.
        query_terms:  List of terms extracted from the issue.
        go_files_only: Restrict to .go files.

    Returns:
        Mapping {file_path: aggregated_score ∈ [0, 1]}.
    """
    aggregated: dict[str, float] = {}
    for term in query_terms:
        result = search_text(repo_path, term, go_files_only=go_files_only)
        for fp, score in result.file_scores.items():
            aggregated[fp] = aggregated.get(fp, 0.0) + score

    # Re-normalise across all terms
    if aggregated:
        max_score = max(aggregated.values())
        aggregated = {fp: s / max_score for fp, s in aggregated.items()}

    return aggregated
