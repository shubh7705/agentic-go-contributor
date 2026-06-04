"""
tools/file_editor.py — Apply code changes to repository files.

Supports two strategies:
  1. Unified diff application (patch format)
  2. Direct full-file replacement

Also provides:
  - diff generation between old and new content
  - safe atomic writes (write to temp → rename)
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


def generate_diff(
    original: str,
    modified: str,
    filename: str = "file",
) -> str:
    """
    Generate a unified diff between original and modified content.

    Args:
        original: Original file content.
        modified: Modified file content.
        filename: File name used in the diff header.

    Returns:
        Unified diff string (may be empty if no changes).
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )
    return diff


def apply_diff(original: str, diff_text: str) -> str:
    """
    Apply a unified diff to file content.

    This is a best-effort implementation. For complex patches with ambiguous
    context, prefer ``replace_content`` or ``apply_hunks`` instead.

    Args:
        original: Original file content.
        diff_text: Unified diff string.

    Returns:
        Patched file content.

    Raises:
        ValueError: If the patch cannot be applied cleanly.
    """
    if not diff_text.strip():
        return original

    try:
        return _apply_unified_diff(original, diff_text)
    except Exception as exc:
        logger.warning("Diff application failed (%s), returning original", exc)
        raise ValueError(f"Could not apply diff: {exc}") from exc


def _apply_unified_diff(original: str, diff_text: str) -> str:
    """Internal: apply a unified diff hunk-by-hunk."""
    lines = original.splitlines(keepends=True)
    result = list(lines)
    offset = 0  # cumulative line-count delta from previous hunks

    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    diff_lines = diff_text.splitlines(keepends=True)
    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        m = hunk_re.match(line)
        if not m:
            i += 1
            continue

        orig_start = int(m.group(1)) - 1 + offset  # 0-indexed
        orig_count = int(m.group(2) or 1)
        i += 1

        hunk_remove: list[str] = []
        hunk_add: list[str] = []

        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            dl = diff_lines[i]
            if dl.startswith("-"):
                hunk_remove.append(dl[1:])
            elif dl.startswith("+"):
                hunk_add.append(dl[1:])
            elif dl.startswith(" "):
                hunk_remove.append(dl[1:])
                hunk_add.append(dl[1:])
            i += 1

        # Replace the hunk region
        end = orig_start + orig_count
        result[orig_start:end] = hunk_add
        offset += len(hunk_add) - orig_count

    return "".join(result)


# ---------------------------------------------------------------------------
# Direct file replacement
# ---------------------------------------------------------------------------


def replace_content(
    repo_path: Path,
    relative_path: str,
    new_content: str,
) -> str:
    """
    Overwrite a repository file with new content (atomic write).

    Args:
        repo_path:     Repository root.
        relative_path: Path relative to the repository root.
        new_content:   New file content.

    Returns:
        Unified diff of the change (for logging / audit).
    """
    full_path = repo_path / relative_path

    # Read original if it exists
    original = ""
    if full_path.exists():
        original = full_path.read_text(encoding="utf-8", errors="replace")

    diff = generate_diff(original, new_content, filename=relative_path)

    # Atomic write: write to temp file in same directory, then rename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=full_path.parent, prefix=".tmp_", suffix=".go"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(new_content)
        os.replace(tmp_path, full_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    lines_added = diff.count("\n+") - diff.count("\n+++")
    lines_removed = diff.count("\n-") - diff.count("\n---")
    logger.info(
        "Wrote %s (+%d / -%d lines)", relative_path, lines_added, lines_removed
    )
    return diff


def apply_proposed_changes(
    repo_path: Path,
    proposed_changes: dict[str, str],
) -> dict[str, str]:
    """
    Apply a batch of proposed file changes to the repository.

    Args:
        repo_path:        Repository root.
        proposed_changes: Mapping of {relative_path: new_file_content}.

    Returns:
        Mapping of {relative_path: diff_string} for each changed file.
    """
    diffs: dict[str, str] = {}
    for rel_path, content in proposed_changes.items():
        try:
            diff = replace_content(repo_path, rel_path, content)
            diffs[rel_path] = diff
        except Exception as exc:
            logger.error("Failed to write %s: %s", rel_path, exc)
            diffs[rel_path] = f"ERROR: {exc}"
    return diffs


# ---------------------------------------------------------------------------
# Patch extraction helpers (for LLM output parsing)
# ---------------------------------------------------------------------------


def extract_code_blocks(llm_response: str) -> dict[str, str]:
    """
    Extract file contents from an LLM response that uses fenced code blocks.

    Supports multiple LLM output styles:

        ```go path/to/file.go
        ...code...
        ```

        // File: path/to/file.go
        (optional blank lines)
        ```go
        ...code...
        ```

        **filename.go** / ### filename.go  before a go fence

    Args:
        llm_response: Raw LLM text output.

    Returns:
        Mapping of {filename: code_content}.
    """
    results: dict[str, str] = {}

    # Pattern 1: ```go path/to/file.go\n...\n```
    pattern1 = re.compile(
        r"```(?:go|golang)\s+([^\n`]+\.go)\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern1.finditer(llm_response):
        filename = m.group(1).strip()
        code = m.group(2)
        results[filename] = code

    # Pattern 2: // File: path/to/file.go  (then \n+ blank lines)  ```go\n...\n```
    # Allow \n+ between marker and opening fence to tolerate LLM whitespace
    pattern2 = re.compile(
        r"//\s*[Ff]ile:\s*([^\n]+\.go)\n+```(?:go|golang)?\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern2.finditer(llm_response):
        filename = m.group(1).strip()
        code = m.group(2)
        results.setdefault(filename, code)

    # Pattern 3: **filename.go** or *filename.go* before a go fence
    pattern3 = re.compile(
        r"\*{1,2}([^\n*`]+\.go)\*{1,2}\s*\n+```(?:go|golang)?\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern3.finditer(llm_response):
        filename = m.group(1).strip()
        code = m.group(2)
        results.setdefault(filename, code)

    # Pattern 4: ### filename.go (markdown heading) before a go fence
    pattern4 = re.compile(
        r"#{1,4}\s+([^\n#`]+\.go)\s*\n+```(?:go|golang)?\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern4.finditer(llm_response):
        filename = m.group(1).strip()
        code = m.group(2)
        results.setdefault(filename, code)

    # Pattern 5: Last-resort — bare ```go\n...\n``` blocks; find the closest
    # filename mentioned in the 300 chars of prose before the fence.
    if not results:
        bare_go = re.compile(r"```(?:go|golang)\n(.*?)```", re.DOTALL)
        file_mention = re.compile(r"([a-zA-Z0-9_/.-]+\.go)")
        for m in bare_go.finditer(llm_response):
            code = m.group(1)
            preamble = llm_response[max(0, m.start() - 300):m.start()]
            names = file_mention.findall(preamble)
            if names:
                filename = names[-1]
                results.setdefault(filename, code)

    return results


def extract_diff_blocks(llm_response: str) -> str:
    """
    Extract unified diff content from fenced diff blocks in an LLM response.

    Args:
        llm_response: Raw LLM text output.

    Returns:
        Concatenated diff string.
    """
    diffs: list[str] = []
    pattern = re.compile(r"```(?:diff|patch)\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(llm_response):
        diffs.append(m.group(1))
    return "\n".join(diffs)
