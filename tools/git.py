"""
tools/git.py — Git repository operations.

Provides:
  - clone_repo()      — Clone a repository into a temp directory
  - checkout_branch() — Create and switch to a new branch
  - get_diff()        — Get the current working-tree diff
  - list_go_files()   — Walk the repo and return all .go file paths
  - read_file()       — Read a single file from the repo
  - read_go_files()   — Read all .go files (path → content)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Generator, Optional

import git
from git import Repo, GitCommandError

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clone / checkout
# ---------------------------------------------------------------------------


def clone_repo(
    url: str,
    *,
    target_dir: Optional[Path] = None,
    depth: int = 1,
    branch: Optional[str] = None,
) -> Path:
    """
    Clone a repository.

    Uses a deterministic subdirectory name based on the URL hash so that
    repeated runs reuse the same clone.

    Args:
        url:        HTTPS or SSH clone URL.
        target_dir: Base directory for clones (default: settings.repo_temp_dir).
        depth:      Shallow-clone depth (1 = only latest commit).
        branch:     Branch to checkout (default: repo default).

    Returns:
        Path to the cloned repository root.
    """
    base = target_dir or settings.repo_temp_dir
    base.mkdir(parents=True, exist_ok=True)

    # Derive a stable directory name from the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    repo_slug = url.rstrip("/").split("/")[-1].removesuffix(".git")
    clone_path = base / f"{repo_slug}-{url_hash}"

    if clone_path.exists():
        logger.info("Reusing existing clone at %s", clone_path)
        try:
            repo = Repo(clone_path)
            repo.remotes.origin.pull()
            logger.debug("Pulled latest changes")
        except GitCommandError as exc:
            logger.warning("Pull failed (%s), continuing with existing clone", exc)
        return clone_path

    logger.info("Cloning %s → %s (depth=%d)", url, clone_path, depth)
    kwargs: dict = {"depth": depth}
    if branch:
        kwargs["branch"] = branch

    try:
        Repo.clone_from(url, clone_path, **kwargs)
    except GitCommandError as exc:
        raise RuntimeError(f"git clone failed: {exc}") from exc

    logger.info("Clone complete: %s", clone_path)
    return clone_path


def checkout_branch(repo_path: Path, branch_name: str) -> None:
    """
    Create and switch to a new branch in the given repository.

    If the branch already exists, it is checked out.

    Args:
        repo_path:   Path to the repository root.
        branch_name: Name of the branch to create/checkout.
    """
    repo = Repo(repo_path)
    if branch_name in [b.name for b in repo.branches]:
        repo.git.checkout(branch_name)
        logger.debug("Checked out existing branch: %s", branch_name)
    else:
        repo.git.checkout("-b", branch_name)
        logger.info("Created and checked out new branch: %s", branch_name)


def get_diff(repo_path: Path) -> str:
    """
    Return the current working-tree diff (unstaged + staged changes).

    Args:
        repo_path: Path to the repository root.

    Returns:
        Unified diff string.
    """
    repo = Repo(repo_path)
    diff = repo.git.diff("HEAD")
    if not diff:
        diff = repo.git.diff()
    return diff


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "vendor", "node_modules", "testdata", ".github"}


def list_go_files(repo_path: Path) -> list[Path]:
    """
    Recursively walk the repository and return all `.go` file paths.

    Skips vendor/, .git/, and other non-source directories.

    Args:
        repo_path: Repository root directory.

    Returns:
        Sorted list of absolute Path objects for .go files.
    """
    go_files: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        # Prune irrelevant dirs in-place to prevent descent
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith("_")]
        for fn in files:
            if fn.endswith(".go"):
                go_files.append(Path(root) / fn)
    return sorted(go_files)


def read_file(repo_path: Path, relative_path: str) -> str:
    """
    Read a file from the repository.

    Args:
        repo_path:     Repository root.
        relative_path: Path relative to the repository root.

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    full_path = repo_path / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"File not found in repo: {relative_path}")
    return full_path.read_text(encoding="utf-8", errors="replace")


def read_go_files(
    repo_path: Path,
    *,
    max_file_size_kb: int = 500,
) -> dict[str, str]:
    """
    Read all Go source files in the repository.

    Args:
        repo_path:       Repository root.
        max_file_size_kb: Skip files larger than this (default 500 KB).

    Returns:
        Mapping of {relative_path: file_content}.
    """
    result: dict[str, str] = {}
    max_bytes = max_file_size_kb * 1024

    for abs_path in list_go_files(repo_path):
        if abs_path.stat().st_size > max_bytes:
            logger.debug("Skipping large file: %s", abs_path)
            continue
        rel = abs_path.relative_to(repo_path).as_posix()
        try:
            result[rel] = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", abs_path, exc)

    logger.debug("Read %d Go files from %s", len(result), repo_path)
    return result


def write_file(repo_path: Path, relative_path: str, content: str) -> None:
    """
    Write content to a file inside the repository.

    Creates parent directories as needed.

    Args:
        repo_path:     Repository root.
        relative_path: Path relative to the repository root.
        content:       New file content.
    """
    full_path = repo_path / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    logger.debug("Wrote %d chars to %s", len(content), relative_path)
