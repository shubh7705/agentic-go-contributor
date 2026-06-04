"""
agents/repository_agent.py — Repository cloning and analysis agent.

Responsibilities:
  1. Clone the repository (from issue data or provided repo URL)
  2. Read all Go source files
  3. Parse with Tree-Sitter to build the repository map
  4. Identify top-level packages and dependencies
  5. Build and persist the FAISS vector index for subsequent retrieval

Output keys written: repo_path, repository_map, logs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import settings
from graph.state import AgentState
from tools.git import clone_repo, read_go_files
from tools.github import parse_issue_url, fetch_repo_info
from tools.tree_sitter_parser import build_repo_map
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_repo_url(state: AgentState) -> str:
    """
    Derive the repository clone URL.

    Priority:
    1. ``state["repo_path"]`` if it already looks like a URL (starts with http)
    2. Parse the ``issue_url`` to infer the repo
    """
    # If a separate repo URL was provided in state (via main.py --repo flag),
    # it gets stored temporarily in repo_path before cloning.
    repo_path_val: str = state.get("repo_path", "")
    if repo_path_val.startswith("http"):
        return repo_path_val

    # Fall back to parsing the issue URL
    issue_url: str = state["issue_url"]
    owner, repo, _ = parse_issue_url(issue_url)
    return f"https://github.com/{owner}/{repo}.git"


def _summarise_repo_map(repo_map: dict[str, dict]) -> str:
    """Build a concise text summary of the repo map for logging."""
    packages: set[str] = set()
    total_funcs = 0
    total_types = 0
    for meta in repo_map.values():
        if meta.get("package"):
            packages.add(meta["package"])
        total_funcs += len(meta.get("functions", [])) + len(meta.get("methods", []))
        total_types += (
            len(meta.get("structs", []))
            + len(meta.get("interfaces", []))
            + len(meta.get("types", []))
        )
    return (
        f"{len(repo_map)} files | "
        f"{len(packages)} packages: {', '.join(sorted(packages)[:15])} | "
        f"{total_funcs} functions/methods | {total_types} types"
    )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_repository_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Repository Agent.

    Clones the repository, parses Go source files, builds a repository map,
    and constructs a FAISS vector index for semantic retrieval.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates.
    """
    logs: list[str] = list(state.get("logs", []))
    logs.append("[RepositoryAgent] Starting …")

    # ── 1. Determine clone URL ───────────────────────────────────────────────
    repo_url = _infer_repo_url(state)
    logs.append(f"[RepositoryAgent] Target repo: {repo_url}")
    logger.info("RepositoryAgent: cloning %s", repo_url)

    # ── 2. Clone ─────────────────────────────────────────────────────────────
    try:
        repo_path: Path = clone_repo(repo_url)
    except Exception as exc:
        logger.error("RepositoryAgent: clone failed: %s", exc)
        logs.append(f"[RepositoryAgent] ERROR cloning: {exc}")
        return {"logs": logs}

    logs.append(f"[RepositoryAgent] Cloned to: {repo_path}")

    # ── 3. Read Go files ─────────────────────────────────────────────────────
    logger.info("RepositoryAgent: reading Go files …")
    file_contents: dict[str, str] = read_go_files(repo_path)
    logs.append(f"[RepositoryAgent] Found {len(file_contents)} Go files")

    # ── 4. Parse with Tree-Sitter ─────────────────────────────────────────────
    logger.info("RepositoryAgent: parsing repository …")
    repo_map = build_repo_map(file_contents)
    summary = _summarise_repo_map(repo_map)
    logs.append(f"[RepositoryAgent] Parsed: {summary}")
    logger.info("RepositoryAgent: %s", summary)

    # ── 5. Build FAISS vector index ──────────────────────────────────────────
    logger.info("RepositoryAgent: building vector index …")
    store = VectorStore()
    try:
        store.build(file_contents, show_progress=False)
        # Persist index next to the clone for reuse
        index_dir = repo_path.parent / f"{repo_path.name}.faiss"
        store.save(index_dir)
        logs.append(
            f"[RepositoryAgent] Vector index built ({store.num_chunks} chunks) "
            f"and saved to {index_dir}"
        )
    except Exception as exc:
        logger.warning("RepositoryAgent: vector indexing failed: %s", exc)
        logs.append(f"[RepositoryAgent] WARNING: vector indexing failed: {exc}")

    logger.info("RepositoryAgent: complete")
    return {
        "repo_path": str(repo_path),
        "repository_map": repo_map,
        "logs": logs,
    }
