"""
agents/retriever_agent.py — Hybrid file retrieval agent.

Responsibilities:
  1. Extract search terms from the issue (keywords from Issue Agent analysis)
  2. Run Ripgrep searches to get keyword scores
  3. Load the FAISS vector index built by Repository Agent
  4. Run hybrid retrieval: score = 0.6 * embedding_score + 0.4 * rg_score
  5. Select top-k files with reasoning

Output keys written: retrieved_files, logs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config.settings import settings
from graph.state import AgentState
from services.llm import chat_completion, build_messages
from services.vector_store import VectorStore
from tools.git import read_go_files
from tools.search import build_rg_scores, search_symbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RANK_SYSTEM = """\
You are a senior Go engineer. Given a GitHub issue and a ranked list of candidate files,
select the most relevant files that must be read or modified to solve the issue.

Return JSON with:
- "selected_files": list of selected file paths (in priority order)
- "reasoning": dict mapping each selected file to a one-line explanation
"""


def _rank_user_prompt(
    issue_title: str,
    issue_body: str,
    candidates: list[tuple[str, float]],
) -> str:
    candidate_list = "\n".join(
        f"{i+1}. {path} (score={score:.3f})"
        for i, (path, score) in enumerate(candidates)
    )
    return f"""\
## Issue
**Title:** {issue_title}

**Body:**
{issue_body[:2000]}

## Candidate Files (by hybrid score)
{candidate_list}

Select up to {settings.retrieval_top_k} files that are most likely to need changes.
Return valid JSON only.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords_from_logs(logs: list[str]) -> list[str]:
    """
    Extract the LLM-generated keyword list stored by the Issue Agent.

    The Issue Agent serialises its analysis as:
      [IssueAgent] ANALYSIS_JSON:<json>
    """
    for log_line in reversed(logs):
        if "[IssueAgent] ANALYSIS_JSON:" in log_line:
            raw = log_line.split("ANALYSIS_JSON:", 1)[1]
            try:
                analysis = json.loads(raw)
                keywords = analysis.get("keywords", [])
                affected = analysis.get("affected_areas", [])
                return keywords + affected
            except json.JSONDecodeError:
                pass
    return []


def _extract_terms_from_text(text: str) -> list[str]:
    """
    Fallback: extract likely Go identifiers from issue text.
    Picks CamelCase words and backtick-quoted strings.
    """
    # Backtick-quoted identifiers
    backtick_terms = re.findall(r"`([^`]+)`", text)
    # CamelCase words (likely types/funcs)
    camel_terms = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)
    # Combine and deduplicate
    all_terms: list[str] = list(dict.fromkeys(backtick_terms + camel_terms))
    return all_terms[:20]


def _load_vector_store(repo_path: Path) -> VectorStore | None:
    """Load FAISS vector store if it was built by the Repository Agent."""
    index_dir = repo_path.parent / f"{repo_path.name}.faiss"
    store = VectorStore()
    try:
        store.load(index_dir)
        return store
    except FileNotFoundError:
        logger.warning("No saved vector store at %s — rebuilding", index_dir)

    # Rebuild on-the-fly
    try:
        file_contents = read_go_files(repo_path)
        store.build(file_contents, show_progress=False)
        store.save(index_dir)
        return store
    except Exception as exc:
        logger.error("Failed to build vector store: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def run_retriever_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Retriever Agent.

    Combines FAISS semantic search with Ripgrep keyword scores to identify
    the most relevant files for the issue.

    Args:
        state: Current AgentState.

    Returns:
        Dict of state updates (``retrieved_files``).
    """
    logs: list[str] = list(state.get("logs", []))
    logs.append("[RetrieverAgent] Starting hybrid retrieval …")

    repo_path = Path(state["repo_path"])
    issue_title: str = state.get("issue_title", "")
    issue_body: str = state.get("issue_body", "")
    query = f"{issue_title}\n\n{issue_body}"

    # ── 1. Extract search terms ───────────────────────────────────────────────
    keywords = _extract_keywords_from_logs(state.get("logs", []))
    if not keywords:
        keywords = _extract_terms_from_text(issue_title + " " + issue_body)
    logs.append(f"[RetrieverAgent] Search terms: {keywords[:15]}")
    logger.info("RetrieverAgent: %d search terms", len(keywords))

    # ── 2. Ripgrep scores ─────────────────────────────────────────────────────
    rg_scores: dict[str, float] = {}
    if keywords:
        try:
            rg_scores = build_rg_scores(repo_path, keywords[:10])
            logs.append(f"[RetrieverAgent] Ripgrep: {len(rg_scores)} files matched")
        except Exception as exc:
            logger.warning("RetrieverAgent: ripgrep failed: %s", exc)
            logs.append(f"[RetrieverAgent] WARNING: ripgrep error: {exc}")

    # ── 3. Load vector store and run semantic search ──────────────────────────
    store = _load_vector_store(repo_path)
    if store is None:
        logs.append("[RetrieverAgent] ERROR: could not build vector store")
        return {"retrieved_files": list(rg_scores.keys())[:settings.retrieval_top_k], "logs": logs}

    try:
        top_files: list[tuple[str, float]] = store.top_files(
            query,
            k=settings.retrieval_top_k * 3,  # over-fetch for LLM re-ranking
            rg_scores=rg_scores,
        )
        logs.append(
            f"[RetrieverAgent] Hybrid search: {len(top_files)} candidates before re-rank"
        )
    except Exception as exc:
        logger.error("RetrieverAgent: vector search failed: %s", exc)
        logs.append(f"[RetrieverAgent] ERROR: vector search: {exc}")
        top_files = [(fp, score) for fp, score in sorted(rg_scores.items(), key=lambda x: x[1], reverse=True)]

    # ── 4. LLM re-ranking ────────────────────────────────────────────────────
    selected_files: list[str] = []
    try:
        messages = build_messages(
            system=_RANK_SYSTEM,
            user=_rank_user_prompt(issue_title, issue_body, top_files[:30]),
        )
        raw = chat_completion(messages, json_mode=True)
        result = json.loads(raw)
        selected_files = result.get("selected_files", [])
        reasoning = result.get("reasoning", {})

        for fp, reason in reasoning.items():
            logs.append(f"[RetrieverAgent] {fp}: {reason}")
    except Exception as exc:
        logger.warning("RetrieverAgent: LLM re-ranking failed (%s) — using raw scores", exc)
        selected_files = [fp for fp, _ in top_files[:settings.retrieval_top_k]]

    # ── 5. Validate that selected files exist ─────────────────────────────────
    existing_files = [
        fp for fp in selected_files
        if (repo_path / fp).exists()
    ]
    if len(existing_files) < len(selected_files):
        missing = set(selected_files) - set(existing_files)
        logs.append(f"[RetrieverAgent] Dropped {len(missing)} non-existent files: {missing}")

    logs.append(
        f"[RetrieverAgent] Final selection: {len(existing_files)} files\n  "
        + "\n  ".join(existing_files)
    )
    logger.info("RetrieverAgent: selected %d files", len(existing_files))

    return {
        "retrieved_files": existing_files,
        "logs": logs,
    }
