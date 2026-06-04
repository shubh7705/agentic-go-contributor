"""
graph/workflow.py — LangGraph StateGraph workflow definition.

Defines the full agent pipeline:

  Issue Agent
    → Repository Agent
      → Retriever Agent
        → Planner Agent
          → Code Agent
            → Validation Agent
              ↙ (pass)         ↘ (fail, max not reached)
           PR Agent          Repair Agent → Validation Agent
              ↓                    ↑_______________|
           END

Repair loop:
  - If validation_passed → go to PR Agent
  - If not passed AND iterations < max → go to Repair Agent
  - If not passed AND iterations >= max → go to PR Agent anyway (best effort)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents.code_agent import run_code_agent
from agents.issue_agent import run_issue_agent
from agents.planner_agent import run_planner_agent
from agents.pr_agent import run_pr_agent
from agents.repair_agent import run_repair_agent, count_repair_iterations
from agents.repository_agent import run_repository_agent
from agents.retriever_agent import run_retriever_agent
from agents.validation_agent import run_validation_agent
from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node names (constants to avoid typos)
# ---------------------------------------------------------------------------

N_ISSUE = "issue_agent"
N_REPO = "repository_agent"
N_RETRIEVER = "retriever_agent"
N_PLANNER = "planner_agent"
N_CODE = "code_agent"
N_VALIDATE = "validation_agent"
N_REPAIR = "repair_agent"
N_PR = "pr_agent"

# ---------------------------------------------------------------------------
# Conditional edge: after validation, go to repair or PR
# ---------------------------------------------------------------------------


def _route_after_validation(state: AgentState) -> Literal["repair_agent", "pr_agent"]:
    """
    Routing function for the conditional edge after Validation Agent.

    Returns:
      "pr_agent"     if validation passed OR max repair iterations reached
      "repair_agent" otherwise
    """
    if state.get("validation_passed", False):
        logger.info("Workflow: validation PASSED → PR Agent")
        return N_PR

    iterations = count_repair_iterations(state.get("logs", []))
    if iterations >= settings.max_repair_iterations:
        logger.warning(
            "Workflow: max repair iterations (%d) reached — proceeding to PR Agent anyway",
            settings.max_repair_iterations,
        )
        return N_PR

    logger.info(
        "Workflow: validation FAILED (iteration %d/%d) → Repair Agent",
        iterations,
        settings.max_repair_iterations,
    )
    return N_REPAIR


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_workflow() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        Compiled StateGraph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node(N_ISSUE, run_issue_agent)
    graph.add_node(N_REPO, run_repository_agent)
    graph.add_node(N_RETRIEVER, run_retriever_agent)
    graph.add_node(N_PLANNER, run_planner_agent)
    graph.add_node(N_CODE, run_code_agent)
    graph.add_node(N_VALIDATE, run_validation_agent)
    graph.add_node(N_REPAIR, run_repair_agent)
    graph.add_node(N_PR, run_pr_agent)

    # ── Linear edges ─────────────────────────────────────────────────────────
    graph.add_edge(START, N_ISSUE)
    graph.add_edge(N_ISSUE, N_REPO)
    graph.add_edge(N_REPO, N_RETRIEVER)
    graph.add_edge(N_RETRIEVER, N_PLANNER)
    graph.add_edge(N_PLANNER, N_CODE)
    graph.add_edge(N_CODE, N_VALIDATE)

    # ── Conditional edge after validation ─────────────────────────────────────
    graph.add_conditional_edges(
        N_VALIDATE,
        _route_after_validation,
        {N_PR: N_PR, N_REPAIR: N_REPAIR},
    )

    # ── Repair → re-validate ──────────────────────────────────────────────────
    graph.add_edge(N_REPAIR, N_VALIDATE)

    # ── PR → END ─────────────────────────────────────────────────────────────
    graph.add_edge(N_PR, END)

    return graph


def compile_workflow():
    """Build and compile the workflow graph."""
    graph = build_workflow()
    return graph.compile()


# Module-level compiled workflow (lazy singleton pattern)
_compiled_workflow = None


def get_workflow():
    """Return the compiled workflow, building it on first call."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = compile_workflow()
    return _compiled_workflow
