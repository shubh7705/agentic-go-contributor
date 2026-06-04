"""
tools/github.py — GitHub API interactions.

Provides:
  - Fetch issue metadata (title, body, labels, comments)
  - Parse repo owner/name from URL
  - List open issues for a repository
  - Post PR descriptions (dry-run by default)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from github import Github, GithubException
from github.Issue import Issue
from github.Repository import Repository

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class IssueData:
    """Structured representation of a GitHub issue."""

    number: int
    title: str
    body: str
    labels: list[str]
    state: str
    url: str
    repo_full_name: str      # e.g. "gin-gonic/gin"
    repo_url: str            # HTTPS clone URL
    comments: list[str]


@dataclass
class RepoInfo:
    """Basic repository metadata."""

    full_name: str           # "owner/repo"
    clone_url: str           # HTTPS clone URL
    default_branch: str
    language: str
    description: str


# ---------------------------------------------------------------------------
# GitHub client singleton
# ---------------------------------------------------------------------------

_gh_client: Optional[Github] = None


def _get_client() -> Github:
    global _gh_client
    if _gh_client is None:
        token = settings.github_token
        _gh_client = Github(login_or_token=token) if token else Github()
        logger.debug("GitHub client initialised (authenticated=%s)", bool(token))
    return _gh_client


# ---------------------------------------------------------------------------
# URL parsing helpers
# ---------------------------------------------------------------------------

_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)"
)

_REPO_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def parse_issue_url(url: str) -> tuple[str, str, int]:
    """
    Parse a GitHub issue URL into (owner, repo, issue_number).

    Raises:
        ValueError: If the URL does not match the expected pattern.
    """
    m = _ISSUE_URL_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Cannot parse issue URL: {url!r}\n"
            "Expected format: https://github.com/<owner>/<repo>/issues/<number>"
        )
    return m.group("owner"), m.group("repo"), int(m.group("number"))


def parse_repo_url(url: str) -> tuple[str, str]:
    """
    Parse a GitHub repository URL into (owner, repo).

    Raises:
        ValueError: If the URL does not match the expected pattern.
    """
    m = _REPO_URL_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Cannot parse repo URL: {url!r}\n"
            "Expected format: https://github.com/<owner>/<repo>"
        )
    return m.group("owner"), m.group("repo")


# ---------------------------------------------------------------------------
# Issue fetching
# ---------------------------------------------------------------------------

def fetch_issue(issue_url: str) -> IssueData:
    """
    Fetch a GitHub issue and return structured data.

    Args:
        issue_url: Full GitHub issue URL.

    Returns:
        IssueData with title, body, labels, and comments.
    """
    owner, repo_name, number = parse_issue_url(issue_url)
    gh = _get_client()

    try:
        repo: Repository = gh.get_repo(f"{owner}/{repo_name}")
        issue: Issue = repo.get_issue(number)
    except GithubException as exc:
        logger.error("GitHub API error: %s", exc)
        raise RuntimeError(f"Failed to fetch issue {issue_url}: {exc}") from exc

    comments: list[str] = []
    try:
        for comment in issue.get_comments():
            if comment.body:
                comments.append(comment.body)
    except GithubException:
        logger.warning("Could not retrieve comments for issue #%d", number)

    data = IssueData(
        number=issue.number,
        title=issue.title or "",
        body=issue.body or "",
        labels=[label.name for label in issue.labels],
        state=issue.state,
        url=issue_url,
        repo_full_name=repo.full_name,
        repo_url=repo.clone_url,
        comments=comments,
    )
    logger.info("Fetched issue #%d: %s", number, data.title)
    return data


def fetch_repo_info(repo_url: str) -> RepoInfo:
    """
    Fetch basic repository metadata.

    Args:
        repo_url: Full GitHub repository URL.

    Returns:
        RepoInfo struct.
    """
    owner, repo_name = parse_repo_url(repo_url)
    gh = _get_client()

    try:
        repo: Repository = gh.get_repo(f"{owner}/{repo_name}")
    except GithubException as exc:
        raise RuntimeError(f"Failed to fetch repo {repo_url}: {exc}") from exc

    return RepoInfo(
        full_name=repo.full_name,
        clone_url=repo.clone_url,
        default_branch=repo.default_branch,
        language=repo.language or "Go",
        description=repo.description or "",
    )
