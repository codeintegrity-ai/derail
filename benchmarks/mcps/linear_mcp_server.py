"""Synthetic Linear MCP for the classifier-evasion benchmark.

Backed by in-memory state that resets at process start.

Run:
    uv run python benchmarks/mcps/linear_mcp_server.py
    LINEAR_BENCH_PORT=5021 uv run python benchmarks/mcps/linear_mcp_server.py

Default URL: http://127.0.0.1:5021/mcp
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("linear-bench")

_DEFAULT_TEAM_ID = "6695fc11-5341-45fd-b428-8d4dc3a11edd"
_DEFAULT_TEAM_KEY = "CON"
_DEFAULT_TEAM_NAME = "Codeintegrity"
_DEFAULT_USER_ID = "00000000-0000-4000-8000-000000000001"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


_TEAMS: dict[str, dict[str, Any]] = {
    _DEFAULT_TEAM_ID: {
        "id": _DEFAULT_TEAM_ID,
        "key": _DEFAULT_TEAM_KEY,
        "name": _DEFAULT_TEAM_NAME,
    },
}

_USERS: dict[str, dict[str, Any]] = {
    _DEFAULT_USER_ID: {
        "id": _DEFAULT_USER_ID,
        "name": "Test User",
        "email": "test@example.com",
        "displayName": "test",
    },
}

_STATES: list[dict[str, Any]] = [
    {"id": "state-backlog", "name": "Backlog", "type": "backlog"},
    {"id": "state-todo", "name": "Todo", "type": "unstarted"},
    {"id": "state-progress", "name": "In Progress", "type": "started"},
    {"id": "state-done", "name": "Done", "type": "completed"},
    {"id": "state-canceled", "name": "Canceled", "type": "canceled"},
]

_ISSUES: dict[str, dict[str, Any]] = {}
_COMMENTS: dict[str, dict[str, Any]] = {}
_LABELS: dict[str, dict[str, Any]] = {}
_PROJECTS: dict[str, dict[str, Any]] = {}
_DOCUMENTS: dict[str, dict[str, Any]] = {}
_ATTACHMENTS: dict[str, dict[str, Any]] = {}
_ISSUE_COUNTER = {"n": 0}


# ---- helpers ----------------------------------------------------------


def _resolve_team(query: str | None) -> dict[str, Any]:
    if not query:
        raise ValueError("team is required")
    for team in _TEAMS.values():
        if query in (team["id"], team["key"], team["name"]):
            return team
    raise ValueError(f"unknown team {query!r}")


def _resolve_state(query: str | None) -> dict[str, Any] | None:
    if not query:
        return None
    needle = query.lower()
    for state in _STATES:
        if state["id"] == query or state["name"].lower() == needle or state["type"].lower() == needle:
            return state
    raise ValueError(f"unknown state {query!r}")


def _resolve_user(query: str | None) -> dict[str, Any] | None:
    if query is None:
        return None
    if query == "me":
        return _USERS[_DEFAULT_USER_ID]
    for user in _USERS.values():
        if query in (user["id"], user["name"], user["email"], user.get("displayName")):
            return user
    raise ValueError(f"unknown user {query!r}")


def _find_issue(query: str) -> dict[str, Any]:
    if query in _ISSUES:
        return _ISSUES[query]
    for issue in _ISSUES.values():
        if issue["identifier"] == query:
            return issue
    raise ValueError(f"Entity not found: Issue - Could not find referenced Issue {query!r}.")


def _next_identifier() -> str:
    _ISSUE_COUNTER["n"] += 1
    return f"{_DEFAULT_TEAM_KEY}-{_ISSUE_COUNTER['n']}"


# ---- teams + users + statuses ----------------------------------------


@mcp.tool()
def list_teams(
    query: Annotated[str | None, "Search query"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    teams = list(_TEAMS.values())
    if query:
        needle = query.lower()
        teams = [t for t in teams if needle in t["name"].lower() or needle in t["key"].lower()]
    return {"teams": teams[:limit]}


@mcp.tool()
def get_team(query: Annotated[str, "Team UUID, key, or name"]) -> dict[str, Any]:
    return {"team": _resolve_team(query)}


@mcp.tool()
def list_users(
    query: Annotated[str | None, "Filter by name or email"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    users = list(_USERS.values())
    if query:
        needle = query.lower()
        users = [u for u in users if needle in u["name"].lower() or needle in u["email"].lower()]
    return {"users": users[:limit]}


@mcp.tool()
def get_user(query: Annotated[str, "User ID, name, email, or 'me'"]) -> dict[str, Any]:
    user = _resolve_user(query)
    if user is None:
        raise ValueError(f"unknown user {query!r}")
    return {"user": user}


@mcp.tool()
def list_issue_statuses(team: Annotated[str, "Team name or ID"]) -> dict[str, Any]:
    _resolve_team(team)
    return {"statuses": list(_STATES)}


# ---- issues ----------------------------------------------------------


@mcp.tool()
def save_issue(
    id: Annotated[str | None, "Issue ID/identifier (only for update)"] = None,
    title: Annotated[str | None, "Issue title (required when creating)"] = None,
    description: Annotated[str | None, "Markdown description"] = None,
    team: Annotated[str | None, "Team name or ID (required when creating)"] = None,
    priority: Annotated[float | None, "0=None,1=Urgent,2=High,3=Normal,4=Low"] = None,
    state: Annotated[str | None, "State type, name, or ID"] = None,
    assignee: Annotated[str | None, "User ID, name, email, or 'me'"] = None,
    labels: Annotated[list[str] | None, "Label names or IDs"] = None,
    project: Annotated[str | None, "Project ID or name"] = None,
    estimate: Annotated[float | None, "Estimate value"] = None,
    links: Annotated[list[dict] | None, "Link attachments [{url, title}]"] = None,
) -> dict[str, Any]:
    if id is None:
        if not title:
            raise ValueError("title is required when creating an issue")
        team_record = _resolve_team(team or _DEFAULT_TEAM_NAME)
        state_record = _resolve_state(state) or _STATES[1]
        assignee_record = _resolve_user(assignee)
        identifier = _next_identifier()
        issue_id = _new_id()
        issue = {
            "id": issue_id,
            "identifier": identifier,
            "title": title,
            "description": description or "",
            "team": team_record,
            "state": state_record,
            "assignee": assignee_record,
            "priority": priority if priority is not None else 0,
            "labels": list(labels or []),
            "project": project,
            "estimate": estimate,
            "links": list(links or []),
            "attachments": [],
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        _ISSUES[issue_id] = issue
        return {"issue": issue}

    issue = _find_issue(id)
    if title is not None:
        issue["title"] = title
    if description is not None:
        issue["description"] = description
    if state is not None:
        issue["state"] = _resolve_state(state)
    if assignee is not None:
        issue["assignee"] = _resolve_user(assignee)
    if priority is not None:
        issue["priority"] = priority
    if labels is not None:
        issue["labels"] = list(labels)
    if project is not None:
        issue["project"] = project
    if estimate is not None:
        issue["estimate"] = estimate
    if links:
        issue["links"] = (issue.get("links") or []) + list(links)
    issue["updatedAt"] = _now()
    return {"issue": issue}


@mcp.tool()
def get_issue(
    id: Annotated[str, "Issue ID or identifier (e.g., CON-1)"],
    includeRelations: Annotated[bool, "Include relations"] = False,
) -> dict[str, Any]:
    return {"issue": _find_issue(id)}


@mcp.tool()
def list_issues(
    team: Annotated[str | None, "Team name or ID"] = None,
    state: Annotated[str | None, "State type, name, or ID"] = None,
    project: Annotated[str | None, "Project ID or name"] = None,
    assignee: Annotated[str | None, "User ID, name, email, or 'me'"] = None,
    query: Annotated[str | None, "Search title/description"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    issues = list(_ISSUES.values())
    if team:
        team_record = _resolve_team(team)
        issues = [i for i in issues if i["team"]["id"] == team_record["id"]]
    if state:
        state_record = _resolve_state(state)
        issues = [i for i in issues if i.get("state", {}).get("id") == state_record["id"]]
    if project:
        issues = [i for i in issues if (i.get("project") or "") == project or (i.get("project") or "") == _resolve_project_id(project)]
    if assignee:
        user = _resolve_user(assignee)
        issues = [i for i in issues if (i.get("assignee") or {}).get("id") == (user or {}).get("id")]
    if query:
        needle = query.lower()
        issues = [
            i
            for i in issues
            if needle in i["title"].lower() or needle in (i.get("description") or "").lower()
        ]
    return {"issues": issues[:limit]}


# ---- comments --------------------------------------------------------


@mcp.tool()
def save_comment(
    body: Annotated[str, "Markdown body"],
    id: Annotated[str | None, "Comment ID (for update)"] = None,
    issueId: Annotated[str | None, "Issue ID/identifier (required when creating)"] = None,
    parentId: Annotated[str | None, "Parent comment ID (for replies)"] = None,
) -> dict[str, Any]:
    if id is None:
        if not issueId:
            raise ValueError("issueId is required when creating a comment")
        issue = _find_issue(issueId)
        comment_id = _new_id()
        comment = {
            "id": comment_id,
            "issueId": issue["id"],
            "issueIdentifier": issue["identifier"],
            "parentId": parentId,
            "body": body,
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        _COMMENTS[comment_id] = comment
        return {"comment": comment}

    if id not in _COMMENTS:
        raise ValueError(f"Entity not found: Comment {id!r}.")
    comment = _COMMENTS[id]
    comment["body"] = body
    comment["updatedAt"] = _now()
    return {"comment": comment}


@mcp.tool()
def list_comments(
    issueId: Annotated[str, "Issue ID or identifier"],
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    issue = _find_issue(issueId)
    comments = [c for c in _COMMENTS.values() if c["issueId"] == issue["id"]]
    return {"comments": comments[:limit]}


# ---- labels ----------------------------------------------------------


@mcp.tool()
def create_issue_label(
    name: Annotated[str, "Label name"],
    description: Annotated[str | None, "Label description"] = None,
    color: Annotated[str | None, "Hex color"] = None,
    teamId: Annotated[str | None, "Team UUID (omit for workspace label)"] = None,
) -> dict[str, Any]:
    label_id = _new_id()
    label = {
        "id": label_id,
        "name": name,
        "description": description,
        "color": color,
        "teamId": teamId,
        "createdAt": _now(),
    }
    _LABELS[label_id] = label
    return {"label": label}


@mcp.tool()
def list_issue_labels(
    name: Annotated[str | None, "Filter by name"] = None,
    team: Annotated[str | None, "Team name or ID"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    labels = list(_LABELS.values())
    if name:
        needle = name.lower()
        labels = [lbl for lbl in labels if needle in lbl["name"].lower()]
    if team:
        team_record = _resolve_team(team)
        labels = [lbl for lbl in labels if lbl.get("teamId") in (None, team_record["id"])]
    return {"labels": labels[:limit]}


# ---- projects --------------------------------------------------------


def _resolve_project_id(query: str) -> str:
    if query in _PROJECTS:
        return query
    for p in _PROJECTS.values():
        if p["name"] == query:
            return p["id"]
    return query


@mcp.tool()
def save_project(
    id: Annotated[str | None, "Project ID (omit to create)"] = None,
    name: Annotated[str | None, "Project name (required when creating)"] = None,
    description: Annotated[str | None, "Project description"] = None,
    team: Annotated[str | None, "Team name or ID"] = None,
    state: Annotated[str | None, "Project state"] = None,
) -> dict[str, Any]:
    if id is None:
        if not name:
            raise ValueError("name is required when creating a project")
        team_record = _resolve_team(team or _DEFAULT_TEAM_NAME)
        project_id = _new_id()
        project = {
            "id": project_id,
            "name": name,
            "description": description or "",
            "team": team_record,
            "state": state or "planned",
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        _PROJECTS[project_id] = project
        return {"project": project}

    if id not in _PROJECTS:
        raise ValueError(f"Entity not found: Project {id!r}.")
    project = _PROJECTS[id]
    if name is not None:
        project["name"] = name
    if description is not None:
        project["description"] = description
    if state is not None:
        project["state"] = state
    project["updatedAt"] = _now()
    return {"project": project}


@mcp.tool()
def get_project(id: Annotated[str, "Project ID or name"]) -> dict[str, Any]:
    project_id = _resolve_project_id(id)
    if project_id not in _PROJECTS:
        raise ValueError(f"Entity not found: Project {id!r}.")
    return {"project": _PROJECTS[project_id]}


@mcp.tool()
def list_projects(
    query: Annotated[str | None, "Search by name"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    projects = list(_PROJECTS.values())
    if query:
        needle = query.lower()
        projects = [p for p in projects if needle in p["name"].lower()]
    return {"projects": projects[:limit]}


# ---- documents -------------------------------------------------------


@mcp.tool()
def save_document(
    id: Annotated[str | None, "Document ID (omit to create)"] = None,
    title: Annotated[str | None, "Document title"] = None,
    content: Annotated[str | None, "Markdown content"] = None,
    project: Annotated[str | None, "Project ID or name"] = None,
) -> dict[str, Any]:
    if id is None:
        if not title:
            raise ValueError("title is required when creating a document")
        doc_id = _new_id()
        doc = {
            "id": doc_id,
            "title": title,
            "content": content or "",
            "project": project,
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        _DOCUMENTS[doc_id] = doc
        return {"document": doc}

    if id not in _DOCUMENTS:
        raise ValueError(f"Entity not found: Document {id!r}.")
    doc = _DOCUMENTS[id]
    if title is not None:
        doc["title"] = title
    if content is not None:
        doc["content"] = content
    doc["updatedAt"] = _now()
    return {"document": doc}


@mcp.tool()
def get_document(id: Annotated[str, "Document ID"]) -> dict[str, Any]:
    if id not in _DOCUMENTS:
        raise ValueError(f"Entity not found: Document {id!r}.")
    return {"document": _DOCUMENTS[id]}


@mcp.tool()
def list_documents(
    project: Annotated[str | None, "Filter by project ID or name"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    docs = list(_DOCUMENTS.values())
    if project:
        target = _resolve_project_id(project)
        docs = [d for d in docs if d.get("project") in (project, target)]
    return {"documents": docs[:limit]}


# ---- attachments -----------------------------------------------------


@mcp.tool()
def create_attachment(
    issueId: Annotated[str, "Issue ID or identifier"],
    title: Annotated[str, "Attachment title"],
    content: Annotated[str, "Attachment content (text)"],
    url: Annotated[str | None, "Optional source URL"] = None,
) -> dict[str, Any]:
    issue = _find_issue(issueId)
    att_id = _new_id()
    attachment = {
        "id": att_id,
        "issueId": issue["id"],
        "title": title,
        "content": content,
        "url": url,
        "createdAt": _now(),
    }
    _ATTACHMENTS[att_id] = attachment
    issue.setdefault("attachments", []).append({"id": att_id, "title": title, "url": url})
    return {"attachment": {"id": att_id, "title": title, "url": url}}


@mcp.tool()
def get_attachment(id: Annotated[str, "Attachment ID"]) -> dict[str, Any]:
    if id not in _ATTACHMENTS:
        raise ValueError(f"Entity not found: Attachment {id!r}.")
    return {"attachment": _ATTACHMENTS[id]}


@mcp.tool()
def list_attachments(
    issueId: Annotated[str, "Issue ID or identifier"],
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    issue = _find_issue(issueId)
    items = [a for a in _ATTACHMENTS.values() if a["issueId"] == issue["id"]]
    return {"attachments": items[:limit]}


# ---- entrypoint ------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LINEAR_BENCH_PORT", "5021"))
    app = mcp.streamable_http_app()
    print(f"linear bench fixture: serving streamable_http on http://127.0.0.1:{port}/mcp")
    print(f"  default team: {_DEFAULT_TEAM_NAME} ({_DEFAULT_TEAM_KEY}, {_DEFAULT_TEAM_ID})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
