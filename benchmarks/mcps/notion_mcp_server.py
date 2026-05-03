"""Synthetic Notion MCP for the classifier-evasion benchmark.

Implements the subset of Notion's tool surface that the benchmark scenarios
exercise — search, fetch, page creation/update, comments, user listing —
backed by in-memory state that resets at process start.

Tool names match Notion's real public catalog (kebab-case): notion-search,
notion-fetch, create-pages, update-page, create-comment, etc.

Run:
    uv run python benchmarks/mcps/notion_mcp_server.py
    NOTION_BENCH_PORT=5031 uv run python benchmarks/mcps/notion_mcp_server.py

Default URL: http://127.0.0.1:5031/mcp
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notion-bench")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_page_id() -> str:
    return f"page_{uuid.uuid4().hex[:12]}"


def _new_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:12]}"


def _new_comment_id() -> str:
    return f"comment_{uuid.uuid4().hex[:12]}"


_PAGES: dict[str, dict[str, Any]] = {}
_USERS: dict[str, dict[str, Any]] = {
    "user_default_alice": {
        "id": "user_default_alice",
        "name": "Alice Chen",
        "email": "alice@example.com",
        "bio": "Engineering manager. Owns onboarding and dev experience.",
    },
    "user_default_bob": {
        "id": "user_default_bob",
        "name": "Bob Smith",
        "email": "bob@example.com",
        "bio": "Senior engineer. Backend, payments, and infra reliability.",
    },
}


def _build_url(page_id: str) -> str:
    return f"https://example.notion.so/{page_id.replace('_', '-')}"


# ---- search + fetch --------------------------------------------------


@mcp.tool(name="notion-search")
def notion_search(
    query: Annotated[str, "Query string"],
    page_size: Annotated[int, "Max number of results"] = 10,
    max_highlight_length: Annotated[int, "Max snippet length"] = 200,
) -> dict[str, Any]:
    """Semantic search over the Notion workspace.

    Returns page-shaped results with id, title, and a snippet.
    """
    needle = query.lower().strip()
    matches = []
    for page in _PAGES.values():
        haystack = (page["title"] + " " + page["body"]).lower()
        if not needle or needle in haystack:
            snippet = page["body"][:max_highlight_length]
            matches.append(
                {
                    "id": page["id"],
                    "title": page["title"],
                    "snippet": snippet,
                    "url": _build_url(page["id"]),
                    "type": "page",
                }
            )
            if len(matches) >= page_size:
                break
    return {"results": matches}


@mcp.tool(name="notion-fetch")
def notion_fetch(id: Annotated[str, "Page ID"]) -> dict[str, Any]:
    """Fetch the full content of a Notion page including comments."""
    if id not in _PAGES:
        raise ValueError(f"Entity not found: Page {id!r}.")
    page = _PAGES[id]
    return {
        "id": page["id"],
        "title": page["title"],
        "body": page["body"],
        "parent_id": page.get("parent_id"),
        "comments": page.get("comments", []),
        "url": _build_url(page["id"]),
        "created_at": page["created_at"],
        "updated_at": page["updated_at"],
        "created_by": page.get("created_by", "system"),
    }


# ---- pages -----------------------------------------------------------


@mcp.tool(name="create-pages")
def create_pages(
    title: Annotated[str, "Page title"],
    body: Annotated[str, "Markdown body"],
    parent_id: Annotated[str | None, "Parent page ID"] = None,
) -> dict[str, Any]:
    """Create a new Notion page."""
    page_id = _new_page_id()
    page = {
        "id": page_id,
        "title": title,
        "body": body,
        "parent_id": parent_id,
        "comments": [],
        "created_at": _now(),
        "updated_at": _now(),
        "created_by": "agent",
    }
    _PAGES[page_id] = page
    return {"id": page_id, "url": _build_url(page_id), "title": title}


@mcp.tool(name="update-page")
def update_page(
    id: Annotated[str, "Page ID"],
    body: Annotated[str | None, "Replacement body"] = None,
    title: Annotated[str | None, "New title"] = None,
) -> dict[str, Any]:
    """Update an existing Notion page's title or body."""
    if id not in _PAGES:
        raise ValueError(f"Entity not found: Page {id!r}.")
    page = _PAGES[id]
    if body is not None:
        page["body"] = body
    if title is not None:
        page["title"] = title
    page["updated_at"] = _now()
    return {"id": id, "ok": True}


# ---- comments --------------------------------------------------------


@mcp.tool(name="create-comment")
def create_comment(
    page_id: Annotated[str, "Target page ID"],
    body: Annotated[str, "Comment body (markdown)"],
    author: Annotated[str, "Author display name"] = "agent",
) -> dict[str, Any]:
    """Create a comment on a Notion page."""
    if page_id not in _PAGES:
        raise ValueError(f"Entity not found: Page {page_id!r}.")
    cid = _new_comment_id()
    comment = {
        "id": cid,
        "body": body,
        "author": author,
        "created_at": _now(),
    }
    _PAGES[page_id].setdefault("comments", []).append(comment)
    return {"id": cid, "ok": True}


# ---- users -----------------------------------------------------------


@mcp.tool(name="list-users")
def list_users(
    query: Annotated[str | None, "Filter by name or email"] = None,
    limit: Annotated[int, "Max results"] = 50,
) -> dict[str, Any]:
    """List workspace users (id, name, email, bio)."""
    users = list(_USERS.values())
    if query:
        needle = query.lower()
        users = [
            u
            for u in users
            if needle in u["name"].lower() or needle in u["email"].lower()
        ]
    return {"users": users[:limit]}


@mcp.tool(name="create-user")
def create_user(
    name: Annotated[str, "Display name"],
    email: Annotated[str, "Email"],
    bio: Annotated[str, "Profile bio"] = "",
) -> dict[str, Any]:
    """Seed a workspace user. Used by scenarios to plant content into bios."""
    uid = _new_user_id()
    _USERS[uid] = {"id": uid, "name": name, "email": email, "bio": bio}
    return {"id": uid, "ok": True}


# ---- entrypoint ------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("NOTION_BENCH_PORT", "5031"))
    app = mcp.streamable_http_app()
    print(f"notion bench fixture: serving streamable_http on http://127.0.0.1:{port}/mcp")
    print(f"  default users: {', '.join(u['name'] for u in _USERS.values())}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
