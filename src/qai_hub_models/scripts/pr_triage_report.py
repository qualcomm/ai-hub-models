# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Markdown and Slack renderers for the pr_triage digest."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pr_triage_types import (
    GROUP_ORDER,
    OTHER,
    PR,
    PUBLIC_GROUP,
    STALE,
    STATUS_ORDER,
)

# Slack allows 100 rows per table and 10,000 characters of table text per message.
SLACK_TABLE_MAX_ROWS = 100
SLACK_TABLE_CHAR_BUDGET = 10_000
SLACK_PR_TITLE_MAX = 60
# The most of the Slack budget the public-repo table may take.
SLACK_PUBLIC_CHAR_BUDGET = 2_000


def group_title(group: str) -> str:
    """The table title; label names are fixed by GitHub, so they're mapped here."""
    if group == OTHER:
        return "Other External PRs"
    return group.replace("Onboarding", "External Onboarding", 1)


def fmt_age(days: float | None) -> str:
    """5h under a day, otherwise 3.2d; empty if there is no date."""
    if days is None:
        return ""
    hours = round(days * 24)
    return f"{hours}h" if hours < 24 else f"{days:.1f}d"


def sort_prs(group: str, prs: list[PR]) -> list[PR]:
    """Longest since the last update first; non-stale groups sort by status first."""
    return sorted(
        prs,
        key=lambda p: (
            0 if group == STALE else STATUS_ORDER.index(p.status),
            -p.days_since_update,
        ),
    )


def pr_label(pr: PR, max_title: int | None = None) -> str:
    """The PR cell: #123 title (author)."""
    title = pr.title
    if max_title is not None and len(title) > max_title:
        title = title[: max_title - 1] + "…"
    return f"#{pr.number} {title} ({pr.author})"


@dataclass
class Column:
    """A report column after the PR column."""

    header: str
    value: Callable[[PR], str]
    slack: dict[str, Any] = field(default_factory=dict)  # Slack column_settings


LAST_UPDATE = Column(
    "Last update", lambda pr: fmt_age(pr.days_since_update), {"align": "right"}
)
GROUP_COLUMNS = [
    Column("Status", lambda pr: pr.status),
    Column("Issues", lambda pr: ", ".join(pr.issues), {"is_wrapped": True}),
    Column("Reviewer(s)", lambda pr: ", ".join(pr.reviewers), {"is_wrapped": True}),
    LAST_UPDATE,
    Column("Last review", lambda pr: fmt_age(pr.days_since_review), {"align": "right"}),
]


@dataclass
class Table:
    """One table in a report. The PR column comes first, then `columns`."""

    title: str
    prs: list[PR]
    columns: list[Column]
    slack_budget: int | None = None  # Slack character cap; None takes what is left

    @property
    def headers(self) -> list[str]:
        return ["PR", *(c.header for c in self.columns)]

    @property
    def slack_column_settings(self) -> list[dict[str, Any]]:
        return [{"is_wrapped": True}, *(c.slack for c in self.columns)]

    def cells(self, pr: PR, max_title: int | None = None) -> list[str]:
        return [pr_label(pr, max_title), *(c.value(pr) for c in self.columns)]


def build_tables(groups: dict[str, list[PR]], public_prs: list[PR]) -> list[Table]:
    """The tables to report: the non-empty groups in order, then the public PRs."""
    tables = [
        Table(group_title(name), sort_prs(name, groups[name]), GROUP_COLUMNS)
        for name in GROUP_ORDER
        if groups[name]
    ]
    if public_prs:
        tables.append(
            Table(PUBLIC_GROUP, public_prs, [LAST_UPDATE], SLACK_PUBLIC_CHAR_BUDGET)
        )
    return tables


def escape_markdown(text: str) -> str:
    """Escape characters that break a markdown link or table."""
    return re.sub(r"([\\\[\]|])", r"\\\1", text)


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown_table(table: Table) -> list[str]:
    lines = [
        f"## {table.title} ({len(table.prs)})",
        "",
        markdown_row(table.headers),
        markdown_row(["---"] * len(table.headers)),
    ]
    for pr in table.prs:
        first, *rest = table.cells(pr, None)
        lines.append(markdown_row([f"[{escape_markdown(first)}]({pr.url})", *rest]))
    return [*lines, ""]


def render_markdown(tables: list[Table], title: str) -> str:
    lines = [f"# {title}", ""]
    for table in tables:
        lines += render_markdown_table(table)
    return "\n".join(lines)


def slack_text_cell(text: str) -> dict[str, Any]:
    return {"type": "raw_text", "text": text or " "}


def slack_rich_cell(element: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "rich_text",
        "elements": [{"type": "rich_text_section", "elements": [element]}],
    }


def slack_bold_cell(text: str) -> dict[str, Any]:
    return slack_rich_cell({"type": "text", "text": text, "style": {"bold": True}})


def slack_link_cell(text: str, url: str) -> dict[str, Any]:
    return slack_rich_cell({"type": "link", "text": text, "url": url})


def slack_rows(table: Table, budget: int) -> tuple[list[list[dict[str, Any]]], int]:
    """The leading PRs that fit in `budget` characters, and the characters used.

    A table that shows no rows is not rendered, so it uses no characters.
    """
    rows: list[list[dict[str, Any]]] = []
    used = sum(len(h) for h in table.headers)
    for pr in table.prs:
        texts = table.cells(pr, SLACK_PR_TITLE_MAX)
        cost = sum(len(t) for t in texts)
        # The header row counts toward Slack's row limit.
        if used + cost > budget or len(rows) + 1 >= SLACK_TABLE_MAX_ROWS:
            break
        used += cost
        rows.append(
            [slack_link_cell(texts[0], pr.url), *map(slack_text_cell, texts[1:])]
        )
    return rows, used if rows else 0


def slack_section(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def render_slack_payload(tables: list[Table]) -> dict[str, Any]:
    """A chat.postMessage body with one table per Table.

    Rows past Slack's table limits are dropped and counted in a closing note.
    """
    # Tables with their own budget are laid out first so the others can't crowd
    # them out. The rest share what is left.
    laid_out = {id(t): slack_rows(t, t.slack_budget) for t in tables if t.slack_budget}
    budget = SLACK_TABLE_CHAR_BUDGET - sum(used for _, used in laid_out.values())
    for table in tables:
        if id(table) not in laid_out:
            laid_out[id(table)] = slack_rows(table, budget)
            budget -= laid_out[id(table)][1]

    blocks = [slack_section(":mag: *Open PR Digest*")]
    hidden = 0
    for table in tables:
        rows, _ = laid_out[id(table)]
        hidden += len(table.prs) - len(rows)
        if rows:
            blocks += [
                slack_section(f"*{table.title}* ({len(table.prs)})"),
                {
                    "type": "table",
                    "column_settings": table.slack_column_settings,
                    "rows": [[slack_bold_cell(h) for h in table.headers], *rows],
                },
            ]
    if hidden:
        note = f"{hidden} more PRs are not shown here. See the full report."
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": note}]}
        )
    return {"text": "Open PR Digest", "blocks": blocks}
