# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

r"""Scrape Slack + GitHub for unanswered customer questions.

Sister to scripts/examples/customer_questions/customer_questions_slack_github_export.py.
That script writes a CSV for human review. This one writes structured JSON
(with thread replies) for the Breeze customer-question agent.

Output schema:
    {
      "scraped_at": "2026-06-08T14:00:00Z",
      "lookback_days": 1,
      "questions": [
        {
          "id": "slack-CXXXXX-1731349200.000100",
          "medium": "slack" | "github",
          "channel_or_repo": "ai-hub-models",
          "thread_url": "https://qualcomm.slack.com/archives/...",
          "submitted_by": "Customer Name",
          "submitted_at": "2026-06-07T18:30:00Z",
          "title": "<github only>",
          "question": "Original message text",
          "thread_replies": [
            {"user": "Name", "is_internal": false, "text": "follow-up..."},
            ...
          ],
          "has_internal_reply": false
        }
      ]
    }

Filtering rules:
  - Internal (Qualcomm) authors are skipped (only customer questions surface).
  - Slack scrape is restricted to an explicit allowlist (BREEZE_AGENT_CHANNELS).
    The sibling CSV exporter in scripts/examples/customer_questions/ still uses
    the broader exclude-list approach in _common.py — the agent path is
    narrower on purpose.
  - Dependabot PRs are skipped.
  - GitHub items are pulled from GITHUB_REPOS with a longer lookback than
    Slack, so an unanswered issue keeps being reminded instead of aging out
    after the Slack window.

Usage:
    export SLACK_BOT_TOKEN=xoxb-***
    python -m scripts.breeze_customer_questions.scrape_questions \
        --lookback-days 1 \
        --max-questions 50 \
        --output "${TMPDIR:-/tmp}/claude/questions.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from slack_sdk import WebClient
from slack_sdk.web.slack_response import SlackResponse

# Run as `python -m ...` or directly. Either way, make _common importable.
_COMMON_DIR = Path(__file__).resolve().parents[1] / "examples" / "customer_questions"
sys.path.insert(0, str(_COMMON_DIR))

from _common import (  # noqa: E402  (sys.path tweak required first)
    DEPENDABOT_LOGINS,
    GITHUB_REPOS,
    get_user_info,
    is_qualcomm_user,
    sanitize_text,
)

logger = logging.getLogger(__name__)


# Explicit allowlist of Slack channels the Breeze agent is permitted to scrape.
# Coordinated with the Slack admin / Breeze team; deliberately narrower than the
# CSV exporter's exclude-list. Edit this list (not _common.py) when scope changes.
BREEZE_AGENT_CHANNELS: list[tuple[str, str]] = [
    ("C0A4W4XN6UQ", "ai-hub-apps"),
    ("C06LGRS3AC8", "ai-hub-models"),
    ("C06KMD15QH4", "general"),
]


def _iso(ts_float: float) -> str:
    return datetime.fromtimestamp(ts_float, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _slack_thread_url(
    client: WebClient, workspace: str, channel_id: str, ts: str
) -> str:
    """Resolve a clickable permalink for a Slack message.

    Prefers Slack's ``chat.getPermalink`` API — the returned URL redirects
    correctly for any user, including those not in the source channel. The
    hand-built ``/archives/<channel_id>/p<ts>`` form only renders as a clickable
    link for channel members, so cross-channel reminder readers see plain text.

    Falls back to the hand-built form on API failure so reminders still post.
    """
    try:
        resp: SlackResponse = client.chat_getPermalink(
            channel=channel_id, message_ts=ts
        )
        data = resp.data if hasattr(resp, "data") else resp  # type: ignore[assignment]
        if isinstance(data, Mapping):
            permalink = data.get("permalink")
            if isinstance(permalink, str) and permalink:
                return permalink
    except Exception as e:
        logger.warning(
            "chat.getPermalink failed for channel=%s ts=%s: %s; "
            "falling back to hand-built archive URL.",
            channel_id,
            ts,
            e,
        )
    ts_compact = ts.replace(".", "")
    return f"https://{workspace}.slack.com/archives/{channel_id}/p{ts_compact}"


def _fetch_thread_replies(
    client: WebClient, channel_id: str, thread_ts: str
) -> list[dict[str, Any]]:
    """Return reply messages (not including the parent) for a Slack thread."""
    replies: list[dict[str, Any]] = []
    try:
        resp: SlackResponse = client.conversations_replies(
            channel=channel_id, ts=thread_ts
        )
    except Exception as e:
        logger.warning(
            "Failed to fetch replies for channel=%s ts=%s: %s",
            channel_id,
            thread_ts,
            e,
        )
        return replies

    data = resp.data if hasattr(resp, "data") else resp  # type: ignore[assignment]
    if not isinstance(data, Mapping):
        return replies

    msgs = data.get("messages", [])
    if not isinstance(msgs, list):
        return replies

    # First message in the response is the parent — skip it.
    for m in msgs[1:]:
        if not isinstance(m, Mapping):
            continue
        user_id = str(m.get("user", "")) if m.get("user") is not None else ""
        user_name, is_internal = get_user_info(client, user_id)
        replies.append(
            {
                "user": user_name,
                "is_internal": is_internal,
                "text": sanitize_text(str(m.get("text", ""))),
            }
        )
    return replies


def _scrape_slack(
    client: WebClient,
    workspace: str,
    oldest_ts: float,
    max_questions: int,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for channel_id, channel_name in BREEZE_AGENT_CHANNELS:
        if len(questions) >= max_questions:
            break
        logger.info("Scanning #%s", channel_name)
        raw = dropped_broadcast = dropped_no_user = dropped_internal = 0
        dropped_no_text = dropped_no_ts = kept = 0

        msg_cursor: str | None = None
        while True:
            if len(questions) >= max_questions:
                break
            try:
                hist: SlackResponse = client.conversations_history(
                    channel=channel_id,
                    cursor=msg_cursor,
                    oldest=str(oldest_ts),
                )
            except Exception as e:
                logger.warning(
                    "conversations_history failed for #%s: %s", channel_name, e
                )
                break
            hdata = hist.data if hasattr(hist, "data") else hist  # type: ignore[assignment]
            if not isinstance(hdata, Mapping):
                break
            msgs = hdata.get("messages", [])
            if not isinstance(msgs, list):
                break

            for m in msgs:
                if len(questions) >= max_questions:
                    break
                if not isinstance(m, Mapping):
                    continue
                raw += 1
                # Skip thread broadcasts: a reply re-posted to the channel
                # view. The original thread is the canonical record; Slack's
                # permalink for a broadcast renders inconsistently for
                # non-channel readers ("view thread" not clickable).
                if m.get("subtype") == "thread_broadcast":
                    dropped_broadcast += 1
                    continue
                user_id = str(m.get("user", "")) if m.get("user") is not None else ""
                if not user_id:
                    dropped_no_user += 1
                    continue
                user_name, is_internal = get_user_info(client, user_id)
                if is_internal:
                    dropped_internal += 1
                    continue

                text = sanitize_text(str(m.get("text", "")))
                if not text:
                    dropped_no_text += 1
                    continue

                ts_raw = m.get("ts")
                try:
                    ts_float = float(ts_raw) if ts_raw is not None else 0.0
                except (TypeError, ValueError):
                    ts_float = 0.0
                if not ts_float:
                    dropped_no_ts += 1
                    continue
                kept += 1

                ts_str = str(ts_raw)
                replies = _fetch_thread_replies(client, channel_id, ts_str)
                has_internal_reply = any(r.get("is_internal") for r in replies)

                questions.append(
                    {
                        "id": f"slack-{channel_id}-{ts_str}",
                        "medium": "slack",
                        "channel_or_repo": channel_name,
                        "thread_url": _slack_thread_url(
                            client, workspace, channel_id, ts_str
                        ),
                        "submitted_by": user_name,
                        "submitted_at": _iso(ts_float),
                        "title": "",
                        "question": text,
                        "thread_replies": replies,
                        "has_internal_reply": has_internal_reply,
                    }
                )

            meta = (
                hdata.get("response_metadata") if isinstance(hdata, Mapping) else None
            )
            next_cursor = (
                str(meta.get("next_cursor"))
                if isinstance(meta, Mapping) and meta.get("next_cursor")
                else None
            )
            if not next_cursor:
                break
            msg_cursor = next_cursor

        logger.info(
            "#%s filter stats: raw=%d kept=%d dropped(broadcast=%d, no_user=%d, "
            "internal=%d, no_text=%d, no_ts=%d)",
            channel_name,
            raw,
            kept,
            dropped_broadcast,
            dropped_no_user,
            dropped_internal,
            dropped_no_text,
            dropped_no_ts,
        )

    return questions


def _scrape_github(since_date: str, max_questions: int) -> list[dict[str, Any]]:
    """Match the CSV exporter's GitHub-fetch behavior, but emit structured JSON."""
    out: list[dict[str, Any]] = []
    for repo in GITHUB_REPOS:
        if len(out) >= max_questions:
            break
        for kind in ("issue", "pr"):
            if len(out) >= max_questions:
                break
            try:
                result = subprocess.run(
                    [
                        "gh",
                        kind,
                        "list",
                        "--repo",
                        repo,
                        # Closed = already resolved; skip. Sister CSV exporter keeps `all`.
                        "--state",
                        "open",
                        "--search",
                        f"created:>={since_date}",
                        "--limit",
                        "200",
                        "--json",
                        "title,body,createdAt,url,comments,author,number",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning("gh fetch failed for %s/%s: %s", repo, kind, e)
                continue

            items = json.loads(result.stdout) if result.stdout.strip() else []
            for item in items:
                if len(out) >= max_questions:
                    break
                author_info = item.get("author", {}) or {}
                author_login = author_info.get("login", "")
                if kind == "pr" and author_login in DEPENDABOT_LOGINS:
                    continue
                # Best-effort customer filter via author profile (mirrors Slack rule).
                if is_qualcomm_user(author_info):
                    continue

                title = sanitize_text(item.get("title", ""))
                body = sanitize_text(item.get("body", "") or "")
                comments = item.get("comments", [])
                comment_count = len(comments) if isinstance(comments, list) else 0

                out.append(
                    {
                        "id": f"github-{repo}-{kind}-{item.get('number', '')}",
                        "medium": "github",
                        "channel_or_repo": repo,
                        "thread_url": item.get("url", ""),
                        "submitted_by": author_info.get("name")
                        or author_login
                        or "Unknown",
                        "submitted_at": item.get("createdAt", ""),
                        "title": title,
                        "question": body,
                        "thread_replies": [],  # comments not pulled in v1
                        "has_internal_reply": comment_count > 0,
                    }
                )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="How many days back to scrape Slack. Default: 1",
    )
    parser.add_argument(
        "--github-lookback-days",
        type=int,
        default=30,
        help="How many days back to scrape open GitHub issues/PRs. Default: 30",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=50,
        help="Hard cap on questions returned (Slack + GitHub combined). Default: 50",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write questions.json",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="qualcomm",
        help="Slack workspace subdomain for thread URLs. Default: qualcomm",
    )
    parser.add_argument(
        "--skip-slack",
        action="store_true",
        help="Skip Slack scrape (useful for GitHub-only test runs).",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub scrape (useful for Slack-only test runs).",
    )
    args = parser.parse_args()

    questions: list[dict[str, Any]] = []

    if not args.skip_slack:
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            raise RuntimeError(
                "SLACK_BOT_TOKEN not set. Export it before running, or pass --skip-slack."
            )
        client = WebClient(token=slack_token)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=args.lookback_days)
        ).timestamp()
        slack_questions = _scrape_slack(
            client,
            workspace=args.workspace,
            oldest_ts=cutoff,
            max_questions=args.max_questions,
        )
        questions.extend(slack_questions)
        logger.info("Slack: %d questions", len(slack_questions))

    if not args.skip_github and len(questions) < args.max_questions:
        since = (
            datetime.now(timezone.utc) - timedelta(days=args.github_lookback_days)
        ).strftime("%Y-%m-%d")
        gh_questions = _scrape_github(since, args.max_questions - len(questions))
        questions.extend(gh_questions)
        logger.info("GitHub: %d questions", len(gh_questions))

    payload = {
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": args.lookback_days,
        "questions": questions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Wrote %d questions to %s", len(questions), args.output)


if __name__ == "__main__":
    main()
