# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Daily triage of open PRs targeting `main`: requests reviewers, closes stale PRs,
and reports what needs review.

Changes to GitHub only happen with --apply; without it the script says what it would do.

Outputs (all optional):
  --full-report-file     every tracked PR
  --summary-report-file  non-draft stale PRs + PRs with status "Needs Review"
  --slack-message-file   the summary as a Slack chat.postMessage body

ACTIONS. Drafts and stale PRs get no review requests. A reviewer list is skipped
if anyone on it has already reviewed or been asked.
  GenAI label        request GENAI_REVIEWERS
  Non-GenAI label    request NON_GENAI_INITIAL_REVIEWERS; once one approves,
                     request NON_GENAI_FOLLOWUP_REVIEWERS instead
  Robotics label     request ROBOTICS_REVIEWERS (whatever other labels the PR has)
  Stale PR           close it with a message, unless the author is in
                     FIRST_PARTY_DEVELOPER_TEAMS or is in BOT_AUTHORS.
                     Closed PRs are left out of the reports.

GROUP. A PR goes in the first group that fits:
  1. Stale                   no PR activity for 14+ days
  2. Onboarding - GenAI      has that label
  3. Onboarding - Non-GenAI  has that label
  4. Onboarding - Robotics   has that label
  5. Other                   no onboarding label, author not in
                             FIRST_PARTY_DEVELOPER_TEAMS

STATUS. First match wins:
  1. Draft                      the PR is a draft
  2. Approved                   a reviewer in REVIEWERS_TEAM approved the PR; there are no requested changes
  3. WIP                        the Issues column has anything other than "CI running" or "changes requested", and
                                  the PR author has not re-requested a review from everyone
  4. Needs Review (India Team)  (Non-GenAI group only) no India reviewer has approved
  5. Needs Review               everything else

ISSUES.
  review comments pending    a comment thread is open & the PR author did not write last reply
  changes requested          a reviewer selected "request changes"
  merge conflicts            merge conflicts
  failing CI                 CI failed
  CI running                 CI still running
  required checks not run    a CI check required to merge never ran

REVIEWER(S). "AIHM reviewers" are the members of REVIEWERS_TEAM plus
NON_GENAI_INITIAL_REVIEWERS (the India reviewers); everyone else is ignored.
  Approved PR      the AIHM reviewers who approved
  Any other PR     the AIHM reviewers who commented; if none have, the ones who
                   were asked to review or are assigned to the PR
Non-GenAI PRs adjust that first:
  an India reviewer approved         leave the India reviewers out
  no India approval yet              if an India reviewer was asked to review,
                                     show only the India reviewers

LAST UPDATE is the author's last activity (open, commit, force-push,
ready-for-review, comment, edit).

LAST REVIEW is a reviewer's last review or
comment, blank if none. Both show hours under a day (5h), otherwise days (3.2d).

Requires the `gh` CLI, authenticated locally via `gh auth login` or in Actions via
GH_TOKEN. The token must be able to read org teams (`read:org`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pr_triage_report import build_tables, render_markdown, render_slack_payload
from pr_triage_types import (
    APPROVED,
    DRAFT,
    GENAI,
    GROUP_ORDER,
    LABEL_GROUPS,
    NEEDS_REVIEW,
    NEEDS_REVIEW_INDIA,
    NON_GENAI,
    OTHER,
    PR,
    PUBLIC_GROUP,
    ROBOTICS,
    STALE,
    WIP,
    Activity,
    Context,
)

# =====================================================================
# 1. RULES
# =====================================================================

# --- People -----------------------------------------------------------

# Members of this team, plus NON_GENAI_INITIAL_REVIEWERS, are the AIHM reviewers.
REVIEWERS_TEAM = "ai-hub-models-reviewers"
# Reviewers for GenAI PRs.
GENAI_REVIEWERS = ["gustlars_QCOM", "shreyajn_QCOM", "weidai_QCOM"]
# Reviewers for robotics PRs.
ROBOTICS_REVIEWERS = ["weidai_QCOM"]
# Initial (first round) reviewers for non-GenAI model onboarding.
NON_GENAI_INITIAL_REVIEWERS = ["krisrajk_QCOM", "souhghos_QCOM", "rchinnan_QCOM"]
# Final approvers (second round review) for non-GenAI model onboarding.
NON_GENAI_FOLLOWUP_REVIEWERS = ["kwatson_QCOM", "ashwmurt_QCOM", "khoin_QCOM"]
# Authors in these teams are our own developers; their PRs aren't triaged as external.
FIRST_PARTY_DEVELOPER_TEAMS = ["tetra-developers", "tetra-nexa-developers-china"]
BOT_AUTHORS = {"qaihm-bot_QCOM"}

# --- Repository -------------------------------------------------------

BASE_BRANCH = "main"
# Open PRs here are listed in their own report section, with no other checks.
PUBLIC_REPO = "qualcomm/ai-hub-models"
STALE_DAYS = 14
STALE_CLOSE_MESSAGE = (
    f"Closing this PR because it has had no activity for {STALE_DAYS}+ days. "
    "Feel free to reopen it when you are ready to continue."
)

ISSUE_COMMENTS = "review comments pending"
ISSUE_CHANGES_REQUESTED = "changes requested"
ISSUE_CONFLICTS = "merge conflicts"
ISSUE_CI_FAILING = "failing CI"
ISSUE_CI_RUNNING = "CI running"
ISSUE_CHECKS_MISSING = "required checks not run"
# Shown in Issues, but alone they leave an unapproved PR at Needs Review.
NOT_WIP_ISSUES = {ISSUE_CI_RUNNING, ISSUE_CHANGES_REQUESTED}

# --- Fetching and retries ---------------------------------------------

PR_PAGE_SIZE = 30

# gh calls are tried MAX_ATTEMPTS times, waiting 5s, 10s, 20s between tries.
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 5
TRANSIENT_ERRORS = (
    "http 5",
    "502",
    "503",
    "504",
    "rate limit",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporary failure",
    "eof",
    "unexpected end of json",
    "bad gateway",
    "service unavailable",
    "gateway time",
    "internal server error",
)

# =====================================================================
# 2. DATA TYPES
# =====================================================================


# =====================================================================
# 3. FETCH (all the network code)
#
# We use GraphQL rather than the REST API because REST is far too slow here.
# Per PR we need each reviewer's latest review, comments, review threads,
# commits, labels, CI status and merge state. REST needs about 5 requests per PR,
# one after another (and has no flag for whether a review thread is resolved),
# while one GraphQL request returns all of that for 30 PRs at once.
# =====================================================================

# Threads fetch only the first and last comments, to stay well under GitHub's
# 500k-node limit.
THREAD_FRAGMENT = """
fragment ThreadFields on PullRequestReviewThread {
  isResolved
  firstComment: comments(first: 1) {
    nodes { author { login __typename } createdAt }
  }
  lastComments: comments(last: 2) {
    nodes { author { login } createdAt }
  }
}
"""

PR_FIELDS = """
        number
        state
        title
        url
        isDraft
        mergeable
        createdAt
        updatedAt
        lastEditedAt
        author { login }
        labels(first: 20) { nodes { name } }
        assignees(first: 10) { nodes { login } }
        reviewRequests(first: 20) {
          nodes { requestedReviewer { ... on User { login } } }
        }
        latestOpinionatedReviews(first: 100) {
          nodes { author { login } state submittedAt }
        }
        latestReviews(first: 100) {
          nodes { author { login } state }
        }
        comments(last: 100) {
          nodes { author { login } createdAt }
        }
        reviewThreads(first: 100) {
          pageInfo { hasNextPage endCursor }
          nodes { ...ThreadFields }
        }
        commits(last: 1) {
          nodes {
            commit {
              committedDate
              statusCheckRollup {
                state
                contexts(first: 100) {
                  totalCount
                  nodes {
                    __typename
                    ... on CheckRun { name }
                    ... on StatusContext { context }
                  }
                }
              }
            }
          }
        }
        timelineItems(
          last: 1
          itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT, READY_FOR_REVIEW_EVENT]
        ) {
          nodes {
            ... on HeadRefForcePushedEvent { createdAt }
            ... on ReadyForReviewEvent { createdAt }
          }
        }
"""

PR_QUERY = (
    """
query($owner: String!, $repo: String!, $base: String, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(states: OPEN, baseRefName: $base, first: $pageSize, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {"""
    + PR_FIELDS
    + """      }
    }
  }
}
"""
    + THREAD_FRAGMENT
)

PR_BY_NUMBER_QUERY = (
    """
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {"""
    + PR_FIELDS
    + """    }
  }
}
"""
    + THREAD_FRAGMENT
)

THREADS_QUERY = (
    """
query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { ...ThreadFields }
      }
    }
  }
}
"""
    + THREAD_FRAGMENT
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def is_transient(stderr: str) -> bool:
    return any(marker in stderr.lower() for marker in TRANSIENT_ERRORS)


def gh(args: list[str]) -> str:
    """Run `gh <args>` and return stdout, retrying transient failures."""
    name = " ".join(args[:2])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(
            ["gh", *args], check=False, capture_output=True, text=True
        )
        if proc.returncode == 0:
            return proc.stdout
        error = proc.stderr.strip()
        if attempt == MAX_ATTEMPTS or not is_transient(error):
            raise RuntimeError(f"gh {name} failed: {error}")
        delay = RETRY_BASE_DELAY_S * 2 ** (attempt - 1)
        log(
            f"gh {name} failed ({error.splitlines()[0]}); "
            f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay}s"
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def gh_json(args: list[str]) -> Any:
    return json.loads(gh(args))


def graphql(query: str, **variables: Any) -> dict[str, Any]:
    """Run a GraphQL query and return its `repository` object."""
    args = ["api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        if value is not None:
            args += ["-F", f"{name}={value}"]
    return gh_json(args)["data"]["repository"]


def fetch_team_members(org: str, slug: str) -> set[str]:
    log(f"Fetching members of {org}/{slug}...")
    out = gh(
        ["api", "--paginate", f"orgs/{org}/teams/{slug}/members", "--jq", ".[].login"]
    )
    members = set(out.split())
    log(f"  {len(members)} members")
    if not members:
        raise RuntimeError(
            f"{org}/{slug} has no visible members; the team may have been renamed "
            "or the token lacks read:org access"
        )
    return members


def fetch_required_checks(owner: str, repo: str) -> set[str]:
    """Names of the checks the base branch requires."""
    log(f"Fetching required checks for {BASE_BRANCH}...")
    try:
        rules = gh_json(["api", f"repos/{owner}/{repo}/rules/branches/{BASE_BRANCH}"])
    except RuntimeError as e:
        log(f"  could not read branch rules ({e}); skipping the required-check test")
        return set()
    required = {
        check["context"]
        for rule in rules
        if rule["type"] == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    }
    log(f"  {len(required)} required checks")
    return required


def fetch_remaining_threads(owner: str, repo: str, gql_pr: dict[str, Any]) -> None:
    """Fetch a PR's review threads beyond the first 100, in place."""
    threads = gql_pr["reviewThreads"]
    while threads["pageInfo"]["hasNextPage"]:
        log(f"  PR #{gql_pr['number']}: {len(threads['nodes'])}+ review threads...")
        page = graphql(
            THREADS_QUERY,
            owner=owner,
            repo=repo,
            number=gql_pr["number"],
            cursor=threads["pageInfo"]["endCursor"],
        )["pullRequest"]["reviewThreads"]
        threads["nodes"] += page["nodes"]
        threads["pageInfo"] = page["pageInfo"]


def fetch_pr(owner: str, repo: str, number: int) -> dict[str, Any]:
    """One PR, in the same shape `fetch_open_prs` returns."""
    log(f"Fetching PR #{number}...")
    gql_pr = graphql(PR_BY_NUMBER_QUERY, owner=owner, repo=repo, number=number)[
        "pullRequest"
    ]
    fetch_remaining_threads(owner, repo, gql_pr)
    return gql_pr


def fetch_open_prs(
    owner: str, repo: str, base: str | None = BASE_BRANCH
) -> list[dict[str, Any]]:
    """Every open PR into `base` (any branch if None), as the dicts that GraphQL
    returns (named `gql_pr` below).

    A `gql_pr` has the fields requested in PR_QUERY (latest reviews, comments, review
    threads, commits, labels, ...) exactly as GitHub sends them. The rest of the
    script turns each one into a `PR`.
    """
    prs: list[dict[str, Any]] = []
    cursor = None
    while True:
        log(f"Fetching open PRs (page {len(prs) // PR_PAGE_SIZE + 1})...")
        page = graphql(
            PR_QUERY,
            owner=owner,
            repo=repo,
            base=base,
            pageSize=PR_PAGE_SIZE,
            cursor=cursor,
        )["pullRequests"]
        for gql_pr in page["nodes"]:
            fetch_remaining_threads(owner, repo, gql_pr)
        prs += page["nodes"]
        log(f"  {len(prs)} PRs fetched so far")
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]


# =====================================================================
# 4. FACTS ABOUT ONE PR
# =====================================================================


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def days_since(now: datetime, then: datetime) -> float:
    return (now - then).total_seconds() / 86400


def login(node: dict[str, Any]) -> str:
    return (node.get("author") or {}).get("login", "")


def pr_author(gql_pr: dict[str, Any]) -> str:
    """The PR author's login; "ghost" for a deleted account."""
    return (gql_pr["author"] or {}).get("login", "ghost")


def is_bot(node: dict[str, Any]) -> bool:
    author = node.get("author") or {}
    return author.get("__typename") == "Bot" or author.get("login") in BOT_AUTHORS


def thread_comments(thread: dict[str, Any]) -> list[dict[str, Any]]:
    """The comments we fetched for a thread: its first and its last two."""
    return [*thread["firstComment"]["nodes"], *thread["lastComments"]["nodes"]]


def activity_events(gql_pr: dict[str, Any]) -> list[tuple[str, datetime]]:
    """(login, time) of each approval or change request, PR comment and thread comment."""
    events = [
        (login(review), parse_ts(review["submittedAt"]))
        for review in gql_pr["latestOpinionatedReviews"]["nodes"]
    ]
    events += [
        (login(comment), parse_ts(comment["createdAt"]))
        for comment in gql_pr["comments"]["nodes"]
    ]
    for thread in gql_pr["reviewThreads"]["nodes"]:
        events += [
            (login(comment), parse_ts(comment["createdAt"]))
            for comment in thread_comments(thread)
        ]
    return events


def awaits_author(thread: dict[str, Any], author: str) -> bool:
    """An open thread started by someone else (not a bot) that the author hasn't
    answered last.
    """
    first = thread["firstComment"]["nodes"]
    if thread["isResolved"] or not first:
        return False
    starter = first[0]
    last = thread["lastComments"]["nodes"][-1]
    return (
        login(starter) not in ("", author)
        and not is_bot(starter)
        and login(last) != author
    )


def collect_activity(
    gql_pr: dict[str, Any], author: str, listed_reviewers: set[str]
) -> Activity:
    events = activity_events(gql_pr)
    author_times = [
        parse_ts(gql_pr["createdAt"]),
        *([parse_ts(gql_pr["lastEditedAt"])] if gql_pr["lastEditedAt"] else []),
        *(parse_ts(c["commit"]["committedDate"]) for c in gql_pr["commits"]["nodes"]),
        *(parse_ts(t["createdAt"]) for t in gql_pr["timelineItems"]["nodes"]),
        *(when for who, when in events if who == author),
    ]
    reviewer_times = [when for who, when in events if who in listed_reviewers]
    return Activity(
        last_author_update=max(author_times),
        last_review=max(reviewer_times, default=None),
        has_unanswered_thread=any(
            awaits_author(t, author) for t in gql_pr["reviewThreads"]["nodes"]
        ),
    )


def latest_review_states(
    gql_pr: dict[str, Any], aihm_reviewers: set[str]
) -> dict[str, str]:
    """Each reviewer's current review state: approval, change request or comment.

    GitHub tracks this per reviewer, so PRs with hundreds of reviews don't need
    paging. A later comment doesn't override an earlier approval or change request.
    """
    states = {
        login(r): r["state"]
        for r in gql_pr["latestReviews"]["nodes"]
        if login(r) in aihm_reviewers
    }
    states.update(
        (login(r), r["state"])
        for r in gql_pr["latestOpinionatedReviews"]["nodes"]
        if login(r) in aihm_reviewers
    )
    return states


def pending_requests(gql_pr: dict[str, Any], aihm_reviewers: set[str]) -> list[str]:
    """Reviewers with a pending review request. Team requests aren't included."""
    requested = (n["requestedReviewer"] for n in gql_pr["reviewRequests"]["nodes"])
    return [r["login"] for r in requested if r and r["login"] in aihm_reviewers]


def requested_reviewers(gql_pr: dict[str, Any], aihm_reviewers: set[str]) -> list[str]:
    """AIHM reviewers with a pending review request or assigned to the PR."""
    assigned = [a["login"] for a in gql_pr["assignees"]["nodes"]]
    logins = [*pending_requests(gql_pr, aihm_reviewers), *assigned]
    return [who for who in dict.fromkeys(logins) if who in aihm_reviewers]


# =====================================================================
# 5. GROUP
# =====================================================================


def pick_group(
    gql_pr: dict[str, Any], labels: set[str], author: str, ctx: Context
) -> str:
    if (ctx.now - parse_ts(gql_pr["updatedAt"])).days >= STALE_DAYS:
        return STALE
    for group in LABEL_GROUPS:
        if group in labels:
            return group
    if author in ctx.developers or author in BOT_AUTHORS:
        return ""
    return OTHER


# =====================================================================
# 6. ISSUES AND STATUS
# =====================================================================


def ci_issues(gql_pr: dict[str, Any], required_checks: set[str]) -> list[str]:
    issues = []
    if gql_pr["mergeable"] == "CONFLICTING":
        issues.append(ISSUE_CONFLICTS)
    rollup = gql_pr["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
    if rollup is None:
        if required_checks:
            issues.append(ISSUE_CHECKS_MISSING)
        return issues
    if rollup["state"] in ("FAILURE", "ERROR"):
        issues.append(ISSUE_CI_FAILING)
    elif rollup["state"] in ("PENDING", "EXPECTED"):
        # A job that hasn't started yet looks like a missing check.
        return [*issues, ISSUE_CI_RUNNING]
    contexts = rollup["contexts"]
    # With more checks than we fetched, we can't tell which are missing.
    fetched_all = contexts["totalCount"] <= len(contexts["nodes"])
    seen = {c.get("name") or c.get("context") for c in contexts["nodes"]}
    if fetched_all and not required_checks <= seen:
        issues.append(ISSUE_CHECKS_MISSING)
    return issues


def find_issues(
    gql_pr: dict[str, Any],
    activity: Activity,
    changes_requested: bool,
    required_checks: set[str],
) -> list[str]:
    issues = []
    if activity.has_unanswered_thread:
        issues.append(ISSUE_COMMENTS)
    if changes_requested:
        issues.append(ISSUE_CHANGES_REQUESTED)
    return [*issues, *ci_issues(gql_pr, required_checks)]


def pick_status(
    is_draft: bool,
    approvers: set[str],
    issues: list[str],
    awaiting_india: bool,
    rerequested: bool,
) -> str:
    """`approvers` are the approvals that count; `awaiting_india` is Non-GenAI
    with no India approval yet. CI running and changes requested don't make a PR
    WIP, and neither do open threads once every reviewer was re-requested.
    """
    if is_draft:
        return DRAFT
    if approvers:
        return APPROVED
    ignored = NOT_WIP_ISSUES | ({ISSUE_COMMENTS} if rerequested else set())
    if set(issues) - ignored:
        return WIP
    return NEEDS_REVIEW_INDIA if awaiting_india else NEEDS_REVIEW


# =====================================================================
# 7. REVIEWERS
# =====================================================================


def pick_reviewers(
    submitted: list[str],
    requested: list[str],
    is_non_genai: bool,
    approvers: set[str],
    nongenai_initial: set[str],
) -> list[str]:
    listed = list(dict.fromkeys([*submitted, *requested]))
    if is_non_genai:
        if approvers & nongenai_initial:
            listed = [r for r in listed if r not in nongenai_initial]
        elif nongenai_initial & set(requested):
            listed = [r for r in listed if r in nongenai_initial]
    return listed


# =====================================================================
# 8. ANALYZE AND ACT
# =====================================================================


def analyze(gql_pr: dict[str, Any], ctx: Context) -> PR:
    author = pr_author(gql_pr)
    labels = {label["name"] for label in gql_pr["labels"]["nodes"]}
    # Members of the two reviewer teams, excluding the PR author.
    aihm_reviewers = ctx.aihm_reviewers - {author}
    # Reviewers currently requested in the UI's Reviewers list (team requests excluded).
    pending = set(pending_requests(gql_pr, aihm_reviewers))
    # Each reviewer's latest review state: approved, changes requested or commented.
    states = latest_review_states(gql_pr, aihm_reviewers)
    # Reviewers who have submitted a review of any kind.
    submitted = set(states)
    # "Last review" only counts activity from pending or submitted reviewers.
    activity = collect_activity(gql_pr, author, pending | submitted)
    changes_requested = "CHANGES_REQUESTED" in states.values()
    issues = find_issues(gql_pr, activity, changes_requested, ctx.required_checks)

    # An approver who is asked to review again no longer counts as approved.
    approvers = {
        who
        for who, state in states.items()
        if state == "APPROVED" and who not in pending
    }
    # Every reviewer who left a non-approving review (change request or comment)
    # is asked again: the PR is back with the reviewers, so open threads stop
    # making it WIP (they still show in Issues).
    awaiting_rereview = submitted - {
        who for who, s in states.items() if s == "APPROVED"
    }
    rerequested = bool(pending) and awaiting_rereview <= pending

    is_non_genai = NON_GENAI in labels
    # On Non-GenAI an India approval only hands the PR on; it doesn't approve it.
    counted = approvers - ctx.nongenai_initial_reviewers if is_non_genai else approvers
    # Outstanding change requests cancel every approval, even if re-requested.
    final_approvers = set() if changes_requested else counted
    status = pick_status(
        gql_pr["isDraft"],
        final_approvers,
        issues,
        awaiting_india=is_non_genai and not approvers,
        rerequested=rerequested,
    )
    if status == APPROVED:
        shown = sorted(final_approvers)
    else:
        requested = requested_reviewers(gql_pr, aihm_reviewers)
        shown = pick_reviewers(
            sorted(submitted),
            requested,
            is_non_genai,
            approvers,
            ctx.nongenai_initial_reviewers,
        )

    return PR(
        number=gql_pr["number"],
        title=gql_pr["title"],
        url=gql_pr["url"],
        author=author,
        group=pick_group(gql_pr, labels, author, ctx),
        status=status,
        issues=issues,
        reviewers=shown,
        days_since_update=days_since(ctx.now, activity.last_author_update),
        days_since_review=(
            None
            if activity.last_review is None
            else days_since(ctx.now, activity.last_review)
        ),
    )


def reviewers_to_request(gql_pr: dict[str, Any], ctx: Context) -> list[str]:
    """Reviewers the labels call for, skipping any group someone has already joined."""
    author = pr_author(gql_pr)
    labels = {label["name"] for label in gql_pr["labels"]["nodes"]}
    everyone = (
        ctx.aihm_reviewers
        | set(GENAI_REVIEWERS)
        | set(ROBOTICS_REVIEWERS)
        | set(NON_GENAI_FOLLOWUP_REVIEWERS)
    )
    states = latest_review_states(gql_pr, everyone)
    pending = set(pending_requests(gql_pr, everyone))
    approved = {who for who, s in states.items() if s == "APPROVED"} - pending

    groups: list[list[str]] = []
    if GENAI in labels:
        groups.append(GENAI_REVIEWERS)
    if NON_GENAI in labels:
        if approved & ctx.nongenai_initial_reviewers:
            groups.append(NON_GENAI_FOLLOWUP_REVIEWERS)
        else:
            groups.append(sorted(ctx.nongenai_initial_reviewers))
    if ROBOTICS in labels:
        groups.append(ROBOTICS_REVIEWERS)

    present = pending | set(states)
    wanted = [
        who
        for group in groups
        if not present & set(group)  # someone from the group is already on the PR
        for who in group
    ]
    return [who for who in dict.fromkeys(wanted) if who != author]


def request_reviewers(
    owner: str, repo: str, number: int, logins: list[str], apply: bool
) -> None:
    if not apply:
        log(f"  #{number}: would request {', '.join(logins)}")
        return
    args = [
        "api",
        "-X",
        "POST",
        f"repos/{owner}/{repo}/pulls/{number}/requested_reviewers",
    ]
    for who in logins:
        args += ["-f", f"reviewers[]={who}"]
    try:
        gh(args)
    except RuntimeError as e:
        log(f"  #{number}: could not request {', '.join(logins)}: {e}")
        return
    log(f"  #{number}: requested {', '.join(logins)}")


def close_stale_pr(owner: str, repo: str, number: int, apply: bool) -> bool:
    """Close a stale PR with a message. False if it stays open."""
    if not apply:
        log(f"  #{number}: would close as stale")
        return False
    try:
        gh(
            [
                *["pr", "close", str(number), "--repo", f"{owner}/{repo}"],
                *["--comment", STALE_CLOSE_MESSAGE],
            ]
        )
    except RuntimeError as e:
        log(f"  #{number}: could not close: {e}")
        return False
    log(f"  #{number}: closed as stale")
    return True


def build_context(owner: str, repo: str) -> Context:
    main_reviewers = fetch_team_members(owner, REVIEWERS_TEAM)
    return Context(
        aihm_reviewers=main_reviewers | set(NON_GENAI_INITIAL_REVIEWERS),
        nongenai_initial_reviewers=set(NON_GENAI_INITIAL_REVIEWERS) - main_reviewers,
        developers=set().union(
            *(fetch_team_members(owner, team) for team in FIRST_PARTY_DEVELOPER_TEAMS)
        ),
        required_checks=fetch_required_checks(owner, repo),
        now=datetime.now(timezone.utc),
    )


def triage_pr(
    owner: str, repo: str, gql_pr: dict[str, Any], ctx: Context, apply: bool
) -> PR | None:
    """Request reviewers for the PR, or close it if stale. None if it was closed."""
    pr = analyze(gql_pr, ctx)
    owned_by_tetra = pr.author in ctx.developers or pr.author in BOT_AUTHORS
    if pr.group == STALE and not owned_by_tetra:
        if close_stale_pr(owner, repo, pr.number, apply):
            return None
    elif pr.status != DRAFT and pr.group != STALE:
        to_request = reviewers_to_request(gql_pr, ctx)
        if to_request:
            request_reviewers(owner, repo, pr.number, to_request, apply)
    return pr


def build_groups(owner: str, repo: str, apply: bool = False) -> dict[str, list[PR]]:
    ctx = build_context(owner, repo)
    gql_prs = fetch_open_prs(owner, repo)
    groups: dict[str, list[PR]] = {name: [] for name in GROUP_ORDER}
    for i, gql_pr in enumerate(gql_prs, 1):
        log(f"Analyzing PR {i}/{len(gql_prs)} (#{gql_pr['number']})")
        pr = triage_pr(owner, repo, gql_pr, ctx, apply)
        if pr and pr.group:
            groups[pr.group].append(pr)
    return groups


def fetch_public_prs(now: datetime) -> list[PR]:
    """Open PRs in PUBLIC_REPO, with no checks, only the time since the last update."""
    owner, repo = PUBLIC_REPO.split("/")
    prs = []
    for gql_pr in fetch_open_prs(owner, repo, base=None):
        author = pr_author(gql_pr)
        activity = collect_activity(gql_pr, author, set())
        prs.append(
            PR(
                number=gql_pr["number"],
                title=gql_pr["title"],
                url=gql_pr["url"],
                author=author,
                group=PUBLIC_GROUP,
                days_since_update=days_since(now, activity.last_author_update),
            )
        )
    return sorted(prs, key=lambda p: -p.days_since_update)


def summarize_groups(groups: dict[str, list[PR]]) -> dict[str, list[PR]]:
    """Non-draft stale PRs, plus PRs with status Needs Review."""
    summary = {}
    for name, prs in groups.items():
        if name == STALE:
            summary[name] = [p for p in prs if p.status != DRAFT]
        else:
            summary[name] = [p for p in prs if p.status == NEEDS_REVIEW]
    return summary


# =====================================================================
# 9. MAIN
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name of the repository (default: $GITHUB_REPOSITORY, else the "
        "repo `gh` resolves for the current directory)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Triage only this PR number and skip the reports.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Request reviewers and close stale PRs on GitHub. Without this, only "
        "log what would be done.",
    )
    parser.add_argument(
        "--full-report-file",
        help="Write the full markdown report (all PRs) to this file.",
    )
    parser.add_argument(
        "--summary-report-file",
        help="Write a markdown report with only stale PRs and PRs awaiting "
        "review to this file.",
    )
    parser.add_argument(
        "--slack-message-file",
        help="Write the summary as a Slack chat.postMessage JSON body (native "
        "table blocks) to this file.",
    )
    parser.add_argument(
        "--step-summary-file",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Append the full report here (defaults to the Actions step summary).",
    )
    return parser.parse_args()


def resolve_repo(repo: str | None) -> str:
    """--repo or $GITHUB_REPOSITORY, else the repo `gh` resolves locally."""
    if repo:
        return repo
    resolved = gh(
        ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    ).strip()
    log(f"Using repo {resolved} (from `gh`). If wrong, pass --repo.")
    return resolved


def main() -> None:
    args = parse_args()
    owner, repo = resolve_repo(args.repo).split("/")

    if args.pr:
        gql_pr = fetch_pr(owner, repo, args.pr)
        if gql_pr["state"] != "OPEN":
            log(f"PR #{args.pr} is {gql_pr['state'].lower()}, nothing to do.")
            return
        triage_pr(owner, repo, gql_pr, build_context(owner, repo), args.apply)
        return

    groups = build_groups(owner, repo, args.apply)
    summary_groups = summarize_groups(groups)
    try:
        public_prs = fetch_public_prs(datetime.now(timezone.utc))
    except Exception as e:
        log(f"Skipping the {PUBLIC_GROUP} section: {e!r}")
        public_prs = []

    full_tables = build_tables(groups, public_prs)
    summary_tables = build_tables(summary_groups, public_prs)

    full_report = render_markdown(full_tables, "Open PR Digest - Full")
    print(full_report)
    if args.step_summary_file:
        with open(args.step_summary_file, "a") as f:
            f.write(full_report + "\n")
    if args.full_report_file:
        Path(args.full_report_file).write_text(full_report + "\n")
    if args.summary_report_file:
        summary = render_markdown(summary_tables, "Open PR Digest - Summary")
        Path(args.summary_report_file).write_text(summary + "\n")
    if args.slack_message_file:
        Path(args.slack_message_file).write_text(
            json.dumps(render_slack_payload(summary_tables))
        )


if __name__ == "__main__":
    main()
