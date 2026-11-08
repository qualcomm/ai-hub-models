# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared data types and names for pr_triage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# --- Groups -----------------------------------------------------------

# Stale is checked first (see pick_group) but listed last.
STALE = "Stale"
GENAI = "Onboarding - GenAI"
NON_GENAI = "Onboarding - Non-GenAI"
ROBOTICS = "Onboarding - Robotics"
OTHER = "Other"
LABEL_GROUPS = [GENAI, NON_GENAI, ROBOTICS]
GROUP_ORDER = [*LABEL_GROUPS, OTHER, STALE]
# Listed after GROUP_ORDER; not a triage group (see fetch_public_prs).
PUBLIC_GROUP = "Public Repo PRs"

# --- Statuses ---------------------------------------------------------

NEEDS_REVIEW = "Needs Review"
NEEDS_REVIEW_INDIA = "Needs Review (India Team)"
APPROVED = "Approved"
WIP = "WIP"
DRAFT = "Draft"
STATUS_ORDER = [NEEDS_REVIEW, NEEDS_REVIEW_INDIA, APPROVED, WIP, DRAFT]


@dataclass
class PR:
    """One row in a report."""

    number: int
    title: str
    url: str
    author: str
    group: str
    days_since_update: float
    status: str = ""
    issues: list[str] = field(default_factory=list)
    reviewers: list[str] = field(default_factory=list)
    days_since_review: float | None = None


@dataclass
class Context:
    """What is shared by every PR in a run."""

    # Both reviewer teams; nobody else counts.
    aihm_reviewers: set[str]
    # NON_GENAI_INITIAL_REVIEWERS, minus anyone also in the main team.
    nongenai_initial_reviewers: set[str]
    developers: set[str]
    required_checks: set[str]
    now: datetime


@dataclass
class Activity:
    """When the author and AIHM reviewers last acted on a PR."""

    last_author_update: datetime
    last_review: datetime | None
    has_unanswered_thread: bool
