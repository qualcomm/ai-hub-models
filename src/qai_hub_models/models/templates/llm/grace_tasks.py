# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Repo-side naming for the Grace metric.

The `grader/` package is shared verbatim with GenAI Lab and holds only scoring
logic. Everything here is this repo's own: the CLI alias, the task set the
prompt-generating evaluators are gated on, the published doc link, and the label
scores are filed under.
"""

from __future__ import annotations

from qai_hub_models.models.templates.llm.grader.grace import (
    GRACE_TASK_NAME,
    GRACE_VERSION,
)

# Convenience alias on the command line: always the latest version.
GRACE_TASK_ALIAS = "grace"

# Public documentation for the prompt set, the grader rubric, and how the score
# is computed. Published alongside the score as the metric's dataset link.
GRACE_DOC_URL = (
    "https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/grace.md"
)

# The label the score is reported under. Versioned, matching this repo's metric
# registry (GRACE1_GRADE, GRACE2_GRADE) so a v1 number and a v2 number stay
# distinguishable in published baselines.
GRACE_METRIC_NAME = f"Grace{GRACE_VERSION}"

# The image + question set, graded the same way.
MULTIMODAL_TASK_NAME = "multimodal_prompts"


def resolve_task_name(task: str) -> str:
    """Map the version-less ``grace`` alias onto the current Grace task name."""
    return GRACE_TASK_NAME if task == GRACE_TASK_ALIAS else task
