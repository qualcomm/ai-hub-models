# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared helper for checking yaml entries against manifest.yaml's enabled paths."""

from __future__ import annotations

from collections.abc import Iterable

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath


def get_release_manifest_support_violations(
    manifest: QAIHMModelManifest,
    yaml_name: str,
    precision: Precision,
    paths: Iterable[ScorecardInferencePath],
) -> list[str]:
    """
    Verifies that the manifest declares:
        - the provided precision is supported
        - the provided precision + path pairings are not disabled due to failures

    If either statement is not true, returns detailed verification failure reasons.
    """
    if precision not in manifest.supported_precisions:
        return [
            f"{manifest.id}: {yaml_name} has {precision}, "
            "which is not a supported precision in manifest.yaml"
        ]

    return [
        f"{manifest.id}: {yaml_name} lists {precision} + {path.name}, "
        "but that path is disabled in manifest.yaml"
        for path in sorted(paths, key=lambda p: p.name)
        if manifest.failure_reason(precision, path.runtime)
    ]
