# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import ToolVersionsByPathYaml


def test_tool_versions_skips_unknown_profile_paths(tmp_path: Path) -> None:
    """A tool-versions.yaml written by older code can name a path that was since
    removed; it must load, keeping every path that still exists.
    """
    yaml_path = tmp_path / "tool-versions.yaml"
    yaml_path.write_text(
        "tool_versions:\n"
        "  qnn_dlc_via_qnn_ep:\n"
        "    qairt: 2.49.0.260730134355\n"
        "  qnn_dlc:\n"
        "    qairt: 2.50.0.260828221209\n"
    )

    loaded = ToolVersionsByPathYaml.from_yaml(yaml_path)

    assert list(loaded.tool_versions) == [ScorecardProfilePath.QNN_DLC]
    qairt = loaded.tool_versions[ScorecardProfilePath.QNN_DLC].qairt
    assert qairt is not None
    assert "2.50.0" in qairt.full_version_with_flavor
