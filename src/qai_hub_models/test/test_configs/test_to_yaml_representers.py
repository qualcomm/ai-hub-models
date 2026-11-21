# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Asserts that ``BaseQAIHMConfig.to_yaml``'s representers stay per-dump.

``add_representer`` is a classmethod that mutates a class-level registry, so
registering on a shared dumper class reconfigures every instance in the
process. That would make one ``flow_lists=True`` dump change the style of every
later dump, anywhere, including unrelated PyYAML users.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qai_hub_models.utils.base_config import BaseQAIHMConfig


class _StyleCfg(BaseQAIHMConfig):
    name: str = "x"
    items: list[str] = []


def _dump(path: Path, flow_lists: bool) -> str:
    cfg = _StyleCfg(name="a", items=["p", "q"])
    cfg.to_yaml(path, write_if_empty=True, flow_lists=flow_lists)
    return path.read_text()


def test_flow_lists_uses_inline_style(tmp_path: Path) -> None:
    assert "items: [p, q]" in _dump(tmp_path / "flow.yaml", flow_lists=True)


def test_flow_lists_does_not_leak_into_later_dumps(tmp_path: Path) -> None:
    _dump(tmp_path / "flow.yaml", flow_lists=True)
    block = _dump(tmp_path / "block.yaml", flow_lists=False)
    assert "items: [p, q]" not in block
    assert "- p" in block


def test_flow_lists_does_not_leak_into_unrelated_yaml(tmp_path: Path) -> None:
    _dump(tmp_path / "flow.yaml", flow_lists=True)
    assert yaml.safe_dump({"x": [1, 2, 3]}) == "x:\n- 1\n- 2\n- 3\n"
    assert yaml.safe_dump({"x": "a\nb"}) == "x: 'a\n\n  b'\n"


def test_newlines_dump_as_literal_block(tmp_path: Path) -> None:
    cfg = _StyleCfg(name="line1\nline2")
    path = tmp_path / "multi.yaml"
    cfg.to_yaml(path, write_if_empty=True)
    assert "name: |-\n  line1\n  line2" in path.read_text()
