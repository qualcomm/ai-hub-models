# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Asserts that ``BaseQAIHMConfig.to_yaml`` produces deterministic output
regardless of the input order of dict keys.

The test loads a model's perf.yaml / numerics.yaml / manifest.yaml, recursively
reverses every dict's key order, parses the scrambled YAML back through the
config class, re-saves via ``to_yaml`` to a tmp file, and asserts the result is
byte-identical to the original on-disk file. If the on-disk file is canonical
and ``to_yaml`` re-canonicalizes correctly, the round-trip is a no-op.

Currently scoped to ``resnet18``. Other models opt in once they've been re-saved
through the canonicalizing ``to_yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qai_hub_models_cli.proto_helpers._common import dump_yaml, safe_load_yaml

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.perf_yaml import (
    QAIHMModelPerf,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

CANONICALIZED_MODELS: tuple[str, ...] = ("resnet18",)


def _reverse_mappings(value: Any) -> Any:
    """Return ``value`` with every dict's key order reversed (recursively)."""
    if isinstance(value, dict):
        return {k: _reverse_mappings(value[k]) for k in reversed(list(value.keys()))}
    if isinstance(value, list):
        return [_reverse_mappings(v) for v in value]
    return value


def _assert_to_yaml_is_order_independent(
    config_cls: type[BaseQAIHMConfig] | None,
    yaml_path: Path,
    tmp_path: Path,
) -> None:
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path.name} does not exist for this model")

    with open(yaml_path) as f:
        data = safe_load_yaml(f)

    scrambled_path = tmp_path / f"scrambled-{yaml_path.name}"
    with open(scrambled_path, "w") as f:
        dump_yaml(_reverse_mappings(data), f)

    out_path = tmp_path / f"rewritten-{yaml_path.name}"
    if config_cls is None:
        QAIHMModelPerf.from_yaml(scrambled_path, yaml_path.parent.name).to_yaml(
            out_path
        )
    else:
        config_cls.from_yaml(scrambled_path).to_yaml(out_path)

    assert out_path.read_text() == yaml_path.read_text(), (
        f"{yaml_path.name}: re-saving a scrambled copy through "
        f"{config_cls.__name__ if config_cls else 'QAIHMModelPerf.to_yaml'} did not reproduce the original file. "
        f"Diff the two files in {tmp_path} for details."
    )


@pytest.mark.parametrize("model_id", CANONICALIZED_MODELS)
def test_perf_yaml_to_yaml_is_order_independent(model_id: str, tmp_path: Path) -> None:
    _assert_to_yaml_is_order_independent(
        None, QAIHM_MODELS_ROOT / model_id / "perf.yaml", tmp_path
    )


@pytest.mark.parametrize("model_id", CANONICALIZED_MODELS)
def test_numerics_yaml_to_yaml_is_order_independent(
    model_id: str, tmp_path: Path
) -> None:
    _assert_to_yaml_is_order_independent(
        QAIHMModelNumerics, QAIHM_MODELS_ROOT / model_id / "numerics.yaml", tmp_path
    )


@pytest.mark.parametrize("model_id", CANONICALIZED_MODELS)
def test_manifest_yaml_to_yaml_is_order_independent(
    model_id: str, tmp_path: Path
) -> None:
    _assert_to_yaml_is_order_independent(
        QAIHMModelManifest, QAIHM_MODELS_ROOT / model_id / "manifest.yaml", tmp_path
    )
