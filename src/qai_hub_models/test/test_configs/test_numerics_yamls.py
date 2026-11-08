# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import tempfile

import pandas as pd
import pytest

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.numerics_yaml import (
    QAIHMModelNumerics,
    get_numerics_yaml_path,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.utils.numerics_yaml_helpers import (
    create_numerics_struct,
    get_chipset_registry,
)
from qai_hub_models.test.test_configs.manifest_path_check import (
    get_manifest_support_violations,
)
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.path_helpers import MODEL_IDS


def test_accuracy_yaml() -> None:
    path_errors: list[str] = []
    for model_id in MODEL_IDS:
        try:
            accuracy = QAIHMModelNumerics.from_model(model_id, not_exists_ok=True)
            if accuracy is None:
                continue
        except Exception as err:
            raise AssertionError(
                f"{model_id} numerics yaml validation failed: {err!s}"
            ) from None

        seen: dict[Precision, set[ScorecardProfilePath]] = {}
        for metric in accuracy.metrics:
            for precisions in metric.device_metric.values():
                for precision, paths in precisions.items():
                    seen.setdefault(precision, set()).update(paths)
        if not seen:
            continue

        manifest = QAIHMModelManifest.from_model(model_id)
        for precision, seen_paths in seen.items():
            path_errors.extend(
                get_manifest_support_violations(
                    manifest, "numerics.yaml", precision, seen_paths
                )
            )

    assert not path_errors, (
        f"{len(path_errors)} numerics.yaml entries conflict with manifest.yaml:\n"
        + "\n".join(path_errors)
    )


def test_yaml_roundtrip() -> None:
    model_id = "resnet18"
    accuracy_yaml_path = get_numerics_yaml_path(model_id)

    # Read from YAML and export back to YAML.
    # Compare both YAML dictionaries to make sure they're the same.
    original_yaml = load_yaml(accuracy_yaml_path)

    # Load Accuracy Object
    accuracy = QAIHMModelNumerics.from_model(model_id, not_exists_ok=True)
    assert accuracy is not None

    # Roundtrip back to dict
    with tempfile.TemporaryDirectory() as tmp:
        test_yaml_path = os.path.join(tmp, "test.yml")
        accuracy.to_yaml(test_yaml_path)
        roundtrip_dict = load_yaml(test_yaml_path)

    assert original_yaml == roundtrip_dict


def _struct_from_intermediates(model_id: str) -> QAIHMModelNumerics:
    sc = QAIHMModelInternalConfig.from_model(model_id)
    struct = create_numerics_struct(
        model_id,
        pd.read_csv(ScorecardArtifact.ACCURACY_CSV.intermediates_path),
        get_chipset_registry(),
        benchmark=sc.manifest.numerics_benchmark,
    )
    assert struct is not None
    return struct


# eyegaze reports Mean Angular Error and hrnet_face reports NME; both are better
# when smaller.
@pytest.mark.parametrize(
    ("model_id", "higher_is_better"),
    [("resnet18", True), ("eyegaze", False), ("hrnet_face", False)],
)
def test_metric_direction_from_accuracy_csv(
    model_id: str, higher_is_better: bool
) -> None:
    struct = _struct_from_intermediates(model_id)
    assert struct.metrics
    for metric in struct.metrics:
        assert metric.higher_is_better is higher_is_better, metric.metric_name


def test_accuracy_yaml_creation() -> None:
    model_id = "resnet18"
    accuracy_yaml_path = get_numerics_yaml_path(model_id)
    original_yaml = load_yaml(accuracy_yaml_path)

    # Create accuracy struct from scratch using accuracy.csv
    new_struct = _struct_from_intermediates(model_id)

    # Write to yaml and load to dict
    with tempfile.TemporaryDirectory() as tmp:
        test_yaml_path = os.path.join(tmp, "test.yml")
        new_struct.to_yaml(test_yaml_path)
        roundtrip_dict = load_yaml(test_yaml_path)

    assert original_yaml == roundtrip_dict
