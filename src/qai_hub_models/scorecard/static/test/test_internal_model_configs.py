# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from qai_hub_models.scorecard.device import DEFAULT_SCORECARD_DEVICE, cs_8_gen_3
from qai_hub_models.scorecard.static.list_models import get_all_static_models
from qai_hub_models.scorecard.static.model_config import (
    DEFAULT_MODELS_DIR,
    ScorecardModelConfig,
)
from qai_hub_models.utils.asset_loaders import load_yaml

TORCHSCRIPT_EXAMPLE_PATH = DEFAULT_MODELS_DIR / "_mobilenetv2_torchscript_example.yaml"


def test_load_validate_example_onnx() -> None:
    model_config = ScorecardModelConfig.from_yaml(
        DEFAULT_MODELS_DIR / "_mobilenetv2_onnx_example.yaml"
    )
    assert model_config.enabled_devices == [cs_8_gen_3, DEFAULT_SCORECARD_DEVICE]

    model_config.enabled_devices.append(DEFAULT_SCORECARD_DEVICE)
    with pytest.raises(ValueError, match="enabled_devices has duplicates"):
        model_config.check_fields(MagicMock())  # type: ignore[operator]


def test_load_validate_example_torchscript() -> None:
    ScorecardModelConfig.from_yaml(TORCHSCRIPT_EXAMPLE_PATH)


def test_yaml_roundtrip() -> None:
    # Read from YAML and export back to YAML.
    # Compare both YAML dictionaries to make sure they're the same.
    x_dict = load_yaml(TORCHSCRIPT_EXAMPLE_PATH)

    # Dict is copied because the constructor modifies it in place
    x = ScorecardModelConfig.from_yaml(TORCHSCRIPT_EXAMPLE_PATH)

    # Roundtrip back to dict
    with tempfile.TemporaryDirectory() as tmp:
        test_yaml_path = os.path.join(tmp, "test.yml")
        x.to_yaml(test_yaml_path)
        roundtrip_dict = load_yaml(test_yaml_path)

    # These were set to the default values and will
    # therefore be excluded from the roundtripped dict.
    del x_dict["channel_first_outputs"]
    del x_dict["precision"]
    del x_dict["enabled_profile_runtimes"]

    assert x_dict == roundtrip_dict


def test_validate_all_models() -> None:
    models = get_all_static_models(include_examples=True)

    model = None
    try:
        for model in models:
            config = ScorecardModelConfig.from_scorecard_model_id(model)
            # Validate Hub Assets as well. This isn't validated on first parse because it's too expensive.
            ScorecardModelConfig.model_validate(
                config, context=dict(validate_hub_assets=True)
            )
            assert config.id == model, (
                f"Model id {config.id} must match the filename {model}"
            )
    except Exception as e:
        raise AssertionError(f"{model} validation failed: {e}") from None
