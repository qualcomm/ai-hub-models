# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from inspect import signature
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import qai_hub as hub
import torch

from qai_hub_models.models._shared.llm.common import cleanup
from qai_hub_models.models.common import Precision
from qai_hub_models.models.llama_v2_7b_chat import Model
from qai_hub_models.models.llama_v2_7b_chat.export import (
    ALL_COMPONENTS,
    BASE_NAME,
    DEFAULT_EXPORT_DEVICE,
    export_model,
)
from qai_hub_models.models.llama_v2_7b_chat.export import main as export_main
from qai_hub_models.models.llama_v2_7b_chat.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_x_elite
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.testing import patch_qai_hub
from qai_hub_models.utils.testing_export_eval import run_llm_compile


def _mock_from_pretrained() -> Mock:
    model = MagicMock()
    # We'll use a single compnent to mock all
    component = MagicMock()
    component.get_input_spec.return_value = {"attention_mask": [[1, 1024]]}
    component.get_output_names.return_value = ["output0", "output1"]
    component.sample_inputs.return_value = {"input0": [np.array([1.0, 2.0])]}

    model.load_model_part.return_value = component
    model.__signature__ = signature(Model.from_pretrained)
    mock_from_pretrained = Mock()
    mock_from_pretrained.__signature__ = signature(Model.from_pretrained)
    mock_from_pretrained.return_value = model
    return mock_from_pretrained


def test_cli_device_with_skips(tmp_path: Path) -> None:
    patch_get_cache = patch(
        "qai_hub_models.utils.model_cache.get_or_create_cached_model",
        return_value=Mock(),
    )
    mock_from_pretrained = _mock_from_pretrained()
    patch_model = patch(
        "qai_hub_models.models.llama_v2_7b_chat.Model.from_pretrained",
        mock_from_pretrained,
    )

    device = hub.Device("Snapdragon X Elite CRD")
    with patch_qai_hub() as mock_hub, patch_model, patch_get_cache:
        export_main(
            [
                "--device",
                device.name,
                "--skip-inferencing",
                "--skip-profiling",
                "--output-dir",
                tmp_path.name,
            ]
        )

        # Compile is called 8 times (4 token parts, 4 prompt parts)
        assert mock_hub.submit_compile_job.call_count == 8
        call_args_list = mock_hub.submit_compile_job.call_args_list
        assert all(c.kwargs["device"].name == device.name for c in call_args_list)

        # Link 4 times
        assert mock_hub.submit_link_job.call_count == 4
        call_args_list = mock_hub.submit_link_job.call_args_list
        assert all(
            c.kwargs["name"] == f"{BASE_NAME}_Llama2_Part{i + 1}_Quantized"
            for i, c in enumerate(call_args_list)
        )

        # Skip profile and inference
        mock_hub.submit_profile_job.assert_not_called()
        mock_hub.submit_inference_job.assert_not_called()
        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_cli_chipset_with_options(tmp_path: Path) -> None:
    patch_get_cache = patch(
        "qai_hub_models.utils.model_cache.get_or_create_cached_model",
        return_value=Mock(),
    )
    mock_from_pretrained = _mock_from_pretrained()
    mock_model = mock_from_pretrained.return_value
    patch_model = patch(
        "qai_hub_models.models.llama_v2_7b_chat.Model.from_pretrained",
        return_value=mock_model,
    )

    chipset = "qualcomm-snapdragon-8-elite"
    with patch_qai_hub() as mock_hub, patch_model, patch_get_cache:
        export_main(
            [
                "--chipset",
                chipset,
                "--output-dir",
                tmp_path.name,
                "--compile-options",
                "compile_extra",
                "--profile-options",
                "profile_extra",
            ]
        )
        parts = 4  # Number of model parts
        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_profile_job.call_count == parts * 2
        assert mock_hub.submit_inference_job.call_count == parts * 2

        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
        ]:
            assert all(
                f"chipset:{chipset}" in call.kwargs["device"].attributes
                for call in mock.call_args_list
            )

        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_cli_default_device_select_component(tmp_path: Path) -> None:
    mock_get_cache = Mock()
    patch_get_cache = patch(
        "qai_hub_models.utils.model_cache.get_or_create_cached_model", mock_get_cache
    )
    mock_from_pretrained = _mock_from_pretrained()
    patch_model = patch(
        "qai_hub_models.models.llama_v2_7b_chat.Model.from_pretrained",
        mock_from_pretrained,
    )

    device = hub.Device(DEFAULT_EXPORT_DEVICE)
    with patch_qai_hub() as mock_hub, patch_model, patch_get_cache:
        export_main(
            [
                "--output-dir",
                tmp_path.name,
                "--max-position-embeddings",
                "128",
                "--model-cache-mode",
                "overwrite",
                "--components",
                "Llama2_Part2_Quantized",
                "Llama2_Part4_Quantized",
            ]
        )
        mock_from_pretrained.assert_called_with(max_position_embeddings=128)

        assert mock_hub.submit_compile_job.call_count == 4
        assert mock_hub.submit_link_job.call_count == 2
        assert mock_hub.submit_profile_job.call_count == 4
        assert mock_hub.submit_inference_job.call_count == 4

        # Check names
        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
        ]:
            cc = mock.call_args_list
            assert cc[0].kwargs["name"] == f"{BASE_NAME}_PromptProcessor_2"
            assert cc[1].kwargs["name"] == f"{BASE_NAME}_TokenGenerator_2"
            assert cc[2].kwargs["name"] == f"{BASE_NAME}_PromptProcessor_4"
            assert cc[3].kwargs["name"] == f"{BASE_NAME}_TokenGenerator_4"

        cc = mock_hub.submit_link_job.call_args_list
        assert cc[0].kwargs["name"] == f"{BASE_NAME}_Llama2_Part2_Quantized"
        assert cc[1].kwargs["name"] == f"{BASE_NAME}_Llama2_Part4_Quantized"

        # Check cache
        for c in mock_get_cache.call_args_list:
            assert c.kwargs["model_name"] == BASE_NAME
            assert c.kwargs["cache_mode"] == CacheMode.OVERWRITE
            # Mock assumes seq len is always 1 (not correct for prompts)
            assert c.kwargs["additional_keys"] == {
                "context_length": "1024",
                "sequence_length": "1",
            }

        call_args_list = mock_hub.submit_compile_job.call_args_list
        assert all(c.kwargs["device"].name == device.name for c in call_args_list)

        assert tmp_path.exists()
        assert tmp_path.is_dir()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    cleanup()
    run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint="DEFAULT",
            model_cls=Model,
            model_name=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            components=ALL_COMPONENTS,
            output_dir="output_dir",
        ),
        skip_compile_options=True,
    )
