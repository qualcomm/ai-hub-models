# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import qai_hub as hub

from qai_hub_models import TargetRuntime
from qai_hub_models.utils.evaluate import pipeline


def test_on_device_inference_uses_built_hub_options() -> None:
    model = mock.MagicMock()
    model.get_input_spec.return_value = {"image": object()}
    model.get_hub_inference_options.return_value = "--qairt_version 2.50"
    model_cls = mock.MagicMock()
    model_cls.get_eval_dataset_classes.return_value = [object]
    model_cls.from_pretrained.return_value = model

    with (
        mock.patch.object(pipeline, "resolve_model_cls", return_value=model_cls),
        mock.patch.object(
            pipeline,
            "resolve_manifest",
            return_value=SimpleNamespace(
                can_use_quantize_job=False, supports_quantization=False
            ),
        ),
        mock.patch.object(pipeline, "get_model_kwargs", return_value={}),
        mock.patch.object(pipeline, "filter_kwargs", return_value={}),
        mock.patch.object(pipeline, "evaluate_on_dataset"),
        mock.patch.object(hub, "get_model"),
        mock.patch.object(pipeline, "AsyncOnDeviceModel") as on_device_model,
    ):
        pipeline.evaluate_model(
            source_dir=Path("/recipe"),
            device=hub.Device("fake"),
            target_runtime=TargetRuntime.QNN_DLC,
            dataset_cls=object,  # type: ignore[arg-type]
            hub_model_id="mabc",
            inference_options="--foo",
        )

    model.get_hub_inference_options.assert_called_once_with(
        TargetRuntime.QNN_DLC, "--foo"
    )
    assert (
        on_device_model.call_args.kwargs["inference_options"] == "--qairt_version 2.50"
    )
