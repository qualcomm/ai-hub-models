# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from typing_extensions import Self
from ultralytics.models import FastSAM
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models import Precision, SampleInputsType
from qai_hub_models.datasets.coco import CocoDataset
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.ultralytics.segmentation_model import (
    UltralyticsSingleClassSegmentor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import InputSpec


class Fast_SAM(UltralyticsSingleClassSegmentor):
    """Exportable FastSAM model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = "") -> Self:
        return cls(cast(SegmentationModel, FastSAM(model=ckpt_name).model))

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 25

    def get_evaluator(self) -> BaseEvaluator:
        from qai_hub_models.models.templates.fastsam.class_agnostic_ar_evaluator import (
            ClassAgnosticARkEvaluator,
        )

        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return ClassAgnosticARkEvaluator(image_height, image_width)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDataset

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "fastsam_s", 1, "image_640.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
