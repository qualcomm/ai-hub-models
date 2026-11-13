# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from typing_extensions import Self
from ultralytics.models import YOLOE as ultralytics_YOLOE
from ultralytics.nn.modules.block import BNContrastiveHead
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models.templates.common import replace_module_recursively
from qai_hub_models.models.templates.ultralytics.segmentation_model import (
    UltralyticsMulticlassSegmentor,
)
from qai_hub_models.models.templates.yolo.model import YoloSegEvalMixin
from qai_hub_models.models.yoloe_seg.model_patches import BNContrastiveHeadInf
from qai_hub_models.utils.labels import get_class_names

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yoloe-v8l-seg.pt",
]
DEFAULT_WEIGHTS = "yoloe-v8l-seg.pt"


class YoloESegmentor(UltralyticsMulticlassSegmentor, YoloSegEvalMixin):
    """Exportable YoloE Prompt-then-detector end-to-end."""

    def __init__(
        self,
        model: SegmentationModel,
        prompt_text: list[str],
    ) -> None:
        super().__init__(model)
        self.prompt_text = prompt_text

    @classmethod
    def from_pretrained(
        cls,
        prompt_text: list[str] | str | None = None,
        ckpt_name: str = DEFAULT_WEIGHTS,
    ) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )

        if prompt_text is None:
            prompt_text = get_class_names("coco")
        elif isinstance(prompt_text, str):
            if prompt_text.endswith(".txt"):
                with open(prompt_text) as f:
                    lines = f.readlines()
                prompt_text = [t.rstrip("\r\n").strip() for t in lines if t.strip()]
            else:
                prompt_text = [t.strip() for t in prompt_text.split(",") if t.strip()]
        # Update local path for ckpt
        yoloe_model = ultralytics_YOLOE(ckpt_name)
        yoloe_model.set_classes(prompt_text)
        model = cast(SegmentationModel, yoloe_model.model)
        replace_module_recursively(model, BNContrastiveHead, BNContrastiveHeadInf)
        return cls(model, prompt_text)
