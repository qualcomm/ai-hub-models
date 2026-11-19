# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps

with patch_mmdet_no_build_deps():
    from mmdet.apis import init_detector
    from mmdet.models.detectors.rtmdet import RTMDet as mmdet_RTMDET

from qai_hub_models.extern.mmengine import (
    patch_mmengine_pkgresources,
    patch_mmengine_torch_load_no_weights_only,
)
from qai_hub_models.models.templates.detection.coco_detection_evaluator import (
    COCODetectionEvaluator,
)
from qai_hub_models.models.templates.yolo.model import Yolo
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import IoType, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

DEFAULT_WEIGHTS = "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
DEFAULT_CONFIG = "rtmdet_m_8xb32-300e_coco.py"

MODEL_LOCAL_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
).fetch()
MODEL_CONFIG_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_CONFIG
).fetch()


class RTMDet(Yolo):
    """Exportable RTMDet bounding box detector, end-to-end."""

    def __init__(
        self, model: mmdet_RTMDET, include_postprocessing: bool = True
    ) -> None:
        super().__init__(model=model)
        self.model: mmdet_RTMDET
        self.stage = [80, 40, 20]
        self.input_shape = 640
        self.include_postprocessing = include_postprocessing
        self.mean = model.data_preprocessor.mean
        self.std = model.data_preprocessor.std

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return COCODetectionEvaluator(
            image_height,
            image_width,
            nms_iou_threshold=0.65,
            score_threshold=0.001,
        )

    @classmethod
    def from_pretrained(cls, include_postprocessing: bool = True) -> Self:
        """RTMDet comes from the MMDet library, so we load using an internal config
        rather than a public weights file
        """
        model = _load_rtmdet_source_model_from_weights(
            str(MODEL_CONFIG_PATH), str(MODEL_LOCAL_PATH)
        )
        return cls(model, include_postprocessing)

    def forward(
        self, image: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """
        Run RTMDet on `image`, and produce a predicted set of bounding boxes and associated class probabilities.
        Forward pass for processing the input image and obtaining the model outputs.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            If self.include_postprocessing is True, returns:
            boxes
                Bounding box locations. Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y).
            scores
                Class scores multiplied by confidence. Shape is [batch, num_preds].
            class_idx
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            If self.include_postprocessing is False, returns the 6 raw
            detection-head maps (decode runs on the host instead; see
            qai_hub_models.models.rtmdet.app.decode_rtmdet_heads):
            cls_80, cls_40, cls_20
                Classification maps, shapes [batch, num_classes, S, S] for S in (80, 40, 20).
            box_80, box_40, box_20
                Box-regression maps, shapes [batch, 4, S, S] for S in (80, 40, 20).
        """
        # Normalize: model expects (pixel_uint8 - mean) / std; input is float [0, 1]
        # Model was trained on BGR images (bgr_to_rgb=False in config).
        # Our input is RGB, so flip channels before applying BGR-ordered mean/std.
        image = image[:, [2, 1, 0], :, :]
        image = (image * 255.0 - self.mean) / self.std
        # Add a cast. model._forward is not typed correctly.
        output = cast(
            tuple[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ],
            self.model._forward(image),
        )

        # When postprocessing is excluded, skip the whole decode (sigmoid / max /
        # argmax / grid / box-corner ops) and return the 6 raw detection-head maps
        # (3 classification + 3 box-regression). The decode then runs on the host
        # (see qai_hub_models.models.rtmdet.app.decode_rtmdet_heads).
        if not self.include_postprocessing:
            cls_scores, bbox_preds = output
            return (*cls_scores, *bbox_preds)

        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        class_idx_list: list[torch.Tensor] = []

        for i, (cls, box) in enumerate(zip(*output, strict=False)):
            # cls: [B, C, H, W] -> [B, H, W, C]; box: [B, 4, H, W] -> [B, H, W, 4]
            cls = cls.permute(0, 2, 3, 1)
            box = box.permute(0, 2, 3, 1)

            cls = F.sigmoid(cls)
            conf = torch.max(cls, dim=3, keepdim=True)[0]
            # HTP cannot narrow ArgMax class IDs from int32 to int8.
            class_idx = torch.argmax(cls, dim=3, keepdim=True).to(torch.int32)

            step = self.input_shape // self.stage[i]
            block_step = (
                torch.arange(self.stage[i], device=box.device, dtype=box.dtype) * step
            )
            block_x = torch.broadcast_to(block_step, [self.stage[i], self.stage[i]])
            block_y = torch.transpose(block_x, 1, 0)
            block_x = torch.unsqueeze(block_x, 0)
            block_y = torch.unsqueeze(block_y, 0)
            block = torch.stack([block_x, block_y], -1)

            # decode box corners
            box_xy1 = block - box[..., :2]
            box_xy2 = block + box[..., 2:4]
            box = torch.cat([box_xy1, box_xy2], dim=-1)

            batch_size = box.shape[0]
            # flatten per output type, keeping semantics separate
            box = box.reshape(batch_size, -1, 4)
            conf = conf.reshape(batch_size, -1)
            class_idx = class_idx.reshape(batch_size, -1)

            boxes_list.append(box)
            scores_list.append(conf)
            class_idx_list.append(class_idx)

        # concat only homogeneous tensors (quantization-friendly)
        boxes = torch.cat(boxes_list, dim=1)
        scores = torch.cat(scores_list, dim=1)
        class_idx = torch.cat(class_idx_list, dim=1)

        return boxes, scores, class_idx

    def get_output_spec(self) -> OutputSpec:
        if self.include_postprocessing:
            outputs = ["boxes", "scores", "class_idx"]
        else:
            # Raw detection-head maps: 3 classification + 3 box-regression, in the
            # order returned by mmdet_RTMDET._forward (see forward()).
            outputs = ["cls_80", "cls_40", "cls_20", "box_80", "box_40", "box_20"]

        # Raw detection-head maps; decode runs on the host (see app.py).
        return {name: TensorSpec(io_type=IoType.TENSOR) for name in outputs}

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 8


def _load_rtmdet_source_model_from_weights(
    model_config_path: str, model_weights_path: str
) -> mmdet_RTMDET:
    with patch_mmengine_torch_load_no_weights_only(), patch_mmengine_pkgresources():
        model = init_detector(
            str(model_config_path), str(model_weights_path), device="cpu"
        )
    assert isinstance(model, mmdet_RTMDET)
    return model
