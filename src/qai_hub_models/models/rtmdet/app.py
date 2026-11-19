# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import torch

from qai_hub_models.models.rtmdet.model import RTMDet
from qai_hub_models.models.templates.yolo.app import YoloObjectDetectionApp


def decode_rtmdet_heads(
    cls_scores: Sequence[torch.Tensor],
    bbox_preds: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode RTMDet's raw detection-head maps into boxes/scores/class_idx.

    Host-side float equivalent of the in-graph decode in ``RTMDet.forward``. Used
    when the model is built with ``include_postprocessing=False`` -- the model
    then skips the decode and emits the 6 raw head maps, and this runs the grid
    decode here instead. The arithmetic mirrors ``RTMDet.forward`` exactly so both
    paths produce identical detections. The per-stage grid sizes (80, 40, 20) and
    640 input side length must match RTMDet.stage / RTMDet.input_shape in model.py.

    Parameters
    ----------
    cls_scores
        Per-stage classification maps, each shape [batch, num_classes, H, W],
        ordered to match ``RTMDet.stage`` (80x80, 40x40, 20x20).
    bbox_preds
        Per-stage box-regression maps, each shape [batch, 4, H, W], same order.

    Returns
    -------
    boxes : torch.Tensor
        Shape [batch, num_preds, 4] where 4 == (left_x, top_y, right_x, bottom_y).
    scores : torch.Tensor
        Confidence per prediction, shape [batch, num_preds].
    class_idx : torch.Tensor
        Most-probable class index per prediction, shape [batch, num_preds].
    """
    stages = (80, 40, 20)
    input_shape = 640
    boxes_list = []
    scores_list = []
    class_idx_list = []
    for i, (cls, box) in enumerate(zip(cls_scores, bbox_preds, strict=True)):
        cls = cls.permute(0, 2, 3, 1)
        box = box.permute(0, 2, 3, 1)

        cls = torch.sigmoid(cls)
        conf = torch.max(cls, dim=3, keepdim=True)[0]
        class_idx = torch.argmax(cls, dim=3, keepdim=True).to(torch.int32)

        step = input_shape // stages[i]
        block_step = torch.arange(stages[i], device=box.device, dtype=box.dtype) * step
        block_x = torch.broadcast_to(block_step, [stages[i], stages[i]])
        block_y = torch.transpose(block_x, 1, 0)
        block_x = torch.unsqueeze(block_x, 0)
        block_y = torch.unsqueeze(block_y, 0)
        block = torch.stack([block_x, block_y], -1)

        box_xy1 = block - box[..., :2]
        box_xy2 = block + box[..., 2:4]
        box = torch.cat([box_xy1, box_xy2], dim=-1)

        batch_size = box.shape[0]
        boxes_list.append(box.reshape(batch_size, -1, 4))
        scores_list.append(conf.reshape(batch_size, -1))
        class_idx_list.append(class_idx.reshape(batch_size, -1))

    boxes = torch.cat(boxes_list, dim=1)
    scores = torch.cat(scores_list, dim=1)
    class_idx = torch.cat(class_idx_list, dim=1)
    return boxes, scores, class_idx


class RTMDetApp(YoloObjectDetectionApp):
    def pre_nms_postprocess(
        self, *predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the raw head maps into boxes/scores/class_idx for NMS.

        When the model excludes postprocessing it emits the 6 raw detection-head
        maps in the order produced by ``mmdet_RTMDET._forward``: the 3
        classification maps followed by the 3 box-regression maps. The grid decode
        runs on the host here (see ``decode_rtmdet_heads``).
        """
        num_stages = len(predictions) // 2
        cls_scores = predictions[:num_stages]
        bbox_preds = predictions[num_stages:]
        return decode_rtmdet_heads(cls_scores, bbox_preds)

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is a valid model input. Image size should be shape
        [batch_size, num_channels, height, width], where height and width are multiples
        of `RTMDet.STRIDE_MULTIPLE`.
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("Pixel Values must be rank 4: [batch, channels, x, y]")

        if (
            pixel_values.shape[2] % RTMDet.STRIDE_MULTIPLE != 0
            or pixel_values.shape[3] % RTMDet.STRIDE_MULTIPLE != 0
        ):
            raise ValueError(
                f"Pixel values must have spatial dimensions (H & W) that are multiples of {RTMDet.STRIDE_MULTIPLE}."
            )
