# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from copy import deepcopy

import torch
from typing_extensions import Self

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.models.gear_guard_net.layers import build_gear_guard_net_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "weights_v1.1.pt"
)


EVALUATOR_NMS_IOU_THRESHOLD = 0.5
EVALUATOR_SCORE_THRESHOLD = 0.7
EVALUATOR_MAP_DEFAULT_LOW_IOU = 0.5
EVALUATOR_MAP_DEFAULT_HIGH_IOU = 0.95
EVALUATOR_MAP_DEFAULT_INCREMENT_IOU = 0.05


class GearGuardNet(BaseModel):
    """GearGuardNet model"""

    def __init__(self, model_cfg: dict, ch: int = 3) -> None:
        """
        Initialize person/face detection model.

        Parameters
        ----------
        model_cfg
            Model configuration
        ch
            Input channels.
        """
        super().__init__()
        self.model, self.save = build_gear_guard_net_model(deepcopy(model_cfg), ch=[ch])

    @staticmethod
    def _decode_scale(
        out: torch.Tensor,
        scale_anchors: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode predictions for a single scale.

        This helper method processes the output from one detection scale, applying
        anchor-based decoding and coordinate transformations.

        Parameters
        ----------
        out
            Output tensor for a single scale with shape [batch, channels, height, width].
        scale_anchors
            Anchor boxes for this scale with shape [num_anchors, 2].
        stride
            Stride value for this scale (8, 16, or 32).

        Returns
        -------
        decoded_predictions : torch.Tensor
            Decoded predictions for this scale with shape [batch, num_predictions, 7],
            where 7 represents [x_center, y_center, width, height, confidence, class_0_score, class_1_score].
        """
        # make detector_output shape [batch, height, width, channels]
        out = out.permute(0, 2, 3, 1)  # [batch, height, width, channels]

        batch, h, w, _ = out.shape

        # Reshape to include anchor dimension for all batches at once
        out = out.reshape(
            batch, h, w, 3, -1
        )  # [batch, height, width, num_anchors, channels]
        ny, nx, na = h, w, 3
        num_classes = out.shape[-1] - 5

        # Create coordinate grids - vectorized
        grid_y, grid_x = torch.meshgrid(
            torch.arange(ny, device=out.device),
            torch.arange(nx, device=out.device),
            indexing="ij",
        )
        # Expand grids to match batch and anchor dimensions
        grid_y = (
            grid_y.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, na)
        )  # [batch, ny, nx, na]
        grid_x = (
            grid_x.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, na)
        )  # [batch, ny, nx, na]

        # Extract predictions - vectorized across all batches
        pred = out  # [batch, ny, nx, na, nc]

        # Compute scores - vectorized
        obj_score = pred[..., 4].sigmoid()  # [batch, ny, nx, na]
        cls_scores = pred[..., 5:].sigmoid()  # [batch, ny, nx, na, num_classes]

        # Flatten spatial and anchor dimensions while keeping batch dimension
        obj_score_flat = obj_score.reshape(batch, -1)  # [batch, ny*nx*na]
        cls_scores_flat = cls_scores.reshape(
            batch, -1, num_classes
        )  # [batch, ny*nx*na, num_classes]
        pred_flat = pred.reshape(batch, -1, pred.shape[-1])  # [batch, ny*nx*na, nc]
        grid_x_flat = grid_x.reshape(batch, -1)  # [batch, ny*nx*na]
        grid_y_flat = grid_y.reshape(batch, -1)  # [batch, ny*nx*na]

        # Get anchor indices for all predictions
        anchor_indices = (
            torch.arange(na, device=out.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(ny, nx, -1)
            .reshape(-1)
        )  # [ny*nx*na]
        valid_anchors = scale_anchors[anchor_indices].to(out.device)  # [ny*nx*na, 2]
        # Expand to batch dimension
        valid_anchors = valid_anchors.unsqueeze(0).expand(
            batch, -1, -1
        )  # [batch, ny*nx*na, 2]

        # Decode bounding boxes - vectorized for all batches and predictions
        bx = (pred_flat[..., 0].sigmoid() * 2 - 0.5 + grid_x_flat) * stride
        by = (pred_flat[..., 1].sigmoid() * 2 - 0.5 + grid_y_flat) * stride
        bw = 4 * pred_flat[..., 2].sigmoid() ** 2 * valid_anchors[..., 0]
        bh = 4 * pred_flat[..., 3].sigmoid() ** 2 * valid_anchors[..., 1]

        # Stack results - vectorized
        scale_results = torch.stack(
            [bx, by, bw, bh, obj_score_flat], dim=2
        )  # [batch, ny*nx*na, 5]

        return torch.cat(
            [scale_results, cls_scores_flat], dim=2
        )  # [batch, ny*nx*na, 5+num_classes]

    @staticmethod
    def decode(
        detector_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode model output to bounding boxes, confidence and class scores.

        This method processes the raw multi-scale outputs from the model and converts them
        into a standardized format of bounding boxes and associated scores. It applies
        anchor-based decoding and coordinate transformations to convert raw predictions
        into a standardized detection format.

        Parameters
        ----------
        detector_output
            Model output consisting of a tuple of 3 tensors (multi-scale outputs).
            Each tensor has shape [batch, channels, height, width], representing
            predictions at different feature map scales.

        Returns
        -------
        detection_results : torch.Tensor
            Detection results with shape (batch_size, num_detections, 7), where:
            - Each detection contains 7 values:
                [x_center, y_center, width, height, confidence, class_0_score, class_1_score]
            - Coordinates (x_center, y_center, width, height) are in absolute pixel values
            - Confidence is the objectness score (0-1)
            - Class scores are values (0-1) for each class
        """
        anchors = torch.tensor(
            [
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]],
            ],
            dtype=torch.float32,
        )
        strides = torch.tensor([8, 16, 32], dtype=torch.float32)

        all_scale_results = [
            GearGuardNet._decode_scale(detector_output[0], anchors[0], strides[0]),
            GearGuardNet._decode_scale(detector_output[1], anchors[1], strides[1]),
            GearGuardNet._decode_scale(detector_output[2], anchors[2], strides[2]),
        ]

        # Concatenate results from all scales along the detection dimension
        return torch.cat(all_scale_results, dim=1)  # [batch, total_detections, 7]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward computation of the GearGuardNet model.

        This method processes input images through the model backbone and detection head,
        then applies postprocessing to generate ready-to-use detection results.

        Parameters
        ----------
        x
            Input image tensor in RGB format with pixel values in range [0-1].
            Shape is [batch_size, channels, height, width] where channels=3.

        Returns
        -------
        torch.Tensor
            Bounding box coordinates with shape [batch_size, num_detections, 4].
            Each box is represented as (x1, y1, x2, y2) in absolute pixel coordinates.
        torch.Tensor
            Detection confidence scores with shape [batch_size, num_detections].
            Each score is the product of objectness confidence and class score.
        torch.Tensor
            Predicted class indices with shape [batch_size, num_detections].
            Values are integer indices representing the detected class.
        """
        # Run backbone model
        y: list[int | None] = []
        for m in self.model:  # type: ignore[attr-defined]
            if m.f != -1:
                x = (
                    y[m.f]  # type: ignore[assignment]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if m.i in self.save else None)  # type: ignore[arg-type]

        # x is now a list of 3 tensors (multi-scale outputs)
        # Each tensor has shape [batch, channels, height, width]
        assert len(x) == 3
        detector_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = tuple(x)

        # Apply decode and postprocessing
        predictions = GearGuardNet.decode(detector_output)
        return detect_postprocess(predictions)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None) -> Self:
        """
        Load model from pretrained weights.

        Parameters
        ----------
        checkpoint_path
            Checkpoint path of pretrained weights.

        Returns
        -------
        model : Self
            The GearGuardNet detection model.
        """
        cfg = {
            "nc": 2,
            "depth_multiple": 0.33,
            "width_multiple": 0.5,
            "anchors": [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ],
            "backbone": [
                [-1, 1, "FusedConvBatchNorm", [64, 6, 2, 2]],
                [-1, 1, "FusedConvBatchNorm", [128, 3, 2]],
                [-1, 3, "DoubleBlazeBlock", [128]],
                [-1, 1, "FusedConvBatchNorm", [256, 3, 2]],
                [-1, 3, "DoubleBlazeBlock", [256]],
                [-1, 1, "FusedConvBatchNorm", [512, 3, 2]],
                [-1, 9, "DoubleBlazeBlock", [512]],
                [-1, 1, "FusedConvBatchNorm", [1024, 3, 2]],
                [-1, 3, "DoubleBlazeBlock", [1024]],
                [-1, 1, "FusedConvBatchNorm", [1024, 3, 1]],
            ],
            "head": [
                [-1, 1, "FusedConvBatchNorm", [512, 1, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 6], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [512]],
                [-1, 1, "FusedConvBatchNorm", [256, 1, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 4], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [256]],
                [-1, 1, "FusedConvBatchNorm", [256, 3, 2]],
                [[-1, 14], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [512]],
                [-1, 1, "FusedConvBatchNorm", [512, 3, 2]],
                [[-1, 10], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [1024]],
                [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],
            ],
        }
        model = cls(cfg)
        checkpoint_to_load = (
            DEFAULT_WEIGHTS if checkpoint_path is None else checkpoint_path
        )
        ckpt = load_torch(checkpoint_to_load)
        model.load_state_dict(ckpt)
        model.eval()
        return model

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 320,
        width: int = 192,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "class_idx"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        input_spec = self.get_input_spec()
        return DetectionEvaluator(
            image_height=input_spec["image"][0][2],
            image_width=input_spec["image"][0][3],
            nms_iou_threshold=EVALUATOR_NMS_IOU_THRESHOLD,
            score_threshold=EVALUATOR_SCORE_THRESHOLD,
            mAP_default_low_iOU=EVALUATOR_MAP_DEFAULT_LOW_IOU,
            mAP_default_high_iOU=EVALUATOR_MAP_DEFAULT_HIGH_IOU,
            mAP_default_increment_iOU=EVALUATOR_MAP_DEFAULT_INCREMENT_IOU,
        )

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["gear_guard_dataset"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "coco_ppe"

    @classmethod
    def get_labels_file_name(cls) -> str | None:
        return "ppe_labels.txt"
