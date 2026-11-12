# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torchvision
from detectron2.modeling import GeneralizedRCNN
from PIL import Image
from qai_hub.client import Device
from torch.nn import functional as F
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.datasets.coco import CocoPanopticSegmentationDataset
from qai_hub_models.models.detectron2_panoptic_seg.evaluator import (
    Detectron2PanopticSegEvaluator,
)
from qai_hub_models.models.detectron2_panoptic_seg.model_patches import (
    gather_roi_align,
    native_roi_align,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.detectron2.model import IMAGE_ADDRESS, Detectron2
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    BboxFormat,
    BboxMetadata,
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_CONFIG = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"

# Lite-MP percentage shared by both components' get_hub_litemp_percentage.
HUB_LITEMP_PERCENTAGE = 20.0

# Default RPN config values from panoptic_fpn_R_50_1x. The per-level pre-NMS
# top-k runs on device (Detectron2PanopticProposalGenerator.forward), so
# RPN_PRE_NMS_TOPK must match the value filter_rpn_proposals reconstructs level
# boundaries with. NMS is data dependent and runs off device.
RPN_PRE_NMS_TOPK = 1000
RPN_NMS_THRESH = 0.7

# Number of proposals fed to the ROI head. All ROI head shapes are static, so
# the app pads/truncates the filtered RPN proposals to exactly this count.
DEFAULT_NUM_PROPOSALS = 200

# Precomputed proposal-generator outputs (FPN feature maps + filtered proposals)
# for the bundled sample image at the default 800x800 spec, used as the ROI
# head's sample inputs so collecting them doesn't re-run the proposal generator.
ROI_HEAD_SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "roi_head_sample_inputs.npz"
)


class Detectron2PanopticProposalGenerator(Detectron2):
    """
    FPN backbone + semantic segmentation head + RPN.

    All outputs are dense and shape static. The RPN's per-level top-k is
    shape static too (k_l = min(N_l, RPN_PRE_NMS_TOPK) is a compile-time
    constant for a fixed input resolution) and runs here, on device, to keep
    the proposals/scores output small; only NMS is data dependent and runs
    off device in filter_rpn_proposals.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        # Keep only the submodules used by forward(); storing the full
        # GeneralizedRCNN would serialize the (unused) ROI head weights too.
        self.pixel_mean = model.pixel_mean
        self.pixel_std = model.pixel_std
        self.backbone = model.backbone
        self.proposal_generator = model.proposal_generator
        self.sem_seg_head = model.sem_seg_head

    @classmethod
    def from_pretrained(cls, config: str = DEFAULT_CONFIG) -> Self:
        # Default to the panoptic config so this component loads standalone;
        # the abstract Detectron2 template only exposes the shared loader.
        with native_roi_align():
            model = cls.load_pretrained_model(config)
        return cls(model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[
        torch.Tensor,  # feature_p2
        torch.Tensor,  # feature_p3
        torch.Tensor,  # feature_p4
        torch.Tensor,  # feature_p5
        torch.Tensor,  # proposals
        torch.Tensor,  # scores
        torch.Tensor,  # sem_seg_logits
    ]:
        """
        Parameters
        ----------
        image
            (1, 3, H, W) float32 RGB [0, 1].

        Returns
        -------
        torch.Tensor
            feature_p2 of shape (1, 256, H//4, W//4).
        torch.Tensor
            feature_p3 of shape (1, 256, H//8, W//8).
        torch.Tensor
            feature_p4 of shape (1, 256, H//16, W//16).
        torch.Tensor
            feature_p5 of shape (1, 256, H//32, W//32).
        torch.Tensor
            proposals of shape (1, num_topk_anchors, 4) xyxy, all RPN levels
            concatenated finest first, each level capped at RPN_PRE_NMS_TOPK
            (see rpn_topk_anchor_counts).
        torch.Tensor
            objectness logits of shape (1, num_topk_anchors).
        torch.Tensor
            semantic segmentation logits of shape (1, 54, H // 4, W // 4),
            emitted at the sem-seg head's common stride of 4.
        """
        image = (image[:, [2, 1, 0]] - (self.pixel_mean / 255)) / (self.pixel_std / 255)
        feature = self.backbone(image)
        sem_seg_logits = self.sem_seg_head.layers(feature)

        # Detectron2 RPN:
        # https://github.com/facebookresearch/detectron2/blob/8a9d885/detectron2/modeling/proposal_generator/rpn.py#L431
        features = [feature[f] for f in self.proposal_generator.in_features]
        anchors = self.proposal_generator.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.proposal_generator.rpn_head(
            features
        )
        pred_objectness_logits = [
            score.permute(0, 2, 3, 1).flatten(1) for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            x.permute(0, 2, 3, 1).reshape(
                x.shape[0], -1, self.proposal_generator.anchor_generator.box_dim
            )
            for x in pred_anchor_deltas
        ]
        proposals_per_level = self.proposal_generator._decode_proposals(
            anchors, pred_anchor_deltas
        )

        # Per-level pre-NMS top-k (shape static — see class docstring).
        # Keeping this on device caps the proposals/scores output at
        # sum(min(N_l, RPN_PRE_NMS_TOPK)) rows instead of the full dense
        # anchor set (e.g. ~160k rows at 800x800), which is what crosses the
        # device boundary for compilation/inference. Batched (not just
        # batch_size=1) so this also works for calibration/sample-input
        # collection, which may run proposal_generator on a dataloader batch.
        batch_size = image.shape[0]
        batch_idx = torch.arange(batch_size).unsqueeze(1)
        topk_proposals_per_level = []
        topk_logits_per_level = []
        for props_l, logits_l in zip(
            proposals_per_level, pred_objectness_logits, strict=True
        ):
            k_l = min(logits_l.shape[1], RPN_PRE_NMS_TOPK)
            topk_logits_l, topk_idx_l = logits_l.topk(k_l, dim=1)
            topk_proposals_per_level.append(props_l[batch_idx, topk_idx_l])
            topk_logits_per_level.append(topk_logits_l)

        return (
            feature["p2"],
            feature["p3"],
            feature["p4"],
            feature["p5"],
            torch.cat(topk_proposals_per_level, dim=1),
            torch.cat(topk_logits_per_level, dim=1),
            sem_seg_logits,
        )

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """
        Deterministic sample input: the bundled sample image.

        Quantized graphs are calibrated on real images, so measuring
        on-device parity with a random-noise image would understate accuracy.

        Parameters
        ----------
        input_spec
            Optional input spec override; defaults to get_input_spec().
        *args
            Unused; accepted for base class compatibility.
        **kwargs
            Unused; accepted for base class compatibility.

        Returns
        -------
        SampleInputsType
            Name -> single-element list of numpy arrays.
        """
        spec = input_spec or self.get_input_spec()
        height, width = (int(x) for x in spec["image"][0][2:])
        return {"image": [_sample_image_chw(height, width)[None]]}

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Lite-MP percentage for mixed-precision quantization."""
        return HUB_LITEMP_PERCENTAGE

    def component_precision(self) -> Precision:
        return Precision.w8a16

    def get_input_spec(
        self,
        batch_size: int = 1,
        generator_height: int = 800,
        generator_width: int = 800,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, generator_height, generator_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(color_format=ColorFormat.RGB),
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            x: TensorSpec()
            for x in [
                "feature_p2",
                "feature_p3",
                "feature_p4",
                "feature_p5",
                "proposals",
                "scores",
                "sem_seg_logits",
            ]
        }


class Detectron2PanopticROIHead(Detectron2):
    """
    FPN ROI head: box prediction + mask prediction.

    Takes 4 FPN feature maps and a fixed-size proposals tensor, assigns each
    proposal to an FPN level (matching detectron2's ROIPooler), and returns
    dense per-proposal boxes, scores, classes, and mask probabilities. All
    shapes are static; score filtering and NMS run in the app.
    """

    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        # Register only forward()'s submodules; the full model would also
        # serialize the unused backbone weights.
        self.roi_heads = model.roi_heads
        self.box_predictor = model.roi_heads.box_predictor

    @classmethod
    def from_pretrained(cls, config: str = DEFAULT_CONFIG) -> Self:
        # Default to the panoptic config so this component loads standalone;
        # the abstract Detectron2 template only exposes the shared loader.
        with native_roi_align():
            model = cls.load_pretrained_model(config)
        return cls(model)

    def forward(
        self,
        feature_p2: torch.Tensor,
        feature_p3: torch.Tensor,
        feature_p4: torch.Tensor,
        feature_p5: torch.Tensor,
        proposals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature_p2
            (1, 256, H//4, W//4) FPN feature map.
        feature_p3
            (1, 256, H//8, W//8) FPN feature map.
        feature_p4
            (1, 256, H//16, W//16) FPN feature map.
        feature_p5
            (1, 256, H//32, W//32) FPN feature map.
        proposals
            (num_proposals, 4) xyxy filtered RPN proposals.

        Returns
        -------
        torch.Tensor
            boxes of shape (1, num_proposals, num_box_classes, 4) xyxy
            per-class refined boxes (num_box_classes is 1 for class-agnostic
            box regression).
        torch.Tensor
            scores of shape (1, num_proposals, num_classes) per-class
            probabilities, background dropped.
        torch.Tensor
            mask_logits of shape (num_proposals, num_mask_classes, mask_h,
            mask_w) per-class mask logits, pooled on the input proposal
            boxes; the app applies sigmoid and pastes with the same proposal
            boxes.
        """
        # This forward pass is implemented for batch_size=1 only.
        features = [feature_p2, feature_p3, feature_p4, feature_p5]
        num_proposals = proposals.shape[0]

        # Box head. The FC layers are expressed as convolutions on the
        # unflattened pooled features (Linear(C*h*w -> D) == Conv2d with an
        # h x w kernel): quantized FullyConnected with K=12544 crashes the
        # HTP DSP at execution, while the equivalent convolutions use native
        # kernels. Weights are identical; results match bit-for-bit in fp32.
        box_features = self._roi_pool(proposals, features, self.roi_heads.box_pooler)
        x = box_features
        for conv in getattr(self.roi_heads.box_head, "conv_norm_relus", []):
            x = conv(x)
        for fc in self.roi_heads.box_head.fcs:
            weight = fc.weight.view(fc.out_features, *x.shape[1:])
            x = F.relu(F.conv2d(x, weight, fc.bias))
        cls_score = self.box_predictor.cls_score
        bbox_pred = self.box_predictor.bbox_pred
        scores = F.conv2d(
            x, cls_score.weight.view(cls_score.out_features, -1, 1, 1), cls_score.bias
        )[:, :, 0, 0]
        proposal_deltas = F.conv2d(
            x, bbox_pred.weight.view(bbox_pred.out_features, -1, 1, 1), bbox_pred.bias
        )[:, :, 0, 0]

        # Detectron2 Fast R-CNN:
        # https://github.com/facebookresearch/detectron2/blob/8a9d885/detectron2/modeling/roi_heads/fast_rcnn.py#L465
        boxes = self.box_predictor.box2box_transform.apply_deltas(
            proposal_deltas, proposals
        )
        if self.box_predictor.use_sigmoid_ce:
            scores = scores.sigmoid()
        else:
            scores = F.softmax(scores, dim=-1)
        scores = scores[:, :-1]  # drop background class

        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = boxes.view(num_proposals, num_bbox_reg_classes, 4)

        # Mask head, pooled on the input proposal boxes (not the refined
        # boxes) so the pooling window matches the fp32 reference.
        # Logits are clamped to +/-8: sigmoid is fully saturated beyond that
        # (within 3e-4 of 0/1), so no information is lost, and the clamp
        # keeps the output range tight for quantization and comparison.
        mask_features = self._roi_pool(proposals, features, self.roi_heads.mask_pooler)
        mask_logits = self.roi_heads.mask_head.layers(mask_features)
        mask_logits = mask_logits.clamp(min=-8.0, max=8.0)

        return boxes[None], scores[None], mask_logits

    def _roi_pool(
        self,
        boxes: torch.Tensor,
        features: list[torch.Tensor],
        pooler: Any,
    ) -> torch.Tensor:
        """
        ROI pooling matching detectron2's ROIPooler with assign_boxes_to_levels,
        implemented with static shapes: every box is pooled on every FPN level
        and the results are combined with a per-level assignment mask.

        Parameters
        ----------
        boxes
            (N, 4) xyxy boxes.
        features
            List of 4 FPN feature maps, finest first.
        pooler
            Detectron2 ROIPooler instance.

        Returns
        -------
        torch.Tensor
            Pooled features of shape (N, C, output_size, output_size).
        """
        # https://github.com/facebookresearch/detectron2/blob/8a9d885/detectron2/modeling/poolers.py#L25
        box_sizes = torch.sqrt(
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        )
        level_assignments = torch.floor(
            pooler.canonical_level
            + torch.log2(box_sizes / pooler.canonical_box_size + 1e-8)
        )
        level_assignments = torch.clamp(
            level_assignments,
            min=pooler.min_level,
            max=pooler.max_level,
        ) - float(pooler.min_level)

        output: torch.Tensor | None = None
        for level, level_pooler in enumerate(pooler.level_poolers):
            pooled = gather_roi_align(
                features[level],
                boxes,
                level_pooler.output_size,
                level_pooler.spatial_scale,
                level_pooler.aligned,
            )
            mask = (level_assignments == level).to(pooled.dtype)
            pooled = pooled * mask[:, None, None, None]
            output = pooled if output is None else output + pooled
        assert output is not None
        return output

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Lite-MP percentage for mixed-precision quantization."""
        return HUB_LITEMP_PERCENTAGE

    def component_precision(self) -> Precision:
        return Precision.w8a8

    def get_input_spec(
        self,
        head_height: int = 200,
        head_width: int = 200,
        num_proposals: int = DEFAULT_NUM_PROPOSALS,
    ) -> InputSpec:
        return {
            "feature_p2": TensorSpec(
                shape=(1, 256, head_height, head_width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "feature_p3": TensorSpec(
                shape=(1, 256, head_height // 2, head_width // 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "feature_p4": TensorSpec(
                shape=(1, 256, head_height // 4, head_width // 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "feature_p5": TensorSpec(
                shape=(1, 256, head_height // 8, head_width // 8),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "proposals": TensorSpec(
                shape=(num_proposals, 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        # Precomputed proposal-generator outputs for the bundled image, loaded
        # from the asset store rather than re-run (random feature maps yield
        # near-tie class scores and meaningless parity).
        spec = input_spec or self.get_input_spec()
        data = load_numpy(ROI_HEAD_SAMPLE_INPUTS)
        sample: SampleInputsType = {}
        for name in spec:
            array = data[name]
            expected = tuple(spec[name][0])
            if array.shape != expected:
                raise ValueError(
                    f"Cached ROI-head sample input '{name}' has shape "
                    f"{array.shape}, expected {expected}; regenerate "
                    f"{ROI_HEAD_SAMPLE_INPUTS.local_cache_path.name} for this spec."
                )
            sample[name] = [array]
        return sample

    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(io_type=IoType.TENSOR),
            "mask_logits": TensorSpec(io_type=IoType.TENSOR),
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True --truncate_64bit_io True"
        return compile_options


class Detectron2PanopticCocoDataset(CocoPanopticSegmentationDataset):
    """COCO panoptic dataset with a smaller on-device per-job sample count.

    The dense per-proposal boxes/scores/mask-logits make each inference job's
    output large, so detectron2_panoptic_seg caps it at 10 samples/job to stay
    under the 2 GB result limit; mask2former keeps the base class's default.
    """

    @staticmethod
    def default_samples_per_job() -> int:
        return 10


class Detectron2PanopticSeg(WorkbenchModelCollection):
    def __init__(
        self,
        proposal_generator: Detectron2PanopticProposalGenerator,
        roi_head: Detectron2PanopticROIHead,
    ) -> None:
        super().__init__(
            {"proposal_generator": proposal_generator, "roi_head": roi_head}
        )
        self.proposal_generator = proposal_generator
        self.roi_head = roi_head

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Detectron2PanopticCocoDataset

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 800,
        width: int = 800,
        num_proposals: int = DEFAULT_NUM_PROPOSALS,
    ) -> ComponentGroup[InputSpec]:
        return ComponentGroup(
            {
                "proposal_generator": self.proposal_generator.get_input_spec(
                    batch_size=batch_size,
                    generator_height=height,
                    generator_width=width,
                ),
                "roi_head": self.roi_head.get_input_spec(
                    head_height=height // 4,
                    head_width=width // 4,
                    num_proposals=num_proposals,
                ),
            }
        )

    def get_component_hub_quantize_options(
        self, component_name: str, precision: Precision, other_options: str = ""
    ) -> str:
        options = super().get_component_hub_quantize_options(
            component_name, precision, other_options
        )
        # The default w8a16 min_max range scheme collapses this model's
        # on-device accuracy (proposal generator PSNR drops from ~50 dB to
        # ~10 dB); the default (MSE-based) scheme calibrates it well. Only drop
        # the default — respect a --range_scheme the caller passed explicitly.
        if precision == Precision.w8a16 and "--range_scheme" not in other_options:
            options = options.replace("--range_scheme min_max", "").strip()
        return options

    @classmethod
    def from_pretrained(cls, config: str = DEFAULT_CONFIG) -> Self:
        # Build the GeneralizedRCNN once and share it between components,
        # rather than each component independently loading its own full
        # copy of the backbone + RPN + ROI heads.
        with native_roi_align():
            model = Detectron2PanopticProposalGenerator.load_pretrained_model(config)
        return cls(
            Detectron2PanopticProposalGenerator(model),
            Detectron2PanopticROIHead(model),
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Detectron2PanopticCocoDataset]

    def get_evaluator(self) -> BaseEvaluator:
        h, w = self.proposal_generator.get_input_spec()["image"][0][2:]
        return Detectron2PanopticSegEvaluator(
            model_image_height=h,
            model_image_width=w,
        )


def _sample_image_chw(height: int, width: int) -> np.ndarray:
    """
    Load the bundled sample image and resize it to (height, width).

    Parameters
    ----------
    height
        Target image height.
    width
        Target image width.

    Returns
    -------
    np.ndarray
        (3, height, width) float32 RGB array in [0, 1].
    """
    image = load_image(IMAGE_ADDRESS).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return app_to_net_image_inputs(image)[1][0].numpy()


def rpn_topk_anchor_counts(
    height: int,
    width: int,
    pre_nms_topk: int = RPN_PRE_NMS_TOPK,
    anchors_per_location: int = 3,
) -> list[int]:
    """
    Per-level anchor counts after the on-device pre-NMS top-k (see
    Detectron2PanopticProposalGenerator.forward): each of the 5 RPN feature
    levels (p2-p6, stride 4 to 64) capped at pre_nms_topk anchors.
    filter_rpn_proposals uses these counts to split the proposal generator's
    (already top-k'd) output back by level.

    Parameters
    ----------
    height
        Input image height.
    width
        Input image width.
    pre_nms_topk
        Per-level top-k cap applied on device.
    anchors_per_location
        Anchors per spatial location (3 aspect ratios for panoptic_fpn_R_50).

    Returns
    -------
    list[int]
        Post-top-k anchor count of each RPN feature level, finest first.
    """
    counts = []
    h, w = math.ceil(height / 4), math.ceil(width / 4)
    for _ in range(5):
        counts.append(min(h * w * anchors_per_location, pre_nms_topk))
        h, w = math.ceil(h / 2), math.ceil(w / 2)
    return counts


def filter_rpn_proposals(
    proposals: torch.Tensor,
    objectness_logits: torch.Tensor,
    image_height: int,
    image_width: int,
    num_proposals: int = DEFAULT_NUM_PROPOSALS,
    pre_nms_topk: int = RPN_PRE_NMS_TOPK,
    nms_thresh: float = RPN_NMS_THRESH,
) -> torch.Tensor:
    """
    Filter RPN proposals, matching detectron2's find_top_rpn_proposals:
    coordinate clamping, zero-area removal, then batched NMS with level ids.
    Runs off device because the surviving proposal count is data dependent.

    The per-level top-k (also part of find_top_rpn_proposals) runs on device
    inside Detectron2PanopticProposalGenerator.forward instead, since it is
    shape static (k_l = min(N_l, pre_nms_topk) is a compile-time constant for
    a fixed input resolution) — keeping it on device avoids passing the full
    dense, unfiltered anchor set (e.g. ~160k rows at 800x800) across the
    device boundary. proposals/objectness_logits here are therefore already
    top-k'd per level; pre_nms_topk and rpn_topk_anchor_counts are used only
    to recover the level boundaries for NMS's level ids.

    Parameters
    ----------
    proposals
        (1, num_anchors, 4) xyxy proposals, all RPN levels concatenated,
        already top-k'd per level on device. Batch dimension must be 1 —
        this function (and the ROI head that consumes its output) only
        supports batch_size=1.
    objectness_logits
        (1, num_anchors) objectness logits, already top-k'd per level on
        device. Batch dimension must be 1.
    image_height
        Model input image height (for clamping and level splitting).
    image_width
        Model input image width (for clamping and level splitting).
    num_proposals
        Fixed number of proposals to return (truncated / zero-padded).
    pre_nms_topk
        Per-level top-k applied on device; must match the value used there
        so the level boundaries here are reconstructed correctly.
    nms_thresh
        NMS IoU threshold.

    Returns
    -------
    torch.Tensor
        (num_proposals, 4) filtered proposals, sorted by objectness,
        zero-padded at the end if fewer survive NMS.
    """
    assert proposals.shape[0] == 1 and objectness_logits.shape[0] == 1, (
        "filter_rpn_proposals only supports batch_size=1; call it once per "
        "sample if the caller's batch dimension is > 1."
    )
    level_counts = rpn_topk_anchor_counts(image_height, image_width, pre_nms_topk)
    props_per_level = torch.split(proposals[0], level_counts)
    logits_per_level = torch.split(objectness_logits[0], level_counts)

    props_list = []
    scores_list = []
    level_ids_list = []
    for level_id, (props_l, logits_l) in enumerate(
        zip(props_per_level, logits_per_level, strict=True)
    ):
        props_l = props_l.clone()
        props_l[:, 0::2] = props_l[:, 0::2].clamp(0, image_width)
        props_l[:, 1::2] = props_l[:, 1::2].clamp(0, image_height)
        valid = (props_l[:, 2] > props_l[:, 0]) & (props_l[:, 3] > props_l[:, 1])
        props_l = props_l[valid]
        logits_l = logits_l[valid]
        props_list.append(props_l)
        scores_list.append(logits_l)
        level_ids_list.append(
            torch.full((props_l.shape[0],), level_id, dtype=torch.int64)
        )

    all_props = torch.cat(props_list, dim=0)
    all_scores = torch.cat(scores_list, dim=0)
    all_levels = torch.cat(level_ids_list, dim=0)

    # torchvision.ops.batched_nms is used directly (not the repo's
    # bounding_box_processing.batched_nms) because we need the surviving keep
    # indices to reorder the proposals by objectness and zero-pad them; the
    # repo helper returns filtered values without the indices.
    keep = torchvision.ops.batched_nms(
        all_props.float(), all_scores.float(), all_levels, nms_thresh
    )
    keep = keep[:num_proposals]
    final_proposals = all_props[keep]

    num_kept = final_proposals.shape[0]
    if num_kept < num_proposals:
        final_proposals = F.pad(final_proposals, (0, 0, 0, num_proposals - num_kept))
    return final_proposals
