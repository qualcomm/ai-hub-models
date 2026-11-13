# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from mmengine.config import Config
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.datasets.nuscenes import NuscenesDataset
from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps
from qai_hub_models.models.bevdet.evaluator import (
    NuscenesObjectDetectionEvaluator,
)
from qai_hub_models.models.bevdet.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.bevdet.external_repos.bevdet.mmdet3d.core.bbox.coders.centerpoint_bbox_coders import (
    CenterPointBBoxCoder,
)
from qai_hub_models.models.bevdet.external_repos.bevdet.mmdet3d.models.backbones.resnet import (
    CustomResNet,
)
from qai_hub_models.models.bevdet.external_repos.bevdet.mmdet3d.models.necks.fpn import (
    CustomFPN,
)
from qai_hub_models.models.bevdet.external_repos.bevdet.mmdet3d.models.necks.lss_fpn import (
    FPN_LSS,
)
from qai_hub_models.models.bevdet.model_patches import (
    CenterHead,
    LSSViewTransformerOptimized,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_numpy,
    load_torch,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec, IoType, OutputSpec, TensorSpec

with patch_mmdet_no_build_deps():
    from mmdet.models.task_modules import BaseBBoxCoder
    from mmdet.registry import MODELS

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# These assets are preprocessed inputs for the model,
# which are source from the nuscene dataset correspond to
# 'scene-0103' with sample_token '3e8750f331d7499e9b5123e9eb70f2e2'
IMAGES = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "imgs.pt")
INV_INTRINS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inv_intrins.pt"
)
SENSOR2KEYEGOS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sensor2keyegos.pt"
)
INV_POST_ROTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inv_post_rots.pt"
)
POST_TRANS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "post_trans.pt"
)

DEPTH = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "depth.npy")
FEAT = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "feat.npy")
BEV_FEATURE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "bev_feature.npy"
)

# Checkpoint is sourced from
# https://github.com/HuangJunJie2017/BEVDet/tree/dev3.0?tab=readme-ov-file#:~:text=30.7-,baidu,-baidu
BEVDET_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "bevdet-r50.pth"
)


class BEVDetEncoder(BaseModel):
    """Image encoder + depth net: cameras -> (depth distribution, context features)."""

    def __init__(
        self,
        img_backbone: BaseModule,
        img_neck: BaseModule,
        img_view_transformer: BaseModule,
    ) -> None:
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.img_view_transformer = img_view_transformer

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        return cast(Self, BEVDet.from_pretrained(ckpt).encoder)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the image encoder and the view-transformer depth_net.

        B = batch size, S = number of cameras, C = 3, H = img height, W = img width

        Parameters
        ----------
        image
            torch.Tensor of shape [B,S*C,H,W] as float32, range[0-1], RGB.

        Returns
        -------
        depth : torch.Tensor
            Per-pixel depth distribution (softmax over D), shape [B, S, D, H', W'].
        feat : torch.Tensor
            Context features, shape [B, S, out_channels, H', W'].
        """
        B, NC, H, W = image.shape
        num_cam = NC // 3
        image = image.view(B * num_cam, 3, H, W)[:, [2, 1, 0], ...]
        image = normalize_image_torchvision(image)

        x = self.img_backbone(image)
        x = self.img_neck(x)
        depth, feat = self.img_view_transformer(x)
        # Restore the batch dim folded into the camera axis so I/O is batch-first.
        depth = depth.view(B, num_cam, *depth.shape[1:])
        feat = feat.view(B, num_cam, *feat.shape[1:])
        return depth, feat

    def get_input_spec(
        self,
        num_cam: int = 6,
        encoder_height: int = 256,
        encoder_width: int = 704,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, num_cam * 3, encoder_height, encoder_width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "depth": TensorSpec(),
            "feat": TensorSpec(),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_torch(IMAGES)
        B, N, C, H, W = image.shape
        image = image.view(B, N * C, H, W).numpy()
        return {"image": [image]}

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
            compile_options += " --truncate_64bit_tensors True"
        return compile_options

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return NuscenesDataset

    def component_precision(self) -> Precision:
        return Precision.w8a8_mixed_int16

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        return 5


class BEVDetPooler(BaseModel):
    """BEV pooling: geometry projection + voxel-index prep + pooling cumsum."""

    def __init__(self, img_view_transformer: BaseModule) -> None:
        super().__init__()
        self.img_view_transformer = img_view_transformer

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        return cast(Self, BEVDet.from_pretrained(ckpt).pooler)

    def forward(
        self,
        depth: torch.Tensor,
        feat: torch.Tensor,
        sensor2keyegos: torch.Tensor,
        inv_intrins: torch.Tensor,
        inv_post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project per-pixel features into the BEV grid.

        Parameters
        ----------
        depth
            Per-pixel depth distribution from the encoder, shape [B, N, D, H, W].
        feat
            Context features from the encoder, shape [B, N, C, H, W].
        sensor2keyegos
            torch.Tensor of shape [B, N, 4, 4] as float32.
        inv_intrins
            torch.Tensor of shape [B, N, 3, 3] as float32.
        inv_post_rots
            torch.Tensor of shape [B, N, 3, 3] as float32.
        post_trans
            torch.Tensor of shape [B, N, 1, 3] as float32.

        Returns
        -------
        bev_feature : torch.Tensor
            BEV feature of shape (B, C, bev_H, bev_W).
        """
        return self.img_view_transformer.pool_to_bev(
            depth, feat, sensor2keyegos, inv_intrins, inv_post_rots, post_trans
        )

    def get_input_spec(
        self,
        num_cam: int = 6,
        depth_bins: int = 59,
        feat_channels: int = 64,
        feat_height: int = 16,
        feat_width: int = 44,
    ) -> InputSpec:
        return {
            "depth": TensorSpec(
                shape=(1, num_cam, depth_bins, feat_height, feat_width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "feat": TensorSpec(
                shape=(1, num_cam, feat_channels, feat_height, feat_width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "sensor2keyegos": TensorSpec(
                shape=(1, num_cam, 4, 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "inv_intrins": TensorSpec(
                shape=(1, num_cam, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "inv_post_rots": TensorSpec(
                shape=(1, num_cam, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "post_trans": TensorSpec(
                shape=(1, num_cam, 1, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"bev_feature": TensorSpec()}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {
            "depth": [load_numpy(DEPTH)],
            "feat": [load_numpy(FEAT)],
            "sensor2keyegos": [load_torch(SENSOR2KEYEGOS).numpy()],
            "inv_intrins": [load_torch(INV_INTRINS).numpy()],
            "inv_post_rots": [load_torch(INV_POST_ROTS).numpy()],
            "post_trans": [load_torch(POST_TRANS).numpy()],
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
            compile_options += " --truncate_64bit_tensors True"
        return compile_options

    def get_hub_inference_options(
        self,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        # Pin to the CPU backend: the fp32 voxel quantization + large-magnitude
        # pooling cumsum cannot run on the HTP.
        inference_options = super().get_hub_inference_options(
            target_runtime, other_inference_options, context_graph_name
        )
        if "--compute_unit" not in inference_options:
            inference_options += " --compute_unit cpu"
        return inference_options

    def component_precision(self) -> Precision:
        # No learned weights and needs fp32 for the cumsum.
        return Precision.float


class BEVDetDecoder(BaseModel):
    """
    BEV encoder + detection head: BEV feature -> raw 3D detection head outputs.

    Takes the BEV feature (produced on the host by ``LSSViewTransformerOptimized.
    pool_to_bev``) and runs the BEV-encoder backbone, neck and CenterHead.
    """

    def __init__(
        self,
        img_bev_encoder_backbone: BaseModule,
        img_bev_encoder_neck: BaseModule,
        pts_bbox_head: BaseModule,
        bbox_coder: BaseBBoxCoder,
    ) -> None:
        super().__init__()
        self.img_bev_encoder_backbone = img_bev_encoder_backbone
        self.img_bev_encoder_neck = img_bev_encoder_neck
        self.pts_bbox_head = pts_bbox_head
        self.bboxcoder = bbox_coder

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        return cast(Self, BEVDet.from_pretrained(ckpt).decoder)

    def forward(
        self,
        bev_feature: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run the BEV encoder and detection head.

        Parameters
        ----------
        bev_feature
            BEV feature of shape (B, C, bev_H, bev_W) = (1, 64, 128, 128).

        Returns
        -------
        reg : torch.Tensor
            2D regression value with the shape of [B, 2, H, W].
        height : torch.Tensor
            Height value with the shape of [B, 1, H, W].
        dim : torch.Tensor
            Size value with the shape of [B, 3, H, W].
        rot : torch.Tensor
            Rotation value with the shape of [B, 2, H, W].
        vel : torch.Tensor
            Velocity value with the shape of [B, 2, H, W].
        heatmap : torch.Tensor
            Heatmap (sigmoid-activated) with the shape of [B, N, H, W].
        """
        x = self.img_bev_encoder_backbone(bev_feature)
        img_feats = self.img_bev_encoder_neck(x)
        ret_dicts = self.pts_bbox_head(img_feats)

        return (
            ret_dicts[0]["reg"],
            ret_dicts[0]["height"],
            ret_dicts[0]["dim"],
            ret_dicts[0]["rot"],
            ret_dicts[0]["vel"],
            ret_dicts[0]["heatmap"].sigmoid(),
        )

    def get_input_spec(
        self,
    ) -> InputSpec:
        return {
            "bev_feature": TensorSpec(
                shape=(1, 64, 128, 128),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "reg": TensorSpec(),
            "height": TensorSpec(),
            "dim": TensorSpec(),
            "rot": TensorSpec(),
            "vel": TensorSpec(),
            "heatmap": TensorSpec(),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"bev_feature": [load_numpy(BEV_FEATURE)]}

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
            compile_options += " --truncate_64bit_tensors True"

        return compile_options

    def get_evaluator(self) -> NuscenesObjectDetectionEvaluator:
        return NuscenesObjectDetectionEvaluator(self.bboxcoder)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [NuscenesDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return NuscenesDataset

    def component_precision(self) -> Precision:
        return Precision.w8a16_mixed_fp16

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 5


class BEVDet(WorkbenchModelCollection):
    def __init__(
        self,
        encoder: BEVDetEncoder,
        pooler: BEVDetPooler,
        decoder: BEVDetDecoder,
    ) -> None:
        super().__init__({"encoder": encoder, "pooler": pooler, "decoder": decoder})
        self.encoder = encoder
        self.pooler = pooler
        self.decoder = decoder
        self.bboxcoder = decoder.bboxcoder

    def get_input_spec(
        self,
        num_cam: int = 6,
        encoder_height: int = 256,
        encoder_width: int = 704,
        depth_bins: int = 59,
        feat_channels: int = 64,
        feat_height: int = 16,
        feat_width: int = 44,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            num_cam=num_cam,
            encoder_height=encoder_height,
            encoder_width=encoder_width,
            depth_bins=depth_bins,
            feat_channels=feat_channels,
            feat_height=feat_height,
            feat_width=feat_width,
        )

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        ckpt = str(BEVDET_CKPT.fetch()) if ckpt is None else ckpt

        config_path = (
            EXTERNAL_REPO_PATHS["bevdet"] / "configs" / "bevdet" / "bevdet-r50.py"
        )
        cfg = Config.fromfile(config_path)
        cfg.model.train_cfg = None

        cfg.model.pts_bbox_head.bbox_coder.pop("type")
        cfg.model.img_neck.pop("type")
        cfg.model.img_bev_encoder_backbone.pop("type")
        cfg.model.img_bev_encoder_neck.pop("type")

        pts_test_cfg = cfg.model.test_cfg["pts"] if cfg.model.test_cfg else None
        cfg.model.pts_bbox_head.update(test_cfg=pts_test_cfg)

        holder = nn.Module()
        holder.img_backbone = MODELS.build(cfg.model.img_backbone)
        holder.img_neck = CustomFPN(**cfg.model.img_neck)
        holder.img_view_transformer = LSSViewTransformerOptimized(
            **cfg.model.img_view_transformer
        )
        holder.img_bev_encoder_backbone = CustomResNet(
            **cfg.model.img_bev_encoder_backbone
        )
        holder.img_bev_encoder_neck = FPN_LSS(**cfg.model.img_bev_encoder_neck)
        holder.pts_bbox_head = CenterHead(**cfg.model.pts_bbox_head)
        load_checkpoint(holder, ckpt, map_location="cpu")
        holder.eval()

        bbox_coder = CenterPointBBoxCoder(**cfg.model.pts_bbox_head.bbox_coder)
        encoder = BEVDetEncoder(
            holder.img_backbone, holder.img_neck, holder.img_view_transformer
        )
        pooler = BEVDetPooler(holder.img_view_transformer)
        decoder = BEVDetDecoder(
            holder.img_bev_encoder_backbone,
            holder.img_bev_encoder_neck,
            holder.pts_bbox_head,
            bbox_coder,
        )
        return cls(encoder, pooler, decoder)

    def get_evaluator(self) -> NuscenesObjectDetectionEvaluator:
        return NuscenesObjectDetectionEvaluator(self.bboxcoder)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [NuscenesDataset]
