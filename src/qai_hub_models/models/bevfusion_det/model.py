# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from mmengine.config import Config
from qai_hub.client import Device
from torch import nn
from torchpack.utils.config import configs
from typing_extensions import Self

from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps
from qai_hub_models.utils.input_spec import OutputSpec

with patch_mmdet_no_build_deps():
    from mmdet.models.layers import PatchMerging

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.models.bevfusion_det.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.bevfusion_det.model_patch import (
    PatchMerging_forward_optimized,
    bev_pool,
    patched_centerhead_get_task_detections,
    patched_get_cam_feats,
    patched_lss_forward,
    patched_topk,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    circle_nms as patched_circle_nms,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    onnx_atan2,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.corevision.window_partitioning import (
    WindowMSA_forward_optimized,
    window_partition_optimized,
    window_reverse_optimized,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_torch,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

with patch_mmdet_no_build_deps():
    from mmdet.models.backbones.swin import ShiftWindowMSA, WindowMSA
    from mmdet.registry import MODELS

    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.core.bbox.coders.centerpoint_bbox_coders import (
        CenterPointBBoxCoder,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.models.backbones.resnet import (
        GeneralizedResNet,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.models.heads.bbox.centerpoint import (
        CenterHead,
        SeparateHead,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.models.necks.generalized_lss import (
        GeneralizedLSSFPN,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.models.necks.lss import (
        LSSFPN,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.models.vtransforms.base import (
        BaseTransform,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.models.vtransforms.lss import (
        LSSTransform,
    )
    from qai_hub_models.models.bevfusion_det.external_repos.bevfusion.mmdet3d.utils.config import (
        recursive_eval,
    )

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 8

# checkpoint from  https://github.com/mit-han-lab/bevfusion/blob/main/README.md#:~:text=41.21-,Link,-LiDAR%2DOnly%20Baseline
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "camera-only-det.pth"
)
BEVFUSION_REPO_PATH = EXTERNAL_REPO_PATHS["bevfusion"]


def _apply_optimizations() -> None:
    """Apply optimized methods for WindowMSA, ShiftWindowMSA, and PatchMerging."""
    WindowMSA.forward = WindowMSA_forward_optimized
    ShiftWindowMSA.window_partition = window_partition_optimized
    ShiftWindowMSA.window_reverse = window_reverse_optimized
    PatchMerging.forward = PatchMerging_forward_optimized


class BEVFusionEncoder1(BaseModel):
    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    @classmethod
    def from_pretrained(cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())) -> Self:
        _apply_optimizations()
        backbone, neck, _, _, _, _ = BEVFusion.load_model(ckpt)
        return cls(backbone, neck)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input images using precomputed mean and std, reshapes them for processing,
        passes them through the backbone (Swin Transformer) and neck (GeneralizedLSSFPN),
        and returns the first feature map.

        Parameters
        ----------
        imgs
            Input tensor of shape (batch_size, 18, height, width) and range of [0-1],
            where 18 = 6 cameras * 3 channels (RGB).

        Returns
        -------
        feature_map : torch.Tensor
            Feature tensor of shape (batch_size, 6, 256, 32, 88).
        """
        B, NC, H, W = imgs.size()
        imgs_reshaped = imgs.reshape(B * (NC // 3), 3, H, W)
        imgs_normalized = normalize_image_torchvision(imgs_reshaped)
        x = self.backbone(imgs_normalized)
        x = self.neck(x)
        return x[0]

    def get_input_spec(
        self, batch_size: int = 1, encoder_height: int = 256, encoder_width: int = 704
    ) -> InputSpec:
        return {
            "imgs": TensorSpec(
                shape=(batch_size, 6 * 3, encoder_height, encoder_width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "feature_map": TensorSpec(),
        }


class BEVFusionEncoder2(BaseModel):
    def __init__(self, vtransform: nn.Module) -> None:
        super().__init__()
        self.vtransform = vtransform

    @classmethod
    def from_pretrained(cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())) -> Self:
        _apply_optimizations()
        _, _, vtransform, _, _, _ = BEVFusion.load_model(ckpt)
        return cls(vtransform)

    def forward(
        self,
        x: torch.Tensor,
        intrins: torch.Tensor,
        camera2lidars: torch.Tensor,
        inv_post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects camera features into BEV space and applies pooling.

        Parameters
        ----------
        x
            Feature tensor of shape (batch_size, 6, 256, 32, 88).
        intrins
            Camera intrinsics of shape (batch_size, 6, 3, 3).
        camera2lidars
            Camera-to-LiDAR transformations of shape (batch_size, 6, 4, 4).
        inv_post_rots
            Inverse rotation matrices of shape (batch_size, 6, 3, 3).
        post_trans
            Post-transformation translations of shape (batch_size, 6, 1, 3).

        Returns
        -------
        pooled_features : torch.Tensor
            Pooled features of shape (1993728, 80).
        lengths : torch.Tensor
            Lengths tensor of shape (59000,).
        geom_feats : torch.Tensor
            Geometric features of shape (2, 59000).
        """
        x, geom_feats, ranks = self.vtransform.forward(
            x, intrins, camera2lidars, inv_post_rots, post_trans
        )
        x, lengths, geom_feats = bev_pool(x, geom_feats, ranks)
        return x, lengths, geom_feats

    def get_input_spec(
        self, batch_size: int = 1, encoder_height: int = 256, encoder_width: int = 704
    ) -> InputSpec:
        return {
            "img": TensorSpec(
                shape=(batch_size, 6, 256, encoder_height // 8, encoder_width // 8),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "intrins": TensorSpec(
                shape=(batch_size, 6, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "camera2lidars": TensorSpec(
                shape=(batch_size, 6, 4, 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "inv_post_rots": TensorSpec(
                shape=(batch_size, 6, 3, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "post_trans": TensorSpec(
                shape=(batch_size, 6, 1, 3),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "pooled_features": TensorSpec(),
            "lengths": TensorSpec(),
            "geom_feats": TensorSpec(),
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


class BEVFusionEncoder3(BaseModel):
    def __init__(self, vtransform: nn.Module) -> None:
        super().__init__()
        self.vtransform = vtransform

    @classmethod
    def from_pretrained(cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())) -> Self:
        _, _, vtransform, _, _, _ = BEVFusion.load_model(ckpt)
        return cls(vtransform)

    def forward(
        self, segment_sums: torch.Tensor, geom_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes differences in segment sums, maps them to a BEV grid using geometric indices,
        and downsamples the result to produce a structured BEV feature map.

        Parameters
        ----------
        segment_sums
            Aggregated features of shape (1, 59000, 80).
        geom_feats
            Geometric indices of shape (1, 2, 59000).

        Returns
        -------
        bev_grid : torch.Tensor
            BEV grid features of shape (1, 80, 256, 256).
        """
        segment_sums = segment_sums.reshape(-1, 80)

        x_pos_vals = geom_feats[0, 0]
        channel_vals = geom_feats[0, 1]

        # Create output tensor with proper device and dtype
        out = torch.zeros(
            (1, 256, 256, 80), dtype=segment_sums.dtype, device=segment_sums.device
        )

        # Clamp indices to valid range to prevent out-of-bounds errors
        x_pos_vals = torch.clamp(x_pos_vals, 0, 255).long()
        channel_vals = torch.clamp(channel_vals, 0, 255).long()
        ba = torch.zeros_like(x_pos_vals)

        out.index_put_((ba, x_pos_vals, channel_vals), segment_sums)
        x = out.permute(0, 3, 1, 2).contiguous()
        return self.vtransform.downsample(x)  # type: ignore[operator]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "segment": TensorSpec(
                shape=(batch_size, 59000, 80),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "geom": TensorSpec(
                shape=(batch_size, 2, 59000),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "bev_grid": TensorSpec(),
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


class BEVFusionDecoder(BaseModel):
    def __init__(self, backbone: nn.Module, neck: nn.Module, heads: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads

    @classmethod
    def from_pretrained(cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())) -> Self:
        _, _, _, backbone, neck, heads = BEVFusion.load_model(ckpt)
        return cls(backbone, neck, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes BEV features through a backbone (GeneralizedResNet), neck (LSSFPN),
        and heads (CenterHead) to generate 3D bounding box predictions, scores, and labels.

        Parameters
        ----------
        x
            BEV feature tensor of shape (batch_size, 80, 128, 128).

        Returns
        -------
        outputs : torch.Tensor
            Concatenated tensor containing all task-specific outputs
            (regression, height, dimension, rotation, velocity, heatmap)
            with shape (batch_size, 70, 128, 128).
        """
        x = self.backbone(x)
        x = self.neck(x)
        ret_dicts = self.heads(x)

        # Pack tensors into a single tensor since torch.export requires a flat output
        tensors = []
        for task_dict in ret_dicts:
            task_dict = task_dict[0]
            for key in ["reg", "height", "dim", "rot", "vel", "heatmap"]:
                tensor = task_dict[key]
                tensors.append(tensor)

        return torch.cat(tensors, dim=1)

    def get_input_spec(
        self, batch_size: int = 1, decoder_height: int = 128, decoder_width: int = 128
    ) -> InputSpec:
        return {
            "x": TensorSpec(
                shape=(batch_size, 80, decoder_height, decoder_width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "outputs": TensorSpec(),
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


class BEVFusion(WorkbenchModelCollection):
    def __init__(
        self,
        encoder1: BEVFusionEncoder1,
        encoder2: BEVFusionEncoder2,
        encoder3: BEVFusionEncoder3,
        decoder: BEVFusionDecoder,
    ) -> None:
        super().__init__(
            {
                "encoder_1": encoder1,
                "encoder_2": encoder2,
                "encoder_3": encoder3,
                "decoder": decoder,
            }
        )
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

        self.decoder = decoder

    def get_input_spec(
        self,
        batch_size: int = 1,
        encoder_height: int = 256,
        encoder_width: int = 704,
        decoder_height: int = 128,
        decoder_width: int = 128,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            encoder_height=encoder_height,
            encoder_width=encoder_width,
            decoder_height=decoder_height,
            decoder_width=decoder_width,
        )

    @staticmethod
    def load_model(
        ckpt: str = str(DEFAULT_WEIGHTS.fetch()),
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        _apply_optimizations()

        config_path = (
            BEVFUSION_REPO_PATH
            / "configs"
            / "nuscenes"
            / "det"
            / "centerhead"
            / "lssfpn"
            / "camera"
            / "256x704"
            / "swint"
            / "default.yaml"
        )

        configs.load(str(config_path), recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None

        def load_state_dict(module: nn.Module, state_dict: dict, prefix: str) -> None:
            module.load_state_dict(
                {
                    k.replace(prefix, ""): v
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
            )

        checkpoint = load_torch(ckpt)
        state_dict = checkpoint["state_dict"]

        encoder_backbone = MODELS.build(cfg.model.encoders.camera.backbone)
        load_state_dict(encoder_backbone, state_dict, "encoders.camera.backbone.")

        neck_cfg = cfg.model.encoders.camera.neck
        neck_cfg.pop("type")
        encoder_neck = GeneralizedLSSFPN(**neck_cfg)
        load_state_dict(encoder_neck, state_dict, "encoders.camera.neck.")

        vtransform_cfg = cfg.model.encoders.camera.vtransform
        vtransform_cfg.pop("type")

        PatchedLSSTransform = type(
            "PatchedLSSTransform", (LSSTransform, BaseTransform), {}
        )

        LSSTransform.forward = patched_lss_forward
        LSSTransform.get_cam_feats = patched_get_cam_feats
        vtransform = PatchedLSSTransform(**vtransform_cfg)

        load_state_dict(vtransform, state_dict, "encoders.camera.vtransform.")

        decoder_bb_cfg = cfg.model.decoder.backbone
        decoder_bb_cfg.pop("type")
        decoder_backbone = GeneralizedResNet(**decoder_bb_cfg)
        load_state_dict(decoder_backbone, state_dict, "decoder.backbone.")

        decoder_neck_cfg = cfg.model.decoder.neck
        decoder_neck_cfg.pop("type")
        decoder_neck = LSSFPN(**decoder_neck_cfg)
        load_state_dict(decoder_neck, state_dict, "decoder.neck.")

        bbox_coder_cfg = cfg.model.heads["object"]["bbox_coder"]
        bbox_coder_cfg.pop("type")
        bbox_coder = CenterPointBBoxCoder(**bbox_coder_cfg)
        bbox_coder.onnx_atan2 = onnx_atan2

        CenterHead._topk = patched_topk
        CenterHead.get_task_detections = patched_centerhead_get_task_detections
        CenterHead.circle_nms = staticmethod(patched_circle_nms)

        cfg.model.heads["object"].pop("type", None)
        cfg.model.heads["object"].pop("bbox_coder", None)

        centerhead_cfg = cfg.model.heads["object"]
        centerhead_cfg["separate_head"] = dict(centerhead_cfg["separate_head"])

        centerhead = CenterHead(
            **centerhead_cfg,
            separate_head_cls=SeparateHead,
            bbox_coder=bbox_coder,
        )

        load_state_dict(centerhead, state_dict, "heads.object.")
        return (
            encoder_backbone,
            encoder_neck,
            vtransform,
            decoder_backbone,
            decoder_neck,
            centerhead,
        )

    @classmethod
    def from_pretrained(cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())) -> Self:
        (
            encoder_backbone,
            encoder_neck,
            vtransform,
            decoder_backbone,
            decoder_neck,
            centerhead,
        ) = cls.load_model(ckpt)
        models = (
            BEVFusionEncoder1(encoder_backbone, encoder_neck),
            BEVFusionEncoder2(vtransform),
            BEVFusionEncoder3(vtransform),
            BEVFusionDecoder(decoder_backbone, decoder_neck, centerhead),
        )
        for model in models[:-1]:
            model.eval()

        return cls(*models)
