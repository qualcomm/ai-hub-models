# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qai_hub_models.models.agpose.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.templates.common import replace_module_recursively
from qai_hub_models.utils.asset_loaders import repo_in_sys_path


@contextmanager
def ag_pose_repo_paths() -> Generator[None, None, None]:
    """
    Context manager that adds all AG-Pose external repo subdirectories
    to sys.path for the duration of the block, then removes them.

    Used by both AGPoseBackbone.from_pretrained (model.py) and
    compute_fps_indices (dataset.py) to import AG-Pose modules and
    PointNet++ CUDA extensions without permanently polluting sys.path.
    """
    repo_path = EXTERNAL_REPO_PATHS["ag_pose"]
    with ExitStack() as stack:
        for p in [
            str(repo_path / "model"),
            str(repo_path),
            str(repo_path / "model" / "pointnet2"),
            str(repo_path / "provider"),
            str(repo_path / "utils"),
        ]:
            stack.enter_context(repo_in_sys_path(p))
        yield


def patch_dinov2_attention(model: nn.Module) -> None:
    """
    Replace DINOv2 scaled-dot-product attention with explicit matmul/softmax
    for ONNX opset-17 export compatibility.
    """
    dinov2_path = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    try:
        with repo_in_sys_path(dinov2_path):
            from dinov2.layers.attention import Attention

        class PatchedDINOv2Attention(Attention):
            """ONNX-compatible DINOv2 Attention using explicit matmul+softmax."""

            def __init__(self, src: Attention) -> None:
                super().__init__(
                    dim=src.qkv.in_features,
                    num_heads=src.num_heads,
                    qkv_bias=src.qkv.bias is not None,
                    proj_bias=src.proj.bias is not None,
                    attn_drop=src.attn_drop,
                    proj_drop=src.proj_drop.p,
                )
                self.load_state_dict(src.state_dict())

            def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
                q, k, v = q.float(), k.float(), v.float()
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = torch.softmax(attn, dim=-1)
                if self.attn_drop > 0.0 and self.training:
                    attn = F.dropout(attn, p=self.attn_drop)
                x = torch.matmul(attn, v).to(x.dtype)
                x = x.transpose(1, 2).contiguous().view(B, N, C)
                return self.proj_drop(self.proj(x))

        replace_module_recursively(model, Attention, PatchedDINOv2Attention)

    except ImportError as err:
        # dinov2_path does not exist: DINOv2 was never downloaded and the
        # DINOv2Stub fallback is active in this environment (e.g. CI with no
        # GitHub access). The ImportError is expected — silently skip.
        # dinov2_path exists: real DINOv2 is present but the import failed.
        # F.scaled_dot_product_attention remains active and ONNX opset-17
        # export will break — raise immediately so this is not a silent regression.
        if os.path.exists(dinov2_path):
            raise RuntimeError(
                f"patch_dinov2_attention: failed to import "
                f"dinov2.layers.attention.Attention even though DINOv2 is "
                f"present at {dinov2_path}. F.scaled_dot_product_attention is "
                f"still active and ONNX opset-17 export will break. "
                f"Check that the DINOv2 repo structure has not changed."
            ) from err


def manual_multihead_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor | None,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace native MultiheadAttention with an ONNX-compatible implementation
    to avoid unsupported attention operators during opset-17 export.
    """
    orig_dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    B, Lq, C = query.shape
    Lk = key.shape[1]
    head_dim = C // num_heads
    scale = head_dim**-0.5
    q = F.linear(
        query,
        in_proj_weight[:C].float(),
        in_proj_bias[:C].float() if in_proj_bias is not None else None,
    )
    k = F.linear(
        key,
        in_proj_weight[C : 2 * C].float(),
        in_proj_bias[C : 2 * C].float() if in_proj_bias is not None else None,
    )
    v = F.linear(
        value,
        in_proj_weight[2 * C :].float(),
        in_proj_bias[2 * C :].float() if in_proj_bias is not None else None,
    )
    q = q.reshape(B, Lq, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, Lk, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, Lk, num_heads, head_dim).transpose(1, 2)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).reshape(B, Lq, C)
    out = F.linear(out, out_proj_weight.float(), out_proj_bias.float())  # type: ignore[union-attr]
    attn_weights = attn.mean(dim=1)
    return out.to(orig_dtype), attn_weights.to(orig_dtype)


def attn_block_forward(
    self: Any,
    kpt_query: torch.Tensor,
    input_feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # -- Begin Qualcomm Change
    kpt_query = torch.clamp(kpt_query, -100.0, 100.0)
    kpt_query2 = F.layer_norm(
        kpt_query.float(),
        self.norm1.normalized_shape,
        self.norm1.weight.float(),
        self.norm1.bias.float(),
        self.norm1.eps,
    ).to(kpt_query.dtype)
    kpt_query2, attn = manual_multihead_attn(
        kpt_query2,
        input_feature,
        input_feature,
        self.multihead_attn.in_proj_weight,
        self.multihead_attn.in_proj_bias,
        self.multihead_attn.out_proj.weight,
        self.multihead_attn.out_proj.bias,
        self.multihead_attn.num_heads,
    )
    kpt_query = kpt_query + self.dropout1(kpt_query2)

    kpt_query = torch.clamp(kpt_query, -100.0, 100.0)
    kpt_query2 = F.layer_norm(
        kpt_query.float(),
        self.norm2.normalized_shape,
        self.norm2.weight.float(),
        self.norm2.bias.float(),
        self.norm2.eps,
    ).to(kpt_query.dtype)
    kpt_query2, _ = manual_multihead_attn(
        kpt_query2,
        kpt_query2,
        kpt_query2,
        self.self_attn.in_proj_weight,
        self.self_attn.in_proj_bias,
        self.self_attn.out_proj.weight,
        self.self_attn.out_proj.bias,
        self.self_attn.num_heads,
    )
    kpt_query = kpt_query + self.dropout2(kpt_query2)

    kpt_query = torch.clamp(kpt_query, -100.0, 100.0)
    kpt_query2 = F.layer_norm(
        kpt_query.float(),
        self.norm3.normalized_shape,
        self.norm3.weight.float(),
        self.norm3.bias.float(),
        self.norm3.eps,
    ).to(kpt_query.dtype)
    kpt_query2 = self.ffn(kpt_query2.float()).to(kpt_query.dtype)
    kpt_query = kpt_query + self.dropout3(kpt_query2)
    # -- End Qualcomm Change
    return kpt_query, attn


def self_attn_block_forward(
    self: Any,
    kpt_query: torch.Tensor,
) -> torch.Tensor:
    # -- Begin Qualcomm Change
    kpt_query = torch.clamp(kpt_query, -100.0, 100.0)
    kpt_query2 = F.layer_norm(
        kpt_query.float(),
        self.norm1.normalized_shape,
        self.norm1.weight.float(),
        self.norm1.bias.float(),
        self.norm1.eps,
    ).to(kpt_query.dtype)
    kpt_query2, _ = manual_multihead_attn(
        kpt_query2,
        kpt_query2,
        kpt_query2,
        self.self_attn.in_proj_weight,
        self.self_attn.in_proj_bias,
        self.self_attn.out_proj.weight,
        self.self_attn.out_proj.bias,
        self.self_attn.num_heads,
    )
    kpt_query = kpt_query + self.dropout1(kpt_query2)

    kpt_query = torch.clamp(kpt_query, -100.0, 100.0)
    kpt_query2 = F.layer_norm(
        kpt_query.float(),
        self.norm2.normalized_shape,
        self.norm2.weight.float(),
        self.norm2.bias.float(),
        self.norm2.eps,
    ).to(kpt_query.dtype)
    kpt_query2 = self.ffn(kpt_query2.float()).to(kpt_query.dtype)
    # -- End Qualcomm Change
    return kpt_query + self.dropout2(kpt_query2)


def patch_manual_multihead_attn(model: nn.Module) -> None:
    """
    Replace AttnBlock and SelfAttnBlock instances in model with ONNX-compatible
    per-instance patched versions using replace_module_recursively.

    Per-instance patching avoids the global class-level side effect of
    overwriting AttnBlock.forward / SelfAttnBlock.forward for the whole process.
    """
    # Imported lazily: ag_pose.model imports a legacy resnet module at
    # module scope. That import raises ModuleNotFoundError during initial
    # package load before the external repo is cloned and sys.path is set up.
    from qai_hub_models.models.agpose.external_repos.ag_pose.model import (
        Net_modules,
    )

    AttnBlock: type[nn.Module] = Net_modules.AttnBlock
    SelfAttnBlock: type[nn.Module] = Net_modules.SelfAttnBlock

    class PatchedAttnBlock(nn.Module):
        """Per-instance ONNX-compatible AttnBlock using manual_multihead_attn."""

        def __init__(self, src: nn.Module) -> None:
            super().__init__()
            src_any: Any = src
            self.__dict__.update(src_any.__dict__)

        def forward(
            self, kpt_query: torch.Tensor, input_feature: torch.Tensor
        ) -> tuple:
            return attn_block_forward(self, kpt_query, input_feature)

    class PatchedSelfAttnBlock(nn.Module):
        """Per-instance ONNX-compatible SelfAttnBlock using manual_multihead_attn."""

        def __init__(self, src: nn.Module) -> None:
            super().__init__()
            src_any: Any = src
            self.__dict__.update(src_any.__dict__)

        def forward(self, kpt_query: torch.Tensor) -> torch.Tensor:
            return self_attn_block_forward(self, kpt_query)

    replace_module_recursively(model, AttnBlock, PatchedAttnBlock)
    replace_module_recursively(model, SelfAttnBlock, PatchedSelfAttnBlock)


class PoseSizeEstimatorBackbone(nn.Module):
    """
    Component A (float) of the split PoseSizeEstimator.

    Runs pts_mlp1 and pts_mlp2 on pts1/pts2, then concatenates with
    pts1_local to produce pre_pose_feat (B, 384, 96). No pose_mlp1 here.

    pose_mlp1, pose_mlp2, translation_estimator, size_estimator,
    rotation_estimator all live in PoseSizeEstimatorRotationHead (float).
    """

    def __init__(self) -> None:
        super().__init__()
        # Conv1d input channels padded 3->4 so per-channel quantized encodings
        # satisfy the QNN HTP constraint that input channels must be a multiple
        # of 4. The fourth channel is always zero-padded at runtime.
        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(4, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(4, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

    def forward(
        self,
        pts1: torch.Tensor,
        pts2: torch.Tensor,
        pts1_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run pts_mlp1/2 and concatenate with pts1_local.

        Parameters
        ----------
        pts1
            kpt_3d of shape (B, kpt_num, 3).
        pts2
            kpt_nocs of shape (B, kpt_num, 3).
        pts1_local
            kpt_feature from GAFA of shape (B, kpt_num, 256).

        Returns
        -------
        pre_pose_feat : torch.Tensor
            Shape (B, 384, 96) — cat(pts_mlp1, pts1_local.T, pts_mlp2).
        """
        # Pad 3-channel point inputs to 4 channels (zero-fill) so the first
        # Conv1d satisfies the QNN HTP per-channel quantization requirement
        # that input channels must be a multiple of 4.
        pts1_t = torch.nn.functional.pad(pts1.transpose(1, 2), (0, 0, 0, 1))
        pts2_t = torch.nn.functional.pad(pts2.transpose(1, 2), (0, 0, 0, 1))
        pts1 = self.pts_mlp1(pts1_t)
        pts2 = self.pts_mlp2(pts2_t)
        return torch.cat([pts1, pts1_local.transpose(1, 2), pts2], dim=1)


class PoseSizeEstimatorRotationHead(nn.Module):
    """
    Component B (float) of the split PoseSizeEstimator.

    Takes pre_pose_feat (B, 384, 96) and centroid c (B, 3) from backbone.
    Runs pose_mlp1 -> global concat -> pose_mlp2 -> all 3 estimators in float.
    """

    def __init__(self) -> None:
        super().__init__()
        # Keep these dimensions aligned with upstream; camera_real.yaml has no
        # configuration section for the estimator dimensions.
        # Import Ortho6d2Mat once at instantiation — stored for reuse in forward()
        # Imported lazily: ag_pose.utils imports a legacy resnet module at
        # module scope. That import raises ModuleNotFoundError during initial
        # package load before the external repo is cloned and sys.path is set up.
        from qai_hub_models.models.agpose.external_repos.ag_pose.utils.rotation_utils import (
            Ortho6d2Mat,
        )

        self._ortho6d2mat = Ortho6d2Mat
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        pre_pose_feat: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run pose estimation head in float precision."""
        pre_pose_feat = pre_pose_feat.float()
        pose_feat = self.pose_mlp1(pre_pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], dim=1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)
        r = self.rotation_estimator(pose_feat)
        a1 = r[:, :3].contiguous()
        a2 = r[:, 3:].contiguous()
        a1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-6)
        a2 = a2 / (torch.norm(a2, dim=-1, keepdim=True) + 1e-6)
        r = self._ortho6d2mat(a1, a2).view(-1, 3, 3)
        t = self.translation_estimator(pose_feat) + c.float()
        s = self.size_estimator(pose_feat)
        return r, t, s


def fold_dinov2_imagenet_norm(dino: nn.Module) -> None:
    """Fold ImageNet normalization into the DINOv2 patch embedding.

    Idempotent: a second call is a no-op (guarded by _norm_folded flag).
    This matters because NetBackbone.__init__ folds during torch.hub.load,
    and AGPoseBackbone.from_pretrained re-folds after load_state_dict restores
    the original patch-embed weights — the second call must not divide by std
    a second time.
    """
    if getattr(dino, "_norm_folded", False):
        return
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    dino_any: Any = dino
    proj: nn.Conv2d = dino_any.patch_embed.proj
    weight = proj.weight.data.clone()
    proj.weight.data = weight / std.view(1, 3, 1, 1)
    assert proj.bias is not None, "DINOv2 patch_embed.proj has no bias"
    proj.bias.data = proj.bias.data - (proj.weight.data * mean.view(1, 3, 1, 1)).sum(
        dim=(1, 2, 3)
    )
    dino._norm_folded = True  # type: ignore[assignment]


class NetBackbone(nn.Module):
    """
    Backbone component of AG-Pose for WorkbenchModelCollection split.

    Identical to Net but uses PoseSizeEstimatorBackbone instead of
    PoseSizeEstimator — does NOT run rotation_estimator.

    forward() returns (pre_pose_feat, c) directly
    instead of end_points dict — no training-time augmentation path.

    Inputs:  same as Net.forward() in eval mode

    Parameters
    ----------
    cfg
        pose_net config namespace from camera_real.yaml.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        # Imported lazily: ag_pose.model.modules imports a legacy resnet module
        # at module scope. That import raises ModuleNotFoundError during initial
        # package load before the external repo is cloned and sys.path is set up.
        # Note: the DINOv2 import in patch_dinov2_attention is also lazy but for
        # a different reason — dinov2 is only importable after torch.hub.load
        # adds facebookresearch_dinov2_main to sys.path at runtime.
        from qai_hub_models.models.agpose.external_repos.ag_pose.model.modules import (
            ModifiedResnet,
            PointNet2MSG,
        )
        from qai_hub_models.models.agpose.external_repos.ag_pose.model.Net_modules import (
            GeometricAwareFeatureAggregator,
            InstanceAdaptiveKeypointDetector,
            NOCS_Predictor,
        )

        self.cat_num = cfg.cat_num
        self.cfg = cfg
        if cfg.rgb_backbone == "resnet":
            self.rgb_extractor = ModifiedResnet()
        elif cfg.rgb_backbone == "dino":
            self.rgb_extractor = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                skip_validation=True,  # skip GitHub API fork-check — hits rate limit in CI
            )
            # -- Begin Qualcomm Change
            fold_dinov2_imagenet_norm(self.rgb_extractor)
            # -- End Qualcomm Change
            for param in self.rgb_extractor.parameters():
                param.requires_grad = False
            self.feature_mlp = nn.Sequential(
                nn.Conv1d(384, 128, 1),
            )
        else:
            raise NotImplementedError
        self.pts_extractor = PointNet2MSG(
            radii_list=[[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]]
        )
        self.IAKD = InstanceAdaptiveKeypointDetector(cfg.IAKD)
        self.GAFA = GeometricAwareFeatureAggregator(cfg.GAFA)
        self.nocs_predictor = NOCS_Predictor(cfg.NOCS_Predictor)
        self.estimator = PoseSizeEstimatorBackbone()

    def forward(
        self,
        inputs: dict,
        fps_indices: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run backbone forward pass.

        Parameters
        ----------
        inputs
            Must contain keys: rgb, pts, choose, category_label.
        fps_indices
            Pre-computed FPS index tensors [fps1, fps2, fps3, fps4].
            If None, FPS is computed internally.

        Returns
        -------
        pre_pose_feat : torch.Tensor
            Shape (B, 384, 96).
        c : torch.Tensor
            Shape (B, 3) — point cloud centroid.
        """
        rgb = inputs["rgb"]
        pts = inputs["pts"].float()
        choose = inputs["choose"]
        cls = inputs["category_label"].reshape(-1)
        c = torch.mean(pts, 1, keepdim=True).float()
        pts = pts.float() - c
        b = pts.size(0)
        index = (
            cls + torch.arange(b, dtype=torch.long, device=cls.device) * self.cat_num
        )
        if self.cfg.rgb_backbone == "resnet":
            rgb_local = self.rgb_extractor(rgb)
        elif self.cfg.rgb_backbone == "dino":
            dino_feature = self.rgb_extractor.forward_features(rgb)["x_prenorm"][:, 1:]  # type: ignore[operator]
            f_dim = dino_feature.shape[-1]
            num_patches = int(dino_feature.shape[1] ** 0.5)
            dino_feature = dino_feature.reshape(
                b, num_patches, num_patches, f_dim
            ).permute(0, 3, 1, 2)
            dino_feature = F.interpolate(
                dino_feature,
                size=(num_patches * 14, num_patches * 14),
                mode="bilinear",
                align_corners=False,
            )
            dino_feature = dino_feature.reshape(b, f_dim, -1)
            rgb_local = self.feature_mlp(dino_feature)
        else:
            raise NotImplementedError
        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous()
        rgb_local = rgb_local.float()
        pts_local = self.pts_extractor(pts, fps_indices=fps_indices)
        pts_local = pts_local.float()
        _, heat_map = self.IAKD(rgb_local, pts_local)
        kpt_3d = torch.bmm(heat_map, pts)
        kpt_feature = torch.bmm(
            heat_map,
            torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2),
        )
        kpt_feature = self.GAFA(
            kpt_feature,
            kpt_3d.detach(),
            torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2),
            pts,
        )
        kpt_nocs = self.nocs_predictor(kpt_feature, index)
        pre_pose_feat = self.estimator(
            kpt_3d.float(),
            kpt_nocs.detach().float(),
            kpt_feature.float(),
        )
        return pre_pose_feat, c.squeeze(1)
