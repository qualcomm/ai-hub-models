# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.basicvsr_pp.dataset import REDS4Dataset
from qai_hub_models.models.basicvsr_pp.model_adaptations import (
    flow_warp,
    pixel_shuffle,
)
from qai_hub_models.models.templates.super_resolution.superres_evaluator import (
    SuperResolutionOutputEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

# Number of intermediate feature channels (matches pretrained REDS4 checkpoint)
MID_CHANNELS = 64
# 2 frames gives each target frame one backward temporal neighbor, improving propagation over T=1.
DEFAULT_NUM_FRAMES = 2
# 64x64: peak VTCM tensor (prop_forward_2 backbone, 320x64x64xfp16 = 2.6MB) fits the HTP's 8MB limit.
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# REDS4 pretrained checkpoint (OpenMMLab public S3)
WEIGHTS_URL = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "basicvsr_plusplus_reds4.pth",
)

# SPyNet backbone checkpoint (required by the model; loaded separately)
SPYNET_WEIGHTS_URL = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "spynet.pth",
)

# Propagation pass names in execution order
PROP_PASSES = ["backward_1", "forward_1", "backward_2", "forward_2"]


class BasicVSRPP(BaseModel):
    """Adds --truncate_64bit_tensors for TFLite compilation."""

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )


class BasicVSRPPFlowNet(BasicVSRPP):
    """SPyNet optical flow sub-model.

    Estimates optical flow between two LR frames.  Runs 2*(T-1) times per
    clip (once forward, once backward for each adjacent frame pair).

    Inputs:  frame_ref (1, 3, H, W),  frame_supp (1, 3, H, W)
    Output:  flow      (1, 2, H, W)
    """

    def __init__(self, spynet: nn.Module) -> None:
        super().__init__()
        self.spynet = spynet

    def forward(
        self, frame_ref: torch.Tensor, frame_supp: torch.Tensor
    ) -> torch.Tensor:
        """Estimate optical flow from frame_supp to frame_ref.

        Parameters
        ----------
        frame_ref:
            Reference frame, shape ``(B, 3, H, W)``, values in ``[0, 1]``.
        frame_supp:
            Supporting (adjacent) frame, same shape as ``frame_ref``.

        Returns
        -------
        torch.Tensor
            Flow field of shape ``(B, 2, H, W)`` in pixels, where channel 0
            is the horizontal displacement and channel 1 is the vertical
            displacement from ``frame_supp`` toward ``frame_ref``.
        """
        return self._compute_flow_htp(frame_ref, frame_supp)

    def _compute_flow_htp(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        """SPyNet forward with bilinear replaced by nearest to avoid HTP DMA errors."""
        h, w = ref.shape[2], ref.shape[3]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref_p = F.interpolate(ref, size=(h_up, w_up), mode="nearest")
        supp_p = F.interpolate(supp, size=(h_up, w_up), mode="nearest")
        flow = self._pyramid_flow(ref_p, supp_p)
        flow = F.interpolate(flow, size=(h, w), mode="nearest")
        flow[:, 0] *= float(w) / float(w_up)
        flow[:, 1] *= float(h) / float(h_up)
        return flow

    def _pyramid_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        """SPyNet compute_flow with bilinear upsample replaced by nearest."""
        mean = cast(torch.Tensor, self.spynet.mean)
        std = cast(torch.Tensor, self.spynet.std)
        refs = [(ref - mean) / std]
        supps = [(supp - mean) / std]
        for _ in range(5):
            refs.append(
                F.avg_pool2d(refs[-1], kernel_size=2, stride=2, count_include_pad=False)
            )
            supps.append(
                F.avg_pool2d(
                    supps[-1], kernel_size=2, stride=2, count_include_pad=False
                )
            )
        refs.reverse()
        supps.reverse()
        n, _, hb, wb = refs[0].shape
        flow = refs[0].new_zeros(n, 2, hb, wb)
        for level, module in enumerate(cast(list, self.spynet.basic_module)):
            if level == 0:
                flow_up = flow
            else:
                # nearest avoids HTP DMA failures; multiply by 2 scales pixel-unit flow
                flow_up = F.interpolate(flow, scale_factor=2, mode="nearest") * 2.0
            flow = flow_up + module(
                torch.cat(
                    [
                        refs[level],
                        flow_warp(
                            supps[level],
                            flow_up.permute(0, 2, 3, 1),
                            interpolation="nearest",
                            padding_mode="border",
                        ),
                        flow_up,
                    ],
                    dim=1,
                )
            )
        return flow

    def get_input_spec(
        self,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        return {
            "frame_ref": TensorSpec(shape=(1, 3, height, width), dtype="float32"),
            "frame_supp": TensorSpec(shape=(1, 3, height, width), dtype="float32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"spynet_flow": TensorSpec(shape=(1, 2, -1, -1), dtype="float32")}

    @classmethod
    def from_pretrained(cls, weights: CachedWebModelAsset = WEIGHTS_URL) -> Self:
        return _load_collection(weights).flownet


class BasicVSRPPFeatExtract(BasicVSRPP):
    """Feature extraction sub-model.

    Extracts spatial features from a single LR frame.  Runs T times per clip.

    Input:  frame (1, 3, H, W)
    Output: feat  (1, 64, H, W)
    """

    def __init__(self, feat_extract: nn.Module) -> None:
        super().__init__()
        self.feat_extract = feat_extract

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.feat_extract(frame)

    def get_input_spec(
        self,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        return {"frame": TensorSpec(shape=(1, 3, height, width), dtype="float32")}

    def get_output_spec(self) -> OutputSpec:
        return {
            "spatial_feat": TensorSpec(shape=(1, MID_CHANNELS, -1, -1), dtype="float32")
        }

    @classmethod
    def from_pretrained(cls, weights: CachedWebModelAsset = WEIGHTS_URL) -> Self:
        return _load_collection(weights).feat_extract


class BasicVSRPPPropStep(BasicVSRPP):
    """One recurrent step of a single propagation pass.

    Wraps SecondOrderDeformableAlignment + ResidualBlocksWithInputConv for one
    pass direction.  Called T times per pass by the app; recurrent state
    (feat_prop, feat_n2) is managed externally.

    Fixed inputs (always present):
      feat_prop     (1, 64, H, W)  — propagated feature from previous step
      feat_n2       (1, 64, H, W)  — propagated feature from 2 steps back
      feat_current  (1, 64, H, W)  — spatial feature of current frame
      flow_n1       (1, 2,  H, W)  — optical flow to nearest neighbor
      flow_n2       (1, 2,  H, W)  — optical flow to second neighbor

    Prior inputs (num_prior_passes tensors, one per already-completed pass):
      prior_feat_0 .. prior_feat_{num_prior_passes-1}  (1, 64, H, W)

    Output:
      feat_prop_out  (1, 64, H, W)
    """

    def __init__(
        self,
        deform_align: nn.Module,
        backbone: nn.Module,
        num_prior_passes: int,
    ) -> None:
        super().__init__()
        self.deform_align = deform_align
        self.backbone = backbone
        self.num_prior_passes = num_prior_passes

    def forward(
        self,
        feat_prop: torch.Tensor,
        feat_n2: torch.Tensor,
        feat_current: torch.Tensor,
        flow_n1: torch.Tensor,
        flow_n2: torch.Tensor,
        *prior_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Run one recurrent propagation step and return the updated feature.

        Parameters
        ----------
        feat_prop:
            Propagated feature from the previous step, ``(B, C, H, W)``.
        feat_n2:
            Feature of the second temporal neighbor, ``(B, C, H, W)``.
        feat_current:
            Feature of the current frame, ``(B, C, H, W)``.
        flow_n1:
            Optical flow to the first (adjacent) neighbor, ``(B, 2, H, W)``.
        flow_n2:
            Optical flow to the second neighbor, ``(B, 2, H, W)``.
        *prior_feats:
            Accumulated propagated features from earlier passes, each
            ``(B, C, H, W)``.  Concatenated with ``feat_current`` and
            ``feat_prop_aligned`` before the backbone residual.

        Returns
        -------
        torch.Tensor
            Updated propagated feature, ``(B, C, H, W)``.
        """
        cond_n1 = flow_warp(
            feat_prop, flow_n1.permute(0, 2, 3, 1), interpolation="nearest"
        )
        cond_n2 = flow_warp(
            feat_n2, flow_n2.permute(0, 2, 3, 1), interpolation="nearest"
        )
        cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
        # Pad cond 192→196 ch so deform_align's input cat (196+2+2=200) is a
        # multiple of 8, satisfying HTP vectorized allocator requirements.
        # Flows stay 2-channel so the offset repeat-scale logic is undisturbed.
        cond_padded = F.pad(cond, (0, 0, 0, 0, 0, 4))
        feat_prop_deform = self.deform_align(
            torch.cat([feat_prop, feat_n2], dim=1), cond_padded, flow_n1, flow_n2
        )
        # Step 0 is detected by an all-zero flow_n1: app.py passes zeros_flow only
        # for step 0, so this matches upstream's `i == 0` skip.  Caveat: a step-1
        # call on perfectly identical frames (flow genuinely zero) also skips
        # alignment — acceptable for this use case.  torch.where keeps the graph
        # export-safe; verify any() lowers on HTP before enabling w8a8.
        has_prior = flow_n1.any().reshape(1, 1, 1, 1).expand_as(feat_prop)
        feat_prop_aligned = torch.where(has_prior, feat_prop_deform, feat_prop)
        feat = torch.cat([feat_current, *prior_feats, feat_prop_aligned], dim=1)
        return feat_prop_aligned + self.backbone(feat)

    def get_input_spec(
        self,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        spec: InputSpec = {
            "feat_prop": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "feat_n2": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "feat_current": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "flow_n1": TensorSpec(shape=(1, 2, height, width), dtype="float32"),
            "flow_n2": TensorSpec(shape=(1, 2, height, width), dtype="float32"),
        }
        for i in range(self.num_prior_passes):
            spec[f"prior_feat_{i}"] = TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            )
        return spec

    def get_output_spec(self) -> OutputSpec:
        return {
            "propagated_feat": TensorSpec(
                shape=(1, MID_CHANNELS, -1, -1), dtype="float32"
            )
        }

    @classmethod
    def from_pretrained(cls, weights: CachedWebModelAsset = WEIGHTS_URL) -> Self:
        raise NotImplementedError(
            "Use BasicVSRPlusPlus.from_pretrained() to load propagation step sub-models."
        )


class BasicVSRPPReconstruction(BasicVSRPP):
    """Reconstruction sub-model.

    Fuses spatial + 4 propagated features and upsamples 4x.  Runs T times.

    Inputs:
      spatial_feat      (1, 64, H, W)
      backward_1_feat   (1, 64, H, W)
      forward_1_feat    (1, 64, H, W)
      backward_2_feat   (1, 64, H, W)
      forward_2_feat    (1, 64, H, W)
      lq_frame          (1, 3,  H, W)  — LR frame for residual skip

    Output:
      hr_frame  (1, 3, 4H, 4W)
    """

    def __init__(
        self,
        reconstruction: nn.Module,
        upsample1: nn.Module,
        upsample2: nn.Module,
        conv_hr: nn.Module,
        conv_last: nn.Module,
    ) -> None:
        super().__init__()
        self.reconstruction = reconstruction
        self.upsample1 = upsample1
        self.upsample2 = upsample2
        self.conv_hr = conv_hr
        self.conv_last = conv_last

    def forward(
        self,
        spatial_feat: torch.Tensor,
        backward_1_feat: torch.Tensor,
        forward_1_feat: torch.Tensor,
        backward_2_feat: torch.Tensor,
        forward_2_feat: torch.Tensor,
        lq_frame: torch.Tensor,
    ) -> torch.Tensor:
        # 5 feature streams concatenated: (1, 5*64=320, H, W)
        hr = torch.cat(
            [
                spatial_feat,
                backward_1_feat,
                forward_1_feat,
                backward_2_feat,
                forward_2_feat,
            ],
            dim=1,
        )
        hr = self.reconstruction(hr)
        up1 = cast(nn.Module, self.upsample1.upsample_conv)
        up2 = cast(nn.Module, self.upsample2.upsample_conv)
        hr = F.leaky_relu(pixel_shuffle(up1(hr), 2), negative_slope=0.1)
        hr = F.leaky_relu(pixel_shuffle(up2(hr), 2), negative_slope=0.1)
        hr = F.leaky_relu(self.conv_hr(hr), negative_slope=0.1)
        # Use nearest upsample instead of bilinear: avoids HTP DMA issues with
        # bilinear resize while preserving the skip-connection shape.
        skip = F.interpolate(lq_frame, scale_factor=4, mode="nearest")
        return self.conv_last(hr) + skip

    def get_input_spec(
        self,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        return {
            "spatial_feat": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "backward_1_feat": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "forward_1_feat": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "backward_2_feat": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "forward_2_feat": TensorSpec(
                shape=(1, MID_CHANNELS, height, width), dtype="float32"
            ),
            "lq_frame": TensorSpec(shape=(1, 3, height, width), dtype="float32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"hr_frame": TensorSpec(shape=(1, 3, -1, -1), dtype="float32")}

    @classmethod
    def from_pretrained(cls, weights: CachedWebModelAsset = WEIGHTS_URL) -> Self:
        return _load_collection(weights).reconstruction


class BasicVSRPlusPlus(WorkbenchModelCollection):
    """BasicVSR++ as a WorkbenchModelCollection of 7 exportable sub-models."""

    def __init__(
        self,
        flownet: BasicVSRPPFlowNet,
        feat_extract: BasicVSRPPFeatExtract,
        prop_backward_1: BasicVSRPPPropStep,
        prop_forward_1: BasicVSRPPPropStep,
        prop_backward_2: BasicVSRPPPropStep,
        prop_forward_2: BasicVSRPPPropStep,
        reconstruction: BasicVSRPPReconstruction,
    ) -> None:
        super().__init__(
            {
                "flownet": flownet,
                "feat_extract": feat_extract,
                "prop_backward_1": prop_backward_1,
                "prop_forward_1": prop_forward_1,
                "prop_backward_2": prop_backward_2,
                "prop_forward_2": prop_forward_2,
                "reconstruction": reconstruction,
            }
        )
        self.flownet = flownet
        self.feat_extract = feat_extract
        self.prop_backward_1 = prop_backward_1
        self.prop_forward_1 = prop_forward_1
        self.prop_backward_2 = prop_backward_2
        self.prop_forward_2 = prop_forward_2
        self.reconstruction = reconstruction

    @classmethod
    def from_pretrained(cls, weights: CachedWebModelAsset = WEIGHTS_URL) -> Self:
        return _load_collection(weights, collection_cls=cls)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [REDS4Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return REDS4Dataset

    def get_evaluator(self) -> BaseEvaluator:
        return SuperResolutionOutputEvaluator()


def _patch_deform_align_conv_offset(conv0: nn.Conv2d) -> None:
    """Insert 4 zero weight columns at index 192 so in_channels becomes a multiple of 8."""
    extra = torch.zeros(
        conv0.out_channels,
        4,
        *conv0.kernel_size,
        device=conv0.weight.device,
        dtype=conv0.weight.dtype,
    )
    conv0.weight = nn.Parameter(
        torch.cat([conv0.weight[:, :192], extra, conv0.weight[:, 192:]], dim=1)
    )
    conv0.in_channels = conv0.in_channels + 4


def _load_collection(
    weights: CachedWebModelAsset = WEIGHTS_URL,
    collection_cls: type = BasicVSRPlusPlus,
) -> Any:
    """Load upstream BasicVSRPlusPlus, extract weights, build sub-models."""
    # Install mmcv v1 shims before any mmedit submodule is imported.
    # This is deferred (not top-level) because it depends on external_repos
    # being cloned and torchvision being available.
    from . import mmcv_shim

    mmcv_shim.install()

    # external_repos imports are deferred: the repos aren't cloned until
    # `qai-hub-models install` runs, so they can't appear at module top-level.
    from .external_repos.BasicVSR_PlusPlus.mmedit.models.backbones.sr_backbones.basicvsr_pp import (
        BasicVSRPlusPlus as _UpstreamModel,
    )

    # Build architecture with SPyNet weights loaded inline
    upstream = _UpstreamModel(
        mid_channels=MID_CHANNELS,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained=str(SPYNET_WEIGHTS_URL.fetch()),
        cpu_cache_length=10000,  # larger than any real sequence; disables CPU offload during export
    )

    # Load main checkpoint
    ckpt = load_torch(weights)
    state_dict = ckpt.get("state_dict", ckpt)
    # MMEditing wraps backbone under 'generator.' prefix when saved from restorer
    if any(k.startswith("generator.") for k in state_dict):
        state_dict = {
            k[len("generator.") :]: v
            for k, v in state_dict.items()
            if k.startswith("generator.")
        }
    upstream.load_state_dict(state_dict, strict=True)
    upstream.eval()

    # Patch each deform_align's conv_offset[0] to accept 200 instead of 196
    # input channels.  HTP requires channel dims to be multiples of 8;
    # cat([extra_feat(192), flow_1(2), flow_2(2)]) = 196 fails (196 % 8 = 4).
    # We pad extra_feat 192→196 in BasicVSRPPPropStep.forward; the cat becomes
    # [cond(192), zeros(4), flow_1(2), flow_2(2)] = 200 = 25x8.
    # The zero weights must be inserted at index 192 (before the flow weights)
    # so that the flow channels still hit the original flow weight columns.
    for pass_name in PROP_PASSES:
        conv_offset = cast(nn.Sequential, upstream.deform_align[pass_name].conv_offset)
        _patch_deform_align_conv_offset(cast(nn.Conv2d, conv_offset[0]))

    flownet = BasicVSRPPFlowNet(upstream.spynet)
    feat_extract = BasicVSRPPFeatExtract(upstream.feat_extract)

    prop_b1 = BasicVSRPPPropStep(
        upstream.deform_align["backward_1"],
        upstream.backbone["backward_1"],
        0,
    )
    prop_f1 = BasicVSRPPPropStep(
        upstream.deform_align["forward_1"],
        upstream.backbone["forward_1"],
        1,
    )
    prop_b2 = BasicVSRPPPropStep(
        upstream.deform_align["backward_2"],
        upstream.backbone["backward_2"],
        2,
    )
    prop_f2 = BasicVSRPPPropStep(
        upstream.deform_align["forward_2"],
        upstream.backbone["forward_2"],
        3,
    )

    recon = BasicVSRPPReconstruction(
        upstream.reconstruction,
        upstream.upsample1,
        upstream.upsample2,
        upstream.conv_hr,
        upstream.conv_last,
    )
    return collection_cls(
        flownet, feat_extract, prop_b1, prop_f1, prop_b2, prop_f2, recon
    )
