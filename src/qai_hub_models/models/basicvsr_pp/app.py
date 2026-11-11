# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Generator, Mapping
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from qai_hub.public_rest_api import DatasetEntries
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from qai_hub_models.datasets.kinetics400.video_utils import (
    normalize as normalize_video,
)
from qai_hub_models.datasets.kinetics400.video_utils import (
    read_video_per_second,
)
from qai_hub_models.models.basicvsr_pp.model import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MID_CHANNELS,
    PROP_PASSES,
    BasicVSRPlusPlus,
)
from qai_hub_models.models.basicvsr_pp.model_adaptations import flow_warp
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_dataset import DatasetSplit, instantiate_dataset
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class BasicVSRPlusPlusApp(CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol):
    """End-to-end BasicVSR++ app that orchestrates the 7 sub-models.

    The collection model components are called in the order required by the
    bidirectional propagation algorithm:

    1. feat_extract       — T calls, one per LR frame
    2. flownet            — 2*(T-1) calls (backward + forward adjacent pairs)
    3. prop_backward_1    — T calls (recurrent, backward direction)
    4. prop_forward_1     — T calls (recurrent, forward direction)
    5. prop_backward_2    — T calls (recurrent, backward direction)
    6. prop_forward_2     — T calls (recurrent, forward direction)
    7. reconstruction     — T calls, one per frame
    """

    def __init__(
        self,
        flownet: ExecutableModelProtocol | AsyncOnDeviceModel,
        feat_extract: ExecutableModelProtocol | AsyncOnDeviceModel,
        prop_backward_1: ExecutableModelProtocol | AsyncOnDeviceModel,
        prop_forward_1: ExecutableModelProtocol | AsyncOnDeviceModel,
        prop_backward_2: ExecutableModelProtocol | AsyncOnDeviceModel,
        prop_forward_2: ExecutableModelProtocol | AsyncOnDeviceModel,
        reconstruction: ExecutableModelProtocol | AsyncOnDeviceModel,
    ) -> None:
        self.flownet = flownet
        self.feat_extract = feat_extract
        self.prop_backward_1 = prop_backward_1
        self.prop_forward_1 = prop_forward_1
        self.prop_backward_2 = prop_backward_2
        self.prop_forward_2 = prop_forward_2
        self.reconstruction = reconstruction

    def predict(self, *args: Any, **kwargs: Any) -> list[Image.Image]:
        return self.upscale_video(*args, **kwargs)

    def upscale_video(
        self,
        lq_frames: torch.Tensor,
    ) -> list[Image.Image]:
        """Upscale a sequence of LR frames 4x.

        Parameters
        ----------
        lq_frames
            ``(T, 3, H, W)`` float32 tensor, values in ``[0, 1]``, RGB.

        Returns
        -------
        list[Image.Image]
            T upscaled PIL images at resolution (4H, 4W).
        """
        hr_tensors = self._run_inference(lq_frames)
        return [
            Image.fromarray(
                (hr.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
                    np.uint8
                )
            )
            for hr in hr_tensors
        ]

    @staticmethod
    def _wait(result: torch.Tensor | AsyncOnDeviceResult) -> torch.Tensor:
        """Resolve a submitted result to a tensor."""
        if isinstance(result, AsyncOnDeviceResult):
            waited = result.wait()
            return waited[0] if isinstance(waited, tuple) else waited
        return result

    @classmethod
    def _call(
        cls, model: ExecutableModelProtocol | AsyncOnDeviceModel, *args: torch.Tensor
    ) -> torch.Tensor:
        """Submit and immediately wait — use only where the result is needed before the next call."""
        return cls._wait(model(*args))

    def _run_inference(self, lq_frames: torch.Tensor) -> list[torch.Tensor]:
        """Core inference pipeline; returns list of (1,3,4H,4W) tensors."""
        t = lq_frames.shape[0]
        frames = [lq_frames[i].unsqueeze(0) for i in range(t)]

        # --- Phase 1: feature extraction (all independent — submit all, wait all) ---
        feat_handles = [self.feat_extract(f) for f in frames]
        spatial_feats = [self._wait(h) for h in feat_handles]

        # --- Phase 2: optical flow (all independent — submit all, wait all) ---
        flow_handles_bwd: list[torch.Tensor | AsyncOnDeviceResult] = []
        flow_handles_fwd: list[torch.Tensor | AsyncOnDeviceResult] = []
        for i in range(t - 1):
            flow_handles_bwd.append(self.flownet(frames[i], frames[i + 1]))
            flow_handles_fwd.append(self.flownet(frames[i + 1], frames[i]))
        flows_bwd = [self._wait(h) for h in flow_handles_bwd]
        flows_fwd = [self._wait(h) for h in flow_handles_fwd]

        zeros_feat = torch.zeros_like(spatial_feats[0])
        zeros_flow = zeros_feat.new_zeros(
            1, 2, spatial_feats[0].shape[2], spatial_feats[0].shape[3]
        )

        # --- Phase 3: 4 propagation passes, each T recurrent steps ---
        prop_models = [
            self.prop_backward_1,
            self.prop_forward_1,
            self.prop_backward_2,
            self.prop_forward_2,
        ]
        is_backward = [True, False, True, False]
        prop_feats: dict[str, list[torch.Tensor]] = {n: [] for n in PROP_PASSES}

        for pass_idx, (pass_name, model, backward) in enumerate(
            zip(PROP_PASSES, prop_models, is_backward, strict=False)
        ):
            if backward:
                frame_indices = list(range(t - 1, -1, -1))
                flows = flows_bwd
            else:
                frame_indices = list(range(t))
                flows = flows_fwd

            prior_pass_names = PROP_PASSES[:pass_idx]

            feat_prop = zeros_feat.clone()
            pass_results: list[torch.Tensor] = [zeros_feat.clone() for _ in range(t)]

            for step, idx in enumerate(frame_indices):
                if step == 0:
                    flow_n1 = zeros_flow.clone()
                    flow_n2 = zeros_flow.clone()
                elif step == 1:
                    flow_n1 = flows[idx] if backward else flows[idx - 1]
                    flow_n2 = zeros_flow.clone()
                else:
                    flow_n1 = flows[idx] if backward else flows[idx - 1]
                    prev_idx = frame_indices[step - 1]
                    flow_n2_prev = flows[prev_idx] if backward else flows[prev_idx - 1]
                    # Upstream composes flows: warp the previous second-neighbor flow
                    # by flow_n1, then add flow_n1 to get the true 2-step composite.
                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2_prev,
                        flow_n1.permute(0, 2, 3, 1),
                        interpolation="nearest",
                    )

                feat_n2 = (
                    pass_results[frame_indices[step - 2]]
                    if step >= 2
                    else zeros_feat.clone()
                )

                prior_feats_step = [prop_feats[n][idx] for n in prior_pass_names]

                feat_prop = self._call(
                    model,
                    feat_prop,
                    feat_n2,
                    spatial_feats[idx],
                    flow_n1,
                    flow_n2,
                    *prior_feats_step,
                )
                pass_results[idx] = feat_prop

            prop_feats[pass_name] = pass_results

        # --- Phase 4: per-frame reconstruction (all independent — submit all, wait all) ---
        recon_handles = [
            self.reconstruction(
                spatial_feats[i],
                prop_feats["backward_1"][i],
                prop_feats["forward_1"][i],
                prop_feats["backward_2"][i],
                prop_feats["forward_2"][i],
                frames[i],
            )
            for i in range(t)
        ]
        return [self._wait(h) for h in recon_handles]

    @classmethod
    def from_model(cls, model: BasicVSRPlusPlus) -> BasicVSRPlusPlusApp:
        return cls(
            flownet=model.flownet,
            feat_extract=model.feat_extract,
            prop_backward_1=model.prop_backward_1,
            prop_forward_1=model.prop_forward_1,
            prop_backward_2=model.prop_backward_2,
            prop_forward_2=model.prop_forward_2,
            reconstruction=model.reconstruction,
        )

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> BasicVSRPlusPlusApp:
        """Construct app from ordered component list [flownet, feat_extract, prop_b1, prop_f1, prop_b2, prop_f2, recon]."""
        assert len(models) == 7, f"Expected 7 components, got {len(models)}"
        return cls(
            flownet=models[0],
            feat_extract=models[1],
            prop_backward_1=models[2],
            prop_forward_1=models[3],
            prop_backward_2=models[4],
            prop_forward_2=models[5],
            reconstruction=models[6],
        )

    @staticmethod
    def _run_two_frame_recurrence(
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        flow_bwd: torch.Tensor,
        flow_fwd: torch.Tensor,
        prop_models: Mapping[str, ExecutableModelProtocol | AsyncOnDeviceModel],
        zeros_feat: torch.Tensor,
        zeros_flow: torch.Tensor,
    ) -> tuple[dict[str, tuple[torch.Tensor, ...]], list[torch.Tensor]]:
        """Run T=2 propagation recurrence; return step-1 inputs and frame-1 outputs.

        Mirrors run_model_for_eval so calibration data comes from the same
        input distribution as real inference.
        """
        is_backward = [True, False, True, False]
        prior_feat_at_0: list[torch.Tensor] = []
        prior_feat_at_1: list[torch.Tensor] = []
        step1_inputs: dict[str, tuple[torch.Tensor, ...]] = {}

        for pass_idx, (pass_name, backward) in enumerate(
            zip(PROP_PASSES, is_backward, strict=False)
        ):
            model = prop_models[pass_name]
            prior_feats_0 = prior_feat_at_0[:pass_idx]
            prior_feats_1 = prior_feat_at_1[:pass_idx]

            if backward:
                feat_prop_1 = cast(
                    torch.Tensor,
                    model(
                        zeros_feat,
                        zeros_feat,
                        feat1,
                        zeros_flow,
                        zeros_flow,
                        *prior_feats_1,
                    ),
                )
                inputs_step1: tuple[torch.Tensor, ...] = (
                    feat_prop_1,
                    zeros_feat,
                    feat0,
                    flow_bwd,
                    zeros_flow,
                    *prior_feats_0,
                )
                feat_prop_0 = cast(torch.Tensor, model(*inputs_step1))
            else:
                feat_prop_0 = cast(
                    torch.Tensor,
                    model(
                        zeros_feat,
                        zeros_feat,
                        feat0,
                        zeros_flow,
                        zeros_flow,
                        *prior_feats_0,
                    ),
                )
                inputs_step1 = (
                    feat_prop_0,
                    zeros_feat,
                    feat1,
                    flow_fwd,
                    zeros_flow,
                    *prior_feats_1,
                )
                feat_prop_1 = cast(torch.Tensor, model(*inputs_step1))

            step1_inputs[f"prop_{pass_name}"] = inputs_step1
            prior_feat_at_0.append(feat_prop_0)
            prior_feat_at_1.append(feat_prop_1)

        return step1_inputs, prior_feat_at_1

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: BasicVSRPlusPlus,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        component = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else component.get_input_spec()
        )
        dataset = instantiate_dataset(
            calibration_dataset_cls, DatasetSplit.TRAIN, input_spec=input_spec
        )
        n = num_samples or dataset.default_samples_per_job()
        torch_dataset = sample_dataset(dataset, n)
        dataloader = DataLoader(torch_dataset, batch_size=1)

        flownet = collection_model.flownet
        feat_extract = collection_model.feat_extract
        prop_models = {
            "backward_1": collection_model.prop_backward_1,
            "forward_1": collection_model.prop_forward_1,
            "backward_2": collection_model.prop_backward_2,
            "forward_2": collection_model.prop_forward_2,
        }

        inputs: list[list[torch.Tensor]] = [[] for _ in range(len(input_spec))]

        with torch.no_grad():
            for lr_clip, _ in dataloader:
                lr_clip = lr_clip.squeeze(0)  # (2, 3, H, W)
                lr0 = lr_clip[0].unsqueeze(0)
                lr1 = lr_clip[1].unsqueeze(0)
                h, w = lr0.shape[2], lr0.shape[3]
                zeros_feat = lr0.new_zeros(1, MID_CHANNELS, h, w)
                zeros_flow = lr0.new_zeros(1, 2, h, w)

                if component_name == "flownet":
                    sample: tuple[torch.Tensor, ...] = (lr0, lr1)
                elif component_name == "feat_extract":
                    sample = (lr0,)
                else:
                    feat0 = feat_extract(lr0)
                    feat1 = feat_extract(lr1)
                    flow_bwd = flownet(lr0, lr1)
                    flow_fwd = flownet(lr1, lr0)
                    # Run the same T=2 recurrence as run_model_for_eval; capture
                    # step-1 inputs per pass so ranges match real inference.
                    step1_inputs, prior_feat_at_1 = cls._run_two_frame_recurrence(
                        feat0,
                        feat1,
                        flow_bwd,
                        flow_fwd,
                        prop_models,
                        zeros_feat,
                        zeros_flow,
                    )
                    if component_name == "reconstruction":
                        sample = (feat1, *prior_feat_at_1, lr1)
                    else:
                        sample = step1_inputs[component_name]

                assert len(sample) == len(input_spec), (
                    f"{component_name}: sample has {len(sample)} tensors "
                    f"but input_spec has {len(input_spec)} keys"
                )
                for i, tensor in enumerate(sample):
                    inputs[i].append(tensor)

        return make_hub_dataset_entries(
            tuple(inputs),  # type: ignore[arg-type]
            list(input_spec.keys()),
        )

    def run_model_for_eval(
        self,
        model_input: Generator | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        """Run the full SR pipeline for one sample and return the HR tensor.

        The dataset returns (lr_clip, hr_frame) where lr_clip is (2, 3, H, W)
        and hr_frame is (3, 4H, 4W) for the second frame (index 1).  We run
        the T=2 pipeline and return the reconstruction for frame index 1, which
        has one backward temporal neighbor for optical-flow context.

        Local path: calls each sub-model as a plain torch module, returns
        (1, 3, 4H, 4W) HR tensor via StopIteration.value.

        On-device path: drives all 7 on-device components, yielding each
        AsyncOnDeviceResult so the harness can track job progress.
        """
        if isinstance(model_input, tuple):
            lr_clip = model_input[0].squeeze(0)  # (1,2,3,H,W) → (2,3,H,W)
            hr_tensors = self._run_inference(lr_clip)
            result = hr_tensors[1]  # frame index 1 — has backward temporal context
            yield (result,)
            return result  # type: ignore[return-value]

        # On-device eval path — T=2.
        lr_batch = next(model_input)
        lr_clip = (
            lr_batch[0] if isinstance(lr_batch, tuple) else cast(torch.Tensor, lr_batch)
        )
        # DataLoader wraps with a batch dim: (1, 2, 3, H, W) → squeeze to (2, 3, H, W).
        lr_clip = lr_clip.squeeze(0)
        # lr_clip is (2, 3, H, W); split into individual frames.
        lr0 = lr_clip[0].unsqueeze(0)  # (1, 3, H, W) — backward neighbor
        lr1 = lr_clip[1].unsqueeze(0)  # (1, 3, H, W) — target frame
        h, w = lr0.shape[2], lr0.shape[3]

        zeros_feat = lr0.new_zeros(1, MID_CHANNELS, h, w)
        zeros_flow = lr0.new_zeros(1, 2, h, w)

        # Phase 1: feature extraction for both frames (submit both, wait both).
        feat0_async = cast(AsyncOnDeviceResult, self.feat_extract(lr0))
        feat1_async = cast(AsyncOnDeviceResult, self.feat_extract(lr1))
        yield feat0_async
        yield feat1_async
        spatial_feat0 = cast(torch.Tensor, feat0_async.wait())
        spatial_feat1 = cast(torch.Tensor, feat1_async.wait())

        # Phase 2: optical flow for the one adjacent pair.
        # Backward: flownet(lr0, lr1) warps lr1 toward lr0.
        # Forward:  flownet(lr1, lr0) warps lr0 toward lr1.
        flow_bwd_async = cast(AsyncOnDeviceResult, self.flownet(lr0, lr1))
        flow_fwd_async = cast(AsyncOnDeviceResult, self.flownet(lr1, lr0))
        yield flow_bwd_async
        yield flow_fwd_async
        flow_bwd = cast(
            torch.Tensor, flow_bwd_async.wait()
        )  # flow at frame index 1 (bwd)
        flow_fwd = cast(
            torch.Tensor, flow_fwd_async.wait()
        )  # flow at frame index 0 (fwd)

        # Phase 3: 4 propagation passes, T=2 each.
        # Backward pass iterates [1, 0]; forward pass iterates [0, 1].
        # We need prop output at frame index 1 for reconstruction.
        prop_models = [
            self.prop_backward_1,
            self.prop_forward_1,
            self.prop_backward_2,
            self.prop_forward_2,
        ]
        is_backward = [True, False, True, False]

        # Track prop outputs per frame so each step gets the right per-frame priors.
        prior_feat_at_0: list[torch.Tensor] = []
        prior_feat_at_1: list[torch.Tensor] = []

        for pass_idx, (prop_model, backward) in enumerate(
            zip(prop_models, is_backward, strict=False)
        ):
            prior_feats_0 = prior_feat_at_0[:pass_idx]
            prior_feats_1 = prior_feat_at_1[:pass_idx]

            if backward:
                # Step 0: frame index 1 (no prior flow/feat)
                prop_async_step0 = cast(
                    AsyncOnDeviceResult,
                    prop_model(
                        zeros_feat,
                        zeros_feat,
                        spatial_feat1,
                        zeros_flow,
                        zeros_flow,
                        *prior_feats_1,
                    ),
                )
                yield prop_async_step0
                feat_prop_1 = cast(torch.Tensor, prop_async_step0.wait())

                # Step 1: frame index 0 (uses flow_bwd, feat_prop from step 0)
                prop_async_step1 = cast(
                    AsyncOnDeviceResult,
                    prop_model(
                        feat_prop_1,
                        zeros_feat,
                        spatial_feat0,
                        flow_bwd,
                        zeros_flow,
                        *prior_feats_0,
                    ),
                )
                yield prop_async_step1
                feat_prop_0 = cast(torch.Tensor, prop_async_step1.wait())
                prior_feat_at_0.append(feat_prop_0)
                prior_feat_at_1.append(feat_prop_1)
            else:
                # Step 0: frame index 0 (no prior flow/feat)
                prop_async_step0 = cast(
                    AsyncOnDeviceResult,
                    prop_model(
                        zeros_feat,
                        zeros_feat,
                        spatial_feat0,
                        zeros_flow,
                        zeros_flow,
                        *prior_feats_0,
                    ),
                )
                yield prop_async_step0
                feat_prop_0 = cast(torch.Tensor, prop_async_step0.wait())

                # Step 1: frame index 1 (uses flow_fwd, feat_prop from step 0)
                prop_async_step1 = cast(
                    AsyncOnDeviceResult,
                    prop_model(
                        feat_prop_0,
                        zeros_feat,
                        spatial_feat1,
                        flow_fwd,
                        zeros_flow,
                        *prior_feats_1,
                    ),
                )
                yield prop_async_step1
                feat_prop_1 = cast(torch.Tensor, prop_async_step1.wait())
                prior_feat_at_0.append(feat_prop_0)
                prior_feat_at_1.append(feat_prop_1)

        b1_feat, f1_feat, b2_feat, f2_feat = prior_feat_at_1

        # Phase 4: reconstruct frame index 1 only.
        recon_async = cast(
            AsyncOnDeviceResult,
            self.reconstruction(spatial_feat1, b1_feat, f1_feat, b2_feat, f2_feat, lr1),
        )
        yield recon_async
        return recon_async


def load_frames_from_video(
    path: str, num_frames: int
) -> tuple[torch.Tensor, list[Image.Image]]:
    """Load consecutive frames from a video file for BasicVSR++ inference."""
    raw = read_video_per_second(path)
    raw = raw[:num_frames]
    video = normalize_video(raw)  # [C, T, H, W]
    video = TF.resize(video, min(DEFAULT_HEIGHT, DEFAULT_WIDTH))
    video = TF.center_crop(video, [DEFAULT_HEIGHT, DEFAULT_WIDTH])
    lq_frames = video.permute(1, 0, 2, 3)  # [T, C, H, W]
    pil_frames = [
        Image.fromarray(
            (lq_frames[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        )
        for i in range(lq_frames.shape[0])
    ]
    return lq_frames, pil_frames


def load_frames_from_dir(
    frames_dir: str, num_frames: int, start_idx: int = 0
) -> tuple[torch.Tensor, list[Image.Image]]:
    """Load PNG/JPG frames from a directory, sorted by filename."""
    all_paths = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    paths = all_paths[start_idx : start_idx + num_frames]
    if not paths:
        raise ValueError(f"No PNG/JPG frames found in {frames_dir!r}")
    pil_frames = [Image.open(p).convert("RGB") for p in paths]
    return pil_list_to_tensor(pil_frames), pil_frames


def pil_list_to_tensor(pil_frames: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images to a (T, 3, H, W) float32 tensor in [0, 1]."""
    arrays = [np.array(img, dtype="float32") / 255.0 for img in pil_frames]
    tensors = [torch.from_numpy(a).permute(2, 0, 1) for a in arrays]
    return torch.stack(tensors)
