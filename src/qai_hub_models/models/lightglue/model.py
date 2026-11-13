# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import torch
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.lightglue.dataset import LightGluePairDataset
from qai_hub_models.models.lightglue.evaluator import LightGlueEvaluator
from qai_hub_models.models.lightglue.external_repos.cvg_lightglue.lightglue.lightglue import (
    LearnableFourierPositionalEncoding,
    MatchAssignment,
    TransformerLayer,
    filter_matches,
    normalize_keypoints,
    sigmoid_log_double_softmax,
)
from qai_hub_models.models.lightglue.postprocess import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_KEYPOINTS,
)
from qai_hub_models.models.superpoint.model import SuperPoint
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Subset of upstream LightGlue.default_conf relevant to this static recipe:
# https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/lightglue.py#L332
_DEFAULT_CONF = {
    "descriptor_dim": 256,
    "n_layers": 9,
    "num_heads": 4,
    "filter_threshold": 0.1,
}

DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "superpoint_lightglue_v0-1_arxiv.pth"
)

# Bundled sample pair, sourced from LightGlue's own assets/ folder (covered
# by the same repo-root Apache-2.0 license as the rest of cvg/LightGlue):
# https://github.com/cvg/LightGlue/tree/eb42fee2d71449efb0aa5c10549752b5d75384d8/assets
IMAGE0_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image0.jpg"
)
IMAGE1_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)

# Precomputed SuperPoint keypoints/descriptors for the bundled sample pair
# above, at MAX_KEYPOINTS -- avoids re-running a second model's inference
# just to produce realistic sample inputs for this one.
SAMPLE_FEATURES_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_features.npz"
)


class LightGlue(BaseModel):
    """LightGlue sparse feature matcher.

    Consumes keypoints and descriptors already produced by a feature
    extractor (SuperPoint) for a pair of images and outputs mutual-nearest-
    neighbour point correspondences between them.

    Unlike the upstream model, depth_confidence/width_confidence early-exit
    and pruning are disabled in this static recipe -- every pair runs all
    n_layers transformer layers, regardless of difficulty.

    Input: keypoints/descriptors/num_keypoints for each of two images, all
        padded to MAX_KEYPOINTS=500 (matching the SuperPoint extractor
        output). Padded slots are excluded from attention via a mask derived
        from num_keypoints.
    Output (all shape (B, MAX_KEYPOINTS)):
        matches0, matches1 — index into the other image's keypoints, or -1
        for both unmatched and padded slots.
        mscores0, mscores1 — match confidence in [0, 1], forced to 0 for
        padded slots.
    """

    def __init__(self, **conf: object) -> None:
        super().__init__()
        cfg = {**_DEFAULT_CONF, **conf}
        self.conf = SimpleNamespace(**cfg)
        # Nothing in this recipe (app.py, extract_padded_features) actually
        # varies these, so they aren't exposed as constructor parameters.
        self.height = DEFAULT_HEIGHT
        self.width = DEFAULT_WIDTH

        d, h, n = self.conf.descriptor_dim, self.conf.num_heads, self.conf.n_layers
        # SuperPoint descriptors are already 256-d == descriptor_dim, so the
        # upstream input_proj collapses to identity (matches checkpoint shape).
        self.input_proj = nn.Identity()
        head_dim = d // h
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)
        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, True) for _ in range(n)]
        )
        # Only the final layer's assignment head is ever used at inference
        # (see forward()), so earlier layers aren't instantiated at all.
        self.log_assignment = MatchAssignment(d)

    @classmethod
    def from_pretrained(
        cls,
        weights: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
    ) -> Self:
        """Load LightGlue from the official superpoint_lightglue checkpoint."""
        if isinstance(weights, CachedWebModelAsset):
            weights_path = str(weights.fetch())
        else:
            weights_path = weights

        model = cls()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # The official checkpoint predates the `transformers.{i}.*` submodule
        # nesting used by the current upstream code (and by this port) — same
        # old->new key rename upstream's own loader applies.
        last = model.conf.n_layers - 1
        for i in range(model.conf.n_layers):
            state_dict = {
                k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn"): v
                for k, v in state_dict.items()
            }
            state_dict = {
                k.replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn"): v
                for k, v in state_dict.items()
            }
        # Only the final layer's log_assignment is instantiated (see
        # __init__), so drop the other layers' weights and un-nest the final
        # one; token_confidence (width-pruning) is never instantiated either.
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("token_confidence."):
                continue
            if k.startswith(f"log_assignment.{last}."):
                k = "log_assignment." + k[len(f"log_assignment.{last}.") :]
            elif k.startswith("log_assignment."):
                continue
            remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # `log_assignment.r` is a stale scale parameter from an older
        # MatchAssignment implementation no longer used anywhere.
        if missing or unexpected != ["log_assignment.r"]:
            raise RuntimeError(
                f"Unexpected checkpoint keys: missing={missing}, unexpected={unexpected}"
            )
        return model

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        num_keypoints0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        num_keypoints1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match keypoints/descriptors between two images.

        Parameters
        ----------
        keypoints0
            Pixel-space (x, y) coordinates for image0, shape (B, MAX_KEYPOINTS, 2).
        descriptors0
            L2-normalised feature vectors for image0, shape (B, MAX_KEYPOINTS, 256).
        num_keypoints0
            Number of valid (unpadded) keypoints in image0, shape (B,).
        keypoints1
            Pixel-space (x, y) coordinates for image1, shape (B, MAX_KEYPOINTS, 2).
        descriptors1
            L2-normalised feature vectors for image1, shape (B, MAX_KEYPOINTS, 256).
        num_keypoints1
            Number of valid (unpadded) keypoints in image1, shape (B,).

        Returns
        -------
        matches0 : torch.Tensor
            Shape (B, MAX_KEYPOINTS), int32. Index into image1's keypoints
            for a mutual match, else -1.
        matches1 : torch.Tensor
            Shape (B, MAX_KEYPOINTS), int32. Index into image0's keypoints
            for a mutual match, else -1.
        mscores0 : torch.Tensor
            Shape (B, MAX_KEYPOINTS), float32. Match confidence in [0, 1].
        mscores1 : torch.Tensor
            Shape (B, MAX_KEYPOINTS), float32. Match confidence in [0, 1].
        """
        b = keypoints0.shape[0]
        m = keypoints0.shape[1]
        n = keypoints1.shape[1]
        device = keypoints0.device

        size = torch.tensor(
            [self.width, self.height], dtype=keypoints0.dtype, device=device
        )
        kpts0 = normalize_keypoints(keypoints0, size).clone()
        kpts1 = normalize_keypoints(keypoints1, size).clone()

        desc0 = self.input_proj(descriptors0.contiguous())
        desc1 = self.input_proj(descriptors1.contiguous())

        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # Explicit head-broadcast dim so mask0 & mask0.transpose(-1,-2) and
        # mask0 & mask1.transpose(-1,-2) (inside TransformerLayer.masked_forward)
        # broadcast correctly against 4D (B, num_heads, seq, seq) attention
        # tensors for any batch size / head count.
        ar0 = torch.arange(m, device=device)[None, None, :, None]
        ar1 = torch.arange(n, device=device)[None, None, :, None]
        mask0 = ar0 < num_keypoints0.view(b, 1, 1, 1)
        mask1 = ar1 < num_keypoints1.view(b, 1, 1, 1)
        # A fully-masked image (num_keypoints==0) makes a mask row all-False,
        # which would NaN its softmax; relies on CrossBlock's nan_to_num call
        # (external_repos/cvg_lightglue/lightglue/lightglue.py) to stay finite.

        for i in range(self.conf.n_layers):
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1
            )

        # log_assignment's own forward() runs its softmax over the full
        # padded (m, n) similarity matrix, so padded rows/columns must be
        # masked out here -- otherwise a padded descriptor can win a real
        # keypoint's argmax after 9 transformer layers of mixing.
        valid0 = ar0[:, 0, :, 0] < num_keypoints0.view(b, 1)
        valid1 = ar1[:, 0, :, 0] < num_keypoints1.view(b, 1)
        mdesc0, mdesc1 = (
            self.log_assignment.final_proj(desc0),
            self.log_assignment.final_proj(desc1),
        )
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        sim = sim.masked_fill(~valid0.unsqueeze(-1), -1e4)
        sim = sim.masked_fill(~valid1.unsqueeze(1), -1e4)
        z0 = self.log_assignment.matchability(desc0)
        z1 = self.log_assignment.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        # filter_matches's argmax can't land on a masked-out column/row, but
        # a fully-padded image (num_keypoints==0) still needs its own output
        # forced to the padded sentinel, since there's no valid target to
        # argmax into in the first place.
        m0 = torch.where(valid0, m0, -1)
        m1 = torch.where(valid1, m1, -1)
        mscores0 = torch.where(valid0, mscores0, 0)
        mscores1 = torch.where(valid1, mscores1, 0)

        return (
            m0.to(torch.int32),
            m1.to(torch.int32),
            mscores0.to(torch.float32),
            mscores1.to(torch.float32),
        )

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        max_keypoints = MAX_KEYPOINTS
        if input_spec is not None:
            _, max_keypoints, _ = input_spec["keypoints0"][0]
        if max_keypoints > MAX_KEYPOINTS:
            raise ValueError(
                f"Sample inputs are only available up to MAX_KEYPOINTS="
                f"{MAX_KEYPOINTS}; requested {max_keypoints}."
            )
        # The cached pair is fully saturated (num_keypoints == MAX_KEYPOINTS
        # on both images), so any prefix of the first max_keypoints entries
        # is still all real (non-padded) keypoints.
        data = np.load(str(SAMPLE_FEATURES_ADDRESS.fetch()))
        return {
            "keypoints0": [data["keypoints0"][:, :max_keypoints]],
            "descriptors0": [data["descriptors0"][:, :max_keypoints]],
            "num_keypoints0": [np.minimum(data["num_keypoints0"], max_keypoints)],
            "keypoints1": [data["keypoints1"][:, :max_keypoints]],
            "descriptors1": [data["descriptors1"][:, :max_keypoints]],
            "num_keypoints1": [np.minimum(data["num_keypoints1"], max_keypoints)],
        }

    def get_evaluator(self) -> BaseEvaluator:
        return LightGlueEvaluator(image_height=self.height, image_width=self.width)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [LightGluePairDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return LightGluePairDataset

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
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"
        return compile_options

    def get_input_spec(self) -> InputSpec:
        k = MAX_KEYPOINTS
        d = self.conf.descriptor_dim
        return {
            "keypoints0": TensorSpec(shape=(1, k, 2), dtype="float32"),
            "descriptors0": TensorSpec(shape=(1, k, d), dtype="float32"),
            "num_keypoints0": TensorSpec(shape=(1,), dtype="int32"),
            "keypoints1": TensorSpec(shape=(1, k, 2), dtype="float32"),
            "descriptors1": TensorSpec(shape=(1, k, d), dtype="float32"),
            "num_keypoints1": TensorSpec(shape=(1,), dtype="int32"),
        }

    def get_output_spec(self) -> OutputSpec:
        k = MAX_KEYPOINTS
        return {
            "matches0": TensorSpec(shape=(1, k), dtype="int32"),
            "matches1": TensorSpec(shape=(1, k), dtype="int32"),
            "mscores0": TensorSpec(shape=(1, k), dtype="float32"),
            "mscores1": TensorSpec(shape=(1, k), dtype="float32"),
        }


class SuperPointLightGlue(WorkbenchModelCollection):
    """Collection of SuperPoint (feature extraction) and LightGlue (matching).

    Each component compiles/profiles independently on device; keypoint
    extraction (NMS/top-k/border-removal) and zero-padding between them stay
    host-side, orchestrated by LightGlueApp.
    """

    def __init__(self, superpoint: SuperPoint, lightglue: LightGlue) -> None:
        super().__init__({"superpoint": superpoint, "lightglue": lightglue})
        self.superpoint = superpoint
        self.lightglue = lightglue

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(SuperPoint.from_pretrained(), LightGlue.from_pretrained())

    def get_evaluator(self) -> BaseEvaluator:
        return self.lightglue.get_evaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return LightGlue.get_eval_dataset_classes()

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return self.lightglue.get_calibration_dataset_cls()
