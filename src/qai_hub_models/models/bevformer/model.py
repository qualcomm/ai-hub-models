# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.extern.mmcv import patch_mmcv_no_extensions
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    TensorSpec,
    make_torch_inputs,
)
from qai_hub_models.utils.onnx.helpers import safe_torch_onnx_export

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "bevformer_tiny_deformable_optimized_exp_86_epoch_24.pth"

MODEL_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
)

# BEV grid + embedding dims for bevformer_tiny. Defined here (the spec owner) and
# imported by app.py; the flattened BEV sequence length is BEV_H * BEV_W.
BEV_H = BEV_W = 50
EMBED_DIMS = 256
BEV_SEQ_LEN = BEV_H * BEV_W  # 2500

# Single cached nuScenes sample used by the demo/test as the frozen input frame.
SAMPLE_TOKEN = "3e8750f331d7499e9b5123e9eb70f2e2"

IMG_INPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, f"img_{SAMPLE_TOKEN}.raw"
)
CAN_BUS_INPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, f"can_bus_{SAMPLE_TOKEN}.raw"
)
LIDAR2IMG_INPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, f"lidar2img_{SAMPLE_TOKEN}.raw"
)


def load_raw(path: str, shape: tuple) -> torch.Tensor:
    """Load a flat float32 ``.raw`` binary file and reshape it.

    Parameters
    ----------
    path
        Path to a ``.raw`` file holding little-endian float32 values.
    shape
        Target shape; the file must contain exactly ``prod(shape)`` values.

    Returns
    -------
    torch.Tensor
        float32 tensor of the given shape.
    """
    data = np.fromfile(path, dtype=np.float32)
    assert data.size == int(np.prod(shape)), (
        f"Shape mismatch for {os.path.basename(path)}: "
        f"got {data.size}, expected {int(np.prod(shape))}"
    )
    return torch.from_numpy(data.reshape(shape))


def load_frame_inputs(
    spec: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch the cached single-frame demo inputs from the asset store.

    Parameters
    ----------
    spec
        Input spec (from ``get_input_spec()``) providing the target shapes
        for the image, can_bus, and lidar2img tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        image: (num_cam, 3, H, W) float32 normalized multi-camera images.
        can_bus: (18,) float32 ego-motion vector.
        lidar2img: (1, num_cam, 4, 4) float32 projection matrices.
    """
    image = load_raw(str(IMG_INPUT.fetch()), spec["image"].shape)
    can_bus = load_raw(str(CAN_BUS_INPUT.fetch()), spec["can_bus"].shape)
    lidar2img = load_raw(str(LIDAR2IMG_INPUT.fetch()), spec["lidar2img"].shape)
    return image, can_bus, lidar2img


class BEVFormer(BaseModel):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        ckpt = str(MODEL_CKPT.fetch()) if ckpt is None else ckpt

        # The BEVFormer plugin modules call mmcv's ext_loader.load_ext('_ext')
        # at import time, and build_model instantiates ops that do the same.
        # mmcv 1.7.0 from PyPI ships without the compiled _ext, so run both
        # inside patch_mmcv_no_extensions(), which stubs the loader and swaps
        # in a torchvision NMS fallback.
        with patch_mmcv_no_extensions():
            import mmcv.ops  # noqa: F401
            from mmdet3d.registry import MODELS
        from mmengine.runner import load_checkpoint

        with patch_mmcv_no_extensions():
            # First import of the external repo triggers the shallow clone +
            # patch; importing the plugin registers BEVFormer's custom modules
            # into the mmdet3d registry so build_model can resolve them.
            from .external_repos import EXTERNAL_REPO_PATHS
            from .external_repos.bevformer.projects import (  # noqa: F401
                mmdet3d_plugin,
            )

            cfg = Config.fromfile(
                str(
                    EXTERNAL_REPO_PATHS["bevformer"]
                    / "projects/configs/bevformer/bevformer_tiny_baseline_optimized.py"
                ),
                lazy_import=False,
            )
            cfg.model.train_cfg = None
            model = MODELS.build(cfg.model)
            # The patched detector's forward() branches on
            # self.export_model: when True it routes to forward_export
            # (the trace-friendly img/use_prev_bev/prev_bev/can_bus/
            # lidar2img signature) instead of the train/test path.
            model.export_model = True

            load_checkpoint(model, ckpt, map_location="cpu")
            model.eval()
        return cls(model)

    def forward(
        self,
        image: torch.Tensor,
        use_prev_bev: torch.Tensor,
        prev_bev: torch.Tensor,
        can_bus: torch.Tensor,
        lidar2img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run BEVFormer inference for a single frame.

        Note
        ----
        ``forward`` does NOT normalize the input. ``image`` must already be
        ImageNet mean/std normalized (mean [123.675, 116.28, 103.53], std
        [58.395, 57.12, 57.375]); passing raw [0, 255] pixels runs without
        error but silently degrades accuracy. Normalization is intentionally
        left outside the model: the ``.raw`` frame inputs are produced by the
        offline nuScenes pipeline that also derives the paired ``can_bus`` and
        ``lidar2img`` tensors, so image normalization is one step of a coupled
        multi-sensor preprocess that cannot be folded into ``forward`` in
        isolation. The w8a16 quantization encodings are calibrated on this
        normalized range, so on-device callers must feed pre-normalized images.
        See ``app.py`` (IMG_MEAN / IMG_STD) for the normalization constants.

        Parameters
        ----------
        image
            (num_cam, 3, H, W) float32. ImageNet mean/std normalized
            multi-camera images (see Note above).
        use_prev_bev
            (1,) float32. Flag: 0.0 = scene start (ignore prev_bev),
            1.0 = use prev_bev for temporal attention.
        prev_bev
            (1, BEV_H, BEV_W, EMBED_DIMS) = (1, 50, 50, 256) float32. BEV feature map
            from the previous frame. Ignored when use_prev_bev == 0.
        can_bus
            (18,) float32. Ego-motion / CAN-bus vector encoding vehicle
            pose and velocity used to align BEV queries across frames.
        lidar2img
            (1, num_cam, 4, 4) float32. Lidar-to-image projection matrices
            for each camera.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            bev_embed: (BEV_SEQ_LEN, 1, EMBED_DIMS) = (2500, 1, 256) float32.
            Flattened BEV feature grid (BEV_H x BEV_W spatial x EMBED_DIMS channels).
            all_cls_scores: (1, 1, 900, 10) float32. Per-query class logits
            for the 10 nuScenes detection categories (pre-sigmoid).
            all_bbox_preds: (1, 1, 900, 10) float32. Per-query box
            parameters, column layout
            [cx, cy, log_w, log_l, cz, log_h, sin_rot, cos_rot, vx, vy].
        """
        out = self.core(image, use_prev_bev, prev_bev, can_bus, lidar2img)
        return (out["bev_embed"], out["all_cls_scores"], out["all_bbox_preds"])

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """Real calibration inputs: the cached single-frame demo .raw frames.

        Parameters
        ----------
        input_spec
            Input spec to sample for. Defaults to get_input_spec().

        Returns
        -------
        SampleInputsType
            Mapping of input name to a single-element list of numpy arrays.
        """
        input_spec = input_spec or self.get_input_spec()
        image, can_bus, lidar2img = load_frame_inputs(input_spec)
        use_prev_bev = torch.zeros(1)
        prev_bev = torch.zeros(1, BEV_H, BEV_W, EMBED_DIMS)
        return {
            "image": [image.numpy()],
            "use_prev_bev": [use_prev_bev.numpy()],
            "prev_bev": [prev_bev.numpy()],
            "can_bus": [can_bus.numpy()],
            "lidar2img": [lidar2img.numpy()],
        }

    @staticmethod
    def get_input_spec(
        num_channel: int = 3,
        num_cam: int = 6,
        height: int = 480,
        width: int = 800,
    ) -> InputSpec:
        # BEVFormer is single-batch by design: `image` uses its leading axis for
        # the cameras (no batch dim), the outputs are fixed at batch 1 (see
        # get_output_spec), and the checkpoint / bevformer_tiny config assume it.
        # So the batch dim of prev_bev / lidar2img is a literal 1 rather than a
        # parameter -- exposing batch_size would imply a batched export path that
        # neither the spec nor the weights support.
        return {
            "image": TensorSpec(
                shape=(num_cam, num_channel, height, width), dtype="float32"
            ),
            "use_prev_bev": TensorSpec(shape=(1,), dtype="float32"),
            "prev_bev": TensorSpec(
                shape=(1, BEV_H, BEV_W, EMBED_DIMS), dtype="float32"
            ),
            "can_bus": TensorSpec(shape=(18,), dtype="float32"),
            "lidar2img": TensorSpec(shape=(1, num_cam, 4, 4), dtype="float32"),
        }

    @staticmethod
    def get_output_spec() -> OutputSpec:
        return {
            "bev_embed": TensorSpec(
                shape=(BEV_SEQ_LEN, 1, EMBED_DIMS), dtype="float32"
            ),
            "all_cls_scores": TensorSpec(shape=(1, 1, 900, 10), dtype="float32"),
            "all_bbox_preds": TensorSpec(shape=(1, 1, 900, 10), dtype="float32"),
        }

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Export the model to ONNX (opset 17).

        Overrides the base-class serialization because BEVFormer uses a
        custom ``ScatterND`` autograd Function (with a ``symbolic`` method
        for ONNX) that ``torch.export`` cannot trace. ONNX export via
        ``safe_torch_onnx_export`` handles the custom op correctly through its
        registered ONNX symbolic.

        Parameters
        ----------
        output_dir
            Directory where the exported ``.onnx`` file will be written.
        input_spec
            Input specification defining tensor shapes and dtypes.
            Defaults to ``get_input_spec()``.

        Returns
        -------
        Path
            Absolute path to the exported ``.onnx`` file.
        """
        input_spec = input_spec or self.get_input_spec()
        output_path = Path(output_dir) / f"{self.name}.onnx"
        self.to("cpu")
        safe_torch_onnx_export(
            self,
            tuple(make_torch_inputs(input_spec)),
            str(output_path),
            input_names=list(input_spec.keys()),
            output_names=self.get_output_names(),
            opset_version=17,
        )
        return output_path
