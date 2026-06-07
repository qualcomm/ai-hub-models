# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# VietOCR's default vgg_transformer recognizer. The recognition CNN backbone
# (vgg19_bn) is the convolution-heavy front end of the model.
VIETOCR_CONFIG_NAME = "vgg_transformer"

# Recognition is performed on fixed-height text-line crops; a representative
# fixed width is pinned for on-device export.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128


def _load_vietocr_cnn() -> torch.nn.Module:
    """Load the pretrained VietOCR vgg_transformer recognizer's CNN backbone."""
    from vietocr.model.transformerocr import VietOCR as _VietOCR
    from vietocr.model.vocab import Vocab
    from vietocr.tool.config import Cfg

    cfg = Cfg.load_config_from_name(VIETOCR_CONFIG_NAME)
    cfg["device"] = "cpu"
    vocab = Vocab(cfg["vocab"])
    model = _VietOCR(
        len(vocab),
        cfg["backbone"],
        cfg["cnn"],
        cfg["transformer"],
        cfg["seq_modeling"],
    ).eval()

    from vietocr.tool.utils import download_weights

    weights = download_weights(cfg["pretrain"])
    model.load_state_dict(torch.load(weights, map_location="cpu"))

    # The CNN backbone is wrapped in `.model` inside VietOCR's CNN module.
    return model.cnn.model if hasattr(model.cnn, "model") else model.cnn


class VietOCR(BaseModel):
    """VietOCR recognition CNN backbone (vgg19_bn) for Vietnamese text.

    VietOCR recognizes Vietnamese text (a character set that includes the full
    precomposed tone vowels, e.g. ế ồ ự ấ ợ). This component is the CNN backbone
    that maps a text-line crop to a per-column feature sequence consumed by the
    downstream transformer recognizer.

    The original backbone tail applies `permute(-1, 0, 1)` and
    `transpose(-1, -2).flatten(2)`. Negative permutation axes and the implied
    dynamic reshape do not export cleanly to a static on-device graph, so the
    tail is rebuilt here with equivalent static, positive-axis operations.
    """

    def __init__(self, vgg: torch.nn.Module) -> None:
        super().__init__()
        self.features = vgg.features
        self.last_conv_1x1 = vgg.last_conv_1x1

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(_load_vietocr_cnn())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run the CNN backbone on a text-line crop.

        Parameters
        ----------
        image
            Pixel values pre-processed for backbone consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        features : torch.Tensor
            Per-column feature sequence. Shape [W', batch, 256].
        """
        x = self.features(image)  # [B, 512, 1, W']
        x = self.last_conv_1x1(x)  # [B, 256, 1, W']
        x = x.transpose(2, 3)  # [B, 256, W', 1]  (static positive axes)
        x = x.flatten(2)  # [B, 256, W']
        # [W', B, 256]   (static positive axes)
        return x.permute(2, 0, 1)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = IMAGE_HEIGHT,
        width: int = IMAGE_WIDTH,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_output_names(self) -> list[str]:
        return ["features"]

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]
