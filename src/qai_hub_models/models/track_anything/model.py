# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.track_anything.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.track_anything.external_repos.track_anything.tracker.model.network import (
    XMem,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_yaml
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import (
    BaseModel,
)
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

XMEM_MODEL = CachedWebModelAsset(
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "XMem-s012.pth",
)

TRACKER_CONFIG_PATH = str(
    EXTERNAL_REPO_PATHS["track_anything"] / "tracker" / "config" / "config.yaml"
)


class TrackAnything(BaseModel):
    def __init__(self, model: XMem) -> None:
        super().__init__(model=model)
        self.model: XMem

    @classmethod
    def from_pretrained(cls) -> Self:
        config = load_yaml(TRACKER_CONFIG_PATH)
        model = XMem(config, XMEM_MODEL.fetch()).eval()
        return cls(model)


class TrackAnythingEncodeKeyWithShrinkage(TrackAnything):
    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything EncodeKey model and return key, shrinkage, selection, f16 for given image.

        Parameters
        ----------
        image
            torch.Tensor of shape [1, 3, height, width], image with value range of [0, 1], RGB channel layout.

        Returns
        -------
        key : torch.Tensor
            torch.Tensor of shape [1, 64, height//16, width//16], encoded key
        shrinkage : torch.Tensor
            torch.Tensor of shape [1, 1, height//16, width//16], shrinkage key
        selection : torch.Tensor
            torch.Tensor of shape [1, 64, height//16, width//16], selection mask
        f16 : torch.Tensor
            torch.Tensor of shape [1, 1024, height//16, width//16], image features
        """
        image = normalize_image_torchvision(image)

        key, shrinkage, selection, f16, _f8, _f4 = self.model.encode_key(
            image,
            need_ek=True,  # encode_key
            need_sk=True,  # shrinkage_key
        )

        return key, shrinkage, selection, f16

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
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
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "key": TensorSpec(),
            "shrinkage": TensorSpec(),
            "selection": TensorSpec(),
            "f16": TensorSpec(),
        }


class TrackAnythingEncodeValue(TrackAnything):
    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        f16: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything Encode_value model and return generated mask for given points

        Parameters
        ----------
        image
            torch.Tensor of shape [1, 3, height, width], image with value range of [0, 1], RGB channel layout.
        mask
            torch.Tensor of shape [1, height, width], mask for first frame
        f16
            torch.Tensor of shape [1, 1024, height//16, width//16], image feature
        hidden_state
            torch.Tensor of shape [1, num_label, 64, height//16, width//16]

        Returns
        -------
        prob : torch.Tensor
            torch.Tensor of shape [2, height, width], predicted probabilities
        value : torch.Tensor
            torch.Tensor of shape [1, num_label, 512, height//16, width//16], encoded value
        hidden : torch.Tensor
            torch.Tensor of shape [1, num_label, 64, height//16, width//16]
        """
        image = normalize_image_torchvision(image)

        new_mask = torch.cat([1 - mask, mask], dim=0).clamp(1e-7, 1 - 1e-7)
        logits = torch.log(new_mask / (1 - new_mask))
        pred_prob_with_bg = torch.nn.functional.softmax(logits, dim=0)

        value, hidden = self.model.encode_value(
            image,
            f16,
            hidden_state,
            pred_prob_with_bg[1:].unsqueeze(0),
            is_deep_update=True,
        )
        return pred_prob_with_bg, value, hidden

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
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
                apply_runtime_channel_reordering=True,
            ),
            "mask": TensorSpec(
                shape=(batch_size, height, width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "f16": TensorSpec(
                shape=(batch_size, 1024, height // 16, width // 16),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "hidden_state": TensorSpec(
                shape=(batch_size, 1, 64, height // 16, width // 16),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "masks": TensorSpec(),
            "value": TensorSpec(),
            "hidden": TensorSpec(),
        }


class TrackAnythingEncodeKeyWithoutShrinkage(TrackAnything):
    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything Encode_Key model and return key, selection, f16, f8, f4 for given image.

        Parameters
        ----------
        image
            torch.Tensor of shape [1, 3, height, width], image with value range of [0, 1], RGB channel layout.

        Returns
        -------
        key : torch.Tensor
            torch.Tensor of shape [1, 64, height//16, width//16], encoded key
        selection : torch.Tensor
            torch.Tensor of shape [1, 64, height//16, width//16], selection mask
        f16 : torch.Tensor
            torch.Tensor of shape [1, 1024, height//16, width//16], image features
        f8 : torch.Tensor
            torch.Tensor of shape [1, 512, height//8, width//8], image features
        f4 : torch.Tensor
            torch.Tensor of shape [1, 256, height//4, width//4], image features
        """
        image = normalize_image_torchvision(image)

        key, _, selection, f16, f8, f4 = self.model.encode_key(
            image,
            need_ek=True,
            need_sk=False,
        )
        return key, selection, f16, f8, f4

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
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
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "key": TensorSpec(),
            "selection": TensorSpec(),
            "f16": TensorSpec(),
            "f8": TensorSpec(),
            "f4": TensorSpec(),
        }


class TrackAnythingSegment(TrackAnything):
    def forward(
        self,
        f16: torch.Tensor,
        f8: torch.Tensor,
        f4: torch.Tensor,
        memory_readout: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything model and return generated mask for given points

        Parameters
        ----------
        f16
            torch.Tensor of shape [1, 1024, height//16, width//16], image features
        f8
            torch.Tensor of shape [1, 512, height//8, width//8], image features
        f4
            torch.Tensor of shape [1, 256, height//4, width//4], image features
        memory_readout
            torch.Tensor of shape [1, num_label, 512, height//16, width//16], memory matched with current key and selection
        hidden_state
            torch.Tensor of shape [1, num_label, 64, height//16, width//16]

        Returns
        -------
        prob : torch.Tensor
            torch.Tensor of shape [2, height, width], predicted probabilities
        hidden : torch.Tensor
            torch.Tensor of shape [1, num_label, 64, height//16, width//16]
        """
        multi_scale_features = (f16, f8, f4)

        # segment the current frame
        hidden, _pred_logits_with_bg, pred_prob_with_bg = self.model.segment(
            multi_scale_features,
            memory_readout,
            hidden_state,
            h_out=True,
            strip_bg=False,
        )
        # remove batch dim
        pred_prob_with_bg = pred_prob_with_bg[0]

        return pred_prob_with_bg, hidden

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
    ) -> InputSpec:
        return {
            "f16": TensorSpec(
                shape=(batch_size, 1024, height // 16, width // 16),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "f8": TensorSpec(
                shape=(batch_size, 512, height // 8, width // 8),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "f4": TensorSpec(
                shape=(batch_size, 256, height // 4, width // 4),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "memory_readout": TensorSpec(
                shape=(batch_size, 1, 512, height // 16, width // 16),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "hidden_state": TensorSpec(
                shape=(batch_size, 1, 64, height // 16, width // 16),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "masks": TensorSpec(),
            "hidden": TensorSpec(),
        }


class TrackAnythingWrapper(WorkbenchModelCollection):
    def __init__(
        self,
        EncodeKeyWithShrinkage: TrackAnythingEncodeKeyWithShrinkage,
        EncodeValue: TrackAnythingEncodeValue,
        EncodeKeyWithoutShrinkage: TrackAnythingEncodeKeyWithoutShrinkage,
        Segment: TrackAnythingSegment,
        config: dict,
    ) -> None:
        super().__init__(
            {
                "encode_key_with_shrinkage": EncodeKeyWithShrinkage,
                "encode_value": EncodeValue,
                "encode_key_without_shrinkage": EncodeKeyWithoutShrinkage,
                "segment": Segment,
            },
        )
        self.EncodeKeyWithShrinkage = EncodeKeyWithShrinkage
        self.EncodeValue = EncodeValue
        self.EncodeKeyWithoutShrinkage = EncodeKeyWithoutShrinkage
        self.Segment = Segment
        self.config = config

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(batch_size=batch_size, height=height, width=width)

    @classmethod
    def from_pretrained(cls) -> Self:
        config = load_yaml(TRACKER_CONFIG_PATH)
        model = XMem(config, XMEM_MODEL.fetch()).eval()
        EncodeKeyWithShrinkage = TrackAnythingEncodeKeyWithShrinkage(model)
        EncodeValue = TrackAnythingEncodeValue(model)
        EncodeKeyWithoutShrinkage = TrackAnythingEncodeKeyWithoutShrinkage(model)
        Segment = TrackAnythingSegment(model)
        return cls(
            EncodeKeyWithShrinkage,
            EncodeValue,
            EncodeKeyWithoutShrinkage,
            Segment,
            config,
        )
