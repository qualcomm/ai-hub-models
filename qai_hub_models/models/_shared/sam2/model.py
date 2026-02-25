# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.modeling.backbones import hieradet
from sam2.modeling.backbones.hieradet import MultiScaleBlock as SAM2_Encoder_Block
from sam2.modeling.sam.transformer import TwoWayAttentionBlock, TwoWayTransformer
from sam2.modeling.sam2_base import SAM2Base as Sam2
from sam2.modeling.sam2_utils import MLP as SAM2MaskDecoderMLP

from qai_hub_models.models._shared.sam.model_patches import (
    Conv2DInplaceLinearSAMMaskDecoderMLP,
    SplitHeadSAMDecoderAttention,
)
from qai_hub_models.models._shared.sam2.model_patches import (
    Conv2DInplaceLinearSAMTransformerMLPBlock,
    SplitHeadSAMEncoderAttention,
    sam_decoder_predict_masks,
    sam_prompt_encoder_embed_points,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.window_partitioning import (
    window_partition_5d,
    window_unpartition_5d,
)

if TYPE_CHECKING:
    from qai_hub_models.models._shared.sam.model_patches import (
        Conv2DInplaceLinearSAMMaskDecoderMLP,
        SplitHeadSAMDecoderAttention,
    )
    from qai_hub_models.models._shared.sam2.model_patches import (
        Conv2DInplaceLinearSAMTransformerMLPBlock,
        SplitHeadSAMEncoderAttention,
    )
from qai_hub_models.models._shared.sam2.model_patches import SAM2Normalize

# Patch Encoder to use 5D Window Partition (rather than 6D)
hieradet.window_partition = window_partition_5d
hieradet.window_unpartition = window_unpartition_5d

BB_FEAT_SIZES = [
    (256, 256),
    (128, 128),
    (64, 64),
]


class SAM2Encoder(BaseModel, ABC):
    """Base class for SAM-based encoders (SAM2, EdgeTAM, etc.)"""

    def __init__(
        self,
        sam2: Sam2,
    ) -> None:
        super().__init__()
        self.sam2 = sam2
        self.normalize = SAM2Normalize()
        self._bb_feat_sizes = BB_FEAT_SIZES

    def forward(
        self,
        Image: torch.Tensor,
        norm_coords: torch.Tensor,  # [num_labels,num_points,2]
        labels: torch.Tensor,  # [num_labels,num_points]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run SAM Image encoder and returns image_embeddings,
        high_res_features1, high_res_features2, sparse_embeddings

        Parameters
        ----------
        Image
            Raw floating point pixel values for encoder consumption.
            3-channel Color Space: RGB, range [0, 1]
        norm_coords
            Point coordinates from input image for segmentation,
            mapped to the resized image with shape [1, N, 2]
        labels
            Point Labels to select/de-select given point for segmentation
            with shape shape [1, N], e.g. Corresponding value is 1
            if this point is to be included, otherwise 0


        Returns
        -------
        image_embeddings : torch.Tensor
            Shape (1, 256, 64, 64).
        high_res_features1 : torch.Tensor
            Shape (1, 32, 256, 256).
        high_res_features2 : torch.Tensor
            Shape (1, 64, 128, 128).
        sparse_embeddings : torch.Tensor
            Shape (1, N+1, 256).
        """
        x = self.normalize(Image)
        backbone_out = self.sam2.forward_image(x)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self._bb_feat_sizes[::-1], strict=False
            )
        ][::-1]
        image_embeddings = feats[2]
        high_res_features1 = feats[0]
        high_res_features2 = feats[1]
        sparse_embedding = self.sam2.sam_prompt_encoder._embed_points(
            norm_coords, labels, pad=True
        )

        return (
            image_embeddings,
            high_res_features1,
            high_res_features2,
            sparse_embedding,
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_points: int = 2,
        encoder_img_height: int = 1024,  # self.sam2.image_size
        encoder_img_width: int = 1024,  # self.sam2.image_size
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "image": (
                (batch_size, 3, encoder_img_height, encoder_img_width),
                "float32",
            ),
            "unnorm_coords": ((1, num_points, 2), "float32"),
            "labels": ((1, num_points), "float32"),
        }

    def _get_input_spec_for_instance(
        self, batch_size: int = 1, num_points: int = 2
    ) -> InputSpec:
        """Override for model.get_input_spec() when called on instances of this class."""
        return self.__class__.get_input_spec(
            batch_size, num_points, self.sam2.image_size, self.sam2.image_size
        )

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["image_embeddings", "high_res_features1", "high_res_features2"]

    @staticmethod
    def get_output_names() -> list[str]:
        return [
            "image_embeddings",
            "high_res_features1",
            "high_res_features2",
            "sparse_embedding",
        ]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "sav"

    @staticmethod
    def get_hub_litemp_percentage(_: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10


class SAM2Decoder(BaseModel, ABC):
    """
    Base class for SAM-based decoders (SAM2, EdgeTAM, etc.)

    This decoder is based on SAM2ImagePredictor.predict from sam2.
    It removes output mask resizing because dynamic shapes are better handled
    as a postprocessing step rather than in the inference framework.
    """

    def __init__(self, sam2: Sam2) -> None:
        super().__init__(sam2)
        self.model: Sam2
        self.mask_decoder = self.model.sam_mask_decoder
        self.prompt_encoder = self.model.sam_prompt_encoder
        self.prompt_encoder_embed_dim: int = self.model.sam_prompt_embed_dim
        self.embed_size = self.prompt_encoder.image_embedding_size
        self._bb_feat_sizes = BB_FEAT_SIZES
        self.high_res_features1_dim = 32
        self.high_res_features2_dim = 64

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor,  # [1, 32, 256, 256]
        high_res_features2: torch.Tensor,  # [1, 64, 128, 128]
        sparse_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run SAM lightweight decoder and return generated mask for the given points.

        Parameters
        ----------
        image_embeddings
            Image embeddings generated by the encoder.
            Shape: [1, 256, 64, 64].
        high_res_features1
            First set of high-resolution features.
            Shape: [1, 32, 256, 256].
        high_res_features2
            Second set of high-resolution features.
            Shape: [1, 64, 128, 128].
        sparse_embedding
            Sparse prompt embeddings (e.g., points/boxes) from the prompt encoder.
            Shape: [1, N+1, 256].

        Returns
        -------
        masks : torch.Tensor
            Low-resolution masks of shape [1, 1, 256, 256].
        scores : torch.Tensor
            IoU predictions of shape [1, 1].
        """
        dense_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        self.mask_decoder.dynamic_multimask_via_stability = False
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        return low_res_masks, iou_predictions

    def _get_input_spec_for_instance(
        self,
        num_points: int = 2,
    ) -> InputSpec:
        """
        Override for model.get_input_spec() when called on instances of this class.

        The initializer for BaseModel will automatically override get_input_spec
        with this function when the class is instantiated.
        """
        return self.__class__.get_input_spec(
            num_points,
            self.prompt_encoder_embed_dim,
            self._bb_feat_sizes[2],
            self._bb_feat_sizes[1],
            self._bb_feat_sizes[0],
            self.high_res_features1_dim,
            self.high_res_features2_dim,
        )

    @staticmethod
    def get_input_spec(
        num_points: int = 2,
        embed_dim: int = 256,
        image_embedding: tuple = (64, 64),
        high_res_featutes2: tuple = (128, 128),
        high_res_featutes1: tuple = (256, 256),
        high_res_features1_dim: int = 32,
        high_res_features2_dim: int = 64,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.

        input_spec: InputSpec = {
            "image_embeddings": ((1, embed_dim, *image_embedding), "float32"),
            "high_res_features1": (
                (1, high_res_features1_dim, *high_res_featutes1),
                "float32",
            ),
            "high_res_features2": (
                (1, high_res_features2_dim, *high_res_featutes2),
                "float32",
            ),
            "sparse_embedding": ((1, num_points + 1, embed_dim), "float32"),
        }
        return input_spec

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image_embeddings", "high_res_features1", "high_res_features2"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["masks"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["masks", "scores"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "sav"

    @staticmethod
    def get_hub_litemp_percentage(_: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10


class SAM2Loader(ABC):
    @classmethod
    @abstractmethod
    def _load_sam2(cls, model_type: str) -> Sam2:
        """
        Load the SAM2 model for the given model type.
        Must be implemented by subclasses.
        """

    @classmethod
    def _patch_sam2_for_qnn_compatibility(cls, sam2: Sam2) -> None:
        """Apply patches to the SAM2 model for compatibility with QNN."""
        ###
        # Patch the graph for compatibility with QNN.
        #
        # All below optimizations either optimize for QNN inference speed,
        # or fix failures that occur when compiling to QNN.
        ###
        if hasattr(sam2.image_encoder.trunk, "blocks"):
            for block in sam2.image_encoder.trunk.blocks:
                assert isinstance(block, SAM2_Encoder_Block)
                block.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(block.mlp)
                block.attn = SplitHeadSAMEncoderAttention(block.attn)

        sam2.sam_mask_decoder.predict_masks = functools.partial(
            sam_decoder_predict_masks, sam2.sam_mask_decoder
        )
        for i in range(len(sam2.sam_mask_decoder.output_hypernetworks_mlps)):
            mlp = cast(
                SAM2MaskDecoderMLP, sam2.sam_mask_decoder.output_hypernetworks_mlps[i]
            )
            sam2.sam_mask_decoder.output_hypernetworks_mlps[i] = (
                Conv2DInplaceLinearSAMMaskDecoderMLP(mlp)
            )

        sam2.sam_mask_decoder.iou_prediction_head = (
            Conv2DInplaceLinearSAMMaskDecoderMLP(
                sam2.sam_mask_decoder.iou_prediction_head
            )
        )

        transformer = cast(TwoWayTransformer, sam2.sam_mask_decoder.transformer)
        transformer.final_attn_token_to_image = SplitHeadSAMDecoderAttention(
            transformer.final_attn_token_to_image
        )
        for block in transformer.layers:
            block = cast(TwoWayAttentionBlock, block)
            block.self_attn = SplitHeadSAMDecoderAttention(block.self_attn)
            block.cross_attn_token_to_image = SplitHeadSAMDecoderAttention(
                block.cross_attn_token_to_image
            )
            block.cross_attn_image_to_token = SplitHeadSAMDecoderAttention(
                block.cross_attn_image_to_token
            )
            block.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(block.mlp)

        sam2.sam_prompt_encoder._embed_points = functools.partial(
            sam_prompt_encoder_embed_points, sam2.sam_prompt_encoder
        )

    @classmethod
    def _initialize_hydra_config(
        cls,
        config_dir: Path | str,
        job_name: str = "sam_inference",
    ) -> None:
        """
        Initialize Hydra configuration from a config directory.

        Parameters
        ----------
        config_dir
            Path to the configuration directory
        job_name
            Name for the Hydra job
        """
        GlobalHydra.instance().clear()
        initialize_config_dir(
            config_dir=str(config_dir),
            job_name=job_name,
            version_base=None,
        )
