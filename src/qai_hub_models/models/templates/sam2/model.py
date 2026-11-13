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
from sam2.modeling.sam2_utils import LayerNorm2d

from qai_hub_models import Precision
from qai_hub_models.models.templates.corevision.window_partitioning import (
    window_partition_5d,
    window_unpartition_5d,
)
from qai_hub_models.models.templates.sam.model_patches import (
    Conv2DInplaceLinearSAMMaskDecoderMLP,
    SplitHeadSAMDecoderAttention,
)
from qai_hub_models.models.templates.sam2.model_patches import (
    Conv2DInplaceLinearSAMTransformerMLPBlock,
    SAM2LayerNorm2d,
    SplitHeadSAMEncoderAttention,
    sam_decoder_predict_masks,
    sam_prompt_encoder_embed_points,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

if TYPE_CHECKING:
    from qai_hub_models.models.templates.sam.model_patches import (
        Conv2DInplaceLinearSAMMaskDecoderMLP,
        SplitHeadSAMDecoderAttention,
    )
    from qai_hub_models.models.templates.sam2.model_patches import (
        Conv2DInplaceLinearSAMTransformerMLPBlock,
        SplitHeadSAMEncoderAttention,
    )
from qai_hub_models.models.templates.sam2.model_patches import SAM2Normalize

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
        image: torch.Tensor,
        norm_coords: torch.Tensor,  # [num_labels,num_points,2]
        labels: torch.Tensor,  # [num_labels,num_points]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run SAM Image encoder and returns image_embeddings,
        high_res_features1, high_res_features2, sparse_embeddings, pix_feat

        Parameters
        ----------
        image
            Raw floating point pixel values for encoder consumption.
            3-channel Color Space: RGB, range [0, 1]
        norm_coords
            Point coordinates from input image for segmentation,
            mapped to the resized image with shape [1, N, 2].
            For tracking frames with no prompt, pass zeros and set
            ``labels`` to -1
        labels
            Point labels with shape [1, N].
            Use 1 for positive, 0 for negative, -1 for "no prompt"
            (tracking frames).

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
        pix_feat : torch.Tensor
            image_embeddings with no_mem_embed added. Shape (1, 256, 64, 64).
            Use this as the decoder's image_embeddings input for the conditioning
            frame. For tracking frames, discard and use memory-conditioned features.
        """
        x = self.normalize(image)
        backbone_out = self.sam2.forward_image(x)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)
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
        no_mem = self.sam2.no_mem_embed.permute(0, 2, 1).view(1, -1, 1, 1)
        pix_feat = image_embeddings + no_mem

        return (
            image_embeddings,
            high_res_features1,
            high_res_features2,
            sparse_embedding,
            pix_feat,
        )

    def get_input_spec(self, batch_size: int = 1, num_points: int = 2) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, self.sam2.image_size, self.sam2.image_size),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
            "unnorm_coords": TensorSpec(
                shape=(1, num_points, 2),
                dtype="float32",
            ),
            "labels": TensorSpec(
                shape=(1, num_points),
                dtype="float32",
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "image_embeddings": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "high_res_features1": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "high_res_features2": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "sparse_embedding": TensorSpec(),
            "pix_feat": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10


class SAM2DecoderBase(BaseModel, ABC):
    """
    Base class for SAM-based decoders (SAM2, EdgeTAM, etc.)

    This decoder is based on SAM2ImagePredictor.predict from sam2.
    It removes output mask resizing because dynamic shapes are better handled
    as a postprocessing step rather than in the inference framework.

    Subclasses implement ``forward`` and ``get_output_spec``.
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
        self.mask_decoder.dynamic_multimask_via_stability = False

    def _run_mask_decoder(
        self,
        image_embeddings: torch.Tensor,
        high_res_features1: torch.Tensor,
        high_res_features2: torch.Tensor,
        sparse_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared mask decoder call used by SAM2Decoder and SAM2VideoDecoder."""
        dense_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )

    def get_input_spec(
        self,
        num_points: int = 2,
    ) -> InputSpec:
        return {
            "image_embeddings": TensorSpec(
                shape=(1, self.prompt_encoder_embed_dim, *self._bb_feat_sizes[2]),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            ),
            "high_res_features1": TensorSpec(
                shape=(1, self.high_res_features1_dim, *self._bb_feat_sizes[0]),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            ),
            "high_res_features2": TensorSpec(
                shape=(1, self.high_res_features2_dim, *self._bb_feat_sizes[1]),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            ),
            "sparse_embedding": TensorSpec(
                shape=(1, num_points + 1, self.prompt_encoder_embed_dim),
                dtype="float32",
            ),
        }

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10


class SAM2Decoder(SAM2DecoderBase):
    """Image decoder returning low-resolution masks and IoU scores."""

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
        low_res_masks, iou_predictions, _, _ = self._run_mask_decoder(
            image_embeddings, high_res_features1, high_res_features2, sparse_embedding
        )
        return low_res_masks, iou_predictions

    def get_output_spec(self) -> OutputSpec:
        return {
            "masks": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "scores": TensorSpec(),
        }


class SAM2VideoDecoder(SAM2DecoderBase):
    """
    Video-capable decoder that also returns obj_ptr for memory tracking.

    Extends SAM2Decoder to capture the SAM output token and produce
    an object pointer vector used for temporal tracking in video.
    """

    def __init__(self, sam2: Sam2) -> None:
        super().__init__(sam2)
        self.obj_ptr_proj = self.model.obj_ptr_proj
        # Fall back to best multi-mask output when single-mask is unstable.
        self.mask_decoder.dynamic_multimask_via_stability = True

    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_features1: torch.Tensor,
        high_res_features2: torch.Tensor,
        sparse_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run SAM decoder and return masks, scores, object pointer, and object score logits.

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
          Sparse prompt embeddings from the prompt encoder.
          Shape: [1, N+1, 256].

        Returns
        -------
        torch.Tensor
          Low-resolution masks of shape [1, 1, 256, 256].
        torch.Tensor
          IoU predictions of shape [1, 1].
        torch.Tensor
          Object pointer of shape [1, 256] for memory tracking.
        torch.Tensor
          Object score logits of shape [1, 1] for CPU-side object-score gating in the app.
        """
        low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = (
            self._run_mask_decoder(
                image_embeddings,
                high_res_features1,
                high_res_features2,
                sparse_embedding,
            )
        )

        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj(sam_output_token)

        return low_res_masks, iou_predictions, obj_ptr, object_score_logits

    def get_output_spec(self) -> OutputSpec:
        return {
            "masks": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "scores": TensorSpec(),
            "obj_ptr": TensorSpec(),
            "object_score_logits": TensorSpec(),
        }


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

        for parent in sam2.modules():
            for name, child in parent.named_children():
                if isinstance(child, LayerNorm2d):
                    setattr(parent, name, SAM2LayerNorm2d(child))

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
