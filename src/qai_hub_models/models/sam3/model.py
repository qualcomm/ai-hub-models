# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import types
from typing import Any, TypeVar

import torch
from qai_hub.client import Device
from sam3 import build_sam3_image_model
from sam3 import model_builder as sam3_model_builder
from sam3.model import vitdet
from sam3.model.decoder import TransformerDecoder
from sam3.model.maskformer_segmentation import UniversalSegmentationHead
from sam3.model.model_misc import (
    DotProductScoring,
    TransformerWrapper,
    inverse_sigmoid,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.sam3_image import Sam3Image as Sam3
from sam3.model.text_encoder_ve import VETextEncoder
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.sam3.model_patches import (
    SAM3Normalize,
    SplitHeadMultiheadAttention,
    SplitHeadResidualAttentionBlock,
    SplitHeadVitDetAttention,
    apply_rope,
    patch_decoder_last_layer_only,
    patch_decoder_rpb_device,
)
from qai_hub_models.models.templates.corevision.window_partitioning import (
    window_partition_5d,
    window_unpartition_5d,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

DEFAULT_MODEL_DEVICE = "cpu"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


def _wrap_mha_attrs(module: torch.nn.Module, attrs: tuple[str, ...]) -> None:
    """Replace each named ``nn.MultiheadAttention`` attr with SplitHead."""
    for attr in attrs:
        mha = getattr(module, attr, None)
        if mha is not None:
            setattr(module, attr, SplitHeadMultiheadAttention(mha))


class SAM3VisionBackbone(BaseModel):
    """SAM3 vision backbone: image -> FPN features."""

    def __init__(
        self,
        normalize: SAM3Normalize,
        vision_model: Sam3DualViTDetNeck,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.vision_model = vision_model

    @classmethod
    def from_pretrained(cls, model_device: str = DEFAULT_MODEL_DEVICE) -> Self:
        return SAM3Loader.load(vision_backbone_cls=cls, model_device=model_device)[1]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime == TargetRuntime.QNN_DLC:
            other_compile_options += " -O2"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the vision backbone and return the three FPN feature maps.

        Parameters
        ----------
        image
            Input image tensor of shape ``(B, 3, H, W)``, float32 in
            ``[0, 1]``.

        Returns
        -------
        backbone_fpn_0 : torch.Tensor
            Highest-resolution FPN level, shape ``(B, 256, H/3.5, W/3.5)``.
        backbone_fpn_1 : torch.Tensor
            Mid-resolution FPN level, shape ``(B, 256, H/7, W/7)``.
        backbone_fpn_2 : torch.Tensor
            Deepest FPN level consumed by the transformer encoder,
            shape ``(B, 256, H/14, W/14)``.
        """
        image = self.normalize(image)
        fpn, _, _, _ = self.vision_model(image)
        return fpn[0], fpn[1], fpn[2]

    def get_input_spec(
        self,
        batch_size: int = 1,
        img_height: int = 1008,
        img_width: int = 1008,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, img_height, img_width), dtype="float32"
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "backbone_fpn_0": TensorSpec(),
            "backbone_fpn_1": TensorSpec(),
            "backbone_fpn_2": TensorSpec(),
        }


class SAM3Head(BaseModel):
    """
    Combined SAM3 head: language backbone + transformer encoder +
    transformer decoder + segmentation head in a single graph.

    Takes the tokenized prompt and the three FPN feature maps from the
    vision backbone; emits the four production outputs.
    """

    def __init__(
        self,
        language_model: VETextEncoder,
        transformer: TransformerWrapper,
        segmentation_head: UniversalSegmentationHead,
        dot_prod_scoring: DotProductScoring,
        vision_pos_enc_2: torch.Tensor,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.transformer = transformer
        self.segmentation_head = segmentation_head
        self.dot_prod_scoring = dot_prod_scoring

        _, d_model, h, w = vision_pos_enc_2.shape
        self._fpn_h, self._fpn_w = int(h), int(w)
        self._mask_h, self._mask_w = self._fpn_h * 4, self._fpn_w * 4
        self._d_model = int(d_model)
        self._seq_len = int(language_model.context_length)

        # Give the decoder the static feat size; reading it off the traced
        # spatial_shapes tensor (int(.item())) breaks torch.export.
        self.transformer.decoder._rpb_feat_size = (self._fpn_h, self._fpn_w)

        # vision_pos_enc_2 is deterministic from (d_model, H, W) — bake
        # the flattened form the encoder/decoder both consume as a
        # graph constant rather than recompute on every call.
        self.register_buffer(
            "_pos_embed",
            vision_pos_enc_2[0:1].flatten(2).permute(2, 0, 1).contiguous(),
            persistent=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_device: str = DEFAULT_MODEL_DEVICE,
    ) -> Self:
        return SAM3Loader.load(
            head_cls=cls,
            model_device=model_device,
        )[2]

    def forward(
        self,
        tokenized: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run language encoder, transformer, and segmentation head end-to-end.

        Parameters
        ----------
        tokenized
            Tokenized text prompt, shape ``(1, seq_len)``, int32/int64.
        backbone_fpn_0
            Highest-resolution FPN feature, shape ``(1, d_model, mask_h, mask_w)``.
        backbone_fpn_1
            Mid-resolution FPN feature, shape ``(1, d_model, mask_h/2, mask_w/2)``.
        backbone_fpn_2
            Deepest FPN feature consumed by the encoder,
            shape ``(1, d_model, fpn_h, fpn_w)``.

        Returns
        -------
        pred_boxes : torch.Tensor
            Predicted boxes in normalized cxcywh,
            shape ``(1, num_queries, 4)``.
        scores : torch.Tensor
            Per-query confidence reduced over the class dim.
            Shape ``(1, num_queries)``.
        pred_masks : torch.Tensor
            Mask logits at quarter-input resolution,
            shape ``(1, num_queries, mask_h, mask_w)``.
        """
        # Language path.
        _, text_memory = self.language_model.encoder(tokenized)
        text_memory = text_memory.transpose(0, 1)
        prompt_sf = self.language_model.resizer(text_memory)
        prompt_mask_bool = (tokenized == 0).bool()

        # Transformer encoder.
        img_feats = [backbone_fpn_2.flatten(2).permute(2, 0, 1)]
        img_pos_embeds = [self._pos_embed]
        prompt_pos_embed = torch.zeros_like(prompt_sf)

        encoder_out = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt_sf,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask_bool,
            feat_sizes=[(self._fpn_h, self._fpn_w)],
            encoder_extra_kwargs=None,
        )
        memory_sf = encoder_out["memory"]

        # Transformer decoder. patch_decoder_last_layer_only makes the
        # decoder return single-layer stacks; [-1] picks the sole entry.
        hs, reference_boxes, dec_presence_out, _ = self.transformer.decoder(
            tgt=self.transformer.decoder.query_embed.weight.unsqueeze(1),
            memory=memory_sf,
            memory_key_padding_mask=None,
            pos=self._pos_embed,
            reference_boxes=None,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt_sf,
            text_attention_mask=prompt_mask_bool,
            apply_dac=False,
        )
        hs_last = hs[-1].transpose(0, 1)
        reference_boxes_last = reference_boxes[-1].transpose(0, 1)
        presence_logit_dec = dec_presence_out[-1].transpose(0, 1)

        pred_logits = self.dot_prod_scoring(
            hs_last.unsqueeze(0), prompt_sf, prompt_mask_bool
        )[0]

        anchor_box_offsets = self.transformer.decoder.bbox_embed(hs_last)
        pred_boxes = (
            inverse_sigmoid(reference_boxes_last) + anchor_box_offsets
        ).sigmoid()

        # Segmentation head.
        encoder_hidden_states = memory_sf
        if self.segmentation_head.cross_attend_prompt is not None:
            tgt2 = self.segmentation_head.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.segmentation_head.cross_attend_prompt(
                query=tgt2,
                key=prompt_sf,
                value=prompt_sf,
                key_padding_mask=prompt_mask_bool,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        backbone_visual_feats = [
            backbone_fpn_0,
            backbone_fpn_1,
            encoder_hidden_states.permute(1, 2, 0).reshape(
                1, self._d_model, self._fpn_h, self._fpn_w
            ),
        ]
        num_queries = hs_last.shape[1]
        pixel_embed = self.segmentation_head.pixel_decoder(backbone_visual_feats)
        instance_embeds = self.segmentation_head.instance_seg_head(pixel_embed)
        mask_embeds = self.segmentation_head.mask_predictor.mask_embed(
            hs_last
        ).unsqueeze(-2)
        mask_pixels = self._mask_h * self._mask_w
        pred_masks = torch.matmul(
            mask_embeds,
            instance_embeds.reshape(1, 1, self._d_model, mask_pixels),
        ).reshape(1, num_queries, self._mask_h, self._mask_w)

        scores = (
            (pred_logits.sigmoid() * presence_logit_dec.sigmoid().unsqueeze(-1))
            .max(dim=-1)
            .values
        )

        return pred_boxes, scores, pred_masks

    def get_input_spec(
        self,
        seq_len: int = 32,
        d_model: int = 256,
        fpn_h: int = 72,
        fpn_w: int = 72,
        mask_h: int = 288,
        mask_w: int = 288,
    ) -> InputSpec:
        return {
            "tokenized": TensorSpec(shape=(1, seq_len), dtype="int32"),
            "backbone_fpn_0": TensorSpec(
                shape=(1, d_model, mask_h, mask_w), dtype="float32"
            ),
            "backbone_fpn_1": TensorSpec(
                shape=(1, d_model, mask_h // 2, mask_w // 2),
                dtype="float32",
            ),
            "backbone_fpn_2": TensorSpec(
                shape=(1, d_model, fpn_h, fpn_w), dtype="float32"
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "pred_boxes": TensorSpec(),
            "scores": TensorSpec(),
            "pred_masks": TensorSpec(),
        }


visionBackboneT = TypeVar("visionBackboneT", bound=SAM3VisionBackbone)
headT = TypeVar("headT", bound=SAM3Head)


class SAM3Loader:
    """Helper class for loading and preparing a QNN-compatible SAM3 model."""

    @classmethod
    def load(
        cls,
        model_device: str = DEFAULT_MODEL_DEVICE,
        vision_backbone_cls: type[visionBackboneT] = SAM3VisionBackbone,  # type: ignore[assignment]
        head_cls: type[headT] = SAM3Head,  # type: ignore[assignment]
    ) -> tuple[Sam3, visionBackboneT, headT]:
        sam3 = cls._load_sam3(model_device)
        cls._patch_sam3_for_qnn_compatibility(sam3)
        vision_backbone = vision_backbone_cls(
            normalize=SAM3Normalize(device=model_device),
            vision_model=sam3.backbone.vision_backbone,
        )
        img_h, img_w = vision_backbone.get_input_spec()["image"][0][-2:]
        head = head_cls(
            language_model=sam3.backbone.language_backbone,
            transformer=sam3.transformer,
            segmentation_head=sam3.segmentation_head,
            dot_prod_scoring=sam3.dot_prod_scoring,
            vision_pos_enc_2=cls._compute_vision_pos_enc_2(
                sam3, model_device, img_h, img_w
            ),
        )
        return sam3, vision_backbone, head

    @classmethod
    def _load_sam3(cls, model_device: str = DEFAULT_MODEL_DEVICE) -> Sam3:
        # Replace upstream's bfloat16-fused addmm+GELU with the unfused
        # fp32 equivalent. The patch is permanent: addmm_act runs on
        # every trunk forward, so restoring it would break inference.
        vitdet.addmm_act = lambda act_cls, linear, x: act_cls()(linear(x))

        # Disable upstream's eager precompute caches — both target
        # device="cuda" at construction time and crash on CPU-only
        # builds. Their lazy paths repopulate on first forward.
        orig_create_pe = sam3_model_builder._create_position_encoding
        orig_decoder_init = TransformerDecoder.__init__

        def _cpu_safe_create_pe(*args: Any, **kwargs: Any) -> Any:
            kwargs["precompute_resolution"] = None
            return orig_create_pe(*args, **kwargs)

        def _no_precompute_decoder_init(
            self: TransformerDecoder, *a: Any, **kw: Any
        ) -> None:
            kw["resolution"] = None
            kw["stride"] = None
            orig_decoder_init(self, *a, **kw)

        sam3_model_builder._create_position_encoding = _cpu_safe_create_pe
        TransformerDecoder.__init__ = _no_precompute_decoder_init
        return build_sam3_image_model(device=model_device)

    @classmethod
    def _compute_vision_pos_enc_2(
        cls,
        sam3: Sam3,
        device: str,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Precompute the deepest-scale 2D positional encoding once.

        Derives the deepest-FPN (H, W) from the vision backbone's
        patch-embed stride and evaluates the sin/cos position encoding
        at that shape. Baked into SAM3Head as a buffer so the sin/cos
        subgraph does not run per-inference.
        """
        stride_h, stride_w = sam3.backbone.vision_backbone.trunk.patch_embed.proj.stride
        h, w = img_h // stride_h, img_w // stride_w
        d_model = sam3.backbone.vision_backbone.convs[2].conv_1x1.out_channels
        probe = torch.zeros(1, d_model, h, w, device=device)
        with torch.no_grad():
            pos = sam3.backbone.vision_backbone.position_encoding(probe).to(probe.dtype)
        return pos.detach().contiguous()

    @classmethod
    def _patch_sam3_for_qnn_compatibility(cls, sam3: Sam3) -> None:
        """Apply patches to the SAM3 model for compatibility with QNN."""
        # Language backbone: split-head residual attention blocks.
        resblocks = sam3.backbone.language_backbone.encoder.transformer.resblocks
        for i, block in enumerate(resblocks):
            resblocks[i] = SplitHeadResidualAttentionBlock(block)

        # Vision backbone: replace complex ops and attention blocks.
        vitdet.window_partition = window_partition_5d
        vitdet.window_unpartition = window_unpartition_5d
        for block in sam3.backbone.vision_backbone.trunk.blocks:
            block.attn = SplitHeadVitDetAttention(block.attn)
            block.attn._apply_rope = types.MethodType(apply_rope, block.attn)

        # Decompose freqs_cis (complex) into freqs_cos/freqs_sin buffers for QNN.
        for module in sam3.modules():
            if hasattr(module, "freqs_cis"):
                freqs_cis = module.freqs_cis
                delattr(module, "freqs_cis")
                module.register_buffer("freqs_cos", freqs_cis.real)
                module.register_buffer("freqs_sin", freqs_cis.imag)

        # Split-head patches on every transformer MHA module.
        for layer in sam3.transformer.encoder.layers:
            _wrap_mha_attrs(layer, ("cross_attn_image", "self_attn"))
        for layer in sam3.transformer.decoder.layers:
            _wrap_mha_attrs(layer, ("cross_attn_image", "self_attn", "ca_text"))
        _wrap_mha_attrs(sam3.segmentation_head, ("cross_attend_prompt",))

        patch_decoder_rpb_device(sam3.transformer.decoder)
        # SAM3's heads only consume ``[-1]`` of the per-layer stacks —
        # patch the forward to materialize only the final layer.
        patch_decoder_last_layer_only(sam3.transformer.decoder)


class SAM3(WorkbenchModelCollection):
    """SAM3: Segment Anything Model 3 with vision-language grounding."""

    def __init__(
        self,
        sam3: Sam3,
        vision_backbone: SAM3VisionBackbone,
        head: SAM3Head,
    ) -> None:
        super().__init__({"vision_backbone": vision_backbone, "head": head})
        self.sam3 = sam3
        self.head = head
        self.vision_backbone = vision_backbone

    def get_input_spec(
        self,
        batch_size: int = 1,
        img_height: int = 1008,
        img_width: int = 1008,
        seq_len: int = 32,
        d_model: int = 256,
        fpn_h: int = 72,
        fpn_w: int = 72,
        mask_h: int = 288,
        mask_w: int = 288,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            img_height=img_height,
            img_width=img_width,
            seq_len=seq_len,
            d_model=d_model,
            fpn_h=fpn_h,
            fpn_w=fpn_w,
            mask_h=mask_h,
            mask_w=mask_w,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_device: str = DEFAULT_MODEL_DEVICE,
    ) -> Self:
        return cls(*SAM3Loader.load(model_device))
