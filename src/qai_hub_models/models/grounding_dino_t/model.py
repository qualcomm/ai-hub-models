# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from functools import lru_cache
from typing import cast

import torch
from qai_hub.client import Device
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerBase
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.grounding_dino_t.dataset import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_TEXT_LEN,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    GroundingDINOCOCODataset,
    build_positive_map,
)
from qai_hub_models.models.grounding_dino_t.evaluator import GroundingDINOEvaluator
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.backbone.swin_transformer import (
    SwinTransformer,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.transformer import (
    Transformer,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.utils import (
    MLP,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.util.misc import (
    inverse_sigmoid as _inverse_sigmoid,
)
from qai_hub_models.models.grounding_dino_t.model_patches import (
    ContrastiveEmbed,
    NestedTensor,
    PositionEmbeddingSineHW,
    _ExportSafeGroupNorm,
    apply_patches,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "groundingdino_swint_ogc.pth",
)


@lru_cache(maxsize=1)
def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load and cache the bert-base-uncased tokenizer shared by all model/app entry points."""
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)


DEFAULT_PRECISION = Precision.float

NUM_FEATURE_LEVELS = 4


def _feature_map_sizes(img_height: int, img_width: int) -> list[tuple[int, int]]:
    """Return the (H, W) of each of the 4 FPN feature levels the vision
    component produces, given an input image of size (img_height, img_width).

    Levels 0-2 come from Swin-T stages at strides 8/16/32; level 3 is an
    extra stride-2 3x3 conv (padding=1) applied to level 2's raw feature.

    Swin's patch merging pads odd spatial dimensions before each 2x downsampling
    step, so each stage uses ceil division. Both dimensions must be multiples of
    32 to guarantee all four levels are integer-sized and agree with the
    fusion decoder's input specs; non-aligned sizes raise ValueError here rather
    than silently producing mismatched shapes downstream.
    """
    if img_height % 32 != 0 or img_width % 32 != 0:
        raise ValueError(
            f"img_height and img_width must be multiples of 32 "
            f"(got {img_height}x{img_width}). "
            f"Swin's patch merging pads odd dims, so non-aligned sizes "
            f"produce mismatched feature-map shapes across components."
        )

    # Swin patch embed: stride-4 Conv2d, then each patch-merging stage pads
    # odd dims -> ceil division (exact here since dims are multiples of 32).
    h0 = math.ceil(math.ceil(img_height / 4) / 2)  # stride 8
    w0 = math.ceil(math.ceil(img_width / 4) / 2)
    h1 = math.ceil(h0 / 2)  # stride 16
    w1 = math.ceil(w0 / 2)
    h2 = math.ceil(h1 / 2)  # stride 32
    w2 = math.ceil(w1 / 2)
    h3 = (h2 + 2 - 3) // 2 + 1  # extra stride-2 conv
    w3 = (w2 + 2 - 3) // 2 + 1
    return [(h0, w0), (h1, w1), (h2, w2), (h3, w3)]


def _add_truncate_64bit_compile_options(
    compile_options: str, target_runtime: TargetRuntime
) -> str:
    """Add --truncate_64bit_tensors/--truncate_64bit_io for non-ONNX targets, which
    otherwise fail to compile.
    """
    if target_runtime == TargetRuntime.ONNX:
        return compile_options
    return compile_options + " --truncate_64bit_tensors True --truncate_64bit_io True"


class _GroundingDINOTComponentMixin(BaseModel):
    """Shared get_hub_compile_options() for the three collection components.

    Subclasses must still override `component_precision()` (each component
    has its own fixed precision); BaseModel's version raises NotImplementedError
    until they do.
    """

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        # The collection export passes the collection-wide precision (e.g.
        # mixed_with_float, which carries no quantize dtype of its own) to
        # every component, so narrow it to this component's own precision.
        if precision in (Precision.mixed, Precision.mixed_with_float):
            precision = self.component_precision()
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        return _add_truncate_64bit_compile_options(compile_options, target_runtime)


class _GroundingDINOTParts(nn.Module):
    """Internal container for the pretrained submodules used by the collection."""

    def __init__(
        self,
        backbone: SwinTransformer,
        position_embedding: PositionEmbeddingSineHW,
        bert: BertModel,
        feat_map: nn.Linear,
        input_proj: nn.ModuleList,
        transformer: Transformer,
        bbox_embed: nn.ModuleList,
        class_embed: nn.ModuleList,
        num_feature_levels: int = NUM_FEATURE_LEVELS,
    ) -> None:
        super().__init__()
        self.backbone_model = backbone
        self.position_embedding = position_embedding
        self.bert = bert
        self.feat_map = feat_map
        self.input_proj = input_proj
        self.transformer = transformer
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed
        self.num_feature_levels = num_feature_levels


class GroundingDINOTLoader:
    """Helper for loading the QNN-compatible GroundingDINO-T module graph."""

    @classmethod
    def load(
        cls,
        ckpt: str | None = None,
        host_device: torch.device | str = torch.device("cpu"),
    ) -> tuple[
        _GroundingDINOTParts,
        GroundingDINOTCollectionVision,
        GroundingDINOTCollectionBert,
        GroundingDINOTCollectionFusionDecoder,
    ]:
        """Build, patch, and split the shared graph into collection components."""
        model_parts = cls._load_model_parts(ckpt).to(torch.device(host_device)).eval()
        return (
            model_parts,
            GroundingDINOTCollectionVision.from_full_model(model_parts),
            GroundingDINOTCollectionBert.from_full_model(model_parts),
            GroundingDINOTCollectionFusionDecoder.from_full_model(model_parts),
        )

    @classmethod
    def _load_model_parts(cls, ckpt: str | None = None) -> _GroundingDINOTParts:
        cls._patch_upstream_classes()

        model_parts = cls._build_model_parts()
        weights_path = ckpt if ckpt is not None else DEFAULT_WEIGHTS.fetch()
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = {
            k.removeprefix("module."): v for k, v in raw.get("model", raw).items()
        }
        apply_patches(
            model_parts,
            state_dict,
            spatial_shapes_list=_feature_map_sizes(IMG_HEIGHT, IMG_WIDTH),
        )

        return model_parts.eval()

    @classmethod
    def _patch_upstream_classes(cls) -> None:
        # Pre-construction patches (P8-P11): see apply_patches()'s docstring
        # in model_patches.py.
        apply_patches()

    @classmethod
    def _build_model_parts(cls) -> _GroundingDINOTParts:
        # Build backbone; depths/num_heads/window_size/embed_dim match the
        # GroundingDINO-SwinT-OGC default config.
        backbone = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.2,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            dilation=False,
        )

        position_embedding = PositionEmbeddingSineHW(
            num_pos_feats=128,
            # temperatureH/W=20 (vs. the class default of 10000) matches
            # GroundingDINO_SwinT_OGC.py's pe_temperatureH/W config.
            temperatureH=20,
            temperatureW=20,
            normalize=True,
        )

        bert = BertModel(
            BertConfig.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
        )
        feat_map = nn.Linear(768, 256)

        # FPN input projections: Swin-T outputs channels [192, 384, 768] at
        # indices 1,2,3; we need 4 levels (256 dim each). GroupNorm uses
        # _ExportSafeGroupNorm (P13 in model_patches.py), not nn.GroupNorm --
        # see model_patches.py's module docstring for why.
        input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(192, 256, kernel_size=1), _ExportSafeGroupNorm(32, 256)
                ),
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=1), _ExportSafeGroupNorm(32, 256)
                ),
                nn.Sequential(
                    nn.Conv2d(768, 256, kernel_size=1), _ExportSafeGroupNorm(32, 256)
                ),
                nn.Sequential(
                    nn.Conv2d(768, 256, kernel_size=3, stride=2, padding=1),
                    _ExportSafeGroupNorm(32, 256),
                ),
            ]
        )

        transformer = Transformer(
            d_model=256,
            nhead=8,
            num_queries=900,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.0,
            num_feature_levels=4,
            enc_n_points=4,
            dec_n_points=4,
            two_stage_type="standard",
            embed_init_tgt=True,
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_text_cross_attention=True,
            text_dropout=0.0,
            fusion_dropout=0.0,
            fusion_droppath=0.1,
            return_intermediate_dec=True,
            learnable_tgt_init=True,
        )

        bbox_embed = nn.ModuleList([MLP(256, 256, 4, 3) for _ in range(6)])
        class_embed = nn.ModuleList(
            [ContrastiveEmbed(max_text_len=MAX_TEXT_LEN) for _ in range(6)]
        )

        # Wire bbox/class embed into decoder and transformer encoder head.
        transformer.decoder.bbox_embed = bbox_embed
        transformer.enc_out_class_embed = ContrastiveEmbed(max_text_len=MAX_TEXT_LEN)
        transformer.enc_out_bbox_embed = MLP(256, 256, 4, 3)

        return _GroundingDINOTParts(
            backbone=backbone,
            position_embedding=position_embedding,
            bert=bert,
            feat_map=feat_map,
            input_proj=input_proj,
            transformer=transformer,
            bbox_embed=bbox_embed,
            class_embed=class_embed,
        )


class GroundingDINOTCollectionVision(_GroundingDINOTComponentMixin):
    """Image half of GroundingDINO-T: Swin backbone + FPN projections.

    Computes the per-level ``srcs`` tensors that the collection graph
    builds from ``image`` alone, flattened into individually-named tensors
    (``src_0..3``) since AI Hub components can't take/return list-shaped I/O.

    ``masks``/``poss`` are deliberately *not* computed or returned here, even
    though the unsplit upstream-style graph computes them alongside
    ``srcs``. At this model's fixed resolution the mask is always all-zero
    by construction (never derived from the letterbox bars ``image`` may
    contain), so both are pure functions of each level's static (H, W) shape and the
    parameter-free ``position_embedding`` module -- not of ``image``'s actual
    values. AI Hub's ONNX-compile-step optimizer recognizes this and folds
    them into constants, but leaves a stale ``value_info`` entry with the
    same name as the (now-constant) graph output, which is an ONNX IR-spec
    violation AI Hub's `QuantizeJob` rejects ("Tensors {...} occur in
    value_info but also in model IO", see
    https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs). Since
    ``mask_i``/``pos_i`` don't depend on anything vision-specific, they're
    instead recomputed inside ``GroundingDINOTCollectionFusionDecoder.forward``
    from ``src_i``'s shape -- as an internal intermediate value (not a declared
    component input/output), so there's no longer any I/O tensor name left
    for AI Hub's optimizer to collide with.
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        input_proj: nn.ModuleList,
        num_feature_levels: int = NUM_FEATURE_LEVELS,
    ) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.input_proj = input_proj
        self.num_feature_levels = num_feature_levels

    @classmethod
    def from_full_model(
        cls, full: _GroundingDINOTParts
    ) -> GroundingDINOTCollectionVision:
        """Wrap references to the relevant submodules of an already-built,
        already QNN-patched module graph. Does not copy weights or
        re-run `apply_patches`.
        """
        return cls(
            full.backbone_model,
            full.input_proj,
            full.num_feature_levels,
        )

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = "DEFAULT") -> Self:
        ckpt = None if checkpoint in ("DEFAULT", "DEFAULT_UNQUANTIZED") else checkpoint
        return cast(Self, GroundingDINOTLoader.load(ckpt)[1])

    def forward(
        self, image: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the Swin backbone + FPN projections on `image`.

        Parameters
        ----------
        image
            NCHW float32 image tensor, shape [bs, 3, IMG_HEIGHT, IMG_WIDTH], range [0, 1].

        Returns
        -------
        src_0 : torch.Tensor
            Level-0 FPN feature tensor.
        src_1 : torch.Tensor
            Level-1 FPN feature tensor.
        src_2 : torch.Tensor
            Level-2 FPN feature tensor.
        src_3 : torch.Tensor
            Level-3 FPN feature tensor.
        """
        image = normalize_image_torchvision(image)

        image_mask = torch.zeros(
            image.shape[0],
            image.shape[2],
            image.shape[3],
            dtype=torch.float32,
            device=image.device,
        )
        samples = NestedTensor(image, image_mask)

        backbone_out = self.backbone_model(samples)

        srcs: list[torch.Tensor] = []
        for level in range(len(backbone_out)):
            feat = backbone_out[level]
            src, _ = feat.decompose()
            srcs.append(self.input_proj[level](src))

        if self.num_feature_levels > len(srcs):
            assert self.num_feature_levels == len(srcs) + 1, (
                "only one extra feature level (beyond the backbone's) is derived here"
            )
            last_key = len(backbone_out) - 1
            raw_last = backbone_out[last_key].tensors
            extra_src = self.input_proj[len(srcs)](raw_last)
            srcs.append(extra_src)

        return (*srcs,)  # type: ignore[return-value]

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, IMG_HEIGHT, IMG_WIDTH),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        spec: OutputSpec = {}
        for i in range(self.num_feature_levels):
            spec[f"src_{i}"] = TensorSpec()
        return spec

    def component_precision(self) -> Precision:
        return Precision.w8a16


class GroundingDINOTCollectionBert(_GroundingDINOTComponentMixin):
    """BERT text-encoder half of GroundingDINO-T.

    Runs the HF `BertModel` text encoder plus the `feat_map` projection to
    produce `encoded_text`.
    """

    def __init__(self, bert: nn.Module, feat_map: nn.Module) -> None:
        super().__init__()
        self.bert = bert
        self.feat_map = feat_map

    @classmethod
    def from_full_model(
        cls, full: _GroundingDINOTParts
    ) -> GroundingDINOTCollectionBert:
        """Wrap references to the relevant submodules of an already-built,
        already QNN-patched module graph. Does not copy weights or
        re-run `apply_patches`.
        """
        return cls(full.bert, full.feat_map)

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = "DEFAULT") -> Self:
        ckpt = None if checkpoint in ("DEFAULT", "DEFAULT_UNQUANTIZED") else checkpoint
        return cast(Self, GroundingDINOTLoader.load(ckpt)[2])

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        text_self_attention_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run BERT text encoding and the feat_map projection.

        Parameters
        ----------
        input_ids
            Tokenized text input ids.
        token_type_ids
            Text token type ids.
        position_ids
            Text position ids.
        text_self_attention_masks
            Text self-attention mask.

        Returns
        -------
        encoded_text : torch.Tensor
            BERT's `last_hidden_state`, projected to the transformer's hidden
            size by `feat_map`.
        """
        # Clamp bounds are bert-base-uncased's BertConfig defaults. The mask is
        # clamped to {0, 1} since HF's extended-attention-mask formula
        # ((1 - mask) * finfo.min) overflows to inf for any other int value.
        bert_out = self.bert(
            input_ids=input_ids.to(torch.int64).clamp(0, 30521),
            attention_mask=text_self_attention_masks.clamp(0, 1),
            token_type_ids=token_type_ids.to(torch.int64).clamp(0, 1),
            position_ids=position_ids.to(torch.int64).clamp(0, 511),
        )
        return self.feat_map(bert_out["last_hidden_state"])

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return {
            "input_ids": TensorSpec(shape=(batch_size, MAX_TEXT_LEN), dtype="int32"),
            "token_type_ids": TensorSpec(
                shape=(batch_size, MAX_TEXT_LEN), dtype="int32"
            ),
            "position_ids": TensorSpec(shape=(batch_size, MAX_TEXT_LEN), dtype="int32"),
            "text_self_attention_masks": TensorSpec(
                shape=(batch_size, MAX_TEXT_LEN, MAX_TEXT_LEN), dtype="int32"
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"encoded_text": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


class GroundingDINOTCollectionFusionDecoder(_GroundingDINOTComponentMixin):
    """Fusion + decoder half of GroundingDINO-T: fusion transformer and
    per-layer box/class heads.

    Consumes the vision component's flattened FPN outputs, the bert
    component's `encoded_text`, and the remaining text tensors, and returns
    `(pred_logits, pred_boxes)` -- identical to the unsplit graph's
    return value.
    """

    def __init__(
        self,
        position_embedding: nn.Module,
        transformer: nn.Module,
        bbox_embed: nn.ModuleList,
        class_embed: nn.ModuleList,
    ) -> None:
        super().__init__()
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed

    @classmethod
    def from_full_model(
        cls, full: _GroundingDINOTParts
    ) -> GroundingDINOTCollectionFusionDecoder:
        """Wrap references to the relevant submodules of an already-built,
        already QNN-patched module graph. Does not copy weights or
        re-run `apply_patches`.
        """
        return cls(
            full.position_embedding, full.transformer, full.bbox_embed, full.class_embed
        )

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = "DEFAULT") -> Self:
        ckpt = None if checkpoint in ("DEFAULT", "DEFAULT_UNQUANTIZED") else checkpoint
        return cast(Self, GroundingDINOTLoader.load(ckpt)[3])

    def forward(
        self,
        encoded_text: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        text_self_attention_masks: torch.Tensor,
        src_0: torch.Tensor,
        src_1: torch.Tensor,
        src_2: torch.Tensor,
        src_3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fusion transformer and box/class heads on the vision
        component's flattened per-level outputs and the bert component's
        `encoded_text`.

        Parameters
        ----------
        encoded_text
            BERT's projected `last_hidden_state`, from `GroundingDINOTCollectionBert`.
        attention_mask
            Text attention mask.
        position_ids
            Text position ids.
        text_self_attention_masks
            Text self-attention mask.
        src_0
            Level-0 FPN feature tensor from `GroundingDINOTCollectionVision`.
        src_1
            Level-1 FPN feature tensor from `GroundingDINOTCollectionVision`.
        src_2
            Level-2 FPN feature tensor from `GroundingDINOTCollectionVision`.
        src_3
            Level-3 FPN feature tensor from `GroundingDINOTCollectionVision`.

        Returns
        -------
        pred_logits : torch.Tensor
            Per-query, per-token contrastive logits from the final decoder layer.
        pred_boxes : torch.Tensor
            Per-query predicted boxes (cx, cy, w, h) from the final decoder layer.
        """
        encoded_text = encoded_text[:, :MAX_TEXT_LEN, :]

        # Slice before casting to bool (not the other way around) -- HTP's
        # StridedSlice op does not support BOOL_8 in/out datatypes, so slicing
        # attention_mask.bool() directly fails compilation with
        # "Unsupported input/output datatypes requested for the HTP Op
        # 'StridedSlice'" on this component's exported graph. Slicing the
        # int32 tensor first (a supported dtype) and casting afterward is
        # numerically identical.
        text_token_mask = attention_mask[:, :MAX_TEXT_LEN].bool()
        pos_ids = position_ids[:, :MAX_TEXT_LEN]
        tsam = text_self_attention_masks[:, :MAX_TEXT_LEN, :MAX_TEXT_LEN]

        text_dict: dict[str, torch.Tensor] = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": pos_ids,
            "text_self_attention_masks": tsam,
        }

        srcs = [src_0, src_1, src_2, src_3]

        # masks/poss are never actually image-dependent (the mask is always
        # all-zero by construction, regardless of any letterbox bars in the
        # image) -- they're recomputed here, as an
        # internal intermediate value, purely from each src_i's static (H, W)
        # shape, rather than being passed in as component inputs. See
        # GroundingDINOTCollectionVision's docstring for why this avoids an
        # ONNX IR-spec violation that AI Hub's QuantizeJob rejects.
        masks: list[torch.Tensor] = []
        poss: list[torch.Tensor] = []
        for src in srcs:
            mask = torch.zeros(
                src.shape[0],
                src.shape[-2],
                src.shape[-1],
                dtype=torch.float32,
                device=src.device,
            )
            masks.append(mask)
            poss.append(
                self.position_embedding(NestedTensor(src, mask.bool())).to(src.dtype)
            )

        hs, reference, _, _, _ = self.transformer(
            srcs, masks, None, poss, None, None, text_dict
        )

        coord_list: list[torch.Tensor] = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(
            reference[:-1], self.bbox_embed, hs, strict=True
        ):
            delta = layer_bbox_embed(layer_hs)
            out = delta + _inverse_sigmoid(layer_ref_sig)
            coord_list.append(out.sigmoid())
        outputs_coord = torch.stack(coord_list)

        outputs_class = torch.stack(
            [
                cls_embed(layer_hs, text_dict)
                for cls_embed, layer_hs in zip(self.class_embed, hs, strict=True)
            ]
        )

        return outputs_class[-1], outputs_coord[-1]

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        spec: InputSpec = {
            "encoded_text": TensorSpec(
                shape=(batch_size, MAX_TEXT_LEN, 256), dtype="float32"
            ),
            "attention_mask": TensorSpec(
                shape=(batch_size, MAX_TEXT_LEN), dtype="int32"
            ),
            "position_ids": TensorSpec(shape=(batch_size, MAX_TEXT_LEN), dtype="int32"),
            "text_self_attention_masks": TensorSpec(
                shape=(batch_size, MAX_TEXT_LEN, MAX_TEXT_LEN), dtype="int32"
            ),
        }
        level_sizes = _feature_map_sizes(IMG_HEIGHT, IMG_WIDTH)
        for i, (h, w) in enumerate(level_sizes):
            spec[f"src_{i}"] = TensorSpec(
                shape=(batch_size, 256, h, w),
                dtype="float32",
            )
        return spec

    def get_output_spec(self) -> OutputSpec:
        return {
            "pred_logits": TensorSpec(),
            "pred_boxes": TensorSpec(),
        }

    def component_precision(self) -> Precision:
        return Precision.w8a16


class GroundingDINOTCollection(WorkbenchModelCollection):
    """GroundingDINO-T collection split into vision, bert, and fusion_decoder
    components. `float` compile all three components at the same
    uniform precision; `mixed_with_float` compiles bert at float and vision +
    fusion_decoder at w8a16.

    `fusion_decoder`'s w8a16 calibration data (GroundingDINOCOCODataset, see
    get_calibration_dataset_cls) always uses the same fixed 80-class COCO
    caption -- only the image varies per calibration sample. The reported
    mAP reflects that same prompt. Short/custom prompts (e.g. the demo's
    default text) exercise text/attention-mask structure the w8a16
    quantizer never saw during calibration and are not separately validated.
    """

    def __init__(
        self,
        vision: GroundingDINOTCollectionVision,
        bert: GroundingDINOTCollectionBert,
        fusion_decoder: GroundingDINOTCollectionFusionDecoder,
    ) -> None:
        super().__init__(
            {"vision": vision, "bert": bert, "fusion_decoder": fusion_decoder}
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | None = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> Self:
        ckpt = None if checkpoint in ("DEFAULT", "DEFAULT_UNQUANTIZED") else checkpoint
        _, vision, bert, fusion_decoder = GroundingDINOTLoader.load(ckpt, host_device)
        return cls(vision, bert, fusion_decoder)

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return GroundingDINOCOCODataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [GroundingDINOCOCODataset]  # type: ignore[type-abstract]

    def get_evaluator(self) -> BaseEvaluator:
        img_height, img_width = self.components["vision"].get_input_spec()["image"][0][
            2:
        ]
        tokenizer = load_tokenizer()
        return GroundingDINOEvaluator(
            image_height=img_height,
            image_width=img_width,
            positive_map=build_positive_map(tokenizer),
        )
