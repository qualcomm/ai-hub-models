# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, SiglipModel
from typing_extensions import Self

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.datasets.imagenet.imagenet_zeroshot import (
    ImagenetZeroshotDataset,
)
from qai_hub_models.models.siglip2.evaluator import (
    SigLIP2Evaluator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)
from qai_hub_models.utils.labels import get_class_names

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "google/siglip2-base-patch16-224"

# Fixed text sequence length used for on-device export.
# SigLIP2 uses a Gemma tokenizer with no built-in max length;
# 64 tokens comfortably covers standard zero-shot classification prompts.
TEXT_SEQ_LEN = 64


@contextlib.contextmanager
def patched_in_projection_packed() -> Generator[None]:
    """
    Avoid unflatten that causes ONNX export failure.
    https://github.com/pytorch/pytorch/issues/135764

    SigLIP2's MultiheadAttentionPoolingHead uses torch.nn.MultiheadAttention,
    whose _in_projection_packed emits onnx::Unsqueeze on a tensor of unknown
    rank. Patching it to use an explicit split avoids the issue, matching the
    approach used in openai_clip.
    """
    original = torch.nn.functional._in_projection_packed  # type: ignore[attr-defined]

    def _patched(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        E = q.size(-1)
        if k is v and q is k:
            proj = F.linear(q, w, b)
            q_proj, k_proj, v_proj = proj.chunk(3, dim=-1)
            return q_proj.contiguous(), k_proj.contiguous(), v_proj.contiguous()
        if k is v:
            # Cross-attention: k == v but q != k (SiglipMultiheadAttentionPoolingHead path).
            # Avoids unflatten+unsqueeze+transpose on unknown rank which breaks ONNX export.
            w_q, w_kv = w.split([E, E * 2])
            b_q, b_kv = (None, None) if b is None else b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            k_proj, v_proj = kv_proj.chunk(2, dim=-1)
            return q_proj, k_proj.contiguous(), v_proj.contiguous()
        return original(q, k, v, w, b)

    torch.nn.functional._in_projection_packed = _patched  # type: ignore[attr-defined]
    try:
        yield
    finally:
        torch.nn.functional._in_projection_packed = original  # type: ignore[attr-defined]


class SigLIP2ImageEncoder(BaseModel):
    """
    SigLIP2 image encoder component.

    Takes a batch of images and returns L2-normalised image embeddings.

    Input contract:
      - image: float32 [B, 3, 224, 224], RGB, values in [0, 1]

    The model normalises the image to [-1, 1] (mean=0.5, std=0.5) internally
    before passing to the vision transformer.
    """

    def __init__(self, model: SiglipModel) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = AutoModel.from_pretrained(weights)
        net.eval()
        return cls(net)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image
            Float32 tensor [B, 3, 224, 224], RGB, values in [0, 1].

        Returns
        -------
        image_embeds : torch.Tensor
            L2-normalised image embeddings, shape [B, EMBED_DIM].
        """
        image = image * 2.0 - 1.0
        with patched_in_projection_packed():
            out = self.model.get_image_features(pixel_values=image)  # type: ignore[operator]
            image_embeds: Tensor = (
                out.pooler_output if hasattr(out, "pooler_output") else out
            )
        return F.normalize(image_embeds, dim=-1)

    def get_input_spec(
        self,
        image_batch_size: int = 1,
        image_height: int = 224,
        image_width: int = 224,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(image_batch_size, 3, image_height, image_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(color_format=ColorFormat.RGB),
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"image_embeds": TensorSpec()}


class SigLIP2TextEncoder(BaseModel):
    """
    SigLIP2 text encoder component.

    Takes a batch of tokenized text prompts and returns L2-normalised text
    embeddings.

    Input contract:
      - input_ids: int32 [N, TEXT_SEQ_LEN], token ids from AutoTokenizer
    """

    def __init__(self, model: SiglipModel) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = AutoModel.from_pretrained(weights)
        net.eval()
        return cls(net)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids
            Int32 tensor [N, TEXT_SEQ_LEN] of token ids.

        Returns
        -------
        text_embeds : torch.Tensor
            L2-normalised text embeddings, shape [N, EMBED_DIM].
        """
        # SigLIP2 is trained without an attention mask — AutoProcessor does not
        # emit one. This is a hard model requirement, not just an export
        # workaround: SigLIP2 pools by taking the hidden state of the last
        # token (last_hidden_state[:, -1, :]), which with padding="max_length"
        # is always a pad token. For that pad token to produce a meaningful
        # pooled representation it must attend to all real tokens — so full
        # attention over all positions including padding is required.
        #
        # We pass an explicit 4D all-zeros additive mask [B, 1, seq, seq].
        # In the additive (bias) convention: 0.0 = attend, -inf = block.
        # An all-zeros mask means "attend to every position".
        #
        # The 4D shape causes _preprocess_mask_arguments (transformers 5.x)
        # to early-exit and return the mask as-is, bypassing
        # create_bidirectional_mask entirely.  This is intentional: it keeps
        # the ONNX graph free of the IsNaN/Where/GatherND ops that
        # _expand_mask introduces when given a 2D input — ops that have no
        # native HTP implementation and cause CPU fallback on-device.
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        attention_mask = torch.zeros(
            bsz,
            1,
            seq_len,
            seq_len,
            dtype=torch.float32,
            device=input_ids.device,
        )
        with patched_in_projection_packed():
            out = self.model.get_text_features(  # type: ignore[operator]
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_embeds: Tensor = (
                out.pooler_output if hasattr(out, "pooler_output") else out
            )
        return F.normalize(text_embeds, dim=-1)

    def get_input_spec(
        self,
        text_batch_size: int = 1,
        text_seq_len: int = TEXT_SEQ_LEN,
    ) -> InputSpec:
        return {
            "input_ids": TensorSpec(
                shape=(text_batch_size, text_seq_len),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"text_embeds": TensorSpec()}


class SigLIP2(WorkbenchModelCollection):
    """
    SigLIP2 vision-language model for zero-shot image classification.

    Composed of two independently deployable components:
      - image_encoder: SigLIP2ImageEncoder
      - text_encoder: SigLIP2TextEncoder

    The app layer combines the normalised embeddings with the learned
    logit_scale and logit_bias to produce sigmoid-loss logits.
    """

    def __init__(
        self,
        image_encoder: SigLIP2ImageEncoder,
        text_encoder: SigLIP2TextEncoder,
        logit_scale: float,
        logit_bias: float,
    ) -> None:
        super().__init__({"image_encoder": image_encoder, "text_encoder": text_encoder})
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        # torch 2.11.0 MKL-DNN has a bug on CPUs without AVX-512 (e.g. AMD EPYC)
        # that causes SIGFPE in Conv2d with kernel >= 4x4. Disabling MKL-DNN
        # falls back to native PyTorch kernels with identical numerics.
        torch.backends.mkldnn.set_flags(_enabled=False)
        net: SiglipModel = AutoModel.from_pretrained(weights)
        net.eval()
        logit_scale = float(net.logit_scale.exp().item())
        logit_bias = float(net.logit_bias.item())
        image_encoder = SigLIP2ImageEncoder(net)
        text_encoder = SigLIP2TextEncoder(net)
        return cls(image_encoder, text_encoder, logit_scale, logit_bias)

    def get_input_spec(self, **kwargs: object) -> ComponentGroup[InputSpec]:
        return ComponentGroup(
            {
                "image_encoder": self.image_encoder.get_input_spec(),
                "text_encoder": self.text_encoder.get_input_spec(),
            }
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetZeroshotDataset]

    def get_evaluator(self) -> SigLIP2Evaluator:
        """Return a SigLIP2 evaluator for ImageNet zero-shot classification.

        Both local and on-device eval paths return raw L2-normalised image
        embeddings from run_model_for_eval.  SigLIP2Evaluator converts them
        to logits using pre-computed text embeddings before scoring.
        """
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_WEIGHTS, use_fast=True)
        prompts = [f"a photo of a {lbl}" for lbl in get_class_names("imagenet")]
        token_out = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=TEXT_SEQ_LEN,
        )
        input_ids = token_out["input_ids"].to(torch.int32)
        with torch.no_grad():
            text_embeds = self.text_encoder(input_ids)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return SigLIP2Evaluator(text_embeds, self.logit_scale, self.logit_bias)
