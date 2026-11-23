# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator

import clip
import torch
import torch.nn.functional as F
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torch import Tensor
from typing_extensions import Self

from qai_hub_models.models.openai_clip.dataset import (
    DEFAULT_IMAGE_SIZE,
    ClipImagenetZeroshotDataset,
)
from qai_hub_models.models.openai_clip.model_patches import (
    _patched_build_attention_mask,
)
from qai_hub_models.utils.asset_loaders import callback_with_retry
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

PRETRAINED_WEIGHTS = "ViT-B/16"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_CONTEXT_LENGTH = 77

_CLIP_NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)


# OpenAI's "Prompt Engineering for ImageNet" notebook https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# ran sequential forward selection over its full 80-template ensemble;
# this 7-template subset scored slightly higher than the full 80, especially for smaller models.
CLIP_ZEROSHOT_TEMPLATES: list[str] = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

# Held-out templates from the CLIP 80-template ensemble not present in
# CLIP_ZEROSHOT_TEMPLATES. Used for text encoder calibration so the
# calibration prompts are disjoint from the eval prompt set.
CLIP_CALIBRATION_TEMPLATES: list[str] = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a blurry photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of the {}.",
    "a sketch of a {}.",
    "a drawing of a {}.",
    "a painting of a {}.",
]


def build_ensembled_text_embeds(
    class_names: list[str],
    templates: list[str],
    encode: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Build one text embedding per class by averaging its embedding across
    multiple prompt templates

    Parameters
    ----------
    class_names
        Class names to build prompts for.
    templates
        Prompt templates; each formatted with a single class name via ``.format``.
    encode
        Maps tokenized prompts ``[N, 77]`` to L2-normalized text embeddings ``[N, D]``.

    Returns
    -------
    text_embeds : torch.Tensor
        L2-normalized text embeddings, one per class. Shape [len(class_names), D].
    """
    prompts = [template.format(name) for name in class_names for template in templates]
    tokens = torch.cat([clip.tokenize(p) for p in prompts])  # [C*T, 77]
    embeds = encode(
        tokens
    )  # [C*T, D] — one job; _run_encoder splits into batch-1 entries
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    embeds = embeds.view(len(class_names), len(templates), -1)
    class_embeds = embeds.mean(dim=1)
    return class_embeds / class_embeds.norm(dim=-1, keepdim=True)


def normalize_clip_image(
    image_tensor: torch.Tensor,
    image_tensor_has_batch: bool = True,
) -> torch.Tensor:
    """
    Normalizes an image tensor using CLIP's channel mean and std constants.

    Parameters
    ----------
    image_tensor
        Image tensor to normalize, values in [0.0, 1.0].
    image_tensor_has_batch
        Whether the tensor has a leading batch dimension.

    Returns
    -------
    normalized_tensor : torch.Tensor
        Normalized image tensor.
    """
    shape = [-1, 1, 1]
    if image_tensor_has_batch:
        shape.insert(0, 1)
    mean = torch.tensor(
        _CLIP_NORMALIZE_MEAN, device=image_tensor.device, dtype=image_tensor.dtype
    ).reshape(*shape)
    std = torch.tensor(
        _CLIP_NORMALIZE_STD, device=image_tensor.device, dtype=image_tensor.dtype
    ).reshape(*shape)
    return (image_tensor - mean) / std


class ClipImageEncoder(BaseModel):
    def __init__(self, clip_model: CLIP) -> None:
        """Wrapper for the CLIP image encoder."""
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode one image.

        Parameters
        ----------
        image
            Raw image tensor, values in [0.0, 1.0] (ToTensor output, no normalization applied).
            Shape: [1, 3, H, W]  Channel Layout: RGB

        Returns
        -------
        image_features : torch.Tensor
            L2-normalized image embeddings. Shape: [1, D]
        """
        with patched_in_projection_packed():
            device = next(self.visual.parameters()).device
            # Get dtype from the first layer (conv1) which is fp16 in mixed-precision models
            first_layer_dtype = self.visual.conv1.weight.dtype
            image = image.to(device=device, dtype=first_layer_dtype)
            image = normalize_clip_image(image)
            image_features = self.visual(image)
        return image_features / image_features.norm(dim=1, keepdim=True)

    def get_input_spec(
        self,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
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
        return {"image_features": TensorSpec()}

    @classmethod
    def from_pretrained(cls) -> ClipImageEncoder:
        return OpenAIClip.from_pretrained().image_encoder


class ClipTextEncoder(BaseModel):
    def __init__(self, clip_model: CLIP) -> None:
        """Wrapper for the CLIP text encoder."""
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        # Register as buffers/parameters so they move with .to(device) and are
        # included in state_dict, but are not nn.Module submodules.
        self.positional_embedding = clip_model.positional_embedding
        self.text_projection = clip_model.text_projection
        self.eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode one text token sequence.

        Parameters
        ----------
        text
            Tokenized text tensor, one caption per forward call.
            Shape: [1, 77]

        Returns
        -------
        text_features : torch.Tensor
            L2-normalized text embeddings. Shape: [1, D]
        """
        with patched_in_projection_packed():
            device = next(self.transformer.parameters()).device
            dtype = next(self.transformer.parameters()).dtype
            clipped_text = torch.clip(text, min=0, max=self.eot_token).to(device)
            x = self.token_embedding(clipped_text).type(dtype)
            x = x + self.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(dtype)
            x = (
                x[torch.arange(x.shape[0]), clipped_text.argmax(dim=-1)]
                @ self.text_projection
            )
        return x / x.norm(dim=1, keepdim=True)

    def get_input_spec(
        self,
        text_batch_size: int = 1,
    ) -> InputSpec:
        return {
            "text": TensorSpec(
                shape=(text_batch_size, DEFAULT_CONTEXT_LENGTH),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"text_features": TensorSpec()}

    @classmethod
    def from_pretrained(cls) -> ClipTextEncoder:
        return OpenAIClip.from_pretrained().text_encoder


class OpenAIClip(WorkbenchModelCollection):
    def __init__(
        self,
        image_encoder: ClipImageEncoder,
        text_encoder: ClipTextEncoder,
        logit_scale: float = 100.0,
    ) -> None:
        super().__init__({"image_encoder": image_encoder, "text_encoder": text_encoder})
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # logit_scale: Temperature scale applied to cosine similarities.
        # Released CLIP weights saturate the training cap at 100.0.
        self.logit_scale = logit_scale

    @classmethod
    def from_pretrained(cls) -> Self:
        def load_clip() -> CLIP:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            CLIP.build_attention_mask = _patched_build_attention_mask
            net, _ = clip.load(PRETRAINED_WEIGHTS, device=device)
            return net

        net = callback_with_retry(num_retries=5, callback=load_clip)
        assert isinstance(net, CLIP)
        logit_scale = float(net.logit_scale.exp().item())
        return cls(ClipImageEncoder(net), ClipTextEncoder(net), logit_scale)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ClipImagenetZeroshotDataset]

    def get_evaluator(self) -> None:
        raise NotImplementedError("Must be implemented by the app")


@contextlib.contextmanager
def patched_in_projection_packed() -> Generator[None]:
    """
    Avoid unflatten that causes ONNX export failure.
    https://github.com/pytorch/pytorch/issues/135764
    """
    original_in_projection_packed = torch.nn.functional._in_projection_packed  # type: ignore[attr-defined]

    def patched_in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        E = q.size(-1)
        if k is v and q is k:
            proj = F.linear(q, w, b)
            proj = proj.view(*proj.shape[:-1], 3, E).permute((2, 0, 1, 3)).contiguous()
            return proj[0], proj[1], proj[2]
        return original_in_projection_packed(q, k, v, w, b)

    torch.nn.functional._in_projection_packed = patched_in_projection_packed  # type: ignore[attr-defined]

    try:
        yield
    finally:
        torch.nn.functional._in_projection_packed = original_in_projection_packed  # type: ignore[attr-defined]
