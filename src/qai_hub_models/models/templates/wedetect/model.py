# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from qai_hub_models import Precision, SampleInputsType
from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps
from qai_hub_models.extern.mmengine import (
    patch_mmengine_pkgresources,
    patch_mmengine_torch_load_no_weights_only,
)
from qai_hub_models.models.templates.wedetect.app import tokenize_class_names
from qai_hub_models.models.templates.wedetect.constants import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_CLASSES,
    INPUT_IMAGE_SIZE,
    TEXT_EMBEDDING_DIM,
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
from qai_hub_models.utils.labels import get_class_names

with patch_mmdet_no_build_deps():
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    from mmengine.config import Config


class BaseTextEncoder(BaseModel):
    """Exportable XLM-RoBERTa text encoder for WeDetect.

    ``forward(input_ids, attention_mask)`` runs the transformer, takes the
    ``[CLS]`` hidden state, projects to 768 dims, and L2-normalises.
    Tokenisation is not compiled — it lives on ``self.tokenizer`` and is
    invoked by the app before this component runs.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        num_classes: int = DEFAULT_NUM_CLASSES,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        mm_base: Any = base_model
        lang_backbone = mm_base.backbone.text_model
        self.model = lang_backbone.model
        self.head = lang_backbone.head
        self.tokenizer: PreTrainedTokenizerBase = lang_backbone.tokenizer
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids
            Shape ``[num_classes, seq_len]`` int32.
        attention_mask
            Shape ``[num_classes, seq_len]`` int32.

        Returns
        -------
        torch.Tensor
            Shape ``[num_classes, 768]`` float32, L2-normalised.
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = out.last_hidden_state[:, 0]
        projected = self.head(cls_hidden)
        return F.normalize(projected, dim=-1)

    def get_input_spec(
        self,
        num_classes: int | None = None,
        seq_len: int | None = None,
    ) -> InputSpec:
        n = num_classes if num_classes is not None else self.num_classes
        s = seq_len if seq_len is not None else self.max_seq_len
        return {
            "input_ids": TensorSpec(
                shape=(n, s),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
            "attention_mask": TensorSpec(
                shape=(n, s),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        spec = input_spec or self.get_input_spec()
        n, s = spec["input_ids"][0]
        input_ids, attention_mask = tokenize_class_names(
            self.tokenizer, get_class_names("coco"), n, s
        )
        return {
            "input_ids": [input_ids.numpy()],
            "attention_mask": [attention_mask.numpy()],
        }

    def component_precision(self) -> Precision:
        return Precision.float

    def get_output_spec(self) -> OutputSpec:
        return {"txt_feats": TensorSpec()}


class BaseDetector(BaseModel):
    """Exportable WeDetect text-conditioned object detector."""

    def __init__(
        self,
        model: torch.nn.Module,
        strides: tuple[int, ...] = (8, 16, 32),
        include_postprocessing: bool = True,
    ) -> None:
        super().__init__(model)
        self.strides = strides
        self.include_postprocessing = include_postprocessing

    @staticmethod
    def _make_grid(
        h: int,
        w: int,
        stride: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        cx = (x + 0.5) * stride
        cy = (y + 0.5) * stride
        return cx, cy

    def forward(
        self,
        image: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Run WeDetect on a batch of images and decode multi-scale predictions.

        Parameters
        ----------
        image
            Input images. Shape ``[B, 3, 640, 640]``, float32, range ``[0, 1]``.
        txt_feats
            Text embeddings from encoder. Shape: ``[B, num_classes, 768]``

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]
            ``(boxes, scores, labels)`` when ``include_postprocessing=True``,
            or ``(boxes, scores)`` otherwise.
        """
        _m: Any = self.model
        img_feats = _m.backbone.forward_image(image)
        if _m.with_neck:
            if _m.mm_neck:
                img_feats = _m.neck(img_feats, txt_feats)
            else:
                img_feats = _m.neck(img_feats)
        cls_heads, reg_heads = _m.bbox_head.forward(img_feats, txt_feats)

        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for cls, reg, stride in zip(cls_heads, reg_heads, self.strides, strict=False):
            B, _C, H, W = cls.shape
            device = cls.device

            score_map = cls.sigmoid()
            scores, labels = score_map.max(dim=1)

            cx, cy = self._make_grid(H, W, stride, device)

            x1 = cx - reg[:, 0] * stride
            y1 = cy - reg[:, 1] * stride
            x2 = cx + reg[:, 2] * stride
            y2 = cy + reg[:, 3] * stride

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            all_boxes.append(boxes.reshape(B, -1, 4))
            all_scores.append(scores.reshape(B, -1))
            all_labels.append(labels.reshape(B, -1))

        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        labels = torch.cat(all_labels, dim=1)

        if not self.include_postprocessing:
            return boxes, scores
        return boxes, scores, labels.to(torch.uint8)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = INPUT_IMAGE_SIZE,
        width: int = INPUT_IMAGE_SIZE,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(color_format=ColorFormat.RGB),
            ),
            "txt_feats": TensorSpec(
                shape=(batch_size, num_classes, TEXT_EMBEDDING_DIM),
                dtype="float32",
            ),
        }

    def get_channel_last_inputs(self) -> list[str]:
        return ["image"]


def load_wedetect_model(
    config_path: str,
    weight_path: str,
    repo_root: str,
) -> torch.nn.Module:
    """Load and prepare a WeDetect base model."""
    register_all_modules()

    cfg = Config.fromfile(config_path)

    # The config stores the XLMRoberta weights as a relative path (e.g. "./xlm-roberta-base/").
    # Resolve it to absolute so HuggingFace can find the files regardless of cwd.
    text_model_cfg = cfg.model.backbone.text_model
    if hasattr(text_model_cfg, "model_name") and not os.path.isabs(
        text_model_cfg.model_name
    ):
        text_model_cfg.model_name = os.path.normpath(
            os.path.join(repo_root, text_model_cfg.model_name)
        )

    with patch_mmengine_torch_load_no_weights_only(), patch_mmengine_pkgresources():
        base_model = init_detector(
            cfg, str(weight_path), device="cpu", palette="random"
        )

    base_model.eval()
    return base_model
