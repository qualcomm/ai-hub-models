# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from torchvision.datasets.coco import CocoDetection
from transformers import AutoTokenizer

from qai_hub_models.datasets.coco.coco import CocoDataset, CocoDatasetClass
from qai_hub_models.datasets.common import DatasetMetadata, DatasetSplit
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    letterbox_resize_pad_pil,
)
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

_LOGGER = logging.getLogger(__name__)

COCO_80_CLASSES: list[str] = get_class_names("coco")

# All-80-class prompt, e.g. "person . bicycle . car . ... . toothbrush ."
COCO_80_CLASS_CAPTION: str = " . ".join(COCO_80_CLASSES) + " ."

# Live here (not model.py) so dataset.py doesn't import back from model.py.
IMG_HEIGHT = 480
IMG_WIDTH = 640
MAX_TEXT_LEN = 256
TOKENIZER_ID = "bert-base-uncased"
# Pinned so tests/demo/scorecard don't depend on HF Hub's current "main" revision.
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def tokenize_caption(
    tokenizer: Any, caption: str, max_text_len: int = MAX_TEXT_LEN
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Tokenise a caption and build special-token-aware attention masks/position ids.

    Delegates segment-boundary logic to upstream's
    ``generate_masks_with_special_tokens_and_transfer_map``, shared with
    ``qai_hub_models.models.grounding_dino_t.app``.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer (e.g. bert-base-uncased).
    caption
        Text prompt to tokenize, e.g. an all-class ". "-joined class list.
    max_text_len
        Tokenizer max length; caption is padded/truncated to this length.

    Returns
    -------
    input_ids : torch.Tensor
        Token ids, shape [1, max_text_len], int32.
    attention_mask : torch.Tensor
        Padding mask, shape [1, max_text_len], int32.
    token_type_ids : torch.Tensor
        BERT segment ids, shape [1, max_text_len], int32.
    position_ids : torch.Tensor
        Per-phrase-segment position ids, shape [1, max_text_len], int32.
    text_self_attention_masks : torch.Tensor
        Segment-aware self-attention mask, shape [1, max_text_len, max_text_len], int32.
    cate_to_token_mask : torch.Tensor
        Shape (num_classes, max_text_len) boolean map from class index to token positions.
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    untruncated_len = len(tokenizer(caption)["input_ids"])
    if untruncated_len > max_text_len:
        _LOGGER.warning(
            f"Caption has {untruncated_len} tokens, exceeding max_text_len="
            f"{max_text_len}; it will be truncated and trailing classes/phrases "
            "will be lost."
        )

    encoded = tokenizer(
        caption,
        max_length=max_text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]  # (1, max_text_len)
    attention_mask = encoded["attention_mask"]  # (1, max_text_len)
    token_type_ids = encoded["token_type_ids"]  # (1, max_text_len)

    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text_self_attention_masks, position_ids, cate_to_token_mask_list = (
        generate_masks_with_special_tokens_and_transfer_map(
            encoded, special_tokens, tokenizer
        )
    )
    # cate_to_token_mask_list[0]: (num_classes, max_text_len) bool
    # Reused by build_positive_map for upstream-consistent token-to-class mapping.
    cate_to_token_mask = cate_to_token_mask_list[0]

    return (
        input_ids.to(torch.int32),  # (1, max_text_len)
        attention_mask.to(torch.int32),  # (1, max_text_len)
        token_type_ids.to(torch.int32),  # (1, max_text_len)
        position_ids.to(torch.int32),  # (1, max_text_len)
        text_self_attention_masks.to(torch.int32),  # (1, max_text_len, max_text_len)
        cate_to_token_mask,  # (num_classes, max_text_len) bool
    )


class GroundingDINOCOCODataset(CocoDataset):
    """
    COCO val2017 dataset for Grounding DINO-T evaluation.

    Inherits CocoDataset for download / validation / label-map handling.

    Parameters
    ----------
    split
        Train or val split.
    max_boxes
        Maximum GT boxes per sample (zero-padded to this size).
    tokenizer
        bert-base-uncased tokenizer.
    input_spec
        Model input spec; determines the (img_height, img_width) this dataset
        letterboxes images to. Defaults to (480, 640) when omitted.
    max_text_len
        Tokenizer max length (default 256).

    Attributes
    ----------
    caption
        All-80-class prompt string e.g. "person . bicycle . ...".
    """

    caption: str

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        max_boxes: int = 100,
        tokenizer: Any = None,
        input_spec: InputSpec | None = None,
        max_text_len: int = MAX_TEXT_LEN,
    ) -> None:
        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, IMG_HEIGHT, IMG_WIDTH))
        }
        self.img_height = input_spec["image"][0][2]
        self.img_width = input_spec["image"][0][3]
        self.max_text_len = max_text_len

        super().__init__(
            split=split,
            input_spec=input_spec,
            max_boxes=max_boxes,
            num_classes=CocoDatasetClass.SUBSET_CLASSES,
        )

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID, revision=TOKENIZER_REVISION
            )

        self.caption = COCO_80_CLASS_CAPTION
        self._tokenizer = tokenizer

        self._cached_text_inputs = tokenize_caption(
            self._tokenizer, self.caption, self.max_text_len
        )

    def _preprocess_image(self, pil_img: PILImage.Image) -> torch.Tensor:
        """Letterbox-resize to (img_width, img_height) and convert to a [0, 1] CHW tensor.

        Uses the shared letterbox_resize_pad_pil helper (also used by
        app.py's inference path) so this stays pixel-exact with the
        evaluator's letterbox unscaling and the demo's preprocessing.

        ImageNet mean/std normalization is applied inside the collection
        graph, not here, matching the repo convention of keeping datasets
        in raw [0, 1] pixel range.
        """
        img = letterbox_resize_pad_pil(pil_img, self.img_height, self.img_width)
        return TF.to_tensor(img)  # (3, img_height, img_width) in [0, 1]

    def __getitem__(  # type: ignore[override]
        self,
        index: int,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Get an item in this dataset.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        model_inputs : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            image_tensor
                Letterboxed image, shape (3, img_height, img_width), RGB in [0, 1].
            input_ids
                Shape (MAX_TEXT_LEN,).
            attention_mask
                Shape (MAX_TEXT_LEN,).
            token_type_ids
                Shape (MAX_TEXT_LEN,).
            position_ids
                Shape (MAX_TEXT_LEN,).
            text_self_attention_masks
                Shape (MAX_TEXT_LEN, MAX_TEXT_LEN).

        ground_truth : tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
            image_id
                Image ID within the original dataset.
            target_height
                Letterboxed image height (img_height). Note: boxes below are
                normalized against the pre-letterbox source image size, not
                this letterboxed size -- the evaluator doesn't use this tuple
                to unnormalize boxes, but a future caller that does must use
                the original (pre-letterbox) image dimensions instead.
            target_width
                Letterboxed image width (img_width). See target_height note.
            boxes
                Shape (max_boxes, 4). Normalized xyxy in [0, 1], relative to
                the pre-letterbox source image size (see target_height note).
            labels
                Shape (max_boxes,). 0-indexed num_classes-class labels.
            num_boxes
                Shape (1,). Number of valid (non-padding) boxes.
        """
        image, target = CocoDetection.__getitem__(self, index)
        src_w, src_h = image.size

        image_tensor = self._preprocess_image(image)

        boxes_list: list[list[float]] = []
        labels_list: list[int] = []

        for ann in target:
            cat_id = int(ann["category_id"])
            if cat_id not in self.class80_label_map:
                continue
            bx, by, bw, bh = ann["bbox"]
            boxes_list.append(
                [
                    bx / src_w,
                    by / src_h,
                    (bx + bw) / src_w,
                    (by + bh) / src_h,
                ]
            )
            labels_list.append(self.class80_label_map[cat_id])

        num_boxes = len(boxes_list)

        if num_boxes == 0:
            boxes = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
            labels = torch.zeros(self.max_boxes, dtype=torch.long)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(
                torch.tensor(boxes_list, dtype=torch.float32),
                (0, 0, 0, self.max_boxes - num_boxes),
                value=0.0,
            )
            labels = F.pad(
                torch.tensor(labels_list, dtype=torch.long),
                (0, self.max_boxes - num_boxes),
                value=0,
            )

        (
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            text_self_attention_masks,
            _,
        ) = self._cached_text_inputs

        return (
            image_tensor,
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            token_type_ids.squeeze(0),
            position_ids.squeeze(0),
            text_self_attention_masks.squeeze(0),
        ), (
            self.ids[index],
            self.img_height,
            self.img_width,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    @classmethod
    def dataset_name(cls) -> str:
        return "grounding_dino_coco"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="COCO val2017 — one sample per image, all 80 classes per query",
        )

    @staticmethod
    def default_samples_per_job() -> int:
        return 300


def build_positive_map(
    tokenizer: Any, max_text_len: int = MAX_TEXT_LEN
) -> torch.Tensor:
    """Build the (num_classes, max_text_len) float32 positive map from a tokenizer alone.

    Does not require COCO assets or dataset initialisation, so it can be
    used directly by get_evaluator() without a GroundingDINOCOCODataset instance.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer (e.g. bert-base-uncased).
    max_text_len
        Token sequence length to build the map against.

    Returns
    -------
    positive_map : torch.Tensor
        Shape (num_classes, max_text_len), float32.
    """
    # Use the upstream transfer map from tokenize_caption — single
    # implementation of token-to-class mapping, numerically consistent
    # with upstream GroundingDINO evaluation.
    *_, cate_to_token_mask = tokenize_caption(
        tokenizer, COCO_80_CLASS_CAPTION, max_text_len
    )
    # cate_to_token_mask: (num_classes, max_text_len) bool
    # Normalise each row by the number of tokens in that class phrase.
    counts = cate_to_token_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    return cate_to_token_mask.float() / counts
