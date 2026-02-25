# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Any

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from qai_hub_models.models._shared.sam2.app import SAM2App, SAM2InputImageLayout
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.testing import assert_most_close


def run_sam2_numerical_test(
    model_cls: type[Any],
    loader_cls: type[Any],
    model_type: str,
    image_address: str,
    mask_comparison_atol: float = 0.005,
    mask_comparison_rtol: float = 0.001,
    squeeze_mask: bool = False,
) -> None:
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    # OOTB SAM Objects
    sam2_without_our_edits = loader_cls._load_sam2(model_type)
    sam2_predictor = SAM2ImagePredictor(sam2_without_our_edits)

    # QAIHM SAMApp
    qaihm_sam2 = model_cls.from_pretrained(model_type)
    qaihm_app = SAM2App(
        qaihm_sam2.encoder.sam2.image_size,
        sam2_predictor.mask_threshold,
        SAM2InputImageLayout["RGB"],
        qaihm_sam2.encoder,
        qaihm_sam2.decoder,
    )

    #
    # Inputs
    #
    input_image_data = np.asarray(load_image(image_address))
    point_coords = torch.tensor([[500, 375], [1100, 600]])
    point_labels = torch.tensor([1, 1])

    #
    # Verify encoder output
    #
    sam2_predictor.set_image(input_image_data)
    (
        qaihm_image_embeddings,
        qaihm_high_res_feat1,
        qaihm_high_res_feat2,
        sparse_embedding,
        input_images_original_size,
    ) = qaihm_app.predict_embeddings(
        input_image_data,
        point_coords,
        point_labels,
    )
    assert_most_close(
        sam2_predictor._features["image_embed"].numpy(),
        qaihm_image_embeddings.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
    assert_most_close(
        sam2_predictor._features["high_res_feats"][0].numpy(),
        qaihm_high_res_feat1.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
    assert_most_close(
        sam2_predictor._features["high_res_feats"][1].numpy(),
        qaihm_high_res_feat2.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )

    # Verify Decoder output
    # Use embeddings from SAM predictor to make sure the inputs to both decoders are the same.

    sam2_pred_masks, sam2_pred_scores, _ = sam2_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

    (
        qaihm_pred_masks,
        qaihm_pred_scores,
    ) = qaihm_app.predict_mask_from_points_and_embeddings(
        qaihm_image_embeddings,
        qaihm_high_res_feat1,
        qaihm_high_res_feat2,
        sparse_embedding,
        input_images_original_size,
    )

    qaihm_pred_masks_np = qaihm_pred_masks.numpy()
    if squeeze_mask:
        qaihm_pred_masks_np = qaihm_pred_masks.squeeze(1).numpy()

    assert_most_close(
        sam2_pred_masks,
        qaihm_pred_masks_np,
        mask_comparison_atol,
        rtol=mask_comparison_rtol,
        atol=mask_comparison_rtol,
    )

    assert_most_close(
        sam2_pred_scores,
        qaihm_pred_scores.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
