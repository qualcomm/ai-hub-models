# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch

from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    nms_cpu,
    xywhr2xyxyr,
)

with patch_mmdet_no_build_deps():
    from mmdet.models.layers import PatchMerging


def bev_pool(
    feats: torch.Tensor,
    coords: torch.Tensor,
    ranks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pools features into BEV space using sorted ranks.

    Adapted from : https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/
    mmdet3d/ops/bev_pool/src/bev_pool_cpu.cpp#L22-47

    Summary
        - The official code uses a custom C++/CUDA op (bev_pool_ext) + QuickCumsum.
        So here we reimplemented pooling using pure PyTorch (no CUDA extensions) and
        the cumulative sum logic is moved outside of the model (into app.py) due to memory efficiency.

    Parameters
    ----------
    feats
        Input features, shape (N, C).
    coords
        Coordinates, shape (N, 3).
    ranks
        Sorted indices, shape (N,).

    Returns
    -------
    x : torch.Tensor
        Pooled features, shape (N', C).
    lengths : torch.Tensor
        Group lengths, shape (N').
    geom_feats : torch.Tensor
        Filtered coordinates, shape (2, N').
    """
    geom_feats = coords

    x = feats

    kept = 1 - (ranks[1:] == ranks[:-1]).int()

    # Filters non-zero elements in kept, 58999 was found to be memory-efficient through testing, as dynamic sizing is slower
    interval_starts = torch.topk(kept, 58999)[1].sort()[0].long()
    starts = torch.concat([torch.tensor([0]).long(), interval_starts])
    geom_feats = geom_feats[:, starts + 1]  # .split(1,0)

    # Custom Bev Pool
    # out = torch.zeros((B * D, H, W, x.size(1))).to(x.device)

    ends = torch.concat([interval_starts, torch.tensor([len(interval_starts)]).long()])

    lengths = ends - starts

    return x, lengths, geom_feats


def patched_centerhead_get_task_detections(
    self: Any,
    num_class_with_bg: int,
    batch_cls_preds: list[torch.Tensor],
    batch_reg_preds: list[torch.Tensor],
    batch_cls_labels: list[torch.Tensor],
    metas: dict | None = None,
    nms_scale: list[float] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """
    Rotate nms for each task.

    Parameters
    ----------
    self
        Instance reference.
    num_class_with_bg
        Number of classes for the current task.
    batch_cls_preds
        Prediction score with the shape of [N].
    batch_reg_preds
        Prediction bbox with the shape of [N, 9].
    batch_cls_labels
        Prediction label with the shape of [N].
    metas
        Meta information of each sample.
    nms_scale
        NMS scale factors.

    Returns
    -------
    list[dict[str, torch.Tensor]]
        Contains the following keys:
        - bboxes: Prediction bboxes after nms with the shape of [N, 9].
        - scores: Prediction scores after nms with the shape of [N].
        - labels: Prediction labels after nms with the shape of [N].

    Notes
    -----
    Source : https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5
    /mmdet3d/models/heads/bbox/centerpoint.py#L759C9-L759C28
    """
    if nms_scale is None:
        nms_scale = [1.0]
    predictions_dicts = []
    post_center_range = self.test_cfg["post_center_limit_range"]
    if len(post_center_range) > 0:
        post_center_range = torch.tensor(
            post_center_range,
            dtype=batch_reg_preds[0].dtype,
            device=batch_reg_preds[0].device,
        )

    for box_preds, cls_preds, cls_labels in zip(
        batch_reg_preds, batch_cls_preds, batch_cls_labels, strict=False
    ):
        # Apply NMS in birdeye view

        # get highest score per prediction, than apply nms
        # to remove overlapped box.
        if num_class_with_bg == 1:
            top_scores = cls_preds.squeeze(-1)
            top_labels = torch.zeros(
                cls_preds.shape[0], device=cls_preds.device, dtype=torch.long
            )

        else:
            top_labels = cls_labels.long()
            top_scores = cls_preds.squeeze(-1)

        if self.test_cfg["score_threshold"] > 0.0:
            thresh = torch.tensor(
                [self.test_cfg["score_threshold"]], device=cls_preds.device
            ).type_as(cls_preds)
            top_scores_keep = top_scores >= thresh
            top_scores = top_scores.masked_select(top_scores_keep)

        if top_scores.shape[0] != 0:
            if self.test_cfg["score_threshold"] > 0.0:
                box_preds = box_preds[top_scores_keep]
                top_labels = top_labels[top_scores_keep]

            # Begin Qualcomm modification
            # Removed dependency on metadata-driven box_type_3d
            bev_box = box_preds[:, :]
            bev_box = bev_box[:, [0, 1, 3, 4, 6]]
            # End Qualcomm modification

            for cls, scale in enumerate(nms_scale):
                cur_bev_box = bev_box[top_labels == cls]
                cur_bev_box[:, [2, 3]] *= scale
                bev_box[top_labels == cls] = cur_bev_box
            boxes_for_nms = xywhr2xyxyr(bev_box)

            # the nms in 3d detection just remove overlap boxes.
            # Begin Qualcomm modification
            # Replaced nms_gpu to custom nms_cpu implementation
            selected = nms_cpu(
                boxes_for_nms,
                top_scores,
                thresh=self.test_cfg["nms_thr"],
                pre_maxsize=self.test_cfg["pre_max_size"],
                post_max_size=self.test_cfg["post_max_size"],
            )
        else:
            selected = []

        # if selected is not None:
        selected_boxes = box_preds[selected]
        selected_labels = top_labels[selected]
        selected_scores = top_scores[selected]

        # finally generate predictions.
        if selected_boxes.shape[0] != 0:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            if post_center_range is not None:
                mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                predictions_dict = dict(
                    bboxes=final_box_preds[mask],
                    scores=final_scores[mask],
                    labels=final_labels[mask],
                )
            else:
                predictions_dict = dict(
                    bboxes=final_box_preds, scores=final_scores, labels=final_labels
                )
        else:
            dtype = batch_reg_preds[0].dtype
            device = batch_reg_preds[0].device
            predictions_dict = dict(
                bboxes=torch.zeros(
                    [0, self.bbox_coder.code_size], dtype=dtype, device=device
                ),
                scores=torch.zeros([0], dtype=dtype, device=device),
                labels=torch.zeros([0], dtype=top_labels.dtype, device=device),
            )

        predictions_dicts.append(predictions_dict)
    return predictions_dicts


def patched_topk(
    self: Any, scores: torch.Tensor, K: int = 80
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get indexes based on scores.

    Selects the top K scores from the heatmap, computes corresponding indices, class IDs,
    and spatial coordinates (x, y) in the feature map.

    Parameters
    ----------
    self
        Instance reference.
    scores
        Scores with shape [B, N, W, H].
    K
        Number of top scores to keep. Defaults to 80.

    Returns
    -------
    topk_score : torch.Tensor
        Selected scores with shape [B, K].
    topk_inds : torch.Tensor
        Selected indices with shape [B, K].
    topk_clses : torch.Tensor
        Selected class indices with shape [B, K].
    topk_ys : torch.Tensor
        Selected y-coordinates with shape [B, K].
    topk_xs : torch.Tensor
        Selected x-coordinates with shape [B, K].
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    # Begin Qualcomm modification:
    # Adjusted index calculation to subtract instead of modulo for efficiency.
    topk_inds = topk_inds.sub(topk_inds // (height * width) * (height * width))
    # End Qualcomm modification

    topk_xs = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()
    topk_ys = topk_inds.sub(topk_inds // width * width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # Begin Qualcomm modification:
    # Changed class calculation to use division without int() for ONNX compatibility.
    topk_clses = topk_ind / torch.tensor(K, dtype=torch.float)
    # End Qualcomm modification
    topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def patched_lss_forward(
    self: Any,
    x: torch.Tensor,
    intrins: torch.Tensor,
    camera2lidars: torch.Tensor,
    inv_post_rots: torch.Tensor,
    post_trans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes camera features and geometric transformations up to ranking.

    Differences from source:
        - Removes multi-sensor inputs (lidar, radar, lidar augmentation matrices).
        - Simplifies geometry computations with per-camera frustum processing.
        - Removes mmdet3d.ops BEV pooling op with Custom Bevpool operation.
        - Optimizes for Qualcomm deployment with single-batch processing and reduced dependencies.

    Parameters
    ----------
    self
        Instance reference.
    x
        Feature tensor of shape (batch_size, 6, 256, 32, 88).
    intrins
        Camera intrinsics of shape (batch_size, 6, 3, 3).
    camera2lidars
        Camera-to-LiDAR transformations of shape (batch_size, 6, 4, 4).
    inv_post_rots
        Inverse rotation matrices of shape (batch_size, 6, 3, 3).
    post_trans
        Post-transformation translations of shape (batch_size, 6, 1, 3).

    Returns
    -------
    x : torch.Tensor
        Feature tensor of shape (-1, 80).
    geom_feats : torch.Tensor
        Geometric features of shape (2, -1).
    ranks : torch.Tensor
        Sorted ranks tensor of shape (-1,).

    Notes
    -----
    Source : https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/
    mmdet3d/models/vtransforms/base.py#L179
    """
    # Begin Qualcomm modification
    camera2lidar_rots = camera2lidars[..., :3, :3]
    camera2lidar_trans = camera2lidars[..., :3, 3]

    cam_feats = cast(list[torch.Tensor], self.get_cam_feats(x.squeeze(0)))

    B, N, _ = camera2lidar_trans.shape
    frustum = self.frustum

    H, L, B, _ = frustum.shape
    W = L * B
    frustum = frustum.view(H, W, 3)

    post_trans = post_trans.view(1, N, 1, 3).permute(0, 3, 2, 1).split(1, -1)
    inv_post_rots = inv_post_rots.permute(2, 3, 0, 1).split(1, -1)
    combine = camera2lidar_rots.matmul(intrins).permute(2, 3, 0, 1).split(1, -1)
    camera2lidar_trans = camera2lidar_trans.view(N, 3, 1, 1).split(1, 0)

    points_list = []
    for i in range(N):
        points = frustum.permute(2, 0, 1) - post_trans[i].repeat(1, 1, H, W)
        points = (points * inv_post_rots[i]).sum(1).unsqueeze(0)
        points = torch.cat(
            (
                points[:, :2, :, :] * points[:, 2:3, :, :],
                points[:, 2:3, :, :],
            ),
            1,
        )
        points = (combine[i] * points).sum(1)
        points += camera2lidar_trans[i].squeeze(0)
        points_list.append(points[0:2])

    geom_feats_list = points_list

    ranks_list = []
    for i in range(len(geom_feats_list)):
        geom_feats = (
            (geom_feats_list[i] - (self.bx[:2] - self.dx[:2] / 2.0).view(2, 1, 1))
            / self.dx[:2].view(2, 1, 1)
        ).int()
        kept = (
            (
                (geom_feats >= torch.tensor(0).int())
                & (geom_feats < self.nx[:2].view(2, 1, 1).int())
            )
            .int()
            .sum(0)
            == torch.tensor(2).int()
        ).int()
        cam_feats[i] = cam_feats[i] * kept.unsqueeze(-1)
        geom_feats = geom_feats * kept
        geom_feats_list[i] = geom_feats.unsqueeze(1)
        geom_feats = geom_feats.split(1)
        ranks = geom_feats[0] * (self.nx[0] * self.nx[2] * B) + geom_feats[1] * (
            self.nx[2] * B
        )
        ranks_list.append(ranks)

    cam_feats_cat = torch.concat(cam_feats, 0)
    geom_feats = torch.concat(geom_feats_list, 1)
    ranks = torch.concat(ranks_list, 0)
    ranks = ranks.reshape(-1)
    ranks, indices = ranks.sort()
    cam_feats_cat = cam_feats_cat.reshape(-1, 80)
    cam_feats_cat = cam_feats_cat[indices]
    geom_feats = geom_feats.reshape(2, -1)[:, indices]
    # End Qualcomm Modification
    return cam_feats_cat, geom_feats, ranks


def patched_get_cam_feats(self: Any, x: torch.Tensor) -> list[torch.Tensor]:
    """
    Extracts camera features and depth using depthnet, combining them for BEV projection.

    Parameters
    ----------
    self
        Instance reference.
    x
        Input tensor of shape (N, C, fH, fW).

    Returns
    -------
    list[torch.Tensor]
        List of feature tensors of shape (1, D, fH*fW, C) for each camera.
    """
    N, _C, fH, fW = x.shape

    # Begin Qualcomm modification
    # Process each camera input separately to optimize memory usage and avoid reshaping entire batch
    x = self.depthnet(x)
    depth = list(x[:, : self.D].softmax(dim=1).split(1, 0))
    other = list(x[:, self.D : (self.D + self.C)].split(1, 0))
    out = []
    for i in range(N):
        depth[i] = depth[i].reshape(self.D, 1, fH, fW).repeat(1, self.C, 1, 1)
        other[i] = other[i].reshape(1, self.C, fH, fW).repeat(self.D, 1, 1, 1)
        out.append(
            (depth[i] * other[i])
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(1, self.D, fH * fW, self.C)
        )
    # End Qualcomm modification

    return out


def PatchMerging_forward_optimized(
    self: PatchMerging, x: torch.Tensor, input_size: tuple[int, int]
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Merge patches for Swin Transformer architecture.

    Parameters
    ----------
    self
        Instance reference.
    x
        Has shape (B, H*W, C_in).
    input_size
        The spatial shape of x, arrange as (H, W).

    Returns
    -------
    x : torch.Tensor
        Has shape (B, Merged_H * Merged_W, C_out).
    out_size : tuple[int, int]
        Spatial shape of x, arrange as (Merged_H, Merged_W).
    """
    B, L, C = x.shape
    assert isinstance(input_size, Sequence), (
        f"Expect input_size is `Sequence` but get {input_size}"
    )

    H, W = input_size
    assert L == H * W, "input feature has wrong size"

    x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

    if self.adap_padding:
        x = self.adap_padding(x)
        H, W = x.shape[-2:]

    # Begin Qualcomm modification - Replace nn.Unfold with manual patch merging
    # Original: x = self.sampler(x)  # Uses nn.Unfold for patch extraction

    # Manual patch merging by alternating row sampling
    even_indices = x[:, :, ::2, :]
    odd_indices = x[:, :, 1::2, :]

    result = torch.stack([even_indices, odd_indices], dim=2)
    result = result.reshape(
        result.shape[0],
        result.shape[1] * result.shape[2],
        result.shape[3],
        result.shape[4],
    )

    # Manual patch merging by alternating column sampling
    even_indices = result[:, :, :, ::2]
    odd_indices = result[:, :, :, 1::2]

    # Combine even and odd columns and flatten
    result = torch.stack([even_indices, odd_indices], dim=3).transpose(2, 3)
    result = result.reshape(result.shape[0], result.shape[1] * result.shape[2], -1)
    x = result
    # End Qualcomm modification

    out_h = (
        H
        + 2 * self.sampler.padding[0]
        - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)
        - 1
    ) // self.sampler.stride[0] + 1
    out_w = (
        W
        + 2 * self.sampler.padding[1]
        - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)
        - 1
    ) // self.sampler.stride[1] + 1

    output_size = (out_h, out_w)
    x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
    x = self.norm(x) if self.norm else x
    x = self.reduction(x)
    return x, output_size
