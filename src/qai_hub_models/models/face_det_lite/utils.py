# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from qai_hub_models.models.templates.corevision.bounding_box_processing import get_iou


class BBox:
    # Bounding Box
    def __init__(
        self,
        label: str,
        xyrb: list[int],
        score: float = 0,
        landmark: list | None = None,
        rotate: bool = False,
    ) -> None:
        """
        A bounding box plus landmarks structure to hold the hierarchical result.

        Parameters
        ----------
        label
            The class label.
        xyrb
            4 list for bbox left, top, right bottom coordinates.
        score
            The score of the detection.
        landmark
            10x2 the landmark of the joints [[x1,y1], [x2,y2]...].
        rotate
            Whether the bounding box is rotated.
        """
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate

        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self) -> str:
        landmark_formated = (
            ",".join([str(item[:2]) for item in self.landmark])
            if self.landmark is not None
            else "empty"
        )
        return (
            f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, "
            f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"
        )

    @property
    def width(self) -> int:
        return self.r - self.x + 1

    @property
    def height(self) -> int:
        return self.b - self.y + 1

    @property
    def box(self) -> list[int]:
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue: list[int]) -> None:
        self.x, self.y, self.r, self.b = newvalue

    @property
    def haslandmark(self) -> bool:
        return self.landmark is not None

    @property
    def xywh(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def nms(objs: list[BBox], iou: float = 0.5) -> list[BBox]:
    """
    Nms function customized to work on the BBox objects list.

    Parameters
    ----------
    objs
        The list of the BBox objects.
    iou
        Intersection over union threshold for NMS.

    Returns
    -------
    filtered_boxes : list[BBox]
        The BBox objects remaining after NMS filtering.
    """
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            # if flags[j] == 0 and obj.iou(objs[j]) > iou:
            if (
                flags[j] == 0
                and get_iou(np.array(obj.box), np.array(objs[j].box)) > iou
            ):
                flags[j] = 1
    return keep


def detect(
    hm: torch.Tensor,
    box: torch.Tensor,
    landmark: torch.Tensor,
    threshold: float = 0.2,
    nms_iou: float = 0.2,
    stride: int = 8,
) -> list[BBox]:
    hm = hm.sigmoid()
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    lens = ((hm == hm_pool).float() * hm).view(1, -1).cpu().shape[1]
    scores_pt, indices = (
        ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(min(lens, 2000))
    )

    _hm_height, hm_width = hm.shape[2:]

    scores_pt = scores_pt.squeeze()
    indices = indices.squeeze()
    ys = list(torch.div(indices, hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores: list[float] = list(scores_pt.data.numpy())

    objs = []
    for cx, cy, score in zip(xs, ys, scores, strict=False):
        if score < threshold:
            break

        x, y, r, b = box[0, :, cy, cx].cpu().data.numpy()
        xyrb: list[int] = (
            (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride  # noqa: RUF005
        ).tolist()
        x5y5 = landmark[0, :, cy, cx].cpu().data.numpy()
        x5y5 = (x5y5 + ([cx] * 5 + [cy] * 5)) * stride

        box_landmark = list(zip(x5y5[:5], x5y5[5:], strict=False))
        objs.append(BBox("0", xyrb=xyrb, score=score, landmark=box_landmark))

    if nms_iou != -1:
        return nms(objs, iou=nms_iou)
    return objs
