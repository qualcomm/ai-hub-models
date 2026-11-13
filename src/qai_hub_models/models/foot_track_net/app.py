# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image
from PIL.Image import fromarray as pil_image_from_array

from qai_hub_models.models.templates.corevision.bounding_box_processing import get_iou
from qai_hub_models.models.templates.corevision.draw import (
    draw_box_from_xyxy,
    draw_points,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.labels import get_class_names

GREEN_COLOR = (170, 255, 0)
RED_COLOR = (255, 0, 0)


def id_to_classname(class_id: int) -> str:
    """Return the class name (e.g. "face"/"person") for the given class id."""
    class_names = get_class_names("foot_track_net")
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    raise RuntimeError(f"Class for id {class_id} not found.")


def restructure_topk(
    scores: torch.Tensor, K: int = 20
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Customized function for top_k specific to FootTrackNet. Will restructure the original coordinates, class id from the floored index.
    After top k operation. this will specifically decoding the coordinates, class from the topk result.

    Parameters
    ----------
    scores
        The heatmap scores in flat shape
    K
        How many top k to be kept.

    Returns
    -------
    topk_scores : torch.Tensor
        The scores list for the top k.
    topk_inds : torch.Tensor
        The index list for the top k.
    topk_clses : torch.Tensor
        The class list for the top k.
    topk_ys : torch.Tensor
        The y coordinate list for the top k.
    topk_xs : torch.Tensor
        The x coordinate list for the top k.
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(
        scores.reshape(batch, -1), min(K, batch * cat * height * width)
    )
    topk_clses = (topk_inds // (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class BBox_landmarks:
    x: float
    y: float
    r: float
    b: float

    def __init__(
        self,
        label: str,
        xyrb: list[int] | npt.NDArray[np.int32],
        score: float = 0,
        landmark: list | np.ndarray | None = None,
        vis: list | np.ndarray | None = None,
    ) -> None:
        """
        A bounding box plus landmarks structure to hold the hierarchical result.

        Parameters
        ----------
        label
            The class label
        xyrb
            4 array or list for bbox left, top, right bottom coordinates
        score
            The score of the detection
        landmark
            17x2 the landmark of the joints [[x1,y1], [x2,y2]...]
        vis
            17 the visiblity of the joints.
        """
        self.label = label
        self.score = score
        self.landmark = landmark
        self.vis = vis
        self.x, self.y, self.r, self.b = xyrb
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    @property
    def label_prop(self) -> str:
        return self.label

    @label_prop.setter
    def label_prop(self, newvalue: str) -> None:
        self.label = newvalue

    @property
    def haslandmark(self) -> bool:
        return self.landmark is not None

    @property
    def box(self) -> list[int | float]:
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue: list[int] | npt.NDArray[np.int32]) -> None:
        self.x, self.y, self.r, self.b = newvalue

    def draw_box(
        self,
        frame: np.ndarray,
        box_text: bool = True,
        color: tuple[int, int, int] = GREEN_COLOR,
    ) -> None:
        """
        Draw the given bbox on the frame.
        Frame should be an RGB, integer [0:255], numpy array of shape [H, W, 3].
        """
        x, y, r, b = (int(bb + 0.5) for bb in np.array(self.box).astype(int))

        # 3DMM adjustment,  reuse the bbox structure
        if self.label_prop == "0":
            cx, cy = (r + x) // 2, (b + y) // 2
            offset = max(r - x, b - y) // 2
            x = int(cx - offset)
            y = int(cy - offset)
            r = int(cx + offset)
            b = int(cy + offset)

        draw_box_from_xyxy(
            frame,
            (x, y),
            (r, b),
            color,
            size=2,
            text=id_to_classname(int(self.label_prop)) if box_text else None,
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmark_idx: int | list[int] | np.ndarray | None = None,
        score_thr: float | list[float] | np.ndarray = 0.05,
        color: tuple[int, int, int] = GREEN_COLOR,
    ) -> None:
        """
        Draw the given landmarks on the frame.
        Frame should be an RGB, integer [0:255], numpy array of shape [H, W, 3].
        """
        if landmark_idx is None:
            landmark_idx = [0, 15, 16]
        if self.landmark is None:
            return

        landmarks = np.asarray(self.landmark)
        if self.vis is not None:
            vis = np.asarray(self.vis)
            landmarks = landmarks[
                np.intersect1d(np.nonzero(np.where(vis >= score_thr)), landmark_idx)
            ]
        if len(landmarks) > 0:
            draw_points(frame, landmarks, color, size=5)


def nms_bbox_landmark(
    objs: list[BBox_landmarks], iou: float = 0.5
) -> list[BBox_landmarks]:
    """
    Nms function customized to work on the BBox_landmarks objects list.

    Parameters
    ----------
    objs
        The list of the BBox_landmarks objects.
    iou
        IOU threshold for NMS.

    Returns
    -------
    nms_result : list[BBox_landmarks]
        The rest of the BBox_landmarks after nms operation.
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
            if (
                flags[j] == 0
                and get_iou(np.array(obj.box), np.array(objs[j].box)) > iou
            ):
                flags[j] = 1
    return keep


def detect_images_multiclass_fb(
    output_hm: torch.Tensor,
    output_tlrb: torch.Tensor,
    output_landmark: torch.Tensor,
    vis: torch.Tensor,
    threshold: list | np.ndarray | None = None,
    stride: int = 4,
    n_lmk: int = 17,
) -> list:
    """
    Get the detection result from the model raw output tensors.

    Parameters
    ----------
    output_hm
        N,C,H,W the model heatmap output.
    output_tlrb
        N,12,H,W the model bbox output.
    output_landmark
        N,34,H,W the model output_landmark output.
    vis
        N,17,H,W the model visiblity output
    threshold
        3 the threshold for each class.
    stride
        The stride of the output map comparing to input.
    n_lmk
        The landmark number.

    Returns
    -------
    detection_result : list
        Detection result.
    """
    if threshold is None:
        threshold = [0.7, 0.7, 0.7]
    _, num_classes, hm_height, hm_width = output_hm.shape
    hm = output_hm[0].reshape(1, num_classes, hm_height, hm_width)
    hm = hm[:, :2]

    tlrb = (
        output_tlrb[0]
        .cpu()
        .data.numpy()
        .reshape(1, num_classes * 4, hm_height, hm_width)
    )

    landmark = output_landmark[0].cpu().data.numpy().reshape(1, -1, hm_height, hm_width)
    vis_np = vis[0].cpu().data.numpy().reshape(1, -1, hm_height, hm_width)
    nmskey = hm

    kscore_pt, kinds_pt, kcls_pt, kys_pt, kxs_pt = restructure_topk(nmskey, 1000)

    kys = kys_pt.cpu().data.numpy().astype(np.int32)
    kxs = kxs_pt.cpu().data.numpy().astype(np.int32)
    kcls = kcls_pt.cpu().data.numpy().astype(np.int32)
    kscore = kscore_pt.cpu().data.numpy().astype(np.float32)
    kinds = kinds_pt.cpu().data.numpy().astype(np.int32)

    key: list[list[np.ndarray]] = [
        [],  # [kys..]
        [],  # [kxs..]
        [],  # [score..]
        [],  # [class..]
        [],  # [kinds..]
    ]

    score_fc = []
    for ind in range(kscore.shape[1]):
        score = kscore[0, ind]
        thr = threshold[kcls[0, ind]]
        if kcls[0, ind] == 0:
            score_fc.append(kscore[0, ind])
        if score > thr:
            key[0].append(kys[0, ind])
            key[1].append(kxs[0, ind])
            key[2].append(score)
            key[3].append(kcls[0, ind])
            key[4].append(kinds[0, ind])

    imboxs = []
    if key[0] is not None and len(key[0]) > 0:
        ky, kx = key[0], key[1]
        classes = key[3]
        scores = key[2]

        for i in range(len(kx)):
            class_ = int(classes[i])
            cx, cy = kx[i], ky[i]
            x1, y1, x2, y2 = tlrb[0, class_ * 4 : (class_ + 1) * 4, cy, cx]
            x1, y1, x2, y2 = (
                np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])
            ) * stride  # back to world

            if class_ == 1:  # face person, only person has landmark otherwise None
                x5y5 = landmark[0, : n_lmk * 2, cy, cx]
                x5y5 = (x5y5 + np.array([cx] * n_lmk + [cy] * n_lmk)) * stride
                boxlandmark = np.array(
                    list(zip(x5y5[:n_lmk], x5y5[n_lmk:], strict=False))
                )
                box_vis = vis_np[0, :, cy, cx].tolist()
            else:
                boxlandmark = None
                box_vis = None
            imboxs.append(
                BBox_landmarks(
                    label=str(class_),
                    xyrb=np.array([x1, y1, x2, y2]),
                    score=scores[i].item(),
                    landmark=boxlandmark,
                    vis=box_vis,
                )
            )
    return imboxs


def postprocess(
    heatmap: torch.Tensor,
    bbox: torch.Tensor,
    landmark: torch.Tensor,
    landmark_visibility: torch.Tensor,
    threshhold: list[float],
    iou_thr: list[float],
) -> tuple[list[BBox_landmarks], list[BBox_landmarks]]:
    """
    Get the detection result from the model raw output tensors.

    Parameters
    ----------
    heatmap
        N,C,H,W the model heatmap output.
    bbox
        N,C,H,W the model bbox output.
    landmark
        N,C,H,W the model output_landmark output.
    landmark_visibility
        N,C,H,W the model visiblity output.
    threshhold
        3 the threshold for each class.
    iou_thr
        3 the iou threshold for each class.

    Returns
    -------
    objs_face : list[BBox_landmarks]
        Face result.
    objs_person : list[BBox_landmarks]
        Person result.
    """
    stride = 4
    num_landmarks = 17
    objs = detect_images_multiclass_fb(
        heatmap,
        bbox,
        landmark,
        threshold=threshhold,
        stride=stride,
        n_lmk=num_landmarks,
        vis=landmark_visibility,
    )

    objs_face = []
    objs_person = []

    for obj in objs:
        label = id_to_classname(int(obj.label_prop))
        if label == "face":
            objs_face.append(obj)
        elif label == "person":
            objs_person.append(obj)

    objs_face = nms_bbox_landmark(objs_face, iou=iou_thr[0])
    objs_person = nms_bbox_landmark(objs_person, iou=iou_thr[1])

    return objs_face, objs_person


def undo_resize_pad_bbox(
    bbox: BBox_landmarks, scale: float, padding: tuple[int, int]
) -> None:
    """
    Undo the resize and pad in place of the BBox_landmarks object.
    operation in place to replace the inner coordinates

    Parameters
    ----------
    bbox
        The BBox_landmarks object to modify.
    scale
        Single scale from original to target image.
    padding
        Left, top padding size
    """
    if bbox.landmark is not None:
        for lmk in bbox.landmark:
            lmk[0] = (lmk[0] - padding[0]) / scale
            lmk[1] = (lmk[1] - padding[1]) / scale
    bbox.x = (bbox.x - padding[0]) / scale
    bbox.y = (bbox.y - padding[1]) / scale
    bbox.r = (bbox.r - padding[0]) / scale
    bbox.b = (bbox.b - padding[1]) / scale


class FootTrackNet_App:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with DDRNet.

    The app uses 1 model:
        * FootTrackNet

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FootTrackNet inference
        * Convert the output to two lists of BBox_landmarks objects for face and body.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        if_norm: bool = True,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            The input model
        if_norm
            If do the image normalization
        input_spec
            Model input spec. If None and model exposes ``get_input_spec()``,
            shape is auto-detected from the model. If None and model has no
            ``get_input_spec()`` (e.g. a traced model), no resize is performed.
        """
        if input_spec is None and hasattr(model, "get_input_spec"):
            input_spec = model.get_input_spec()
        if input_spec is not None:
            _, _, h, w = input_spec["image"][0]
            self.compiled_image_input_size: tuple[int, int] | None = (h, w)
        else:
            self.compiled_image_input_size = None
        self.model = model
        self.threshhold = [0.6, 0.7, 0.7]  # threshold for each detector, 0.6 original
        self.iou_thr = [0.2, 0.5, 0.5]  # iou threshold
        self.if_norm = if_norm

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[BBox_landmarks], list[BBox_landmarks]]:
        return self.predict_bbox_landmarks(*args, **kwargs)

    def _predict_bbox_landmarks(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image
    ) -> tuple[list[np.ndarray], list[BBox_landmarks], list[BBox_landmarks]]:
        # Convert from PIL / torch/ etc. to NHWC, RGB numpy frames, which is the required input type.
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        assert NCHW_fp32_torch_frames.shape[0] == 1, (
            "This app supports only a batch size of 1."
        )

        # Center-pad & scale image to fit compiled network input image size.
        if self.compiled_image_input_size:
            NCHW_fp32_torch_frames, scale, padding = resize_pad(
                NCHW_fp32_torch_frames, self.compiled_image_input_size
            )
        else:
            scale, padding = 1, (0, 0)

        heatmap, bbox, landmark, landmark_visibility = self.model(
            NCHW_fp32_torch_frames
        )
        objs_face, objs_person = postprocess(
            heatmap, bbox, landmark, landmark_visibility, self.threshhold, self.iou_thr
        )

        # Translate coordinates back to the original image.
        for obj_face in objs_face:
            undo_resize_pad_bbox(obj_face, scale, padding)
        for obj_person in objs_person:
            undo_resize_pad_bbox(obj_person, scale, padding)

        return NHWC_int_numpy_frames, objs_face, objs_person

    def predict_bbox_landmarks(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image
    ) -> tuple[list[BBox_landmarks], list[BBox_landmarks]]:
        """
        Return two lists,  objs_face, objs_person.
        Each list contains the object of BBox_landmarks which contains the bbox and landmark info. Please refer to BBox definition.

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both BGR channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), BGR channel layout

        Returns
        -------
        objs_face : list[BBox_landmarks]
            A list of BBox_landmarks for face.
        objs_person : list[BBox_landmarks]
            A list of BBox_landmarks for person.
        """
        _, objs_face, objs_person = self._predict_bbox_landmarks(pixel_values_or_image)
        return objs_face, objs_person

    def predict_and_draw_bbox_landmarks(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image
    ) -> Image:
        """
        Predict BBoxes + Coordinates and draw them on the input image.

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both BGR channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), BGR channel layout

        Returns
        -------
        annotated_image : Image
            PIL Image with boxes & landmarks drawn.
        """
        NHWC_int_numpy_frames, objs_face, objs_person = self._predict_bbox_landmarks(
            pixel_values_or_image
        )

        frame = NHWC_int_numpy_frames[0].copy()
        for obj in objs_person:
            obj.draw_box(frame, color=RED_COLOR)
            obj.draw_landmarks(frame, score_thr=0, color=RED_COLOR)

        for obj in objs_face:
            obj.draw_box(frame)
            obj.draw_landmarks(frame)

        return pil_image_from_array(frame)
