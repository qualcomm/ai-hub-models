# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import Any, cast

import numpy as np
import torch

from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps

with patch_mmdet_no_build_deps():
    from mmdet.models.task_modules import BaseBBoxCoder
from PIL import Image
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.bevdet.model import BEVDet
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    circle_nms,
    draw_3d_bbox,
    rotation_3d_in_axis,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    get_post_rot_and_tran,
)
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

OBJECT_CLASSES = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}


class BEVDetApp(CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol):
    """
    This class is required to perform end to end inference for BEVDet Model

    For a given images input with intrinsics and extrinsics, the app will:
        * pre-process the inputs (convert to range[0, 1])
        * Run the inference
        * Convert the reg, height, dim, rot, vel, heatmap into 3D_bboxes
        * Draw 3D_bbox in the image and return it.
    """

    def __init__(
        self,
        encoder: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        pooler: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            torch.Tensor,
        ],
        decoder: Callable[
            [torch.Tensor],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        bbox_coder: BaseBBoxCoder,
        model_input_shape: tuple[int, int] = (256, 704),
        score_threshold: float = 0.4,
        nms_threshold: float = 4.0,
        nms_post_max_size: int = 500,
    ) -> None:
        """
        Initialize BEVDetApp

        Parameters
        ----------
        encoder
            BEVDet encoder component (image -> (depth, feat)).
        pooler
            BEVDet pooler component, which runs the geometry + voxel pooling:
            (depth, feat, sensor2keyegos, inv_intrins, inv_post_rots, post_trans)
            -> BEV feature.
        decoder
            BEVDet decoder component (BEV feature -> raw detection head outputs).
        bbox_coder
            CenterPointBBoxCoder for BEVDet Model.
        model_input_shape
            (height, width) of the model input.
        score_threshold
            Default is 0.4.
        nms_threshold
            Default is 4.0,
        nms_post_max_size
            Default is 500
        """
        self.encoder = encoder
        self.pooler = pooler
        self.decoder = decoder
        self.bbox_coder = bbox_coder
        self.model_input_shape = model_input_shape
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_post_max_size = nms_post_max_size

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image.Image]:
        # See predict_3d_boxes_from_images.
        return self.predict_3d_boxes_from_images(*args, **kwargs)

    def predict_3d_boxes_from_images(
        self,
        images_list: list[Image.Image],
        intrins_list: list[np.ndarray],
        sensor2egos_list: list[np.ndarray],
        ego2globals_list: list[np.ndarray],
        raw_output: bool = False,
    ) -> np.ndarray | list[Image.Image]:
        """
        Run the BevDet model and predict 3d bounding boxes

        Parameters
        ----------
        images_list
            List of PIL Images in RGB format from each camera.
        intrins_list
            List of np.ndarray with shape [3,3] as float32.
            Camera intrinsic matrix for each camera.
            Used to project 3D points in camera frames to 2D image coordinates.
        sensor2egos_list
            List of np.ndarray with shape [4,4] as float32.
            sensor2ego transformation matrix for each camera
            converts from camera sensor to ego-vehicle coordinate frame.
        ego2globals_list
            List of np.ndarray with shape [4,4] as float32.
            ego2global transformation matrix for each camera
            converts from ego-vehicle to world coordinate frame.
        raw_output
            If true, returns corners. Otherwise returns output_images.

        Returns
        -------
        corners_or_output_images : np.ndarray | list[Image.Image]
            If raw_output is True:
                Corners of 3D bounding boxes with shape (N, 8, 3)
                where N is number of bounding boxes.
            If raw_output is False:
                List of PIL images with 3D bounding boxes overlaid.
        """
        sensor2egos = torch.tensor(sensor2egos_list)
        ego2globals = torch.tensor(ego2globals_list)

        # bug in source repo of bevdet
        # model is trained on camera front left as key ego,
        # it should be camera front as mentioned in paper,
        # using camera front left to maintain the accuracy
        global2keyego = torch.inverse(ego2globals[0])

        # transformation matix to convert from camera sensor
        # to ego-vehicle at front camera coordinate frame
        sensor2keyegos = global2keyego @ ego2globals @ sensor2egos
        sensor2keyegos = sensor2keyegos.unsqueeze(0)

        imgs, inv_post_rots, post_trans = self.preprocess_images(images_list)

        # used to project 2D image coordinates to 3D points
        inv_intrins = torch.inverse(torch.tensor(intrins_list)).unsqueeze(0)

        # model supports only single batch
        assert imgs.shape[0] == 1

        # Encoder (device): images -> per-pixel depth distribution + context features.
        depth, feat = self.encoder(imgs)

        # Pooler (CPU): geometry projection + voxel index prep + BEV-pool cumsum.
        # Runs on the QNN CPU backend because the HTP cannot reproduce the
        # float->int voxel quantization (fp16 corrupts ~11% of voxel ranks at
        # +-70m coords, breaking detection). No learned compute leaves the device.
        bev_feature = self.pooler(
            depth,
            feat,
            sensor2keyegos,
            inv_intrins,
            inv_post_rots,
            post_trans,
        )

        # Decoder (device): BEV feature -> raw detection head outputs.
        reg, height, dim, rot, vel, heatmap = self.decoder(bev_feature)

        corners_pt, scores_pt, labels_pt = self.get_bboxes(
            reg, height, dim, rot, vel, heatmap
        )[0]

        if raw_output:
            return corners_pt.detach().numpy()

        # Filter based on confidence score
        indices = scores_pt >= self.score_threshold
        corners, labels = (
            corners_pt[indices].numpy(),
            labels_pt[indices].int().numpy(),
        )

        corners_camfrontego = corners.reshape(-1, 3)
        corners_camfrontego = np.concatenate(
            [corners_camfrontego, np.ones([corners_camfrontego.shape[0], 1])], axis=1
        )

        output_images = []

        for k, pil_img in enumerate(images_list):
            # transform corners from front camera ego-vehicle frame to image coordinates
            corners_sensor = corners_camfrontego @ np.linalg.inv(sensor2egos_list[k]).T
            image_coor = (corners_sensor[:, :3] @ intrins_list[k].T).reshape(-1, 8, 3)

            # remove out-of-boundary coordinates
            valid_indices = np.all(image_coor[..., 2] > 0, axis=1)
            image_coor = image_coor[..., :2] / image_coor[..., 2:3]

            image = np.array(pil_img)
            image = draw_3d_bbox(
                image,
                image_coor[valid_indices],
                labels[valid_indices],
                OBJECT_CLASSES,
            )
            output_images.append(Image.fromarray(image))

        return output_images

    def preprocess_images(
        self, images_list: list[Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform pil image to image tensor

        Parameters
        ----------
        images_list
            List of PIL Image in RGB format

        Returns
        -------
        image_tensor : torch.Tensor
            Shape [B, N*C, H, W] where B=1, N=6, C=3.
            Pre-processed image with range[0-1].
        inv_post_rots : torch.Tensor
            Shape [B, N, 3, 3].
            Inverse post rotation matrix in camera coordinate system.
        post_trans : torch.Tensor
            Shape [B, N, 1, 3].
            Post translation tensor in camera coordinate system.
        """
        post_tran_list = []
        post_rot_list = []
        images_tensor_list = []
        for img in images_list:
            W, H = img.size
            fH, fW = self.model_input_shape

            resize = float(fW) / float(W)
            resize_dims = (int(W * resize), int(H * resize))
            crop_h = resize_dims[1] - fH
            crop_w = int(max(0, resize_dims[0] - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            img = img.resize(resize_dims).crop(crop)
            post_rot, post_tran = get_post_rot_and_tran(
                resize=resize, crop=crop, rotate=0
            )

            _, img_tensor = app_to_net_image_inputs(img)
            images_tensor_list.append(img_tensor)

            post_rot_list.append(post_rot)
            post_tran_list.append(post_tran)

        post_rots = torch.stack(post_rot_list)
        inv_post_rots = torch.inverse(post_rots).unsqueeze(0)
        post_trans = torch.stack(post_tran_list).reshape(1, len(images_list), 1, 3)
        image_tensor = torch.concat(images_tensor_list, dim=1)
        return image_tensor, inv_post_rots, post_trans

    def get_bboxes(
        self,
        reg: torch.Tensor,
        height: torch.Tensor,
        dim: torch.Tensor,
        rot: torch.Tensor,
        vel: torch.Tensor,
        heatmap: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate bboxes from bbox head predictions.

        Parameters
        ----------
        reg
            2D regression value with the shape of [B, 2, H, W].
        height
            Height value with the shape of [B, 1, H, W].
        dim
            Size value with the shape of [B, 3, H, W].
        rot
            Rotation value with the shape of [B, 2, H, W].
        vel
            Velocity value with the shape of [B, 2, H, W].
        heatmap
            Heatmap with the shape of [B, N, H, W].

        Returns
        -------
        detections_list : list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            Decoded bbox corners with shape (Num_pred, 8, 3)
            where 8 corners has 3 coordinates (x, y, z)
            scores with shape (Num_pred,) and
            labels with shape (Num_pred,) after nms.
        """
        # Decode bboxes from the given inputs
        # https://github.com/HuangJunJie2017/BEVDet/blob/26144be7c11c2972a8930d6ddd6471b8ea900d13/mmdet3d/core/bbox/coders/centerpoint_bbox_coders.py#L117
        decoded_outputs = self.bbox_coder.decode(
            heatmap,
            rot[:, 0].unsqueeze(1),
            rot[:, 1].unsqueeze(1),
            height,
            torch.exp(dim),
            vel,
            reg=reg,
            task_id=0,
        )

        ret = []
        for i in range(len(decoded_outputs)):
            boxes3d = decoded_outputs[i]["bboxes"]
            scores = decoded_outputs[i]["scores"]
            labels = decoded_outputs[i]["labels"]
            centers = boxes3d[:, [0, 1]]
            boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
            selected_indices = circle_nms(
                boxes.detach().numpy(),
                thresh=self.nms_threshold,
                post_max_size=self.nms_post_max_size,
            )

            bboxes = boxes3d[selected_indices]
            scores = scores[selected_indices]
            labels = labels[selected_indices]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            dims = bboxes[:, 3:6]
            corners_norm = torch.from_numpy(
                np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
            )

            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
            # use relative origin [0.5, 0.5, 0]
            corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
            corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

            # rotate around z axis
            corners = rotation_3d_in_axis(corners, bboxes[:, 6], axis=2)
            corners += bboxes[:, :3].view(-1, 1, 3)
            ret.append((corners, scores, labels))
        return ret

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> BEVDetApp:
        torch_model = BEVDet.from_pretrained()
        return cls(
            models[0],  # type: ignore[arg-type]
            models[1],  # type: ignore[arg-type]
            models[2],  # type: ignore[arg-type]
            torch_model.bboxcoder,
        )

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult]
        | tuple[torch.Tensor, ...]
        | torch.Tensor,
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        image, sensor2keyegos, inv_intrins, inv_post_rots, post_trans = model_input

        enc_output = self.encoder(cast(torch.Tensor, image))
        yield enc_output

        if isinstance(enc_output, AsyncOnDeviceResult):
            depth, feat = enc_output.wait()
            pool_output = self.pooler(
                depth.split(model_batch_size, dim=0),
                feat.split(model_batch_size, dim=0),
                sensor2keyegos,
                inv_intrins,
                inv_post_rots,
                post_trans,
            )
        else:
            depth, feat = enc_output
            pool_output = (
                self.pooler(
                    depth,
                    feat,
                    cast(torch.Tensor, sensor2keyegos),
                    cast(torch.Tensor, inv_intrins),
                    cast(torch.Tensor, inv_post_rots),
                    cast(torch.Tensor, post_trans),
                ),
            )
        yield pool_output

        if isinstance(pool_output, AsyncOnDeviceResult):
            bev_feature = cast(torch.Tensor, pool_output.wait())
            dec_output = self.decoder(bev_feature.split(model_batch_size, dim=0))
        else:
            dec_output = self.decoder(*pool_output)

        yield dec_output
        return dec_output

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        batch_size = get_batch_size(input_spec) or 1

        encoder = collection_model.components["encoder"]
        pooler = collection_model.components["pooler"]
        enc_spec = (input_specs or {}).get("encoder", encoder.get_input_spec())

        calibration_dataset_cls = encoder.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=enc_spec,
        )
        num_samples = num_samples or dataset.default_num_calibration_samples()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]
        with torch.no_grad():
            for sample_input, _ in dataloader:
                image, sensor2keyegos, inv_intrins, inv_post_rots, post_trans = (
                    sample_input
                )
                if component_name == "encoder":
                    sample_input = (image,)
                else:
                    depth, feat = encoder(image)
                    pool_input = (
                        depth,
                        feat,
                        sensor2keyegos,
                        inv_intrins,
                        inv_post_rots,
                        post_trans,
                    )
                    if component_name == "pooler":
                        sample_input = pool_input
                    elif component_name == "decoder":
                        sample_input = (pooler(*pool_input),)
                    else:
                        raise ValueError(f"Invalid component name: {component_name}")
                for i, tensor in enumerate(sample_input):
                    inputs[i].append(tensor)
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
