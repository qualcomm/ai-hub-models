# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision.models import vgg
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.kitti import KittiDataset
from qai_hub_models.models.deepbox.evaluator import Kitti3DDetectionEvaluator
from qai_hub_models.models.deepbox.external_repos.boundingbox_3d.torch_lib import (
    Model as TorchLibModel,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.yolov3.model import YoloV3
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import (
    BaseModel,
)
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

VGG_WEIGHTS_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "epoch_10.pkl"
)
DEFAULT_YOLO_WEIGHTS = "yolov3u.pt"


class Yolo2DDetection(YoloV3):
    """
    Exportable YoloV3 2D bounding box detector for DeepBox.

    Detects objects (cars, pedestrians, etc.) in a driving scene image.
    Input is an RGB image [0, 1]; output is
    [bounding boxes, confidence scores, class indices].
    """

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 416,
        width: int = 416,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return super()._sample_inputs_impl(input_spec or self.get_input_spec())


class VGG3DDetection(BaseModel):
    """
    VGG19-BN based 3D bounding box estimator for DeepBox.

    Takes a 224x224 crop of a detected object and predicts its 3D orientation,
    dimensions, and location. Returns [orient, conf, dim].
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model=model)

    @classmethod
    def from_pretrained(cls, ckpt_path: str = "DEFAULT") -> Self:
        if ckpt_path == "DEFAULT":
            ckpt_path = str(VGG_WEIGHTS_ASSET.fetch())

        my_vgg = vgg.vgg19_bn(pretrained=True)
        vgg_model = TorchLibModel.Model(features=my_vgg.features, bins=2)
        checkpoint = load_torch(ckpt_path)
        vgg_model.load_state_dict(checkpoint["model_state_dict"])
        return cls(vgg_model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass on the VGG 3D detection model.

        Parameters
        ----------
        image
            RGB image of range [0, 1] and shape [1, 3, H, W].

        Returns
        -------
        orient : torch.Tensor
            Orientation prediction. Shape is [1, bins, 2], where bins=2 and
            2 represents (cos, sin) of the local orientation angle.
        conf : torch.Tensor
            Confidence prediction for each orientation bin. Shape is [1, bins],
            where bins=2.
        dim : torch.Tensor
            Dimension prediction (height, width, length offsets). Shape is [1, 3].
        """
        # The original implementation of DeepBox applies RGB torchvision constants to BGR input images.
        image_bgr = torch.flip(image, dims=[1])
        norm_image_bgr = normalize_image_torchvision(image_bgr)
        out = self.model(norm_image_bgr)
        return out[0], out[1], out[2]

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 224,
        width: int = 224,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
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
        return {
            "orient": TensorSpec(),
            "conf": TensorSpec(),
            "dim": TensorSpec(),
        }


class DeepBox(WorkbenchModelCollection):
    def __init__(
        self, yolo_2d_det: Yolo2DDetection, vgg_3d_det: VGG3DDetection
    ) -> None:
        super().__init__(
            {"yolo_2d_detection": yolo_2d_det, "vgg_3d_detection": vgg_3d_det}
        )
        self.yolo_2d_det = yolo_2d_det
        self.vgg_3d_det = vgg_3d_det

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return KittiDataset

    def get_input_spec(
        self, batch_size: int = 1, height: int = 416, width: int = 416
    ) -> ComponentGroup[InputSpec]:
        return ComponentGroup(
            {
                "yolo_2d_detection": self.yolo_2d_det.get_input_spec(
                    batch_size=batch_size, height=height, width=width
                ),
                "vgg_3d_detection": self.vgg_3d_det.get_input_spec(
                    batch_size=batch_size
                ),
            }
        )

    @classmethod
    def from_pretrained(
        cls,
        yolo_ckpt: str = DEFAULT_YOLO_WEIGHTS,
        vgg_ckpt_path: str = "DEFAULT",
    ) -> Self:
        vgg_net = VGG3DDetection.from_pretrained(vgg_ckpt_path)
        return cls(Yolo2DDetection.from_pretrained(yolo_ckpt), vgg_net)

    def get_evaluator(self) -> BaseEvaluator:
        return Kitti3DDetectionEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [KittiDataset]
