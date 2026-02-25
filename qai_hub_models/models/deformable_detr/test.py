# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models._shared.detr.app import DETRApp
from qai_hub_models.models.deformable_detr.demo import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.deformable_detr.demo import main as demo_main
from qai_hub_models.models.deformable_detr.model import DeformableDETR
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "detr_test_image.jpg"
)


def test_task() -> None:
    net = DeformableDETR.from_pretrained()
    img = load_image(IMAGE_ADDRESS)
    _, _, label, _ = DETRApp(net).predict(img, threshold=0.75)
    assert set(label.numpy()) == {75, 17}


@pytest.mark.trace
def test_trace() -> None:
    model = DeformableDETR.from_pretrained()
    net = model.convert_to_torchscript()
    img = load_image(IMAGE_ADDRESS)
    (h, w) = model.get_input_spec()["image"][0][2:]
    _, _, label, _ = DETRApp(net, h, w).predict(img, threshold=0.75)
    assert set(label.numpy()) == {17}


def test_demo() -> None:
    demo_main(is_test=True)
