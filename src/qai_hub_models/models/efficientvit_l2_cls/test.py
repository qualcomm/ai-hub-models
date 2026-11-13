# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch

from qai_hub_models.models.efficientvit_l2_cls.demo import main as demo_main
from qai_hub_models.models.efficientvit_l2_cls.model import EfficientViT
from qai_hub_models.models.templates.corevision.image_processing import (
    IMAGENET_TRANSFORM,
)
from qai_hub_models.models.templates.imagenet_classifier.app import (
    ImagenetClassifierApp,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    TEST_IMAGENET_CLASS,
    TEST_IMAGENET_IMAGE,
)
from qai_hub_models.utils.asset_loaders import load_image


def test_task() -> None:
    model = EfficientViT.from_pretrained()
    app = ImagenetClassifierApp(model, transform=IMAGENET_TRANSFORM)
    probabilities = app.predict(load_image(TEST_IMAGENET_IMAGE))
    assert torch.argmax(probabilities, dim=0) == TEST_IMAGENET_CLASS
    assert probabilities[TEST_IMAGENET_CLASS].item() > 0.39


def test_demo() -> None:
    demo_main(is_test=True)
