# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest
import torch
from torchvision import transforms

from qai_hub_models.models.templates.corevision.image_processing import (
    IMAGENET_TRANSFORM,
)
from qai_hub_models.models.templates.imagenet_classifier.app import (
    ImagenetClassifierApp,
    preprocess_image,
)
from qai_hub_models.models.templates.imagenet_classifier.model import (
    MODEL_ASSET_VERSION,
    ImagenetClassifier,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.test_helpers import assert_most_close

GROUP_NAME = "imagenet_classifier"
TEST_IMAGENET_IMAGE = CachedWebModelAsset.from_asset_store(
    GROUP_NAME, MODEL_ASSET_VERSION, "dog.jpg"
)

# Class "Samoyed" from https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
TEST_IMAGENET_CLASS = 258


@pytest.fixture(scope="module")
def imagenet_sample_torch() -> torch.Tensor:
    """
    Returns
    -------
    sample_image : torch.Tensor
        Preprocessed (normalized etc) image as torch.Tensor with shape [1, 3, 224, 224].
    """
    img = load_image(TEST_IMAGENET_IMAGE, desc="imagenet_classifier")
    return preprocess_image(img)


def run_imagenet_classifier_test(
    model: ImagenetClassifier,
    model_name: str,
    asset_version: int = 2,
    probability_threshold: float = 0.7,
    diff_tol: float = 0.0,
    rtol: float = 0.0,
    atol: float = 1e-4,
    transform: transforms.Compose = IMAGENET_TRANSFORM,
    expected_class: int = TEST_IMAGENET_CLASS,
) -> None:
    """
    Evaluates the classifier on a test image and validates the output.

    Parameters
    ----------
    model
        The model to evaluate.
    model_name
        Identifier used to lookup the expected output file.
    asset_version
        Version of the expected output file to lookup.
    probability_threshold
        If the predicited probability for the correct class
        is below this threshold, the method throws an error.
    diff_tol
        Float in range [0,1] representing the maximum percentage of
        the probabilities that can differ from the ground truth while
        still having the test pass.
    rtol
        Relative tolerance allowed for two numbers to be "close".
    atol
        Absolute tolerance allowed for two numbers to be "close".
    transform
        Torchvision transform to apply to the image before inference.
        Defaults to the standard 224x224 ImageNet transform.
    expected_class
        Expected top-1 ImageNet class index. Defaults to TEST_IMAGENET_CLASS (258, Samoyed).
        Override for models where the correct transform yields a different top-1 on the test image.
    """
    img = load_image(TEST_IMAGENET_IMAGE)
    app = ImagenetClassifierApp(model, transform=transform)
    probabilities = app.predict(img)
    expected_output = CachedWebModelAsset.from_asset_store(
        model_name, asset_version, "expected_out.npy"
    )
    expected_out = load_numpy(expected_output)
    assert_most_close(probabilities.numpy(), expected_out, diff_tol, rtol, atol)

    predicted_class = torch.argmax(probabilities, dim=0)
    predicted_probability = probabilities[expected_class].item()
    assert predicted_probability > probability_threshold, (
        f"Predicted probability {predicted_probability:.3f} is below the threshold {probability_threshold}."
    )
    assert predicted_class == expected_class, (
        f"Model predicted class {predicted_class} when correct class was {expected_class}."
    )
