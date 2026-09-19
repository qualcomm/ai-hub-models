# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest
import torch

from qai_hub_models.models.mobile_vit.app import MobileVITApp
from qai_hub_models.models.mobile_vit.demo import demo as demo_main
from qai_hub_models.models.mobile_vit.model import MODEL_ID, MobileVIT
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.test_helpers import assert_most_close

GROUP_NAME = "mobile_vit"
MODEL_ASSET_VERSION = 2
TEST_IMAGE = CachedWebModelAsset.from_asset_store(
    GROUP_NAME, MODEL_ASSET_VERSION, "dog.jpg"
)

# Class "Samoyed" from https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
TEST_CLASS = 258


def run_classifier_test(
    model: MobileVIT,
    model_name: str,
    asset_version: int = MODEL_ASSET_VERSION,
    probability_threshold: float = 0.7,
    diff_tol: float = 0.0,
    rtol: float = 0.0,
    atol: float = 1e-4,
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
    """
    img = load_image(TEST_IMAGE)
    app = MobileVITApp(model)
    probabilities = app.predict(img)

    expected_output = CachedWebModelAsset.from_asset_store(
        model_name, asset_version, "expected_out.npy"
    )
    expected_out = load_numpy(expected_output)
    assert_most_close(probabilities.numpy(), expected_out, diff_tol, rtol, atol)

    predicted_class = torch.argmax(probabilities, dim=0)
    predicted_probability = probabilities[TEST_CLASS].item()
    assert predicted_probability > probability_threshold, (
        f"Predicted probability {predicted_probability:.3f} is below the threshold {probability_threshold}."
    )
    assert predicted_class == TEST_CLASS, (
        f"Model predicted class {predicted_class} when correct class was {TEST_CLASS}."
    )


def run_classifier_trace_test(
    model: MobileVIT,
    diff_tol: float = 0.005,
    rtol: float = 0.0,
    atol: float = 1e-4,
    is_quantized: bool = False,
    check_trace: bool = True,
) -> None:
    img = load_image(TEST_IMAGE)
    app = MobileVITApp(model)
    if not is_quantized:
        trace_app = MobileVITApp(model.convert_to_torchscript(check_trace=check_trace))
    else:
        trace_app = MobileVITApp(model.convert_to_torchscript())
    probabilities = app.predict(img)
    trace_probs = trace_app.predict(img)
    assert_most_close(probabilities.numpy(), trace_probs.numpy(), diff_tol, rtol, atol)


def test_task() -> None:
    run_classifier_test(
        MobileVIT.from_pretrained(),
        MODEL_ID,
        probability_threshold=0.45,
        diff_tol=0.005,
        atol=0.02,
        rtol=0.2,
    )


@pytest.mark.trace
def test_trace() -> None:
    run_classifier_trace_test(MobileVIT.from_pretrained())


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
