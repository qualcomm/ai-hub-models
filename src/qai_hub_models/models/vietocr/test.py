# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.vietocr.demo import main as demo_main
from qai_hub_models.models.vietocr.model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VietOCR,
)
from qai_hub_models.scorecard.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    model = VietOCR.from_pretrained()
    spec = model.get_input_spec()
    assert spec["image"][0] == (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    image = torch.rand(*spec["image"][0])
    features = np.asarray(model(image))

    # Output is a per-column feature sequence: [W', batch, 256].
    assert features.ndim == 3
    assert features.shape[1] == 1
    assert features.shape[2] == 256


@skip_clone_repo_check
def test_static_export_graph() -> None:
    """The rebuilt backbone tail must export to a static ONNX graph with no
    negative-axis Transpose (the reason the original tail did not compile).
    """
    import onnx

    model = VietOCR.from_pretrained()
    image = torch.rand(*model.get_input_spec()["image"][0])
    path = "/tmp/vietocr_test.onnx"
    torch.onnx.export(
        model,
        image,
        path,
        input_names=["image"],
        output_names=["features"],
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True)
    for node in onnx_model.graph.node:
        if node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
                    assert all(p >= 0 for p in attr.ints)


def test_demo() -> None:
    demo_main(is_test=True)
