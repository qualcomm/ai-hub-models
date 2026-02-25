# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import datetime
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pytest
import qai_hub as hub
from qai_hub.hub import _global_client

from qai_hub_models.utils.onnx import helpers as onnx_helpers
from qai_hub_models.utils.onnx.helpers import (
    ONNX_MAX_COMPATIBLE_VERSION,
    ONNX_PACKAGE_NAME,
    ONNXBundle,
    download_and_unzip_workbench_onnx_model,
    generate_wrapper_onnx_file,
    verify_onnx_export_is_compatible_with_ai_hub,
)
from qai_hub_models.utils.runtime_torch_wrapper import ModelIODetails


def test_onnx_bundle_from_path(tmp_path: Path) -> None:
    onnx_graph_path = tmp_path / "model.onnx"
    onnx_weights_path = tmp_path / "model.data"
    aimet_encodings_path = tmp_path / "model_seq1_cl4096.encodings"

    onnx_graph_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name is None
    assert bundle.onnx_weights_path is None
    assert bundle.aimet_encodings_name is None
    assert bundle.aimet_encodings_path is None

    onnx_weights_path.touch()
    bundle = ONNXBundle.from_bundle_path(bundle_path=tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name == "model.data"
    assert bundle.onnx_weights_path == onnx_weights_path
    assert bundle.aimet_encodings_name is None
    assert bundle.aimet_encodings_path is None

    aimet_encodings_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name == "model.data"
    assert bundle.onnx_weights_path == onnx_weights_path
    assert bundle.aimet_encodings_name == "model_seq1_cl4096.encodings"
    assert bundle.aimet_encodings_path == aimet_encodings_path
    assert bundle.qairt_bin_name is None
    assert bundle.qairt_context_binary_path is None

    qairt_bin_path = tmp_path / "model.bin"
    qairt_bin_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name == "model.data"
    assert bundle.onnx_weights_path == onnx_weights_path
    assert bundle.aimet_encodings_name == "model_seq1_cl4096.encodings"
    assert bundle.aimet_encodings_path == aimet_encodings_path
    assert bundle.qairt_bin_name == "model.bin"
    assert bundle.qairt_context_binary_path == qairt_bin_path

    onnx_weights_path_2 = tmp_path / "apple_model_2.data"
    onnx_weights_path_2.touch()
    with pytest.raises(ValueError, match="more than 1 ONNX weight file"):
        ONNXBundle.from_bundle_path(tmp_path)
    onnx_weights_path_2.unlink()

    aimet_encodings_path_2 = tmp_path / "aimet_2.encodings"
    aimet_encodings_path_2.touch()
    with pytest.raises(ValueError, match="more than 1 AIMET encodings file"):
        ONNXBundle.from_bundle_path(tmp_path)
    aimet_encodings_path_2.unlink()

    qairt_bin_path_2 = tmp_path / "model_2.bin"
    qairt_bin_path_2.touch()
    with pytest.raises(ValueError, match="more than 1 QAIRT context binary file"):
        ONNXBundle.from_bundle_path(tmp_path)
    qairt_bin_path_2.unlink()

    onnx_graph_path_2 = tmp_path / "model_2.onnx"
    onnx_weights_path_2 = tmp_path / "model_2.data"
    aimet_encodings_path_2 = tmp_path / "model_2.encodings"
    onnx_graph_path_2.touch()
    onnx_weights_path_2.touch()
    aimet_encodings_path_2.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path, model_name="model_2")
    assert bundle.onnx_graph_name == "model_2.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path_2
    assert bundle.onnx_weights_name == "model_2.data"
    assert bundle.onnx_weights_path == onnx_weights_path_2
    assert bundle.aimet_encodings_name == "model_2.encodings"
    assert bundle.aimet_encodings_path == aimet_encodings_path_2


def test_onnx_bundle_move(tmp_path: Path) -> None:
    onnx_graph_path = tmp_path / "model.onnx"
    onnx_weights_path = tmp_path / "model.data"
    aimet_encodings_path = tmp_path / "model_seq1_cl4096.encodings"

    dst = tmp_path / "test"
    os.makedirs(dst)
    onnx_graph_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    out = bundle.move(dst, "model2")
    assert bundle == out
    assert out.onnx_graph_path == dst / "model2.onnx"
    assert out.onnx_weights_path is None
    assert out.aimet_encodings_path is None
    assert not os.path.exists(onnx_graph_path)
    assert os.path.exists(out.onnx_graph_path)

    dst = tmp_path / "test2"
    os.makedirs(dst)
    onnx_graph_path.touch()
    onnx_weights_path.touch()
    aimet_encodings_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    out = bundle.move(dst, "model2")
    assert bundle == out
    assert out.onnx_graph_path == dst / "model2.onnx"
    assert out.onnx_weights_path == dst / "model2.data"
    assert out.aimet_encodings_path == dst / "model2.encodings"
    assert not os.path.exists(onnx_graph_path)
    assert os.path.exists(out.onnx_graph_path)
    assert not os.path.exists(onnx_weights_path)
    assert out.onnx_weights_path and os.path.exists(out.onnx_weights_path)
    assert not os.path.exists(aimet_encodings_path)
    assert out.aimet_encodings_path and os.path.exists(out.aimet_encodings_path)

    dst = tmp_path / "test3"
    os.makedirs(dst)
    onnx_graph_path.touch()
    onnx_weights_path.touch()
    aimet_encodings_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    out = bundle.move(dst, "model2", copy=True)
    assert out.onnx_graph_path == dst / "model2.onnx"
    assert out.onnx_weights_path == dst / "model2.data"
    assert out.aimet_encodings_path == dst / "model2.encodings"
    assert os.path.exists(onnx_graph_path)
    assert os.path.exists(out.onnx_graph_path)
    assert os.path.exists(onnx_weights_path)
    assert out.onnx_weights_path and os.path.exists(out.onnx_weights_path)
    assert os.path.exists(aimet_encodings_path)
    assert out.aimet_encodings_path and os.path.exists(out.aimet_encodings_path)


def test_onnx_bundle_move_with_qairt_bin(tmp_path: Path) -> None:
    """Test moving ONNX bundles that include QAIRT context binary files."""
    onnx_graph_path = tmp_path / "model.onnx"
    qairt_bin_path = tmp_path / "model.bin"

    # Create ONNX model with EPContext node
    generate_wrapper_onnx_file(
        graph_name="test_graph",
        onnx_output_path=onnx_graph_path,
        onnx_input_specs={"input": ((1, 3, 224, 224), onnx.TensorProto.FLOAT)},
        onnx_output_specs={"output": ((1, 1000), onnx.TensorProto.FLOAT)},
        qnn_context_bin_path="./model.bin",
        qairt_version="1.0.0",
    )
    qairt_bin_path.write_bytes(b"dummy bin content")

    # Test move with rename
    dst = tmp_path / "test"
    os.makedirs(dst)
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    assert bundle.qairt_bin_name == "model.bin"
    assert bundle.qairt_context_binary_path == qairt_bin_path

    out = bundle.move(dst, "model2")
    assert out.qairt_bin_name == "model2_qairt_context.bin"
    assert out.qairt_context_binary_path == dst / "model2_qairt_context.bin"
    assert not os.path.exists(qairt_bin_path)
    assert out.qairt_context_binary_path and os.path.exists(
        out.qairt_context_binary_path
    )

    # Verify the bin content was preserved
    assert out.qairt_context_binary_path.read_bytes() == b"dummy bin content"

    # Verify the ONNX model was updated to reference the new bin file name
    model = onnx.load(str(out.onnx_graph_path))
    ep_nodes = [n for n in model.graph.node if n.op_type == "EPContext"]
    assert len(ep_nodes) == 1
    ep_cache_attr = next(
        (a for a in ep_nodes[0].attribute if a.name == "ep_cache_context"), None
    )
    assert ep_cache_attr is not None
    assert ep_cache_attr.s.decode() == "./model2_qairt_context.bin"

    # Test copy (not move) with qairt bin
    dst2 = tmp_path / "test2"
    os.makedirs(dst2)
    onnx_graph_path2 = tmp_path / "model3.onnx"
    qairt_bin_path2 = tmp_path / "model3.bin"
    generate_wrapper_onnx_file(
        graph_name="test_graph",
        onnx_output_path=onnx_graph_path2,
        onnx_input_specs={"input": ((1, 3, 224, 224), onnx.TensorProto.FLOAT)},
        onnx_output_specs={"output": ((1, 1000), onnx.TensorProto.FLOAT)},
        qnn_context_bin_path="./model3.bin",
        qairt_version="1.0.0",
    )
    qairt_bin_path2.write_bytes(b"dummy bin content 2")

    bundle2 = ONNXBundle.from_bundle_path(tmp_path, model_name="model3")
    out2 = bundle2.move(dst2, "model4", copy=True)
    assert out2.qairt_bin_name == "model4_qairt_context.bin"
    assert out2.qairt_context_binary_path == dst2 / "model4_qairt_context.bin"
    # Original should still exist since we used copy=True
    assert os.path.exists(qairt_bin_path2)
    assert out2.qairt_context_binary_path and os.path.exists(
        out2.qairt_context_binary_path
    )


def test_download_and_unzip_workbench_onnx_model() -> None:
    class PatchedModel(hub.Model):
        """Hub Model patched so download() copies a local file rather than downloading from the internet."""

        def __init__(self, local_file: str | os.PathLike) -> None:
            super().__init__(
                _global_client,
                "dummy_id",
                datetime.datetime.now(),
                hub.SourceModelType.ONNX,
                "dummy_name",
                {},
                None,
                False,
                None,
                {},
                {},
            )
            self.local_file = local_file

        def download(self, filename: str) -> str:
            dst = os.path.join(filename, os.path.basename(self.local_file))
            shutil.copyfile(self.local_file, dst)
            return dst

    #
    # Download structure:
    # my_zip.zip
    #    zip_folder
    #        model_asdf.onnx
    #        model.data
    #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        zippath = tmppath / "model.zip"
        with zipfile.ZipFile(zippath, "w") as zipf:
            zipf.writestr("model_asbsd_onnx/model_asdf.onnx", "")
            zipf.writestr("model_asbsd_onnx/model.data", "")
        out = download_and_unzip_workbench_onnx_model(
            PatchedModel(zippath), tmpdir, "test_model"
        )
        assert out.onnx_graph_path.exists()
        assert out.onnx_graph_path == tmppath / "test_model.onnx"
        assert out.onnx_weights_path and out.onnx_weights_path.exists()
        assert out.onnx_weights_path == tmppath / "test_model.data"
        assert out.aimet_encodings_path is None

    #
    # Download structure:
    # my_zip.zip
    #    model.onnx
    #    model_blah.encodings
    #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        zippath = tmppath / "model.zip"
        with zipfile.ZipFile(zippath, "w") as zipf:
            zipf.writestr("model.onnx", "")
            zipf.writestr("model_blah.encodings", "")
        out = download_and_unzip_workbench_onnx_model(PatchedModel(zippath), tmpdir)
        assert out.onnx_graph_path.exists()
        assert out.onnx_graph_path == tmppath / "dummy_name.onnx"
        assert out.onnx_weights_path is None
        assert out.aimet_encodings_path and out.aimet_encodings_path.exists()
        assert out.aimet_encodings_path == tmppath / "dummy_name.encodings"

    #
    # Download structure:
    # model.onnx
    #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        model_path = tmppath / "model.onnx"
        model_path.touch()
        out = download_and_unzip_workbench_onnx_model(PatchedModel(model_path), tmpdir)
        assert out.onnx_graph_path.exists()
        assert out.onnx_graph_path == tmppath / "dummy_name.onnx"
        assert out.onnx_weights_path is None
        assert out.aimet_encodings_path is None


def test_verify_onnx_export_is_compatible_with_ai_hub() -> None:
    ec_patch = mock.patch("qai_hub_models.utils.onnx.helpers.ONNX_ENV_CHECKED", False)
    err_patch = mock.patch("qai_hub_models.utils.onnx.helpers.ONNX_ENV_ERROR", None)

    with (
        ec_patch,
        err_patch,
    ):
        with pytest.raises(
            ValueError,
            match=r"Package 'onnx' is not installed in your python environment\..*",
        ):
            verify_onnx_export_is_compatible_with_ai_hub({})
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert (
            onnx_helpers.ONNX_ENV_ERROR is not None
            and onnx_helpers.ONNX_ENV_ERROR.startswith(
                "Package 'onnx' is not installed in your python environment."
            )
        )

    with (
        ec_patch,
        err_patch,
    ):
        with pytest.raises(
            ValueError,
            match=r"Installed onnx package \(onnx==1.45.0\) is too new for compatibility with AI Hub Workbench\..*",
        ):
            verify_onnx_export_is_compatible_with_ai_hub({ONNX_PACKAGE_NAME: "1.45.0"})
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert (
            onnx_helpers.ONNX_ENV_ERROR is not None
            and onnx_helpers.ONNX_ENV_ERROR.startswith(
                "Installed onnx package (onnx==1.45.0) is too new for compatibility with AI Hub Workbench."
            )
        )

    with (
        ec_patch,
        err_patch,
    ):
        verify_onnx_export_is_compatible_with_ai_hub(
            {ONNX_PACKAGE_NAME: ONNX_MAX_COMPATIBLE_VERSION}
        )
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert onnx_helpers.ONNX_ENV_ERROR is None


def test_generate_wrapper_onnx_file(tmp_path: Path) -> None:
    """Test ONNX wrapper generation"""
    out_path = tmp_path / "wrapper.onnx"

    # For testing purposes, we mix tuple and ModelIODetails specs
    input_specs: dict[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ] = {
        "fp_input": ((1, 3, 224, 224), onnx.TensorProto.FLOAT),
        "q_input": ModelIODetails(
            shape=(1, 3, 224, 224),
            dtype=np.dtype("uint8"),
            qdq_params=ModelIODetails.QDQParams(scale=0.5, zero_point=128),
        ),
    }

    output_specs: dict[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ] = {
        "fp_output": ((1, 1000), onnx.TensorProto.FLOAT),
        "q_output": ModelIODetails(
            shape=(1, 1000),
            dtype=np.dtype("uint8"),
            qdq_params=ModelIODetails.QDQParams(scale=0.05, zero_point=0),
        ),
    }

    generate_wrapper_onnx_file(
        graph_name="my_graph",
        onnx_output_path=str(out_path),
        onnx_input_specs=input_specs,
        onnx_output_specs=output_specs,
        qnn_context_bin_path="ctx.bin",
        qairt_version="1.0.0",
    )

    model = onnx.load(str(out_path))
    graph = model.graph

    # EPContext node exists
    ep_nodes = [n for n in graph.node if n.op_type == "EPContext"]
    assert len(ep_nodes) == 1
    ep_node = ep_nodes[0]
    assert ep_node.name == "my_graph"
    assert b"ctx.bin" in [
        a.s for a in ep_node.attribute if a.name == "ep_cache_context"
    ]

    # Check QDQ nodes exist
    op_types = [n.op_type for n in graph.node]
    assert "DequantizeLinear" in op_types
    assert "QuantizeLinear" in op_types

    # Initializers
    init_names = {i.name for i in graph.initializer}
    assert "q_input_scale" in init_names
    assert "q_output_scale" in init_names

    # Inputs/Outputs
    input_names = [vi.name for vi in graph.input]
    output_names = [vi.name for vi in graph.output]
    assert "fp_input" in input_names
    assert "q_input" in input_names
    assert "fp_output" in output_names
    assert "q_output" in output_names
