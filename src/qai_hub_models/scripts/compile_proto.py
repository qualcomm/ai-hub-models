# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from qai_hub_models.utils.path_helpers import QAIHM_REPO_ROOT

# grpcio-tools 1.62.3 bundles protoc 25.1. Keep it in sync with PROTOC_VERSION in
# .github/workflows/release_website.yml.
EXPECTED_GRPCIO_TOOLS_VERSION = "1.62.3"
EXPECTED_MYPY_PROTOBUF_VERSION = "3.6.0"

try:
    _grpcio_tools_version = _pkg_version("grpcio-tools")
    assert _grpcio_tools_version == EXPECTED_GRPCIO_TOOLS_VERSION, (
        f"Expected grpcio-tools=={EXPECTED_GRPCIO_TOOLS_VERSION} for code generation, got {_grpcio_tools_version}"
    )
except PackageNotFoundError as e:
    if sys.version_info >= (3, 13):
        raise ValueError(
            f"grpcio-tools was not found on your system. The required version ({EXPECTED_GRPCIO_TOOLS_VERSION}) is not installable on python 3.13 and above; use python 3.12 or older."
        ) from e
    raise

_mypy_protobuf_version = _pkg_version("mypy-protobuf")
assert _mypy_protobuf_version == EXPECTED_MYPY_PROTOBUF_VERSION, (
    f"Expected mypy-protobuf=={EXPECTED_MYPY_PROTOBUF_VERSION} for code generation, got {_mypy_protobuf_version}"
)

from grpc_tools import protoc  # noqa: E402

PROTO_SRC_DIR = QAIHM_REPO_ROOT / "proto"
PROTO_OUT_DIR = QAIHM_REPO_ROOT / "cli" / "qai_hub_models_cli" / "proto"

PKG = "qai_hub_models_cli.proto"
BARE_IMPORT_RE = re.compile(r"^(import \w+_pb2)", re.MULTILINE)
SHARED_IMPORT_RE = re.compile(r"^from shared (import \w+_pb2)", re.MULTILINE)


def _fix_imports(path: str) -> None:
    """Rewrite bare `import x_pb2` to `from qai_hub_models_cli.proto import x_pb2`
    and `from shared import x_pb2` to `from qai_hub_models_cli.proto.shared import x_pb2`
    so the generated files work as a proper Python package.
    """
    with open(path) as f:
        text = f.read()
    fixed = BARE_IMPORT_RE.sub(rf"from {PKG} \1", text)
    fixed = SHARED_IMPORT_RE.sub(rf"from {PKG}.shared \1", fixed)
    if fixed != text:
        with open(path, "w") as f:
            f.write(fixed)


def _fix_shared_imports(path: str) -> None:
    """For files inside shared/, rewrite `from shared import x_pb2` and bare
    `import x_pb2` to `from qai_hub_models_cli.proto.shared import x_pb2`.
    """
    with open(path) as f:
        text = f.read()
    fixed = SHARED_IMPORT_RE.sub(rf"from {PKG}.shared \1", text)
    fixed = BARE_IMPORT_RE.sub(rf"from {PKG}.shared \1", fixed)
    if fixed != text:
        with open(path, "w") as f:
            f.write(fixed)


def _is_generated(name: str) -> bool:
    return name.endswith(("_pb2.py", "_pb2.pyi"))


def compile_proto() -> None:
    # Clean only generated protobuf files, preserving __init__.py and any
    # hand-written modules in the output directory.
    if PROTO_OUT_DIR.exists():
        for item in PROTO_OUT_DIR.iterdir():
            if item.is_dir():
                for child in item.iterdir():
                    if _is_generated(child.name):
                        child.unlink()
            elif _is_generated(item.name):
                item.unlink()

    proto_files = sorted(PROTO_SRC_DIR.glob("*.proto"))
    proto_files += sorted(PROTO_SRC_DIR.glob("shared/*.proto"))
    if not proto_files:
        raise FileNotFoundError(f"No .proto files found in {PROTO_SRC_DIR}")

    PROTO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROTO_OUT_DIR / "shared").mkdir(parents=True, exist_ok=True)
    (PROTO_OUT_DIR / "shared" / "__init__.py").touch()

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={PROTO_SRC_DIR}",
            f"--python_out={PROTO_OUT_DIR}",
            f"--mypy_out={PROTO_OUT_DIR}",
        ]
        + [str(p) for p in proto_files],
    )
    if result != 0:
        raise RuntimeError(f"protoc failed with exit code {result}")

    modified_files: list[str] = []
    for generated in PROTO_OUT_DIR.glob("*_pb2.py"):
        _fix_imports(str(generated))
        modified_files.append(str(generated))
    for generated in PROTO_OUT_DIR.glob("*_pb2.pyi"):
        _fix_imports(str(generated))
        modified_files.append(str(generated))
    for generated in (PROTO_OUT_DIR / "shared").glob("*_pb2.py"):
        _fix_shared_imports(str(generated))
        modified_files.append(str(generated))
    for generated in (PROTO_OUT_DIR / "shared").glob("*_pb2.pyi"):
        _fix_shared_imports(str(generated))
        modified_files.append(str(generated))

    os.environ["SKIP"] = "mypy-src,mypy-cli"
    subprocess.run(["pre-commit", "run", "--files", *modified_files], check=False)


if __name__ == "__main__":
    compile_proto()
