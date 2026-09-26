# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import argparse
import warnings

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.bge_large import MODEL_ID, Model
from qai_hub_models.utils.args import evaluate_parser
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline
from qai_hub_models.utils.export.context import resolve_recipe_dir

SUPPORTED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.TFLITE,
        TargetRuntime.QNN_DLC,
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.ONNX,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
    Precision.w8a8: [
        TargetRuntime.TFLITE,
        TargetRuntime.QNN_DLC,
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.ONNX,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


DEFAULT_EVAL_DEVICE = "Samsung Galaxy S25 (Family)"

evaluate_model = select_evaluate_pipeline(resolve_recipe_dir(MODEL_ID))


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this model's evaluate script.

    Exposed so the qai-hub-models CLI dispatcher can reuse the model's native
    parser without re-running main().
    """
    return evaluate_parser(
        model_cls=Model,
        supported_dataset_classes=Model.get_eval_dataset_classes(),
        supported_precision_runtimes=SUPPORTED_PRECISION_RUNTIMES,
        default_device=DEFAULT_EVAL_DEVICE,
    )


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.bge_large.evaluate` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models evaluate bge_large` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = build_parser().parse_args()
    evaluate_model(**vars(args))


if __name__ == "__main__":
    main()
