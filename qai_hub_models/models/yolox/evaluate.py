# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import warnings

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.yolox import MODEL_ID, Model
from qai_hub_models.models.yolox.export import export_model
from qai_hub_models.utils.args import (
    evaluate_parser,
    get_input_spec_kwargs,
    get_model_kwargs,
)
from qai_hub_models.utils.evaluate import evaluate_on_dataset
from qai_hub_models.utils.inference import compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec


def main() -> None:
    warnings.filterwarnings("ignore")
    eval_datasets = Model.eval_datasets()
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a16: [
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a8: [
            TargetRuntime.ONNX,
        ],
        Precision.w8a8_mixed_int16: [
            TargetRuntime.ONNX,
        ],
    }

    parser = evaluate_parser(
        model_cls=Model,
        supported_datasets=eval_datasets,
        supported_precision_runtimes=supported_precision_runtimes,
        default_device="Samsung Galaxy S25 (Family)",
    )
    args = parser.parse_args()

    model_kwargs = get_model_kwargs(Model, vars(args))
    input_spec_kwargs = get_input_spec_kwargs(Model, vars(args))

    if len(eval_datasets) == 0:
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_model(
            device=args.device,
            target_runtime=args.target_runtime,
            skip_downloading=True,
            skip_profiling=True,
            compile_options=args.compile_options,
            profile_options=args.profile_options,
            **{**model_kwargs, **input_spec_kwargs},
        )
        return

    compiled_model: hub.Model | None = None
    quantized_model: hub.Model | None = None
    if not args.skip_device_accuracy or args.compute_quant_cpu_accuracy:
        if args.hub_model_id is not None:
            compiled_model = hub.get_model(args.hub_model_id)
        else:
            compiled_model = compile_model_from_args(
                MODEL_ID, args, {**model_kwargs, **input_spec_kwargs}
            )
        if args.precision != Precision.float:
            quantized_model = compiled_model

    torch_model = Model.from_pretrained(**get_model_kwargs(Model, vars(args)))
    input_spec: InputSpec | None = None
    if not args.skip_torch_accuracy and not compiled_model and not quantized_model:
        input_spec = torch_model.get_input_spec(**input_spec_kwargs)

    evaluate_on_dataset(
        evaluator_func=torch_model.get_evaluator,
        dataset_name=args.dataset_name,
        input_spec=input_spec,
        torch_model=torch_model,
        compiled_model=compiled_model if not args.skip_device_accuracy else None,
        quantized_model=quantized_model if args.compute_quant_cpu_accuracy else None,
        hub_device=args.device,
        samples_per_job=args.samples_per_job,
        num_samples=args.num_samples,
        seed=args.seed,
        profile_options=args.profile_options,
        use_cache=args.use_dataset_cache,
    )


if __name__ == "__main__":
    main()
