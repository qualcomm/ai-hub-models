# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Standalone script to compare local CPU inference outputs with on-device
inference outputs from AI Hub.

This script provides detailed numerical comparison metrics for debugging
model accuracy issues after export/compilation.

Usage:
    python -m qai_hub_models.scripts.compare_inference --model resnet50

"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import qai_hub as hub
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.utils.args import (
    _add_device_args,
    add_precision_arg,
    add_target_runtime_arg,
    get_model_kwargs,
    get_parser,
)
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.compare import METRICS_FUNCTIONS, torch_inference
from qai_hub_models.utils.export.context import resolve_model_cls, resolve_recipe_dir
from qai_hub_models.utils.export.dispatch import select_pipeline
from qai_hub_models.utils.export.result import (
    CollectionExportResult,
    LegacyCollectionExportResult,
)
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.path_helpers import MODEL_IDS
from qai_hub_models.utils.printing import compute_inference_metrics
from qai_hub_models.utils.transpose_channel import transpose_channel_last_to_first

ALL_METRICS = ",".join(k for k in METRICS_FUNCTIONS if k not in ("top1", "top5"))


def get_component_inference_job(
    export_result: Any,
    component: str,
) -> hub.InferenceJob | None:
    """Return the inference job for one component of a collection export result.

    ``CollectionExportResult`` and ``LegacyCollectionExportResult`` store
    per-component jobs under different shapes; this resolves both to a single
    inference job (or None if the export skipped inferencing). Multi-graph
    collection models are rejected before reaching this point, since they have
    no per-component inference job.
    """
    if isinstance(export_result, LegacyCollectionExportResult):
        return export_result.components[component].inference_job
    if isinstance(export_result, CollectionExportResult):
        if export_result.inference_jobs is None:
            return None
        return export_result.inference_jobs[component]
    raise TypeError(f"Unexpected collection export result: {type(export_result)}")


def export(
    model_id: str,
    device: hub.Device,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    precision: Precision = Precision.float,
    component: str | None = None,
    **additional_model_kwargs: Any,
) -> tuple[WorkbenchModel, hub.InferenceJob | None]:
    """Export a model (or one collection component) with on-device inference.

    Pass ``component`` to export a single component of a collection model;
    otherwise the whole (non-collection) model is exported. Returns the local
    model together with its on-device inference job (``None`` if the export
    skipped inferencing). Raises ``CLIUserError`` if ``model_id`` does not resolve
    to a recipe, and ``ValueError`` if it is a multi-graph model (no on-device
    inference path) or if ``component`` is inconsistent with the model kind.
    """
    source_dir = resolve_recipe_dir(model_id)
    model_cls: Any = resolve_model_cls(source_dir)

    # Multi-graph models (sharded LLMs, e.g. qwen VL) have no on-device
    # inference path -- their export pipelines hardcode skip_inferencing -- so
    # there is nothing to compare against local CPU inference.
    is_multi_graph = issubclass(
        model_cls, (MultiGraphWorkbenchModel, MultiGraphCollectionModel)
    )
    if is_multi_graph:
        raise ValueError(
            f"Model {model_id} is a multi-graph model, which has no on-device "
            "inference path to compare against."
        )

    # Validate the --component argument against the model kind.
    is_collection = issubclass(model_cls, CollectionModel)
    if is_collection and not component:
        raise ValueError(
            f"Model {model_id} is a collection model. Use --component to specify one."
        )
    if not is_collection and component:
        raise ValueError(
            f"Model {model_id} is not a collection model, "
            "--component is not applicable."
        )

    export_model = select_pipeline(source_dir)
    export_kwargs: dict[str, Any] = dict(
        device=device,
        target_runtime=target_runtime,
        precision=precision,
        skip_profiling=True,
        skip_inferencing=False,
        skip_downloading=True,
        skip_summary=True,
        **additional_model_kwargs,
    )
    model_kwargs = get_model_kwargs(
        model_cls, dict(**additional_model_kwargs, precision=precision)
    )

    if is_collection:
        assert component is not None  # validated above
        export_result = export_model(components=[component], **export_kwargs)
        inference_job = get_component_inference_job(export_result, component)
        model = model_cls.from_pretrained(**model_kwargs).components[component]
        if not isinstance(model, WorkbenchModel):
            raise TypeError(f"Component {component} is not a WorkbenchModel")
        return model, inference_job

    export_result = export_model(**export_kwargs)
    return model_cls.from_pretrained(**model_kwargs), export_result.inference_job


def compare_inference(
    model: WorkbenchModel,
    inference_job: hub.InferenceJob,
    metrics: str = ALL_METRICS,
    outputs_to_skip: list[int] | None = None,
    channel_last_outputs: list[str] | None = None,
    **additional_model_kwargs: Any,
) -> None:
    """
    Compare a model's local CPU inference against its on-device outputs.

    Parameters
    ----------
    model
        The local model to run on CPU.
    inference_job
        The completed on-device inference job to compare against.
    metrics
        Comma-separated list of metrics to compute.
    outputs_to_skip
        List of output indices to skip.
    channel_last_outputs
        List of output names that are in channel-last format on device.
    **additional_model_kwargs
        Additional kwargs for model.get_input_spec.
    """
    # Download on-device results
    device_outputs = inference_job.download_output_data()
    if device_outputs is None:
        raise RuntimeError("Failed to download inference results.")

    output_names = list(model.get_output_spec())

    # Transpose channel-last outputs if needed
    if channel_last_outputs:
        device_outputs = transpose_channel_last_to_first(
            channel_last_outputs, device_outputs
        )

    # Run local inference
    input_spec = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )
    sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)

    torch_out = torch_inference(model, sample_inputs, return_channel_last_output=False)

    compute_inference_metrics(
        device_outputs,
        torch_out,
        output_names,
        outputs_to_skip=outputs_to_skip,
        metrics=metrics,
        print_result=True,
        inference_job=inference_job,
    )


def main() -> None:
    warnings.filterwarnings("ignore")

    parser = get_parser()
    parser.description = (
        "Compare local CPU inference with on-device inference from AI Hub."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help=f"Model ID (e.g., resnet50). Available: {', '.join(MODEL_IDS[:5])}...",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=ALL_METRICS,
        help="Comma-separated list of metrics. Available: "
        + ALL_METRICS
        + f". Default: {ALL_METRICS}",
    )

    parser.add_argument(
        "--outputs-to-skip",
        type=str,
        default=None,
        help="Comma-separated list of output indices to skip (e.g., '0,2').",
    )

    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="For collection models, specify which component to compare.",
    )

    parser.add_argument(
        "--channel-last-outputs",
        type=str,
        default=None,
        help="Comma-separated list of output names in channel-last format on device.",
    )

    _add_device_args(parser, default_device="Samsung Galaxy S25 (Family)")
    add_target_runtime_arg(
        parser,
        helpmsg="Target runtime for compilation.",
        default=TargetRuntime.TFLITE,
    )
    add_precision_arg(
        parser,
        supported_precisions={Precision.float, Precision.w8a8, Precision.w8a16},
        default_if_arg_explicitly_passed=Precision.w8a8,
        default=Precision.float,
    )

    args = parser.parse_args()

    # Validate metrics
    for m in args.metrics.split(","):
        if m not in METRICS_FUNCTIONS:
            parser.error(
                f"Unknown metric: {m}. Available: {', '.join(METRICS_FUNCTIONS.keys())}"
            )

    # Parse list arguments
    outputs_to_skip: list[int] | None = None
    if args.outputs_to_skip:
        outputs_to_skip = [int(x.strip()) for x in args.outputs_to_skip.split(",")]

    channel_last_outputs: list[str] | None = None
    if args.channel_last_outputs:
        channel_last_outputs = [x.strip() for x in args.channel_last_outputs.split(",")]

    try:
        model, inference_job_opt = export(
            model_id=args.model,
            device=args.device,
            target_runtime=args.target_runtime,
            precision=args.precision,
            component=args.component,
        )
    except (ValueError, CLIUserError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if inference_job_opt is None:
        print("Error: Export did not produce an inference job.")
        sys.exit(1)

    compare_inference(
        model=model,
        inference_job=inference_job_opt,
        metrics=args.metrics,
        outputs_to_skip=outputs_to_skip,
        channel_last_outputs=channel_last_outputs,
    )


if __name__ == "__main__":
    main()
