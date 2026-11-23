# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Generator, Iterable
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import IO, Any, TypeGuard

import numpy as np
import pandas as pd
import qai_hub as hub
from prettytable import PrettyTable
from qai_hub.public_rest_api import DatasetEntries
from tabulate import tabulate

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.compare import METRICS_FUNCTIONS, generate_comparison_metrics
from qai_hub_models.utils.device import OperatingSystem, OperatingSystemType
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT
from qai_hub_models.utils.qai_hub_helpers import get_device_and_chipset_name

_INFO_DASH = "-" * 60

# Comparison metrics (PSNR, MSE, ...) and accuracy percentages are truncated to
# a fixed number of significant figures wherever they are printed or written, so
# export output and scorecard CSVs stay free of floating point noise.
METRIC_SIG_FIGS = 4
ACCURACY_SIG_FIGS = 3


def format_sig_figs(value: float, sig_figs: int) -> str:
    """Render ``value`` truncated to ``sig_figs`` significant figures."""
    return f"{float(value):.{sig_figs}g}"


def format_metric(value: float) -> str:
    """Render a comparison metric (PSNR, MSE, MAE, ...)."""
    return format_sig_figs(value, METRIC_SIG_FIGS)


def format_accuracy(value: float) -> str:
    """Render an accuracy percentage."""
    return format_sig_figs(value, ACCURACY_SIG_FIGS)


def is_float(value: object) -> TypeGuard[float]:
    """True for Python and numpy floats.

    numpy scalars are not ``float`` subclasses (except float64), so a bare
    isinstance check silently skips them and leaks unrounded values.
    """
    return isinstance(value, (float, np.floating))


def print_tool_versions(
    tool_versions: ToolVersions | None,
    tool_versions_are_from_inference_job: bool = False,
) -> None:
    """Print tool versions."""
    print(_INFO_DASH)
    if tool_versions is not None:
        version_type = (
            "compilation and on-device inference"
            if tool_versions_are_from_inference_job
            else "compilation"
        )
        print(
            f"The versions of tools (eg. SDKs, runtimes) used during {version_type} via AI Hub Workbench:"
        )
        print(tool_versions)
        if not tool_versions_are_from_inference_job:
            print(
                "NOTE: Some tool (eg. SDK, runtime) versions required to run this model may be missing, as those tools were not needed for compilation. "
                "Run an inference job on the model to get the full set of tools versions required to run the model on-device."
            )
    print(_INFO_DASH)


def print_with_box(data: list[str]) -> None:
    """
    Print input list with box around it as follows
    +-----------------------------+
    | list data 1                 |
    | list data 2 that is longest |
    | data                        |
    +-----------------------------+
    """
    size = max(len(line) for line in data)
    size += 2
    print("+" + "-" * size + "+")
    for line in data:
        print("| {:<{}} |".format(line, size - 2))
    print("+" + "-" * size + "+")


def compute_inference_metrics(
    inference_result: DatasetEntries,
    torch_out: list[np.ndarray],
    output_names: list[str] | None = None,
    outputs_to_skip: list[int] | None = None,
    metrics: str = "psnr",
    print_result: bool = False,
    inference_job: hub.InferenceJob | None = None,
) -> pd.DataFrame:
    """
    Compare on-device outputs against a torch baseline and return the metrics.

    Parameters
    ----------
    inference_result
        On-device outputs, keyed by output name.
    torch_out
        Torch baseline outputs, ordered to match ``output_names``.
    output_names
        Outputs to compare. Defaults to every key in ``inference_result``.
    outputs_to_skip
        Indices into ``output_names`` to drop from the comparison.
    metrics
        Comma-separated metric names, e.g. ``"psnr,mse"``.
    print_result
        If set, also prints the on-device vs. local-cpu comparison table.
    inference_job
        Job the outputs came from, used for the printed header and trailing
        URL. Ignored unless ``print_result`` is set.

    Returns
    -------
    df : pd.DataFrame
        One row per output, indexed by output name, with a ``shape`` column plus
        one column per requested metric.
    """
    if output_names is None:
        output_names = list(inference_result.keys())
    missing = [n for n in output_names if n not in inference_result]
    if missing:
        raise KeyError(
            f"Output names {missing} not found in inference_result. "
            f"Available keys: {list(inference_result.keys())}"
        )
    inference_data = [
        np.concatenate(inference_result[out_name], axis=0) for out_name in output_names
    ]
    df_eval = generate_comparison_metrics(
        torch_out, inference_data, names=output_names, metrics=metrics
    )
    for output_idx in outputs_to_skip or []:
        if output_idx < len(output_names):
            df_eval = df_eval.drop(output_names[output_idx])

    if print_result:
        _print_inference_metrics(df_eval, inference_job)

    return df_eval


def _print_inference_metrics(
    df_eval: pd.DataFrame, inference_job: hub.InferenceJob | None
) -> None:
    def custom_float_format(x: object) -> str | object:
        if is_float(x):
            return format_metric(x)
        return x

    formatted_df = df_eval.applymap(custom_float_format)  # pyright: ignore[reportCallIssue]

    print(
        "\nComparing on-device vs. local-cpu inference"
        + (f" for {inference_job.name.title()}." if inference_job is not None else "")
    )
    print(tabulate(formatted_df, headers="keys", tablefmt="grid"))
    print()

    # Print explainers for each eval metric
    for m in df_eval.columns.drop("shape"):
        print(f"- {m}:", METRICS_FUNCTIONS[m][1])

    if inference_job is not None:
        last_line = f"More details: {inference_job.url}"
        print()
        print(last_line)


_BYTES_PER_MB = 1024 * 1024


def print_profile_metrics_from_job(
    inference_job: hub.InferenceJob,
    profile_data: dict[str, Any],
) -> None:
    compute_unit_counts = Counter(
        op.get("compute_unit", "UNK").lower() for op in profile_data["execution_detail"]
    )
    npu_ops = compute_unit_counts.get("npu", 0)
    gpu_ops = compute_unit_counts.get("gpu", 0)
    cpu_ops = compute_unit_counts.get("cpu", 0)

    execution_summary = profile_data["execution_summary"]
    low_mem_bytes, high_mem_bytes = execution_summary["inference_memory_peak_range"]
    mem_min_mb = low_mem_bytes // _BYTES_PER_MB
    mem_max_mb = high_mem_bytes // _BYTES_PER_MB
    inf_time_ms = execution_summary["estimated_inference_time"] / 1000

    runtime = TargetRuntime.from_hub_model_type(inference_job.model.model_type)
    device_name = inference_job.device.name

    # Read OS from hub.Device.attributes
    os_str = None
    for attr in inference_job.device.attributes:
        if attr.startswith("os:"):
            os_type = OperatingSystemType[attr.split(":")[-1].upper()]
            os_obj = OperatingSystem(ostype=os_type, version=inference_job.device.os)
            os_str = str(os_obj)
            break

    rows = [
        ["Device", f"{device_name} ({os_str})" if os_str else device_name],
        ["Runtime", runtime.name],
        [
            "Estimated inference time (ms)",
            "<0.1" if inf_time_ms < 0.1 else f"{inf_time_ms:.1f}",
        ],
        ["Estimated peak memory usage (MB)", f"[{mem_min_mb}, {mem_max_mb}]"],
        ["Total # Ops", str(npu_ops + gpu_ops + cpu_ops)],
        [
            "Compute Unit(s)",
            f"npu ({npu_ops} ops) gpu ({gpu_ops} ops) cpu ({cpu_ops} ops)",
        ],
    ]
    table = PrettyTable(align="l", header=False, border=False, padding_width=0)
    for label, value in rows:
        table.add_row([label, f": {value}"])

    print(f"\n{_INFO_DASH}")
    print(f"Performance results on-device for {inference_job.name.title()}.")
    print(_INFO_DASH)
    print(table.get_string())
    print(_INFO_DASH)
    print(f"More details: {inference_job.url}\n")


def print_on_target_demo_cmd(
    compile_job: hub.CompileJob | hub.LinkJob | Iterable[hub.CompileJob | hub.LinkJob],
    model_folder: Path,
    device: hub.Device,
) -> None:
    """Outputs a command that will run a model's demo script via inference job."""
    model_folder = model_folder.resolve()
    if not isinstance(compile_job, Iterable):
        compile_job = [compile_job]

    target_model_id = []
    for job in compile_job:
        assert job.wait().success
        target_model = job.get_target_model()
        assert target_model is not None
        target_model_id.append(target_model.model_id)

    target_model_id_str = ",".join(target_model_id)
    # An in-tree folder's name is its model id; a standalone recipe needs the
    # path, which dispatch resolves the same way.
    target = (
        model_folder.name
        if model_folder.parent == QAIHM_MODELS_ROOT.resolve()
        else model_folder
    )
    print(
        f"\nRun compiled model{'s' if len(target_model_id) > 1 else ''} on a hosted device on sample data using:"
    )
    print(
        f"qai-hub-models demo {target} --eval-mode on-device --hub-model-id {target_model_id_str} ",
        end="",
    )
    device_name, chipset = get_device_and_chipset_name(device)
    if chipset:
        print(f"--chipset {chipset}\n")
    else:
        print(f'--device "{device_name}"\n')


def print_file_tree_changes(
    base_dir: str,
    files_unmodified: list[str],
    files_added: list[str] | None = None,
    files_removed: list[str] | None = None,
) -> list[str]:
    """
    Given a set of absolute paths, prints the file tree with modifications highlighted.

    Parameters
    ----------
    base_dir
        The "top level" directory in which all files live.
    files_unmodified
        ABSOLUTE paths to files in base_dir that are not modified.
    files_added
        ABSOLUTE paths to files in base_dir that will be added.
    files_removed
        ABSOLUTE paths to files in base_dir that will be removed.

    Returns
    -------
    output_lines : list[str]
        Output lines (return value mainly used for unit testing)

    Raises
    ------
    AssertionError
        If any file path is not contained within base_dir.
    """
    if files_removed is None:
        files_removed = []
    if files_added is None:
        files_added = []
    changed = len(files_added) > 0 or len(files_removed) > 0
    outlines = [f"--- File Tree {'Changes' if changed else ' (Unchanged)'} ---"]

    # Get all files
    all_files_set = set(files_unmodified)
    all_files_set.update(files_removed)
    all_files_set.update(files_added)
    all_files = sorted(all_files_set)

    # Collect starting level
    base_dir = base_dir.removesuffix("/")
    base_level = base_dir.count(os.sep)
    last_level = base_level
    last_folder = None
    outlines.append(base_dir)

    for file in all_files:
        assert file.startswith(base_dir)

        level = file.count(os.sep)
        indent = file.count(os.sep) - base_level - 1
        folder = os.path.dirname(file)

        # If the level increases, or the folder name changes at the same level,
        # this is a new folder. Print the folder name.
        if (
            level > last_level or (level == last_level and last_folder != folder)
        ) and folder != base_dir:
            outlines.append("")
            outlines.append(f"{' ' * 4 * (indent)}{os.path.basename(folder)}/")
        elif level == last_level - 1:
            # pop back to previous folder, presumably to list more regular files
            outlines.append("")

        last_folder = folder
        last_level = level

        # Print file
        addl_info = ""
        added = file in files_added
        removed = file in files_removed
        if added and removed:
            addl_info = "-+ "
        elif added:
            addl_info = "+ "
        elif removed:
            addl_info = "- "
        outlines.append(f"{' ' * 4 * (indent + 1)}{addl_info}{os.path.basename(file)}")

    for line in outlines:
        print(line)

    return outlines


@contextmanager
def suppress_stdout() -> Generator[IO[str], None, None]:
    """A context manager that redirects stdout to devnull"""
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull) as out:
        yield out
