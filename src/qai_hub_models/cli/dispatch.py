# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Heavy-side dispatcher for ``qai-hub-models <script> <target>`` subcommands.

Resolves *target* (a recipe folder path or a bare installed model id) to a
:class:`Path` once at the top layer, then hands it to the export/evaluate
pipelines and to the CLI parsers.
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import shlex
import sys
from pathlib import Path
from typing import cast

from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.cli.configure_dataset import main as configure_dataset_main
from qai_hub_models.cli.generate_files import main as generate_files_main
from qai_hub_models.cli.install import main as install_main
from qai_hub_models.cli.upload_to_hf import main as upload_to_hf_main
from qai_hub_models.cli.validate import main as validate_main
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.args import evaluate_parser, export_parser
from qai_hub_models.utils.asset_loaders import (
    check_disabled_path_warning,
    check_unpublished_model_warning,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline
from qai_hub_models.utils.export.context import (
    RecipeSourceUnavailableError,
    import_recipe_module,
    resolve_manifest,
    resolve_model_cls,
    resolve_publish_status,
    resolve_recipe_dir,
)
from qai_hub_models.utils.export.dispatch import select_pipeline

_PROMPT_STATUSES = (MODEL_STATUS.UNPUBLISHED, MODEL_STATUS.PENDING)


def _confirm_run_ok(source_dir: Path, args: argparse.Namespace | None = None) -> bool:
    """Return False iff the user declines a pre-run warning.

    Two prompts, both skipped in dev mode / CI / pytest.

    **Unpublished recipe.** Only in-tree ``unpublished`` and ``pending`` recipes
    prompt: the recipe is in the Qualcomm catalog but has not cleared review, so
    the warning's offer of no support is accurate.

    External recipes have no publish status and never prompt. The warning would
    tell authors their own recipe might not meet standards Qualcomm never applied
    to it, on every single run.

    **Known-failing path.** When ``args`` carries the resolved precision and
    runtime, a pair the manifest records a failure for prompts as well. Such a
    pair stays selectable on purpose — the record can be stale — but running one
    unwarned means paying for Hub jobs the recipe already expects to fail.
    """
    if (
        resolve_publish_status(source_dir) in _PROMPT_STATUSES
        and not check_unpublished_model_warning()
    ):
        return False
    return args is None or _confirm_path_ok(resolve_manifest(source_dir), args)


def _passing_paths(
    manifest: QAIHMModelManifest, paths: dict[Precision, list[TargetRuntime]]
) -> dict[Precision, list[TargetRuntime]]:
    """``paths`` minus every pair the manifest records a failure for."""
    return {
        precision: [
            runtime
            for runtime in runtimes
            if manifest.failure_reason(precision, runtime) is None
        ]
        for precision, runtimes in paths.items()
    }


def _confirm_path_ok(manifest: QAIHMModelManifest, args: argparse.Namespace) -> bool:
    """Return False iff the user declines running a path recorded as failing."""
    precision = getattr(args, "precision", None)
    runtime = getattr(args, "target_runtime", None)
    if precision is None or runtime is None:
        return True
    reason = manifest.failure_reason(precision, runtime)
    if reason is None:
        return True
    return check_disabled_path_warning(f"{precision} + {runtime.value}", reason)


def build_export_parser_for(source_dir: Path) -> argparse.ArgumentParser:
    """Build the export parser for the recipe at *source_dir*."""
    manifest = resolve_manifest(source_dir)
    export_paths = manifest.get_supported_paths_for_export()
    parser = export_parser(
        model_cls=resolve_model_cls(source_dir),
        export_fn=select_pipeline(source_dir),
        supported_precision_runtimes=export_paths,
        default_export_device=manifest.default_device,
        omit_precision=manifest.separate_quantize_script,
    )
    parser.set_preferred_precision_runtimes(_passing_paths(manifest, export_paths))
    return parser


def build_evaluate_parser_for(source_dir: Path) -> argparse.ArgumentParser:
    """Build the evaluate parser for the recipe at *source_dir*."""
    manifest = resolve_manifest(source_dir)
    model_cls = cast(type[BaseModel], resolve_model_cls(source_dir))
    supports_quant_cpu = (
        manifest.can_use_quantize_job and manifest.supports_quantization
    )
    export_paths = manifest.get_supported_paths_for_export()
    parser = evaluate_parser(
        model_cls=model_cls,
        supported_dataset_classes=model_cls.get_eval_dataset_classes(),
        supported_precision_runtimes=export_paths,
        uses_quantize_job=supports_quant_cpu,
        num_calibration_samples=manifest.num_calibration_samples
        if manifest.num_calibration_samples
        else None,
        default_device=manifest.default_device,
    )
    parser.set_preferred_precision_runtimes(_passing_paths(manifest, export_paths))
    parser.add_argument(
        "--local-model-dir",
        dest="local_model_dir",
        type=str,
        default=None,
        help="Run precompiled_qnn_onnx model(s) on this machine instead of on "
        "Workbench. For a single model, the directory holding its wrapper .onnx "
        "and .bin; for a collection model, a directory with one subdirectory "
        "per component.",
    )
    return parser


def _run_llm_evaluate(source_dir: Path, forwarded: list[str]) -> None:
    """Run the recipe's own evaluate.py, which evaluates on the host GPU/CPU.

    The generic evaluate parser offers --device / --chipset, but LLM
    evaluation never runs on a Workbench device.
    """
    if not (source_dir / "evaluate.py").exists():
        raise RecipeSourceUnavailableError(
            f"Recipe `{source_dir.name}` ships no evaluate script in this "
            "installation, so it cannot be evaluated locally."
        )
    module_name = f"{import_recipe_module(source_dir).__name__}.evaluate"
    if not _confirm_run_ok(source_dir):
        return
    saved_argv = sys.argv
    sys.argv = [module_name, *forwarded]
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved_argv


def run_model_script(model_id: str | Path, script: str, forwarded: list[str]) -> None:
    """Run the given script for the given recipe target.

    Parameters
    ----------
    model_id
        A recipe target — either a folder path (``my_model``, ``Path``, or
        path string) or a bare installed model id (``"yolov8_det"``).
        Kept named ``model_id`` for call-site compatibility. The lean CLI
        rejects ``script="install"``/``"generate-files"``/``"validate"``
        with a folder before this is called. For
        ``script="configure-dataset"`` this is not a recipe at all but a
        dotted dataset class path, forwarded verbatim.
    script
        Script name: ``"demo"``, ``"quantize"``, ``"export"``, ``"evaluate"``, ``"install"``,
        ``"generate-files"``, ``"validate"``, ``"upload-to-hf"``, or
        ``"configure-dataset"``.
    forwarded
        Argv tail handed to the target's parser.
    """
    try:
        _run_model_script(model_id, script, forwarded)
    except (ModuleNotFoundError, ImportError) as e:
        model_id = str(model_id)
        rerun = shlex.join(["qai-hub-models", script, model_id, *forwarded])
        raise CLIUserError(
            f"{type(e).__name__} encountered during {script}:\n    {e}\n\n"
            f"You might be missing required dependencies for {model_id}. "
            f"Try running the following:\n"
            f"   qai-hub-models install {shlex.quote(model_id)}\n"
            f"   {rerun}"
        ) from e


def _run_model_script(model_id: str | Path, script: str, forwarded: list[str]) -> None:
    if script == "install":
        install_main([str(model_id), *forwarded])
        return

    if script == "generate-files":
        generate_files_main([str(model_id), *forwarded])
        return

    if script == "validate":
        validate_main([str(model_id), *forwarded])
        return

    if script == "upload-to-hf":
        upload_to_hf_main([str(model_id), *forwarded])
        return

    # Not a recipe target: this one's first positional is a dotted dataset class
    # path, which the lean CLI passes through untouched.
    if script == "configure-dataset":
        configure_dataset_main([str(model_id), *forwarded])
        return

    source_dir = resolve_recipe_dir(model_id)
    if script in ("demo", "quantize"):
        module_name = f"{import_recipe_module(source_dir).__name__}.{script}"
        module = importlib.import_module(module_name)
        if not _confirm_run_ok(source_dir):
            return
        # These scripts build their own parser internally and read sys.argv
        # rather than taking argv, so the tail is handed over that way.
        saved_argv = sys.argv
        sys.argv = [f"{source_dir.name}.{script}", *forwarded]
        try:
            if hasattr(module, "main"):
                module.main()
            else:
                # Some recipes only have an `if __name__ == "__main__":` block.
                sys.modules.pop(module_name)
                runpy.run_module(module_name, run_name="__main__")
        finally:
            sys.argv = saved_argv
        return

    if script == "export":
        parser = build_export_parser_for(source_dir)
        parser.prog = f"qai_hub_models export {source_dir.name}"
        args = parser.parse_args(forwarded)
        if not _confirm_run_ok(source_dir, args):
            return
        select_pipeline(source_dir)(**vars(args))
        return

    if script == "evaluate":
        if resolve_manifest(source_dir).model_type_llm:
            _run_llm_evaluate(source_dir, forwarded)
            return
        # Recipes with a bespoke evaluate (LLMs) add flags the generic parser
        # lacks, so their own build_parser()/main() pair wins when present.
        evaluate_module = (
            importlib.import_module(
                f"{import_recipe_module(source_dir).__name__}.evaluate"
            )
            if (source_dir / "evaluate.py").exists()
            else None
        )
        if evaluate_module is not None and hasattr(evaluate_module, "build_parser"):
            parser = evaluate_module.build_parser()
            manifest = resolve_manifest(source_dir)
            parser.set_preferred_precision_runtimes(
                _passing_paths(manifest, manifest.get_supported_paths_for_export())
            )
            run = evaluate_module.main
        else:
            parser = build_evaluate_parser_for(source_dir)
            run = lambda args: select_evaluate_pipeline(source_dir)(**vars(args))  # noqa: E731
        parser.prog = f"qai-hub-models evaluate {source_dir.name}"
        args = parser.parse_args(forwarded)
        if not _confirm_run_ok(source_dir, args):
            return
        run(args)
        return

    raise ValueError(
        "This function currently only supports demo, quantize, evaluate, export, install, "
        "generate-files, validate, and upload-to-hf."
    )
