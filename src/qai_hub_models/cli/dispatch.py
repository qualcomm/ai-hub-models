# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Heavy-side dispatcher for ``qai_hub_models <script> <model_id>`` subcommands.

Imports ``qai_hub_models.models.<model_id>.<script>``, builds its native
argparse parser via ``build_parser()``, parses the forwarded args, then calls
``main(parsed_args)``.

The lean CLI is responsible for resolving display name -> model id and
checking the model exists in the installed package before calling in.
"""

from __future__ import annotations

import argparse
from typing import cast

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.args import evaluate_parser, export_parser
from qai_hub_models.utils.asset_loaders import (
    check_unpublished_model_warning,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline
from qai_hub_models.utils.export.dispatch import (
    ResolvedModel,
    resolve_model,
    select_pipeline,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS


def _confirm_run_ok(model_id_or_module: str) -> bool:
    """Return False if an unpublished/unknown model should not proceed.

    Known unpublished ids get a confirmation prompt via
    :func:`check_unpublished_model_warning`. Module paths outside
    ``MODEL_IDS`` execute untrusted code on import and require the same
    confirmation.

    Parameters
    ----------
    model_id_or_module
        Model ID or importable dotted module path.

    Returns
    -------
    bool
        True to proceed, False to abort.
    """
    if model_id_or_module in MODEL_IDS:
        try:
            manifest_status = QAIHMModelManifest.from_model(model_id_or_module).status
            status = (
                manifest_status.value if manifest_status is not None else "unpublished"
            )
        except ValueError:
            status = "unpublished"
        if status == "published":
            return True
    return check_unpublished_model_warning()


def build_export_parser_for(resolved: ResolvedModel) -> argparse.ArgumentParser:
    """Build the export parser from an already-resolved model.

    Parameters
    ----------
    resolved
        Model metadata from :func:`resolve_model`.

    Returns
    -------
    argparse.ArgumentParser
        The model's native export argument parser.
    """
    return export_parser(
        model_cls=resolved.model_cls,
        export_fn=select_pipeline(resolved),
        supported_precision_runtimes=resolved.manifest.get_supported_paths_for_export(),
        default_export_device=resolved.manifest.default_device,
        omit_precision=resolved.manifest.separate_quantize_script,
    )


def build_export_parser(model_id_or_module: str) -> argparse.ArgumentParser:
    """Build the argparse parser for exporting this model.

    Equivalent to the `build_parser()` function that was previously generated
    in each `models/<model_id>/export.py`.

    Parameters
    ----------
    model_id_or_module
        Model ID (e.g. ``"mobilenet_v2"``) or an importable dotted module path.

    Returns
    -------
    argparse.ArgumentParser
        The model's native export argument parser.
    """
    return build_export_parser_for(resolve_model(model_id_or_module))


def run_export(model_id_or_module: str, args: argparse.Namespace) -> None:
    """Run export for the given model with parsed arguments.

    Equivalent to the `main(args)` function that was previously generated
    in each `models/<model_id>/export.py`.

    Parameters
    ----------
    model_id_or_module
        Model ID (e.g. ``"mobilenet_v2"``) or an importable dotted module path.
    args
        Parsed arguments from ``build_export_parser``.
    """
    if not _confirm_run_ok(model_id_or_module):
        return
    resolved = resolve_model(model_id_or_module)
    select_pipeline(resolved)(resolved.model_id, **vars(args))


def build_evaluate_parser_for(resolved: ResolvedModel) -> argparse.ArgumentParser:
    """Build the evaluate parser from an already-resolved model.

    Parameters
    ----------
    resolved
        Model metadata from :func:`resolve_model`.

    Returns
    -------
    argparse.ArgumentParser
        The model's native evaluate argument parser.
    """
    model_cls = cast(type[BaseModel], resolved.model_cls)
    return evaluate_parser(
        model_cls=model_cls,
        supported_dataset_classes=model_cls.get_eval_dataset_classes(),
        supported_precision_runtimes=resolved.manifest.get_supported_paths_for_export(),
        uses_quantize_job=resolved.supports_quant_cpu,
        num_calibration_samples=resolved.manifest.num_calibration_samples
        if resolved.manifest.num_calibration_samples
        else None,
        default_device=resolved.manifest.default_device,
    )


def build_evaluate_parser(model_id_or_module: str) -> argparse.ArgumentParser:
    """Build the argparse parser for evaluating this model.

    Equivalent to the `build_parser()` function that was previously generated
    in each `models/<model_id>/evaluate.py`.

    Parameters
    ----------
    model_id_or_module
        Model ID (e.g. ``"mobilenet_v2"``) or an importable dotted module path.

    Returns
    -------
    argparse.ArgumentParser
        The model's native evaluate argument parser.
    """
    return build_evaluate_parser_for(resolve_model(model_id_or_module))


def run_evaluate(model_id_or_module: str, args: argparse.Namespace) -> None:
    """Run evaluate for the given model with parsed arguments.

    Equivalent to the `main(args)` function that was previously generated
    in each `models/<model_id>/evaluate.py`.

    Parameters
    ----------
    model_id_or_module
        Model ID (e.g. ``"mobilenet_v2"``) or an importable dotted module path.
    args
        Parsed arguments from ``build_evaluate_parser``.
    """
    if not _confirm_run_ok(model_id_or_module):
        return
    resolved = resolve_model(model_id_or_module)
    select_evaluate_pipeline(resolved)(resolved.model_id, **vars(args))


def run_model_script(model_id: str, script: str, forwarded: list[str]) -> None:
    """Run the given script for the model.

    Parameters
    ----------
    model_id
        Model directory name (e.g. ``"mobilenet_v2"``). Must already be
        validated against ``MODEL_IDS`` by the caller.
    script
        Script name (``"export"`` or ``"evaluate"``).
    forwarded
        Argv tail handed to the model's parser.
    """
    resolved = resolve_model(model_id)
    if script == "export":
        parser = build_export_parser_for(resolved)
        parser.prog = f"qai_hub_models export {model_id}"
        args = parser.parse_args(forwarded)
        if not _confirm_run_ok(model_id):
            return
        select_pipeline(resolved)(resolved.model_id, **vars(args))
        return

    if script == "evaluate":
        parser = build_evaluate_parser_for(resolved)
        parser.prog = f"qai_hub_models evaluate {model_id}"
        args = parser.parse_args(forwarded)
        if not _confirm_run_ok(model_id):
            return
        select_evaluate_pipeline(resolved)(resolved.model_id, **vars(args))
        return

    raise ValueError("This function currently only supports evaluate and export.")
