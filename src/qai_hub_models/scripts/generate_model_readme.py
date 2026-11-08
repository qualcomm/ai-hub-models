# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, UNPUBLISHED_MODEL_WARNING
from qai_hub_models.utils.path_helpers import MODEL_IDS

GENIEX_RUNTIMES = (TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
README_TEMPLATE = jinja_env.get_template("model_readme_template.j2")


def main() -> None:
    """Generate a README for each model, and save it in <model_dir>/README.md."""
    for model_id in MODEL_IDS:
        generate_and_write_model_readme(model_id)


def _has_machine_gated_entries(manifest: QAIHMModelManifest) -> bool:
    """True when the manifest declares any GPU-specific pre/post pip command.

    The README-side effect is a note telling the reader the recipe has GPU
    requirements. The actual CPU/GPU split is handled at install time by
    the ``qai-hub-models install`` CLI, which auto-detects CUDA.
    """
    return any(
        c.machine != "any"
        for c in manifest.pre_pip_install_commands + manifest.post_pip_install_commands
    )


def get_shared_template_args(manifest: QAIHMModelManifest) -> dict[str, Any]:
    """
    Get template arguments shared between regular README and HF model card.

    Parameters
    ----------
    manifest
        Model info object.

    Returns
    -------
    template_args: dict[str, Any]
        Dictionary of shared template arguments.
    """
    # {genie_url} (legacy Genie) and {geniex_url} (GenieX) are distinct
    # placeholders pointing at different tutorials. A section that mixes both is
    # almost always a typo (e.g. using {genie_url} while expecting the GenieX
    # URL), and the wrong link would render silently. Reject it at generation
    # time instead.
    additional_readme_section = manifest.additional_readme_section
    if (
        "{genie_url}" in additional_readme_section
        and "{geniex_url}" in additional_readme_section
    ):
        raise ValueError(
            f"{manifest.id}: additional_readme_section mixes the legacy "
            "{genie_url} and {geniex_url} placeholders. Use only one."
        )

    orchestrator_runtimes = set(manifest.orchestrator_runtimes)
    has_geniex_runtime = any(r in orchestrator_runtimes for r in GENIEX_RUNTIMES)
    has_genie_runtime = TargetRuntime.GENIE in orchestrator_runtimes
    # restrict_model_sharing means artifacts cannot be redistributed, so the
    # user must run the export step themselves before pointing GenieX at a
    # local path. That is exactly what needs_local_export gates in the README.
    needs_local_export = manifest.restrict_model_sharing

    # The macro and additional_readme_section both emit a "Deploying X
    # on-device" heading; rendering both would produce duplicate sections.
    if has_geniex_runtime and additional_readme_section:
        raise ValueError(
            f"{manifest.id}: has both a GenieX orchestrator runtime and an "
            "additional_readme_section. Remove the additional_readme_section "
            "so the deploying_on_device macro renders the deployment guidance."
        )

    return {
        # Model info
        "model_id": manifest.id,
        "model_name": manifest.name,
        "model_description": manifest.description,
        "model_url": f"{ASSET_CONFIG.models_website_url}/models/{manifest.id}",
        "additional_readme_section": additional_readme_section.replace(
            "{genie_url}", ASSET_CONFIG.genie_url
        ).replace("{geniex_url}", ASSET_CONFIG.geniex_url),
        # Source and references
        "source_repo": manifest.source_repo,
        "research_paper_title": manifest.research_paper_title,
        "research_paper_url": manifest.research_paper,
        "license_url": manifest.license,
        # Flags
        "include_gen_ai_terms": manifest.is_gen_ai_model,
        "voice_ai_sdk": manifest.voice_ai_sdk,
        "voice_ai_url": (
            ASSET_CONFIG.get_voice_ai_url(manifest.voice_ai_sdk)
            if manifest.voice_ai_sdk is not None
            else None
        ),
        # On-device deployment section (driven by orchestrator_runtimes)
        "has_geniex_runtime": has_geniex_runtime,
        "has_genie_runtime": has_genie_runtime,
        "needs_local_export": has_geniex_runtime and needs_local_export,
        "geniex_url": ASSET_CONFIG.geniex_url,
        "geniex_quickstart_url": ASSET_CONFIG.geniex_quickstart_url,
        "genie_url": ASSET_CONFIG.genie_url,
    }


def generate_and_write_model_readme(model_id: str) -> Path:
    """Generate a README for this model from the jinja template and write it."""
    sc = QAIHMModelInternalConfig.from_model(model_id)
    manifest = sc.manifest
    assert manifest.headline is not None

    # Convert precisions to strings for Jinja template. Populated for both
    # quantize-job models and separate_quantize_script (LLM) models; the
    # macro decides whether to render the demo's --quantize hint vs. the
    # DEFAULT_<PRECISION> checkpoint list based on separate_quantize_script.
    supported_precisions = None
    if manifest.supported_precisions and (
        manifest.can_use_quantize_job or manifest.separate_quantize_script
    ):
        supported_precisions = [str(p) for p in manifest.supported_precisions]

    template_vars = get_shared_template_args(manifest)
    template_vars.update(
        {
            # Model info
            "model_status": sc.status.value,
            "unpublished_model_warning": UNPUBLISHED_MODEL_WARNING,
            "model_headline": manifest.headline.strip("."),
            "model_package": manifest.get_package_name(),
            "model_assets_shareable": not manifest.restrict_model_sharing,
            "default_runtime": manifest.default_runtime.value,
            "default_precision": str(manifest.default_precision),
            "supported_precisions": supported_precisions,
            "separate_quantize_script": manifest.separate_quantize_script,
            # Package installation
            "has_machine_gated_entries": _has_machine_gated_entries(manifest),
            "python_version_gte": manifest.python_version_greater_than_or_equal_to,
            "python_version_lt": manifest.python_version_less_than,
            # Flags
            "include_example_and_usage": not manifest.skip_example_usage,
            "has_on_target_demo": manifest.has_on_target_demo,
            "local_device_deployment": manifest.local_device_deployment,
            # System-level dependencies installation instructions
            "readme_install_system_deps": manifest.readme_install_system_deps,
        }
    )

    readme_path = manifest.get_readme_path()
    with open(readme_path, "w") as readme_file:
        readme_file.write(README_TEMPLATE.module.render(**template_vars))  # type: ignore[attr-defined]
    return readme_path


if __name__ == "__main__":
    main()
