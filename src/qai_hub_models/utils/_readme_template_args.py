# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Any

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG

GENIEX_RUNTIMES = (TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP)


def has_machine_gated_entries(manifest: QAIHMModelManifest) -> bool:
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
