# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.utils._readme_template_args import (
    get_shared_template_args,
    has_machine_gated_entries,
)
from qai_hub_models.utils.asset_loaders import UNPUBLISHED_MODEL_WARNING
from qai_hub_models.utils.path_helpers import MODEL_IDS

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
            "has_machine_gated_entries": has_machine_gated_entries(manifest),
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
