# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from qai_hub_models.configs._info_yaml_enums import MODEL_DOMAIN_USE_CASES
from qai_hub_models.configs.manifest_yaml import (
    MODEL_DOMAIN,
    MODEL_STATUS,
    MODEL_USE_CASE,
    QAIHMModelManifest,
)
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_PACKAGE_NAME,
    QAIHM_PACKAGE_SRC_ROOT,
    QAIHM_REPO_ROOT,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
GLOBAL_README_TEMPLATE = jinja_env.get_template("global_readme_template.j2")


def _get_model_row(model: QAIHMModelManifest) -> dict[str, str]:
    """Build a row dict for a single model entry."""
    assert model.id is not None
    assert model.name is not None
    readme_path = str(model.get_readme_path(Path("src") / QAIHM_PACKAGE_NAME))
    if os.path.exists(model.get_perf_yaml_path()):
        model_url = str(ASSET_CONFIG.get_website_url(model.id, relative=False))
    else:
        model_url = readme_path
    return {
        "name": model.name,
        "model_url": model_url,
        "package_name": model.get_package_name(),
        "readme_path": readme_path,
    }


def _get_model_directory(models: list[QAIHMModelManifest]) -> list[dict[str, Any]]:
    """Build structured model directory data grouped by domain and use case."""
    domains = []
    ordered_domains = sorted(
        MODEL_DOMAIN, key=lambda d: d != MODEL_DOMAIN.GENERATIVE_AI
    )
    for domain in ordered_domains:
        domain_models = [m for m in models if m.domain == domain]
        if not domain_models:
            continue

        use_cases: list[MODEL_USE_CASE] = MODEL_DOMAIN_USE_CASES[domain]
        sections: list[dict[str, Any]] = []
        if not use_cases:
            sections.append(
                {
                    "title": None,
                    "rows": [_get_model_row(m) for m in domain_models],
                }
            )
        else:
            for use_case in use_cases:
                uc_models = sorted(
                    [m for m in domain_models if m.use_case == use_case],
                    key=lambda x: x.name or "",
                )
                if uc_models:
                    sections.append(
                        {
                            "title": use_case.value,
                            "rows": [_get_model_row(m) for m in uc_models],
                        }
                    )

        domains.append({"name": domain.value, "sections": sections})

    return domains


def generate_global_readme(
    models: list[QAIHMModelManifest],
    repo_root: Path,
    pypi_root: Path,
) -> tuple[Path, Path]:
    """Generate the repo README and PyPI README by filling in the model table.

    *models* must already be filtered to published models.
    Returns (repo_readme_path, pypi_readme_path).
    """
    model_directory = _get_model_directory(models)

    paths: list[Path] = []
    for root, include_relative_links in [
        (repo_root, True),
        (pypi_root, False),
    ]:
        readme = GLOBAL_README_TEMPLATE.module.render(  # type: ignore[attr-defined]
            model_directory=model_directory,
            include_relative_links=include_relative_links,
        )
        readme_path = root / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)
        paths.append(readme_path)

    return paths[0], paths[1]


def main() -> None:
    """Generate the global README with a summary table of published models."""
    models = [
        sc.manifest
        for sc in (QAIHMModelInternalConfig.from_model(mid) for mid in MODEL_IDS)
        if sc.status is MODEL_STATUS.PUBLISHED
    ]
    generate_global_readme(models, QAIHM_REPO_ROOT, QAIHM_PACKAGE_SRC_ROOT)


if __name__ == "__main__":
    main()
