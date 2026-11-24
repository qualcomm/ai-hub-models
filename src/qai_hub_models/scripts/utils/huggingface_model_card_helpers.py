# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any

import jinja2
from qai_hub_models_cli.proto_helpers._common import dump_yaml
from qai_hub_models_cli.versions import CURRENT_VERSION

from qai_hub_models._version import __version__ as qaihm_version
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.proto_helpers import (
    precision_from_proto,
    runtime_from_proto,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.core.configs import get_model_release_assets
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.chipset_helpers import WEBSITE_CHIPSET_ORDER
from qai_hub_models.utils._platform_helpers import (
    platform_chipsets,
    platform_devices,
)
from qai_hub_models.utils._readme_template_args import get_shared_template_args
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    QAIHM_WEB_ASSET,
)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
MODEL_CARD_TEMPLATE = JINJA_ENV.get_template("hf_model_card_template.j2")
DEPRECATED_MODEL_CARD_TEMPLATE = JINJA_ENV.get_template(
    "hf_deprecated_model_card_template.j2"
)


def _chipset_sort_key(chipset_id: str) -> int:
    """Return sort index for chipset based on WEBSITE_CHIPSET_ORDER."""
    try:
        return WEBSITE_CHIPSET_ORDER.index(chipset_id)
    except ValueError:
        return len(WEBSITE_CHIPSET_ORDER)


def get_download_links_rows(model_id: str) -> list[dict[str, Any]]:
    """
    Build download links table data as a list of row dictionaries.

    Parameters
    ----------
    model_id
        Model identifier (e.g., "resnet50").

    Returns
    -------
    rows: list[dict[str, Any]]
        List of dicts with keys: precision, runtime, chipset, tool_versions, download_url.
    """
    release_assets = get_model_release_assets(
        model_id, _set_url_version=CURRENT_VERSION
    )
    if release_assets is None:
        return []

    chipsets = platform_chipsets()
    rows: list[dict[str, Any]] = []
    for asset in release_assets.assets:
        if asset.HasField("chipset"):
            chipset_info = chipsets.get(asset.chipset)
            chipset_display = (
                chipset_info.marketing_name if chipset_info else asset.chipset
            )
            chipset_sort = _chipset_sort_key(asset.chipset)
        else:
            chipset_display = "Universal"
            chipset_sort = -1  # Universal comes first
        rows.append(
            {
                "precision": str(precision_from_proto(asset.precision)),
                "runtime": runtime_from_proto(asset.runtime).name,
                "chipset": chipset_display,
                "_chipset_sort": chipset_sort,
                "tool_versions": ToolVersions.from_proto(asset.tool_versions),
                "download_url": asset.download_url,
            }
        )

    # Sort by runtime, then precision, then chipset (using WEBSITE_CHIPSET_ORDER)
    rows.sort(key=lambda r: (r["runtime"], r["precision"], r["_chipset_sort"]))
    return rows


def get_performance_table_rows(
    perf_info: QAIHMModelPerf,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Build performance table data as a list of row dictionaries.

    Returns a tuple of (rows, is_llm) where rows is a list of dicts with keys:
    component, precision, device, chipset, runtime, and performance metrics.
    """
    rows: list[dict[str, Any]] = []
    is_llm = False

    devices = platform_devices(include_similar_devices=True)
    chipsets = platform_chipsets(include_similar_devices=True)

    for key, entry in perf_info.entries.items():
        device_chipset = devices[key.device.reference_device_name].chipset

        base_row: dict[str, Any] = {
            "component": key.component,
            "precision": str(key.precision),
            "chipset": chipsets[device_chipset].marketing_name,
            "_chipset_sort": _chipset_sort_key(device_chipset),
            "runtime": key.path.runtime.name,
        }

        if entry.llm_metrics:
            # LLM metrics - one row per (context_length, desired_compute_unit).
            is_llm = True
            for ctx in entry.llm_metrics:
                if not ctx.tokens_per_second:
                    # Decode produced 0 tokens/sec (e.g. ran out of memory
                    # mid-generation at this context length), not a
                    # measurement worth publishing.
                    continue
                row = base_row.copy()
                row["context_length"] = ctx.context_length
                row["tokens_per_second"] = str(ctx.tokens_per_second)
                row["compute_unit"] = ctx.desired_compute_unit
                rows.append(row)
        else:
            assert entry.HasField("metrics")
            metrics = entry.metrics
            assert metrics.HasField("estimated_peak_memory_range_mb")
            assert metrics.inference_time_milliseconds
            mem = metrics.estimated_peak_memory_range_mb
            row = base_row.copy()
            row["inference_time"] = f"{metrics.inference_time_milliseconds} ms"
            row["peak_memory"] = f"{mem.min} - {mem.max} MB"
            row["compute_unit"] = metrics.primary_compute_unit
            rows.append(row)

    # Sort by model, then runtime, then precision, then chipset, then context_length (using WEBSITE_CHIPSET_ORDER)
    rows.sort(
        key=lambda r: (
            r["component"],
            r["runtime"],
            r["precision"],
            r["_chipset_sort"],
            r.get("context_length", 0),
        )
    )
    return rows, is_llm


def generate_hf_model_LICENSE(manifest: QAIHMModelManifest) -> str:
    string_stream = StringIO()
    if manifest.license is not None:
        print(
            f"The license of the original trained model can be found at {manifest.license}.",
            file=string_stream,
        )
    return string_stream.getvalue()


def generate_hf_manifest(
    model_id: str,
    version: str,
) -> QAIHMModelReleaseAssets | None:
    """
    Generate a manifest with download URLs for a model's release assets.

    Parameters
    ----------
    model_id
        Model identifier (e.g., "resnet50").
    version
        Version string for the release assets (e.g., "0.45.0").

    Returns
    -------
    manifest: QAIHMModelReleaseAssets | None
        Manifest with download URLs populated, or None if model has no release assets.
    """
    release_assets = get_model_release_assets(model_id, _set_url_version=version)
    if release_assets is None:
        return None

    manifest = QAIHMModelReleaseAssets.from_proto(release_assets)
    manifest.version = version
    return manifest


def generate_hf_model_card(
    manifest: QAIHMModelManifest,
    model_perf: QAIHMModelPerf,
    model_card_template: jinja2.Template = MODEL_CARD_TEMPLATE,
) -> str:
    """Generate a model_card for this model from the Jinja template."""
    assert manifest.id is not None
    # Get shared template args
    context = get_shared_template_args(manifest)

    # Generate HuggingFace metadata YAML
    string_stream = StringIO()
    dump_yaml(manifest.get_hugging_face_metadata(), string_stream)

    # Build performance table rows
    perf_rows, is_llm = get_performance_table_rows(model_perf)

    # Build download links table rows
    download_links = []
    if not manifest.restrict_model_sharing:
        download_links = get_download_links_rows(manifest.id)

    # Determine if we should show purchase statement
    assert manifest.id is not None
    internal_config = QAIHMModelInternalConfig.from_model(manifest.id)
    show_purchase_statement = (
        internal_config.llm_details is not None
        and internal_config.llm_details.call_to_action
        == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
    )

    # Add HF-specific template args
    context.update(
        {
            # Metadata
            "hugging_face_metadata": string_stream.getvalue(),
            # Model info
            "use_case": str(manifest.use_case).lower().capitalize(),
            "technical_details": manifest.technical_details,
            "restrict_model_sharing": manifest.restrict_model_sharing,
            # Assets
            "static_image": ASSET_CONFIG.get_web_asset_url(
                manifest.id, QAIHM_WEB_ASSET.STATIC_IMG
            ),
            "qaihm_model_url": ASSET_CONFIG.get_qaihm_repo(manifest.id, relative=False),
            # Flags
            "is_llm": is_llm,
            "is_precompiled": manifest.is_precompiled,
            "show_purchase_statement": show_purchase_statement,
            # Performance data
            "perf_rows": perf_rows,
            # Download links
            "download_links": download_links,
        }
    )

    return model_card_template.module.render(**context)  # type: ignore[attr-defined]


def write_hf_model_card_and_license(
    manifest: QAIHMModelManifest,
    model_perf: QAIHMModelPerf,
    output_dir: str | os.PathLike,
) -> None:
    """
    Generate both the model card and LICENSE content for HuggingFace;
    output_dir will be ready for direct release to HuggingFace.
    """
    assert manifest.id is not None
    model_card = generate_hf_model_card(manifest, model_perf)
    license_text = generate_hf_model_LICENSE(manifest)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the model card and license to disk.
    model_card_path = output_dir / "README.md"
    with open(model_card_path, "w") as model_card_file:
        model_card_file.write(model_card)

    license_path = output_dir / "LICENSE"
    with open(license_path, "w") as license_file:
        license_file.write(license_text)

    # Write release_assets.json with download URLs for release assets.
    if not manifest.restrict_model_sharing:
        release_assets = generate_hf_manifest(manifest.id, qaihm_version)
        if release_assets is not None:
            release_assets_path = output_dir / "release_assets.json"
            release_assets.to_json(release_assets_path)


def write_deprecated_hf_model_card(output_dir: str | os.PathLike) -> None:
    """
    Write a deprecated model card to output_dir.
    output_dir will be ready for direct release to HuggingFace.
    """
    deprecated_model_card = DEPRECATED_MODEL_CARD_TEMPLATE.module.render()  # type: ignore[attr-defined]
    with open(Path(output_dir) / "README.md", "w", encoding="utf-8") as f:
        f.write(deprecated_model_card)
