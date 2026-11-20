# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Maps the legacy (pre-proto) manifest.yaml format to the ``ModelManifest`` proto's dict form."""

from __future__ import annotations

from typing import Any

from qai_hub_models_cli.proto import model_manifest_pb2

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
    VOICE_AI_SDK,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.proto_helpers import (
    call_to_action_to_proto,
    domain_to_proto,
    form_factor_to_proto,
    license_to_proto,
    precision_to_proto,
    runtime_to_proto,
    status_to_proto,
    tag_to_proto,
    use_case_to_proto,
    voice_ai_sdk_to_proto,
)
from qai_hub_models.utils.device import FormFactor

_MACHINE_TO_PROTO_NAME = {
    "any": model_manifest_pb2.ModelManifest.PipCommand.Machine.Name(
        model_manifest_pb2.ModelManifest.PipCommand.MACHINE_UNSPECIFIED
    ),
    "gpu": model_manifest_pb2.ModelManifest.PipCommand.Machine.Name(
        model_manifest_pb2.ModelManifest.PipCommand.MACHINE_GPU
    ),
    "cpu": model_manifest_pb2.ModelManifest.PipCommand.Machine.Name(
        model_manifest_pb2.ModelManifest.PipCommand.MACHINE_CPU
    ),
}


def _pip_command(entry: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"command": entry}
    return {
        "command": entry["command"],
        "machine": _MACHINE_TO_PROTO_NAME[entry.get("machine", "any")],
    }


def _technical_detail(key: str, value: str | float) -> dict[str, Any]:
    if isinstance(value, int):
        return {"key": key, "int_value": value}
    if isinstance(value, float):
        return {"key": key, "float_value": value}
    return {"key": key, "string_value": value}


def _disabled_paths(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            **reasons,
            "precision": precision_to_proto(Precision.parse(str(precision))),
            "runtime": runtime_to_proto(TargetRuntime(runtime)),
        }
        for precision, runtimes in data.items()
        for runtime, reasons in runtimes.items()
    ]


def legacy_manifest_to_proto_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrite a legacy manifest.yaml dict into the dict form of a ``ModelManifest`` proto.

    Keys the proto has no field for (e.g. ``lm_quantization_details``, which now
    lives in lm_recipe.yaml) are left in place; parse with
    ``ignore_unknown_fields=True`` to drop them.

    Parameters
    ----------
    data
        The parsed legacy manifest.yaml.

    Returns
    -------
    proto_dict : dict[str, Any]
        A dict that ``json_format.ParseDict`` can load into a ``ModelManifest``.
    """
    out = dict(data)

    if (domain := out.pop("domain", None)) is not None:
        out["domain"] = domain_to_proto(MODEL_DOMAIN(domain))
    if (use_case := out.pop("use_case", None)) is not None:
        out["use_case"] = use_case_to_proto(MODEL_USE_CASE(use_case))
    if (status := out.pop("status", None)) is not None:
        out["status"] = status_to_proto(MODEL_STATUS(status))
    for field in ("form_factors", "private_perf_form_factors"):
        if (factors := out.pop(field, None)) is not None:
            out[field] = [form_factor_to_proto(FormFactor(f)) for f in factors]
    if (llm_details := out.pop("llm_details", None)) is not None:
        out["llm_details"] = {
            **llm_details,
            "call_to_action": call_to_action_to_proto(
                LLM_CALL_TO_ACTION(llm_details["call_to_action"])
            ),
        }
    if (license_type := out.pop("license_type", None)) is not None:
        out["license_type"] = license_to_proto(MODEL_LICENSE(license_type))
    if (license_url := out.pop("license", None)) is not None:
        out["license_url"] = license_url
    if (sdk := out.pop("voice_ai_sdk", None)) is not None:
        out["voice_ai_sdk"] = voice_ai_sdk_to_proto(VOICE_AI_SDK(sdk))
    out["tags"] = [tag_to_proto(MODEL_TAG(t)) for t in out.pop("tags", None) or []]
    out["supported_precisions"] = [
        precision_to_proto(Precision.parse(str(p)))
        for p in out.pop("supported_precisions", None) or []
    ]
    out["orchestrator_runtimes"] = [
        runtime_to_proto(TargetRuntime(r))
        for r in out.pop("orchestrator_runtimes", None) or []
    ]
    out["disabled_paths"] = _disabled_paths(out.pop("disabled_paths", None) or {})
    out["runtime_technical_details"] = [
        {
            "runtime": runtime_to_proto(TargetRuntime(runtime)),
            "technical_details": [
                _technical_detail(k, v) for k, v in sorted(details.items())
            ],
        }
        for runtime, details in (
            out.pop("runtime_technical_details", None) or {}
        ).items()
    ]
    out["external_repos"] = [
        {"name": name, **repo}
        for name, repo in (out.pop("external_repos", None) or {}).items()
    ]
    for field in ("pre_pip_install_commands", "post_pip_install_commands"):
        out[field] = [_pip_command(c) for c in out.pop(field, None) or []]
    out["outputs_to_skip_validation"] = {
        str(k): v
        for k, v in (out.pop("outputs_to_skip_validation", None) or {}).items()
    }
    return out
