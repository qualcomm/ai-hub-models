# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""One in-tree model's full description: its manifest plus its internal config.

The website feed and the catalog need fields from both files. Rather than let either
schema reach across to the other, this pairs them by composition and owns the output
that spans them (``to_proto``, ``to_website_info_yaml``, publication gating).

Deliberately not a proxy: ``info.manifest.name`` and ``info.internal.technical_details``
say which file a field lives in, and a ``__getattr__`` forwarder would hide that.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qai_hub_models_cli.proto import info_pb2

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import (
    QAIHMModelManifest,
    TechnicalDetails,
)
from qai_hub_models.configs.proto_helpers import (
    call_to_action_from_proto,
    call_to_action_to_proto,
    domain_from_proto,
    domain_to_proto,
    form_factor_from_proto,
    form_factor_to_proto,
    license_from_proto,
    license_to_proto,
    runtime_from_proto,
    runtime_to_proto,
    status_from_proto,
    status_to_proto,
    tag_from_proto,
    tag_to_proto,
    use_case_from_proto,
    use_case_to_proto,
    voice_ai_sdk_from_proto,
    voice_ai_sdk_to_proto,
)
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.utils.base_config import dump_yaml_dict
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

__all__ = ["WebsiteModelInfo"]


class WebsiteModelInfo:
    def __init__(
        self, manifest: QAIHMModelManifest, internal: QAIHMModelInternalConfig
    ) -> None:
        self.manifest = manifest
        self.internal = internal

    @classmethod
    def from_model(cls, model_id: str) -> WebsiteModelInfo:
        internal_config = QAIHMModelInternalConfig.from_model(model_id)
        return cls(internal_config.manifest, internal_config)

    @property
    def is_published(self) -> bool:
        return self.internal.status is MODEL_STATUS.PUBLISHED

    @classmethod
    def from_manifest(cls, manifest: QAIHMModelManifest) -> WebsiteModelInfo:
        """Pair an already-loaded (possibly mutated) manifest with its internal config.

        For the scorecard collectors, which write disabled_paths and must not use the
        shared cached manifest that ``from_model`` hands out.
        """
        assert manifest.id is not None
        return cls(manifest, QAIHMModelInternalConfig.from_model(manifest.id))

    @staticmethod
    def _technical_details_to_proto(
        details: TechnicalDetails,
    ) -> list[info_pb2.ModelInfo.TechnicalDetail]:
        protos = []
        for key, val in details.items():
            td = info_pb2.ModelInfo.TechnicalDetail(key=key)
            if isinstance(val, int):
                td.int_value = val
            elif isinstance(val, float):
                td.float_value = val
            else:
                td.string_value = str(val)
            protos.append(td)
        return protos

    def to_proto(self, aihm_version: str) -> info_pb2.ModelInfo:
        manifest, internal = self.manifest, self.internal
        assert manifest.is_full_model_manifest, (
            "to_proto only valid for model manifests"
        )
        assert manifest.domain is not None
        assert manifest.use_case is not None
        assert manifest.license_type is not None
        technical_details = self._technical_details_to_proto(internal.technical_details)

        runtime_technical_details = [
            info_pb2.ModelInfo.RuntimeTechnicalDetails(
                runtime=runtime_to_proto(runtime),
                technical_details=self._technical_details_to_proto(details),
            )
            for runtime, details in manifest.runtime_technical_details.items()
        ]

        llm_details = None
        if internal.llm_details is not None:
            llm_details = info_pb2.ModelInfo.LLMDetails(
                call_to_action=call_to_action_to_proto(
                    internal.llm_details.call_to_action
                ),
                genie_compatible=internal.llm_details.genie_compatible,
                geniex_llamacpp_compatible=internal.llm_details.geniex_llamacpp_compatible,
            )

        numerics_benchmark = None
        if manifest.numerics_benchmark is not None:
            numerics_benchmark = manifest.numerics_benchmark.to_proto()

        return info_pb2.ModelInfo(
            aihm_version=aihm_version,
            id=manifest.id,
            name=manifest.name,
            status=status_to_proto(internal.status),
            status_reason=internal.status_reason,
            headline=manifest.headline,
            domain=domain_to_proto(manifest.domain),
            description=manifest.description,
            use_case=use_case_to_proto(manifest.use_case),
            tags=[tag_to_proto(t) for t in manifest.tags],
            applicable_scenarios=internal.applicable_scenarios,
            related_models=internal.related_models,
            technical_details=technical_details,
            runtime_technical_details=runtime_technical_details,
            license_type=license_to_proto(manifest.license_type),
            model_maker_id=internal.model_maker_id,
            research_paper=manifest.research_paper,
            research_paper_title=manifest.research_paper_title,
            source_repo=manifest.source_repo,
            license_url=manifest.license,
            has_static_banner=internal.has_static_banner,
            has_animated_banner=internal.has_animated_banner,
            imsdk_supported=internal.imsdk_supported,
            restrict_model_sharing=manifest.restrict_model_sharing,
            numerics_benchmark=numerics_benchmark,
            model_type_llm=manifest.model_type_llm,
            llm_details=llm_details,
            private_perf_form_factors=[
                form_factor_to_proto(ff)
                for ff in (internal.private_perf_form_factors or [])
            ],
            voice_ai_sdk=voice_ai_sdk_to_proto(manifest.voice_ai_sdk),
        )

    @staticmethod
    def _technical_details_from_proto(
        details: Iterable[info_pb2.ModelInfo.TechnicalDetail],
    ) -> dict[str, str | int | float]:
        return {
            td.key: next(v for f, v in td.ListFields() if f.name != "key")
            for td in details
        }

    def to_website_info_yaml(self, path: Path, aihm_version: str = "") -> Path:
        """
        Write the info.yaml the website importer reads.

        The fields come from to_proto(); enums and key/value lists are converted
        back to the flat string format the importer expects.
        """
        proto = self.to_proto(aihm_version)
        data: dict[str, Any] = {
            "id": proto.id,
            "name": proto.name,
            "status": status_from_proto(proto.status).value,
            "status_reason": proto.status_reason or None,
            "headline": proto.headline,
            "domain": domain_from_proto(proto.domain).value,
            "description": proto.description,
            "use_case": use_case_from_proto(proto.use_case).value,
            "tags": [tag_from_proto(t).value for t in proto.tags],
            "applicable_scenarios": list(proto.applicable_scenarios),
            "related_models": list(proto.related_models),
            "technical_details": self._technical_details_from_proto(
                proto.technical_details
            ),
            "runtime_technical_details": {
                runtime_from_proto(r.runtime).value: self._technical_details_from_proto(
                    r.technical_details
                )
                for r in proto.runtime_technical_details
            },
            "license_type": license_from_proto(proto.license_type).value,
            "license": proto.license_url or None,
            "model_maker_id": proto.model_maker_id or None,
            "research_paper": proto.research_paper or None,
            "research_paper_title": proto.research_paper_title or None,
            "source_repo": proto.source_repo or None,
            "has_static_banner": proto.has_static_banner,
            "has_animated_banner": proto.has_animated_banner,
            "imsdk_supported": proto.imsdk_supported,
            "restrict_model_sharing": proto.restrict_model_sharing,
            "model_type_llm": proto.model_type_llm,
            "voice_ai_sdk": getattr(
                voice_ai_sdk_from_proto(proto.voice_ai_sdk), "value", None
            ),
        }
        if proto.private_perf_form_factors:
            data["private_perf_form_factors"] = [
                form_factor_from_proto(ff).value
                for ff in proto.private_perf_form_factors
            ]
        if proto.HasField("numerics_benchmark"):
            nb = proto.numerics_benchmark
            data["numerics_benchmark"] = {
                "dataset_name": nb.dataset_name,
                "metric_name": nb.metric_name,
                "value": nb.value,
                "unit": nb.unit,
                "source": nb.source,
            }
        if proto.HasField("llm_details"):
            llm = proto.llm_details
            data["llm_details"] = {
                "call_to_action": call_to_action_from_proto(llm.call_to_action).value,
                "genie_compatible": llm.genie_compatible,
                "geniex_llamacpp_compatible": llm.geniex_llamacpp_compatible,
            }
        dump_yaml_dict({k: v for k, v in data.items() if v is not None}, path)
        return path

    def can_promote_to_published(self) -> tuple[bool, str]:
        """
        Check whether this model meets all prerequisites for promotion to PUBLISHED.

        Returns (True, "") if promotion is safe, or (False, reason) if not.
        """
        manifest, internal = self.manifest, self.internal
        assert manifest.license_type is not None
        assert manifest.id is not None
        if manifest.license_type.is_non_commerical:
            return (
                False,
                f"Models with license {manifest.license_type!s} cannot be published",
            )

        if not internal.has_static_banner:
            return False, "model has no static banner asset"

        if not manifest.supports_at_least_1_runtime:
            return False, "model does not support at least one export path"

        must_have_assets = not manifest.restrict_model_sharing
        if internal.llm_details:
            must_have_assets = (
                must_have_assets
                and internal.llm_details.call_to_action
                in {
                    LLM_CALL_TO_ACTION.DOWNLOAD,
                    LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
                }
            )

        if must_have_assets and not os.path.exists(
            QAIHM_MODELS_ROOT / manifest.id / "release-assets.yaml"
        ):
            return False, "no release assets available"

        return True, ""
