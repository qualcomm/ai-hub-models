# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""One in-tree model's full description: its manifest plus its internal config.

The website feed and the catalog need fields from both files. Rather than let either
schema reach across to the other, this pairs them by composition and owns the output
that spans them (``to_proto``, ``to_internal_proto``, publication gating).

Deliberately not a proxy: ``info.manifest.name`` and ``info.internal.status``
say which file a field lives in, and a ``__getattr__`` forwarder would hide that.
"""

from __future__ import annotations

import os

from qai_hub_models_cli.proto import model_manifest_pb2

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.proto_helpers import (
    call_to_action_to_proto,
    form_factor_to_proto,
    status_to_proto,
)
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
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

    def to_proto(self, aihm_version: str) -> model_manifest_pb2.ModelManifest:
        """Release ``ModelManifest`` with every field filled in, defaults included."""
        manifest = self.manifest
        assert manifest.is_full_model_manifest, (
            "to_proto only valid for model manifests"
        )
        assert manifest.domain is not None
        assert manifest.use_case is not None
        assert manifest.license_type is not None

        msg = manifest.to_proto(include_defaults=True)
        msg.aihm_version = aihm_version
        return msg

    def to_internal_proto(self, aihm_version: str) -> model_manifest_pb2.ModelManifest:
        """Release ``ModelManifest`` plus the private ``_internal`` fields."""
        internal = self.internal
        msg = self.to_proto(aihm_version)
        msg._internal.CopyFrom(
            model_manifest_pb2.ModelManifest.Internal(
                status=status_to_proto(internal.status),
                status_reason=internal.status_reason,
                applicable_scenarios=internal.applicable_scenarios,
                related_models=internal.related_models,
                model_maker_id=internal.model_maker_id,
                has_static_banner=internal.has_static_banner,
                has_animated_banner=internal.has_animated_banner,
                imsdk_supported=internal.imsdk_supported,
                private_perf_form_factors=[
                    form_factor_to_proto(ff)
                    for ff in (internal.private_perf_form_factors or [])
                ],
                standalone_components=internal.standalone_components,
            )
        )
        if internal.llm_details is not None:
            msg._internal.llm_details.CopyFrom(
                model_manifest_pb2.ModelManifest.Internal.LLMDetails(
                    call_to_action=call_to_action_to_proto(
                        internal.llm_details.call_to_action
                    ),
                    genie_compatible=internal.llm_details.genie_compatible,
                    geniex_llamacpp_compatible=internal.llm_details.geniex_llamacpp_compatible,
                )
            )
        return msg

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
