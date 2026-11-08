# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from qai_hub_models_cli.proto import release_assets_pb2
from qai_hub_models_cli.proto_helpers._common import (
    read_proto_yaml,
    write_proto_yaml,
)

from qai_hub_models import Precision
from qai_hub_models.configs.proto_helpers import (
    precision_from_proto,
    precision_to_proto,
    runtime_from_proto,
    runtime_to_proto,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

# llama.cpp assets are GGUF urls a human types in; nothing regenerates them, so
# a scoped drop that removed one would lose it permanently (see bc87e013f4).
NEVER_DROPPED = frozenset({ScorecardProfilePath.GENIEX_LLAMACPP})


class QAIHMModelReleaseAssets(BaseQAIHMConfig):
    """Schema for model release_assets.yaml files."""

    # Version of QAIHM (only present in manifest.yaml, not release-assets.yaml)
    version: str | None = None

    class AssetDetails(BaseQAIHMConfig):
        # Key for object in the AI Hub Models S3 Bucket
        # (see qai_hub_models/utils/aws.py)
        # Optional for manifest.yaml (uses download_url instead)
        s3_key: str | None = None

        # Tool versions used to generate this asset.
        tool_versions: ToolVersions | None = None

        # Optional: Public download URL (populated for manifest generation)
        download_url: str | None = None

        @classmethod
        def from_proto(
            cls, proto: release_assets_pb2.ModelReleaseAssets.AssetDetails
        ) -> QAIHMModelReleaseAssets.AssetDetails:
            return cls(
                s3_key=proto.s3_key if proto.HasField("s3_key") else None,
                tool_versions=(
                    ToolVersions.from_proto(proto.tool_versions)
                    if proto.HasField("tool_versions")
                    else None
                ),
                download_url=proto.download_url or None,
            )

    class PrecisionDetails(BaseQAIHMConfig):
        universal_assets: dict[
            ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails
        ] = Field(default_factory=dict)
        chipset_assets: dict[
            str,
            dict[ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails],
        ] = Field(default_factory=dict)

    precisions: dict[Precision, QAIHMModelReleaseAssets.PrecisionDetails] = Field(
        default_factory=dict
    )

    @property
    def empty(self) -> bool:
        return not self.precisions

    @property
    def has_ephemeral_s3_keys(self) -> bool:
        """Whether any asset's s3_key sits under the auto-purged ephemeral prefix.

        Ephemeral assets are uploaded to ``ephemeral_test_assets/`` and pruned
        by S3 lifecycle after 7 days; canonical assets live under
        ``pre_release_assets/`` and are kept indefinitely. Per-model
        release-assets.yaml files must only contain canonical entries — committing
        an ephemeral key would leave the manifest pointing at a soon-deleted zip.
        """
        for prec_details in self.precisions.values():
            for asset in prec_details.universal_assets.values():
                if asset.s3_key and asset.s3_key.startswith("ephemeral_test_assets/"):
                    return True
            for path_dict in prec_details.chipset_assets.values():
                for asset in path_dict.values():
                    if asset.s3_key and asset.s3_key.startswith(
                        "ephemeral_test_assets/"
                    ):
                        return True
        return False

    def add_asset(
        self,
        details: QAIHMModelReleaseAssets.AssetDetails,
        precision: Precision,
        chipset: str | None,
        path: ScorecardProfilePath,
    ) -> None:
        if precision not in self.precisions:
            self.precisions[precision] = QAIHMModelReleaseAssets.PrecisionDetails()
        if chipset:
            if chipset not in self.precisions[precision].chipset_assets:
                self.precisions[precision].chipset_assets[chipset] = {}
            self.precisions[precision].chipset_assets[chipset][path] = details
        else:
            self.precisions[precision].universal_assets[path] = details

    def drop_entries_in_scope(
        self,
        scope: set[tuple[Precision, str | None, ScorecardProfilePath]],
    ) -> None:
        """Delete every asset whose (precision, chipset, path) key is in scope.

        Paths in NEVER_DROPPED are exempt: they are hand-authored source data
        living in a generated file, so no run may delete one.
        """
        empty_precisions: list[Precision] = []
        for precision, prec_details in self.precisions.items():
            in_scope = {
                (chipset, path)
                for (p, chipset, path) in scope
                if p == precision and path not in NEVER_DROPPED
            }
            if not in_scope:
                continue
            for path in [
                p for p in prec_details.universal_assets if (None, p) in in_scope
            ]:
                prec_details.universal_assets.pop(path, None)
            empty_chipsets: list[str] = []
            for chipset, path_dict in prec_details.chipset_assets.items():
                for path in [p for p in path_dict if (chipset, p) in in_scope]:
                    path_dict.pop(path, None)
                if not path_dict:
                    empty_chipsets.append(chipset)
            for chipset in empty_chipsets:
                prec_details.chipset_assets.pop(chipset, None)
            if not prec_details.universal_assets and not prec_details.chipset_assets:
                empty_precisions.append(precision)
        for precision in empty_precisions:
            self.precisions.pop(precision, None)

    def merge_from(self, other: QAIHMModelReleaseAssets) -> None:
        """Fold *other* into self, with other's entries taking precedence on collision."""
        for precision, prec_details in other.precisions.items():
            if precision not in self.precisions:
                self.precisions[precision] = QAIHMModelReleaseAssets.PrecisionDetails()
            self_prec = self.precisions[precision]
            self_prec.universal_assets.update(prec_details.universal_assets)
            for chipset, path_dict in prec_details.chipset_assets.items():
                if chipset not in self_prec.chipset_assets:
                    self_prec.chipset_assets[chipset] = {}
                self_prec.chipset_assets[chipset].update(path_dict)

    def get_asset(
        self,
        precision: Precision,
        chipset: str | None,
        path: ScorecardProfilePath,
    ) -> QAIHMModelReleaseAssets.AssetDetails | None:
        if precision not in self.precisions:
            return None
        if chipset:
            if chipset not in self.precisions[precision].chipset_assets:
                return None
            return self.precisions[precision].chipset_assets[chipset].get(path)
        return self.precisions[precision].universal_assets.get(path)

    def to_proto(
        self,
        aihm_version: str,
        model_id: str,
        include_internal_fields: bool = False,
    ) -> release_assets_pb2.ModelReleaseAssets:
        """
        Build a ``ModelReleaseAssets`` proto.

        By default this is the published release proto: only assets with a
        ``download_url``, never the private ``s3_key``. With
        ``include_internal_fields`` it is the committed-file form instead: every
        asset, with its ``s3_key``, and ``tool_versions`` set only when present.
        """
        proto = release_assets_pb2.ModelReleaseAssets(
            aihm_version=aihm_version, model_id=model_id
        )
        for precision, prec_details in self.precisions.items():
            groups = [
                (None, prec_details.universal_assets),
                *prec_details.chipset_assets.items(),
            ]
            for chipset, path_dict in groups:
                for path, asset in path_dict.items():
                    if not include_internal_fields and not asset.download_url:
                        continue
                    entry = proto.assets.add(
                        precision=precision_to_proto(precision),
                        runtime=runtime_to_proto(path.runtime),
                    )
                    if chipset:
                        entry.chipset = chipset
                    if asset.download_url is not None:
                        entry.download_url = asset.download_url
                    if asset.tool_versions is not None:
                        entry.tool_versions.CopyFrom(asset.tool_versions.to_proto())
                    if include_internal_fields:
                        if asset.s3_key is not None:
                            entry.s3_key = asset.s3_key
                    else:
                        entry.tool_versions.SetInParent()
        return proto

    @classmethod
    def from_proto(
        cls, proto: release_assets_pb2.ModelReleaseAssets
    ) -> QAIHMModelReleaseAssets:
        """Build the in-memory representation from a ``ModelReleaseAssets`` proto."""
        # Insert in (precision, chipset, path) name order, the order the old
        # name-keyed YAML loaded in, so the release proto's order is unchanged.
        entries = sorted(
            (
                (
                    precision_from_proto(a.precision),
                    a.chipset if a.HasField("chipset") else None,
                    ScorecardProfilePath.from_runtime(runtime_from_proto(a.runtime)),
                    a,
                )
                for a in proto.assets
            ),
            key=lambda e: (str(e[0]), e[1] or "", str(e[2])),
        )
        assets = cls()
        for precision, chipset, path, asset in entries:
            assets.add_asset(
                cls.AssetDetails.from_proto(asset), precision, chipset, path
            )
        return assets

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike, create_empty_if_no_file: bool = False
    ) -> QAIHMModelReleaseAssets:
        """Read a release-assets.yaml, the YAML form of a ``ModelReleaseAssets`` proto."""
        if create_empty_if_no_file and (
            not os.path.exists(path) or os.path.getsize(path) == 0
        ):
            return cls()
        return cls.from_proto(
            read_proto_yaml(path, release_assets_pb2.ModelReleaseAssets)
        )

    def to_yaml(self, path: str | os.PathLike, **kwargs: Any) -> bool:  # type: ignore[override]
        """
        Write these assets to ``path`` as the YAML form of a ``ModelReleaseAssets`` proto.

        If there are no assets, any existing file at ``path`` is deleted instead.

        Parameters
        ----------
        path
            Where to write the file.
        **kwargs
            Unused. Accepted so the signature matches ``BaseQAIHMConfig.to_yaml``.

        Returns
        -------
        written : bool
            True if the file was written.
        """
        path = Path(path)
        proto = self.to_proto("", "", include_internal_fields=True)
        if not proto.assets:
            path.unlink(missing_ok=True)
            return False
        proto.assets.sort(key=lambda a: (a.precision, a.chipset, a.runtime))
        write_proto_yaml(proto, path)
        return True

    @classmethod
    def from_model(
        cls: type[QAIHMModelReleaseAssets], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelReleaseAssets:
        assets_path = QAIHM_MODELS_ROOT / model_id / "release-assets.yaml"
        if not_exists_ok and not os.path.exists(assets_path):
            return QAIHMModelReleaseAssets()
        return cls.from_yaml(assets_path)

    def to_model_yaml(self, model_id: str) -> Path:
        out = QAIHM_MODELS_ROOT / model_id / "release-assets.yaml"
        self.to_yaml(out)
        return out
