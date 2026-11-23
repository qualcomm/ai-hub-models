# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Helpers for working with the ``ModelPerf`` proto (the in-memory and on-disk form of perf.yaml).

``ModelPerf.performance_metrics`` is a flat list of ``PerformanceDetails`` entries.
``QAIHMModelPerf`` wraps it and indexes each entry by a ``QAIHMModelPerf.Key``
(precision, component, device and profile path).
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from qai_hub_models_cli.proto import model_perf_pb2
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
from qai_hub_models.scorecard.device import ScorecardDevice, compute_peer_chipsets
from qai_hub_models.scorecard.devices_and_chipsets_yaml import (
    _load_similar_devices_raw,
)
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.chipset_helpers import (
    get_supported_devices,
    sorted_chipsets,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


@cache
def _similar_device_names() -> frozenset[str]:
    """Names of "similar" (non-workbench) devices excluded from the perf proto."""
    return frozenset(_load_similar_devices_raw().devices.keys())


MIXED_PRECISIONS = frozenset({Precision.mixed, Precision.mixed_with_float})


def _entry_sort_key(entry: model_perf_pb2.ModelPerf.PerformanceDetails) -> tuple:
    return (entry.precision, entry.component, entry.device, entry.runtime)


class QAIHMModelPerf:
    """
    A model's perf data (the in-memory form of perf.yaml).

    The ``ModelPerf`` proto is kept in ``_proto`` and should only be changed through
    this class's methods. Use ``entries`` to read the per-job entries, keyed by the
    ``Key`` that identifies them, and ``to_proto()`` to get a copy of the proto.
    """

    @dataclass(frozen=True)
    class Key:
        """Identifies an entry of a model's perf data."""

        precision: Precision
        component: str
        device: ScorecardDevice
        path: ScorecardInferencePath

        @classmethod
        def from_details(
            cls, details: model_perf_pb2.ModelPerf.PerformanceDetails
        ) -> QAIHMModelPerf.Key:
            """The key an entry's own precision, component, device and runtime identify."""
            return cls(
                precision_from_proto(details.precision),
                details.component,
                ScorecardDevice.parse(details.device),
                ScorecardInferencePath.from_runtime(
                    runtime_from_proto(details.runtime)
                ),
            )

        def stamp(
            self, details: model_perf_pb2.ModelPerf.PerformanceDetails
        ) -> model_perf_pb2.ModelPerf.PerformanceDetails:
            """A copy of ``details`` with this key's fields set."""
            out = model_perf_pb2.ModelPerf.PerformanceDetails()
            out.CopyFrom(details)
            out.precision = precision_to_proto(self.precision)
            out.component = self.component
            out.device = str(self.device)
            out.runtime = runtime_to_proto(self.path.runtime)
            return out

    def __init__(
        self, model_id: str, proto: model_perf_pb2.ModelPerf | None = None
    ) -> None:
        self.model_id = model_id
        self._proto = proto if proto is not None else model_perf_pb2.ModelPerf()
        self._entries: dict[
            QAIHMModelPerf.Key, model_perf_pb2.ModelPerf.PerformanceDetails
        ] = {}
        self._reindex()

    # -- state ---------------------------------------------------------------

    def _reindex(self) -> None:
        """Rebuild ``_entries`` from the proto. Call after changing the proto's entry list."""
        self._entries = {
            QAIHMModelPerf.Key.from_details(entry): entry
            for entry in self._proto.performance_metrics
        }

    @property
    def entries(
        self,
    ) -> Mapping[QAIHMModelPerf.Key, model_perf_pb2.ModelPerf.PerformanceDetails]:
        """Read-only view of every entry, keyed by the ``Key`` that identifies it."""
        return MappingProxyType(self._entries)

    @property
    def supported_devices(self) -> RepeatedScalarFieldContainer[str]:
        return self._proto.supported_devices

    @property
    def supported_chipsets(self) -> RepeatedScalarFieldContainer[str]:
        return self._proto.supported_chipsets

    @property
    def empty(self) -> bool:
        return not (self.supported_chipsets or self.supported_devices or self._entries)

    def apply_similar_devices(self, mapping: dict[str, tuple[str, list[str]]]) -> None:
        """
        Duplicate performance entries from supported devices to similar unsupported devices.

        Any pre-existing similar-device entry is overwritten from the first reference
        device that currently has results, so a re-run against a perf.yaml already
        populated by a prior call refreshes the borrowed metrics rather than leaving
        them frozen at the earlier values. This matters for the LLM pipeline, where
        apply_llm_perf_updates.py loads the committed perf.yaml before calling this.

        Parameters
        ----------
        mapping
            Dict of unsupported_device_name -> (real_chipset, list of reference device names).
            The first reference device with results is used as the perf source.
            real_chipset is added to ``supported_chipsets`` whenever the similar
            device lands in ``supported_devices``.
        """
        groups: dict[
            tuple[int, str], list[model_perf_pb2.ModelPerf.PerformanceDetails]
        ] = defaultdict(list)
        for entry in self._proto.performance_metrics:
            groups[(entry.precision, entry.component)].append(entry)

        new_entries: list[model_perf_pb2.ModelPerf.PerformanceDetails] = []
        for entries in groups.values():
            by_device: dict[str, list[model_perf_pb2.ModelPerf.PerformanceDetails]] = (
                defaultdict(list)
            )
            for entry in entries:
                by_device[entry.device].append(entry)

            result = {
                device: list(device_entries)
                for device, device_entries in by_device.items()
            }
            for unsupported_name, (_, reference_names) in mapping.items():
                for name in reference_names:
                    if name not in by_device:
                        continue
                    copies = []
                    for entry in by_device[name]:
                        entry_copy = model_perf_pb2.ModelPerf.PerformanceDetails()
                        entry_copy.CopyFrom(entry)
                        entry_copy.device = unsupported_name
                        copies.append(entry_copy)
                    result[unsupported_name] = copies
                    break
            for device_entries in result.values():
                new_entries.extend(device_entries)
        self._set_entries(new_entries)

        reference_to_similar: dict[str, list[str]] = {}
        for unsupported_name, (_, reference_names) in mapping.items():
            for name in reference_names:
                reference_to_similar.setdefault(name, []).append(unsupported_name)

        existing_names = set(self.supported_devices)
        new_devices: list[str] = []
        inserted: set[str] = set()
        for device in self.supported_devices:
            new_devices.append(device)
            for unsupported_name in reference_to_similar.get(device, []):
                if (
                    unsupported_name not in existing_names
                    and unsupported_name not in inserted
                ):
                    new_devices.append(unsupported_name)
                    inserted.add(unsupported_name)
        self.supported_devices[:] = new_devices

        # Every similar device in supported_devices contributes its chipset.
        present_similar = set(self.supported_devices) & set(mapping)
        existing_chipsets = set(self.supported_chipsets)
        new_chipsets = {
            mapping[name][0]
            for name in present_similar
            if mapping[name][0] not in existing_chipsets
        }
        if new_chipsets:
            self.supported_chipsets[:] = sorted_chipsets(
                existing_chipsets | new_chipsets
            )

    def apply_compute_peer_chipsets(self) -> None:
        """
        Credit interchangeable compute chipsets (see ``compute_peer_chipsets``).

        Additive only: devices already listed are kept, so QDC-only devices absent
        from the Hub device query survive. The non-LLM scorecard writer gets this
        from ``ScorecardDevice.extended_supported_chipsets``; the LLM pipeline needs
        it applied separately, and must not pick up that method's mobile proxying.
        """
        existing_chipsets = set(self.supported_chipsets)
        peers = {c for chip in existing_chipsets for c in compute_peer_chipsets(chip)}
        if not (new_chipsets := peers - existing_chipsets):
            return

        self.supported_chipsets[:] = sorted_chipsets(peers)
        existing_names = set(self.supported_devices)
        # Appended rather than re-sorted: supported_devices may hold similar devices
        # that are absent from Hub, and sorting resolves each chipset through Hub.
        self.supported_devices.extend(
            str(d)
            for d in get_supported_devices(new_chipsets)
            if str(d) not in existing_names
        )

    def _set_entries(
        self, entries: Iterable[model_perf_pb2.ModelPerf.PerformanceDetails]
    ) -> None:
        """Replace all entries with copies of ``entries`` (which may alias the current ones)."""
        copies = []
        for entry in entries:
            entry_copy = model_perf_pb2.ModelPerf.PerformanceDetails()
            entry_copy.CopyFrom(entry)
            copies.append(entry_copy)
        del self._proto.performance_metrics[:]
        self._proto.performance_metrics.extend(copies)
        self._reindex()

    def drop_entries(self, should_drop: Callable[[QAIHMModelPerf.Key], bool]) -> None:
        """Delete every entry whose key makes ``should_drop`` return True."""
        self._set_entries(
            entry for key, entry in self._entries.items() if not should_drop(key)
        )

    def sort_entries(self) -> None:
        """Sort entries in place into the stable on-disk order."""
        self._proto.performance_metrics.sort(key=_entry_sort_key)
        self._reindex()

    def has_component(self, precision: Precision, component: str) -> bool:
        return any(
            key.precision == precision and key.component == component
            for key in self._entries
        )

    def get_or_create_entry(
        self, key: QAIHMModelPerf.Key
    ) -> model_perf_pb2.ModelPerf.PerformanceDetails:
        if (entry := self._entries.get(key)) is not None:
            return entry
        metrics: RepeatedCompositeFieldContainer[
            model_perf_pb2.ModelPerf.PerformanceDetails
        ] = self._proto.performance_metrics
        entry = metrics.add(
            precision=precision_to_proto(key.precision),
            component=key.component,
            device=str(key.device),
            runtime=runtime_to_proto(key.path.runtime),
        )
        self._entries[key] = entry
        return entry

    def update_entry(
        self,
        details: model_perf_pb2.ModelPerf.PerformanceDetails,
        component_precision: Precision | None = None,
    ) -> model_perf_pb2.ModelPerf.PerformanceDetails:
        """
        Set the entry that ``details`` identifies to ``details``.

        The entry is identified by the precision, component, device and runtime set
        on ``details`` (see ``Key.stamp``). Adds a new entry if there isn't one yet,
        otherwise replaces the existing one.

        Parameters
        ----------
        details
            The new contents of the entry, with its identifying fields set.
        component_precision
            The precision this component uses. Required for every component of a
            mixed-precision model (the model's precision alone doesn't say what each
            component uses), and must not be passed otherwise.

        Returns
        -------
        entry : model_perf_pb2.ModelPerf.PerformanceDetails
            The entry that was added or replaced.

        Raises
        ------
        ValueError
            If ``component_precision`` is missing for a mixed precision, or given for
            any other precision.
        """
        key = QAIHMModelPerf.Key.from_details(details)
        is_mixed = key.precision in MIXED_PRECISIONS
        if is_mixed and component_precision is None:
            raise ValueError(
                f"component_precision is required for {key.precision} component {key.component}."
            )
        if not is_mixed and component_precision is not None:
            raise ValueError(
                f"component_precision is only allowed for mixed precisions, not {key.precision}."
            )
        entry = self.get_or_create_entry(key)
        entry.CopyFrom(details)
        if component_precision is not None:
            entry.component_precision = precision_to_proto(component_precision)
        return entry

    def drop_entries_in_scope(
        self,
        scope: set[tuple[Precision, ScorecardInferencePath, ScorecardDevice]],
        only_components: set[str] | None = None,
    ) -> None:
        """
        Delete every entry whose (precision, path, device) key is in scope.

        Parameters
        ----------
        scope
            The (precision, path, device) keys to drop.
        only_components
            Restrict the drop to these component names. A hybrid LLM's perf.yaml has two
            writers -- the QDC pipeline owns the consolidated backbone entry and the
            scorecard owns the standalone components -- and each must only clear its own,
            or whichever runs second deletes the other's results.
        """
        self.drop_entries(
            lambda key: (key.precision, key.path, key.device) in scope
            and (only_components is None or key.component in only_components)
        )

    def drop_entries_with_mismatched_aot(
        self,
        requires_aot_prepare: bool,
        only_components: set[str] | None = None,
    ) -> None:
        """
        Delete entries for runtimes the scorecard never tests given ``requires_aot_prepare``.

        The scoped merge preserves entries outside this run's scope, so flipping the
        flag would otherwise leave the old family's numbers in perf.yaml forever.
        Orchestrator runtimes are exempt: they are gated by ``orchestrator_runtimes``.

        Parameters
        ----------
        requires_aot_prepare
            The model's manifest flag. If set, non-AOT entries are dropped; otherwise AOT entries are.
        only_components
            Restrict the drop to these component names (see ``drop_entries_in_scope``).
        """
        self.drop_entries(
            lambda key: (only_components is None or key.component in only_components)
            and not key.path.runtime.is_orchestrator_runtime
            and key.path.runtime.is_aot_compiled != requires_aot_prepare
        )

    def drop_chipsets_without_assets(self, model_id: str) -> None:
        """
        Drop claimed chipsets this model ships no context binary for.

        Subtractive only, and a no-op when the model publishes a universal
        asset: that one file runs on any chipset via CPU fallback, so its reach
        isn't enumerable from the asset list. Call before
        ``apply_similar_devices``, which re-adds the borrowed devices on top.

        Parameters
        ----------
        model_id
            Model whose ``release-assets.yaml`` lists the published assets.
        """
        assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
        asset_chipsets: set[str] = set()
        for precision_details in assets.precisions.values():
            if precision_details.universal_assets:
                return
            for chipset in precision_details.chipset_assets:
                asset_chipsets |= compute_peer_chipsets(chipset)
        if not asset_chipsets:
            return

        supported_chipsets = set(self.supported_chipsets) & asset_chipsets
        self.supported_chipsets[:] = sorted_chipsets(supported_chipsets)
        self.supported_devices[:] = [
            str(d) for d in get_supported_devices(supported_chipsets)
        ]

    @classmethod
    def from_model(cls, model_id: str, not_exists_ok: bool = False) -> QAIHMModelPerf:
        perf_path = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
        if not_exists_ok and not os.path.exists(perf_path):
            return cls(model_id)
        return cls.from_yaml(perf_path, model_id)

    def to_model_yaml(self, model_id: str) -> Path:
        out = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
        self.to_yaml(out)
        return out

    def to_proto(self) -> model_perf_pb2.ModelPerf:
        """A copy of the underlying ``ModelPerf`` proto."""
        out = model_perf_pb2.ModelPerf()
        out.CopyFrom(self._proto)
        return out

    def to_release_proto(
        self,
        aihm_version: str,
        model_id: str,
        exclude_similar_devices: bool = True,
    ) -> model_perf_pb2.ModelPerf:
        """
        Build a copy of the proto stamped with the release version and model id.

        When *exclude_similar_devices* is True (default), "similar" devices
        (those from similar_devices.yaml, whose perf is borrowed from a reference
        device rather than measured) are dropped from both the entries and
        ``supported_devices``.
        """
        out = self.copy()
        out._proto.aihm_version = aihm_version
        out._proto.model_id = model_id
        if exclude_similar_devices:
            similar = _similar_device_names()
            out.drop_entries(lambda key: str(key.device) in similar)
            out.supported_devices[:] = [
                d for d in out.supported_devices if d not in similar
            ]
        return out.to_proto()

    def copy(self) -> QAIHMModelPerf:
        return QAIHMModelPerf(self.model_id, self.to_proto())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QAIHMModelPerf) and self._proto == other._proto

    # Mutable and compared by value, so it must not be hashable.
    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike,
        model_id: str,
        create_empty_if_no_file: bool = False,
    ) -> QAIHMModelPerf:
        """Read a perf.yaml, which is the YAML form of a ``ModelPerf`` proto."""
        if create_empty_if_no_file and (
            not os.path.exists(path) or os.path.getsize(path) == 0
        ):
            return cls(model_id)
        return cls(model_id, read_proto_yaml(path, model_perf_pb2.ModelPerf))

    def to_yaml(self, path: str | os.PathLike) -> bool:
        """
        Write this perf to ``path`` as the YAML form of a ``ModelPerf`` proto.

        Entries are sorted so regenerated files have a stable order. If this perf
        is empty, any existing file at ``path`` is deleted instead.

        Parameters
        ----------
        path
            Where to write the file.

        Returns
        -------
        written : bool
            True if the file was written.
        """
        path = Path(path)
        if self.empty:
            path.unlink(missing_ok=True)
            return False
        proto = self.to_proto()
        for entry in proto.performance_metrics:
            if not entry.tool_versions.ListFields():
                entry.ClearField("tool_versions")
        proto.performance_metrics.sort(key=_entry_sort_key)
        write_proto_yaml(proto, path)
        return True
