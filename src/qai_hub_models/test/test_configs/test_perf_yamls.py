# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.devices_and_chipsets_yaml import (
    SCORECARD_DEVICE_YAML_PATH,
    SIMILAR_DEVICES_YAML_PATH,
    DevicesAndChipsetsYaml,
    _load_similar_devices_raw,
    load_similar_devices,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.test.test_configs.manifest_path_check import (
    get_manifest_support_violations,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS


def test_perf_yaml() -> None:
    # DevicesAndChipsetsYaml defines the devices valid for use with the AI Hub Models website.
    dc = DevicesAndChipsetsYaml.load()
    valid_devices = dc.devices
    valid_chipsets = dc.chipsets

    def _validate_device(device: ScorecardDevice) -> None:
        # Verify the given device is valid for use in the AI Hub Models website.
        if device.reference_device_name not in valid_devices:
            raise ValueError(
                f"Invalid device '{device.reference_device_name}'. Device must be listed in {SCORECARD_DEVICE_YAML_PATH}.\n"
                "You may need to re-generate the valid device list via `python qai_hub_models/models/generate_scorecard_device_yaml.py`"
            )

    def _validate_chipset(chipset_name: str) -> None:
        # Verify the given chipsets is valid for use in the AI Hub Models website.
        if chipset_name not in valid_chipsets:
            raise ValueError(
                f"Invalid chipset '{chipset_name}'. Chipset must be listed in {SCORECARD_DEVICE_YAML_PATH}.\n"
                "You may need to re-generate the valid device list via `python qai_hub_models/models/generate_scorecard_device_yaml.py`"
            )

    path_errors: list[str] = []
    model_id = ""
    try:
        for model_id in MODEL_IDS:
            perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
            model_name: str | None = None
            manifest = QAIHMModelManifest.from_model(model_id)

            # Verify all devices are valid AI Hub Workbench devices.
            for chipset in perf.supported_chipsets:
                _validate_chipset(chipset)

            for device in perf.supported_devices:
                _validate_device(device)

            for precision, precision_perf in perf.precisions.items():
                measured_paths: set[ScorecardProfilePath] = set()
                for component_detail in precision_perf.components.values():
                    for (
                        device,
                        device_paths,
                    ) in component_detail.performance_metrics.items():
                        _validate_device(device)
                        measured_paths.update(device_paths)

                path_errors.extend(
                    get_manifest_support_violations(
                        manifest, "perf.yaml", precision, measured_paths
                    )
                )

                # If there is 1 component, make sure it matches the model name.
                if len(precision_perf.components) == 1:
                    if not model_name:
                        model_name = manifest.name
                    component_name = next(iter(precision_perf.components))
                    if component_name != model_name:
                        raise ValueError(  # noqa: TRY301
                            f"If model has 1 component, the component name (found: {component_name}) should match the model name (expected: {model_name})"
                        )
                # For LLMs, check if the performance details are complete
                if model_name is not None:
                    for runtime_performance_details in precision_perf.components[
                        model_name
                    ].performance_metrics.values():
                        for performance_details in runtime_performance_details.values():
                            # Validate LLM metrics if present
                            if performance_details.llm_metrics is not None:
                                assert len(performance_details.llm_metrics) > 0, (
                                    "For LLM models, at least one context length entry must be provided"
                                )
                                for ctx in performance_details.llm_metrics:
                                    assert ctx.context_length is not None, (
                                        "For LLM models, context length value must be provided"
                                    )
                                    assert ctx.tokens_per_second is not None, (
                                        "For LLM models, tokens per second must be provided"
                                    )

    except Exception as err:
        raise AssertionError(
            f"{model_id} perf yaml validation failed: {err!s}"
        ) from None

    assert not path_errors, (
        f"{len(path_errors)} perf.yaml entries conflict with manifest.yaml:\n"
        + "\n".join(path_errors)
    )


def test_similar_devices_chipsets_resolve() -> None:
    """Every chipset and reference_chipset referenced from similar_devices.yaml
    must be defined in devices_and_chipsets.yaml's chipsets: section, otherwise
    perf-yaml propagation can silently drop entries.
    """
    raw = _load_similar_devices_raw()
    catalog_chipsets = set(DevicesAndChipsetsYaml.load().chipsets)
    # similar_devices.yaml may define non-workbench chipsets locally; those
    # propagate into devices_and_chipsets.yaml via codegen, so we treat both
    # sources as valid for the lookup.
    valid_chipsets = catalog_chipsets | set(raw.chipsets)

    for device_name, entry in raw.devices.items():
        assert entry.chipset in valid_chipsets, (
            f"similar device {device_name!r}: chipset {entry.chipset!r} not "
            f"defined in {SCORECARD_DEVICE_YAML_PATH} or {SIMILAR_DEVICES_YAML_PATH}"
        )
        if entry.reference_chipset is not None:
            assert entry.reference_chipset in valid_chipsets, (
                f"similar device {device_name!r}: reference_chipset "
                f"{entry.reference_chipset!r} not defined in "
                f"{SCORECARD_DEVICE_YAML_PATH} or {SIMILAR_DEVICES_YAML_PATH}"
            )


def test_platform_to_proto_excludes_similar_devices() -> None:
    """The platform proto omits similar devices/chipsets, except the allowlist."""
    from qai_hub_models.scorecard.devices_and_chipsets_yaml import (
        ALLOWED_SIMILAR_DEVICES,
    )

    dc = DevicesAndChipsetsYaml.load()
    similar_names = {name for name, d in dc.devices.items() if d.reference_chipset}
    excluded = similar_names - ALLOWED_SIMILAR_DEVICES
    assert excluded and similar_names >= ALLOWED_SIMILAR_DEVICES  # sanity

    proto = dc.to_proto("0.99.0")
    proto_devices = {d.name for d in proto.devices}
    assert not (proto_devices & excluded)
    assert proto_devices >= ALLOWED_SIMILAR_DEVICES
    # qualcomm-qcs8275 (the allowlisted device's chipset) survives; a purely
    # similar chipset (e.g. qualcomm-sa8255p) is pruned.
    proto_chipsets = {c.name for c in proto.chipsets}
    assert "qualcomm-qcs8275" in proto_chipsets
    assert "qualcomm-sa8255p" not in proto_chipsets

    # With the flag off, similar devices are retained.
    proto_all = dc.to_proto("0.99.0", exclude_similar_devices=False)
    assert excluded <= {d.name for d in proto_all.devices}


def test_apply_similar_devices_idempotent() -> None:
    """A second apply on a perf.yaml that has already had similar devices applied
    must be a no-op for both ``supported_devices`` and ``supported_chipsets``.
    """
    mapping = load_similar_devices()

    perf = QAIHMModelPerf.from_model("inception_v3")
    perf.apply_similar_devices(mapping)
    after_devices = [str(d) for d in perf.supported_devices]
    after_chipsets = list(perf.supported_chipsets)

    perf.apply_similar_devices(mapping)

    assert after_devices == [str(d) for d in perf.supported_devices]
    assert after_chipsets == list(perf.supported_chipsets)


def test_to_proto_excludes_similar_devices() -> None:
    """The built perf proto omits all similar devices (perf is borrowed)."""
    mapping = load_similar_devices()
    similar_names = set(mapping)

    perf = QAIHMModelPerf.from_model("inception_v3")
    perf.apply_similar_devices(mapping)
    # Sanity check: applying similar devices puts at least one into the config.
    assert {str(d) for d in perf.supported_devices} & similar_names

    proto = perf.to_proto("0.99.0", "inception_v3")
    proto_devices = set(proto.supported_devices) | {
        r.device for r in proto.performance_metrics
    }
    # No similar device survives in the built proto.
    assert not (proto_devices & similar_names)

    # With the flag off, everything is retained.
    proto_all = perf.to_proto("0.99.0", "inception_v3", exclude_similar_devices=False)
    assert similar_names & set(proto_all.supported_devices)


def test_apply_similar_devices_refreshes_stale_entry() -> None:
    """A similar-device entry seeded with stale metrics must be overwritten
    on re-apply. This is what the LLM pipeline relies on:
    apply_llm_perf_updates.py loads the committed perf.yaml (which already
    has similar-device entries from a prior run) and expects
    apply_similar_devices to refresh them from the freshly upserted
    reference device numbers.
    """
    mapping = load_similar_devices()

    perf = QAIHMModelPerf.from_model("inception_v3")

    # Plant a distinctive sentinel in a similar device's slot on top of a
    # reference that inception_v3 already has perf for, then verify the
    # sentinel is gone after apply_similar_devices.
    stale_time = 999999.0
    seeded_similar: str | None = None

    for prec in perf.precisions.values():
        for comp in prec.components.values():
            existing_by_name = {str(d): d for d in comp.performance_metrics}
            for similar_name, (_, reference_names) in mapping.items():
                for ref_name in reference_names:
                    if ref_name not in existing_by_name:
                        continue
                    ref_bucket = comp.performance_metrics[existing_by_name[ref_name]]
                    similar_device = ScorecardDevice(
                        name=similar_name,
                        reference_device_name=similar_name,
                        register=False,
                    )
                    stale_bucket = {
                        path: details.model_copy(
                            update={"inference_time_milliseconds": stale_time},
                            deep=True,
                        )
                        for path, details in ref_bucket.items()
                    }
                    comp.performance_metrics[similar_device] = stale_bucket
                    seeded_similar = similar_name
                    break
                if seeded_similar:
                    break
            if seeded_similar:
                break
        if seeded_similar:
            break

    assert seeded_similar is not None, (
        "test setup: no similar device found with a reference in inception_v3's perf"
    )

    perf.apply_similar_devices(mapping)

    for prec in perf.precisions.values():
        for comp in prec.components.values():
            for device, bucket in comp.performance_metrics.items():
                if str(device) != seeded_similar:
                    continue
                for details in bucket.values():
                    assert details.inference_time_milliseconds != stale_time, (
                        f"stale similar-device entry for {seeded_similar} was "
                        f"not overwritten by apply_similar_devices"
                    )


def test_empty_perf_to_yaml_deletes_existing_file(tmp_path: Path) -> None:
    """QAIHMModelPerf.to_yaml deletes the on-disk file when the perf serializes
    empty (all defaults). collect_scorecard_results relies on the caller
    checking .empty before writing so a shard that has no data for a model
    (e.g. its perf jobs ran in a different test split) doesn't silently blow
    away the committed perf.yaml -- which then fails QAIHMModelInfo validation
    ("All published models must have a perf.yaml") for every published model.

    This test pins the delete-on-empty behavior so the check in
    process_e2e_recipe_model is not accidentally dropped.
    """
    path = tmp_path / "perf.yaml"
    path.write_text("supported_devices:\n- Samsung Galaxy S25\n")
    assert path.exists()

    empty_perf = QAIHMModelPerf()
    assert empty_perf.empty
    written = empty_perf.to_yaml(path)

    assert not written
    assert not path.exists()


def test_apply_similar_devices_adds_real_chipset() -> None:
    """When perf data is duplicated onto a similar device, its real chipset
    should be added to ``supported_chipsets``.
    """
    mapping = load_similar_devices()

    perf = QAIHMModelPerf.from_model("inception_v3")
    # Strip any pre-existing similar-device entries so we observe the real-chipset
    # insertion fresh, and verify it doesn't re-trigger after a second apply.
    similar_names = set(mapping)
    real_chipsets = {real for real, _ in mapping.values()}
    perf.supported_devices = [
        d for d in perf.supported_devices if str(d) not in similar_names
    ]
    perf.supported_chipsets = [
        c for c in perf.supported_chipsets if c not in real_chipsets
    ]
    for prec in perf.precisions.values():
        for comp in prec.components.values():
            for d in list(comp.performance_metrics):
                if str(d) in similar_names:
                    del comp.performance_metrics[d]

    perf.apply_similar_devices(mapping)

    # IQ-8275 EVK mirrors SA7255P ADP, which inception_v3 has perf on.
    assert "qualcomm-qcs8275" in perf.supported_chipsets

    # Idempotent for the chipset list as well.
    chips_after = list(perf.supported_chipsets)
    perf.apply_similar_devices(mapping)
    assert chips_after == list(perf.supported_chipsets)


def test_apply_similar_devices_chipset_matches_device_insertion() -> None:
    """A similar device inserted into ``supported_devices`` must also add its
    chipset to ``supported_chipsets``, even without a matching entry in
    ``performance_metrics``. Regression test for the AIHM filter mismatch.
    """
    mapping = load_similar_devices()

    perf = QAIHMModelPerf.from_model("inception_v3")
    similar_names = set(mapping)
    real_chipsets = {real for real, _ in mapping.values()}
    perf.supported_devices = [
        d for d in perf.supported_devices if str(d) not in similar_names
    ]
    perf.supported_chipsets = [
        c for c in perf.supported_chipsets if c not in real_chipsets
    ]

    supported_names = {str(d) for d in perf.supported_devices}
    metric_names: set[str] = set()
    for prec in perf.precisions.values():
        for comp in prec.components.values():
            metric_names.update(str(d) for d in comp.performance_metrics)

    target_similar: str | None = None
    for similar_name, (_, reference_names) in mapping.items():
        if any(r in supported_names for r in reference_names) and not any(
            r in metric_names for r in reference_names
        ):
            target_similar = similar_name
            break

    assert target_similar is not None, "test setup: no matching similar device"
    target_chipset = mapping[target_similar][0]

    perf.apply_similar_devices(mapping)

    assert target_similar in {str(d) for d in perf.supported_devices}
    assert target_chipset in perf.supported_chipsets
