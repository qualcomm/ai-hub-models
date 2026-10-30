# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import copy

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.model_disable_reasons import (
    ModelDisableReasons,
    ModelDisableReasonsMapping,
)
from qai_hub_models.scorecard import (
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.scorecard.execution_helpers import (
    get_model_test_precisions,
)
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.chipset_helpers import (
    get_supported_devices,
    sorted_chipsets,
)
from qai_hub_models.scorecard.results.numerics_diff import (
    NumericsDiff,
)
from qai_hub_models.scorecard.results.scorecard_summary import (
    ScorecardExportTestSummary,
)

# Maximum acceptable inference time (milliseconds).
# Above this inference time, a model will not be published.
MAX_ACCEPTABLE_INFERENCE_TIME_MS = 4000


def _clean_old_failure_reasons(
    precisions: list[Precision],
    manifest: QAIHMModelManifest,
    clean_general: bool,
    clean_accuracy: bool,
) -> None:
    """In the code gen config, delete failure reasons for all enabled runtimes + given precision pairs."""
    passing_precisions: list[Precision] = []
    for precision, reasons_by_runtime in manifest.disabled_paths.data.items():
        if precision in precisions:
            for path in ScorecardProfilePath:
                if (not path.is_published or path.enabled) and (
                    reasons := reasons_by_runtime.get(path.runtime)
                ):
                    if clean_general:
                        reasons.scorecard_failure = None
                    if clean_accuracy:
                        reasons.scorecard_accuracy_failure = None
                    if not reasons.has_failure:
                        reasons_by_runtime.pop(path.runtime)

        # Delete precisions that are no longer valid
        if precision not in manifest.supported_precisions:
            runtimes_to_remove = []
            for runtime, reasons in list(reasons_by_runtime.items()):
                if clean_general:
                    reasons.scorecard_failure = None
                if clean_accuracy:
                    reasons.scorecard_accuracy_failure = None
                if not reasons.has_failure:
                    runtimes_to_remove.append(runtime)
            for runtime in runtimes_to_remove:
                reasons_by_runtime.pop(runtime)

        if not reasons_by_runtime:
            passing_precisions.append(precision)

    for precision in passing_precisions:
        manifest.disabled_paths.data.pop(precision)


def update_code_gen_failure_reasons(
    summaries: list[ScorecardExportTestSummary],
    enabled_test_paths: dict[Precision, list[ScorecardProfilePath]],
    manifest: QAIHMModelManifest,
) -> None:
    """
    Updates the provided manifest's disabled_paths to reflect job failures in the provided summaries.
    <path>_scorecard_failure will be set if certain jobs fail, and will be unset if no failing jobs are found.

    If relevant jobs can't be found in the given scorecard summaries, then no changes are made to the config.
    """
    default_device = ScorecardDevice.get(manifest.default_device)
    if not default_device.enabled:
        return

    supported_precisions = list(enabled_test_paths.keys())
    _clean_old_failure_reasons(
        precisions=supported_precisions,
        manifest=manifest,
        clean_general=True,
        clean_accuracy=False,
    )

    for summary in summaries:
        params = summary.params
        assert params.precision is not None
        assert params.device is not None
        assert isinstance(params.path, ScorecardProfilePath)
        if params.device != default_device:
            continue
        if not params.path.is_published:
            continue

        disable_reasons = manifest.disabled_paths.get_disable_reasons(
            params.precision, params.path.runtime
        )
        if disable_reasons.scorecard_failure:
            continue
        if params.precision not in enabled_test_paths:
            continue
        if params.path not in enabled_test_paths[params.precision]:
            continue

        # We enable a model if it works on the default device.
        if failure_reason := summary.get_failure_reason():
            disable_reasons.scorecard_failure = failure_reason


def update_code_gen_accuracy_failure_reasons(
    model_id: str, manifest: QAIHMModelManifest, model_diff: NumericsDiff
) -> None:
    supported_precisions = get_model_test_precisions(
        model_id,
        manifest.supported_precisions,
        can_use_quantize_job=manifest.can_use_quantize_job,
    )

    _clean_old_failure_reasons(
        precisions=supported_precisions,
        manifest=manifest,
        clean_general=False,
        clean_accuracy=True,
    )

    for disabled_path in model_diff.device_vs_float_greater_than_enablement_threshold:
        diff_model_id = disabled_path[0]
        precision = disabled_path[4]
        path = disabled_path[5]
        if (
            diff_model_id != model_id
            or not path.is_published
            or not path.enabled
            or precision not in supported_precisions
        ):
            continue

        if precision not in manifest.disabled_paths.data:
            manifest.disabled_paths.data[precision] = {}
        reasons_by_runtime = manifest.disabled_paths.data[precision]
        if path.runtime not in reasons_by_runtime:
            reasons_by_runtime[path.runtime] = ModelDisableReasons()
        reasons = reasons_by_runtime[path.runtime]
        reasons.scorecard_accuracy_failure = f"Torch and On-device accuracy diff ({disabled_path[9]}) above threshold ({disabled_path[10]})"

    for benchmark_failure in model_diff.benchmark_failures:
        diff_model_id = benchmark_failure[0]
        if diff_model_id != model_id:
            continue

        accuracy_type = benchmark_failure[3]
        bf_precision = benchmark_failure[4]
        bf_path = benchmark_failure[5]

        if accuracy_type == "device":
            assert bf_precision is not None and bf_path is not None
            if (
                not bf_path.is_published
                or not bf_path.enabled
                or bf_precision not in supported_precisions
            ):
                continue
            if bf_precision not in manifest.disabled_paths.data:
                manifest.disabled_paths.data[bf_precision] = {}
            reasons_by_runtime = manifest.disabled_paths.data[bf_precision]
            if bf_path.runtime not in reasons_by_runtime:
                reasons_by_runtime[bf_path.runtime] = ModelDisableReasons()
            reasons = reasons_by_runtime[bf_path.runtime]
            reasons.scorecard_accuracy_failure = f"Device accuracy ({benchmark_failure[7]}) deviates from benchmark ({benchmark_failure[8]}) by {benchmark_failure[9]}, threshold: {benchmark_failure[10]}"

        elif accuracy_type == "torch":
            # Torch accuracy failure affects all paths for all precisions
            for p in supported_precisions:
                for sc_path in ScorecardProfilePath:
                    if not sc_path.is_published or not sc_path.enabled:
                        continue
                    if p not in manifest.disabled_paths.data:
                        manifest.disabled_paths.data[p] = {}
                    reasons_by_runtime = manifest.disabled_paths.data[p]
                    if sc_path.runtime not in reasons_by_runtime:
                        reasons_by_runtime[sc_path.runtime] = ModelDisableReasons()
                    reasons = reasons_by_runtime[sc_path.runtime]
                    reasons.scorecard_accuracy_failure = f"Torch accuracy ({benchmark_failure[7]}) deviates from benchmark ({benchmark_failure[8]}) by {benchmark_failure[9]}, threshold: {benchmark_failure[10]}"


def update_model_publish_status(manifest: QAIHMModelManifest) -> bool:
    """Update the model publishing status based on failure reasons. Returns true if the status was changed, false otherwise."""
    # Update model status & reason, if applicable
    can_promote, reason = manifest.can_promote_to_published()
    if manifest.status == MODEL_STATUS.PENDING:
        if can_promote:
            manifest.status = MODEL_STATUS.PUBLISHED
            manifest.status_reason = None
            print(f"{manifest.id} | Set model to PUBLISHED")
            return True
        # Model has passing runtimes but is missing other requirements
        print(f"{manifest.id} | Skipping promotion to PUBLISHED: {reason}")
    elif manifest.status == MODEL_STATUS.PUBLISHED:
        if not can_promote:
            # Demote PUBLISHED to PENDING if no longer eligible
            manifest.status = MODEL_STATUS.PENDING
            manifest.status_reason = reason
            print(f"{manifest.id} | Set model to PENDING ({reason})")
            return True

    return False


def _also_strip_genie_if_geniex_fails(
    runtimes: list[TargetRuntime],
) -> list[TargetRuntime]:
    """Also strip Genie when GenieX (QAIRT) is stripped, matching QAIHMModelManifest.failure_reason."""
    if TargetRuntime.GENIEX_QAIRT in runtimes and TargetRuntime.GENIE not in runtimes:
        return [*runtimes, TargetRuntime.GENIE]
    return runtimes


def _runtimes_to_strip_for_device(
    failure_reasons: ModelDisableReasonsMapping,
    precision: Precision,
    device_name: str | None,
) -> list[TargetRuntime]:
    """Runtimes whose entries to drop for this precision on the given Hub device (None = not device-specific)."""
    return _also_strip_genie_if_geniex_fails(
        [
            runtime
            for runtime, reasons in (failure_reasons.data.get(precision) or {}).items()
            if reasons.failure_reason_for_device(device_name)
        ]
    )


def _device_names_for_chipset(chipset: str) -> set[str]:
    """Hub device names of scorecard devices on this chipset, resolved without querying Hub."""
    yaml_devices = DevicesAndChipsetsYaml.load().devices
    return {
        device.device_name
        for device in ScorecardDevice.all_devices(check_available_in_hub=False)
        if (details := yaml_devices.get(device.reference_device_name))
        and details.chipset == chipset
    }


def _runtimes_to_strip_for_chipset(
    failure_reasons: ModelDisableReasonsMapping,
    precision: Precision,
    chipset: str | None,
) -> list[TargetRuntime]:
    """Runtimes whose assets to drop for this precision on the given chipset (None = universal asset)."""
    chipset_devices = _device_names_for_chipset(chipset) if chipset else set()
    out: list[TargetRuntime] = []
    for runtime, reasons in (failure_reasons.data.get(precision) or {}).items():
        scoped = [d for d in reasons.disable_devices if d in chipset_devices]
        if reasons.failure_reason_for_device(scoped[0] if scoped else None):
            out.append(runtime)
    return _also_strip_genie_if_geniex_fails(out)


def remove_numerics_failures(
    numerics: QAIHMModelNumerics, failure_reasons: ModelDisableReasonsMapping
) -> QAIHMModelNumerics:
    """
    Drop all device + runtime + precision pairs from the numerics YAML for which a failure reason exists.

    Parameters
    ----------
    numerics
        The numerics YAML to modify.

    failure_reasons
        The failure reasons to consider.

    Returns
    -------
    QAIHMModelNumerics
        New numerics.yaml with failing device + runtime + precisions pairs removed.
    """
    metrics: list[QAIHMModelNumerics.MetricDetails] = []
    for metric in numerics.metrics:
        new_metrics_by_device: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ] = {}

        for device, metrics_by_precision in metric.device_metric.items():
            new_metrics_by_precision: dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ] = {}

            for precision, metrics_by_path in metrics_by_precision.items():
                runtimes_to_strip = _runtimes_to_strip_for_device(
                    failure_reasons, precision, device.device_name
                )
                new_metrics_by_path = {
                    path: copy.deepcopy(metrics_by_path[path])
                    for path in metrics_by_path
                    if path.runtime not in runtimes_to_strip
                }
                if new_metrics_by_path:
                    new_metrics_by_precision[precision] = new_metrics_by_path

            if new_metrics_by_precision:
                new_metrics_by_device[device] = new_metrics_by_precision

        if new_metrics_by_device:
            metrics.append(
                QAIHMModelNumerics.MetricDetails(
                    dataset_name=metric.dataset_name,
                    dataset_link=metric.dataset_link,
                    dataset_split_description=metric.dataset_split_description,
                    metric_name=metric.metric_name,
                    metric_description=metric.metric_description,
                    metric_unit=metric.metric_unit,
                    metric_range=copy.copy(metric.metric_range),
                    higher_is_better=metric.higher_is_better,
                    metric_enablement_threshold=metric.metric_enablement_threshold,
                    benchmark_value=metric.benchmark_value,
                    num_partial_samples=metric.num_partial_samples,
                    partial_torch_metric=metric.partial_torch_metric,
                    device_metric=new_metrics_by_device,
                )
            )

    return QAIHMModelNumerics(metrics=metrics)


def remove_perf_failures(
    perf: QAIHMModelPerf, failure_reason: ModelDisableReasonsMapping
) -> QAIHMModelPerf:
    """
    Drop all device + runtime + precision pairs from the perf YAML for which a failure reason exists.

    Parameters
    ----------
    perf
        The perf YAML to modify.
    failure_reason
        The failure reasons to consider.

    Returns
    -------
    QAIHMModelPerf
        New perf.yaml with failing device + runtime + precisions pairs removed.
    """
    supported_chipsets: set[str] = set()
    precisions: dict[Precision, QAIHMModelPerf.PrecisionDetails] = {}

    for precision, perf_precision_components in perf.precisions.items():
        components: dict[str, QAIHMModelPerf.ComponentDetails] = {}
        for (
            component,
            component_details,
        ) in perf_precision_components.components.items():
            performance_metrics: dict[
                ScorecardDevice,
                dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
            ] = {}
            for device, perf_by_path in component_details.performance_metrics.items():
                runtimes_to_strip = _runtimes_to_strip_for_device(
                    failure_reason, precision, device.device_name
                )
                path_perf = {
                    path: copy.deepcopy(perf_by_path[path])
                    for path in perf_by_path
                    if path.runtime not in runtimes_to_strip
                }
                if path_perf:
                    performance_metrics[device] = path_perf
                    if device.available_in_hub:
                        supported_chipsets.update(
                            device.supported_chipsets_for_paths(path_perf)
                        )
                    else:
                        yaml = DevicesAndChipsetsYaml.load()
                        if details := yaml.devices.get(device.reference_device_name):
                            supported_chipsets.add(details.chipset)

            if performance_metrics:
                components[component] = QAIHMModelPerf.ComponentDetails(
                    performance_metrics=performance_metrics,
                )

        if components:
            precisions[precision] = QAIHMModelPerf.PrecisionDetails(
                components=components
            )

    return QAIHMModelPerf(
        supported_devices=get_supported_devices(supported_chipsets),
        supported_chipsets=sorted_chipsets(supported_chipsets),
        precisions=precisions,
    )


def remove_asset_failures(
    assets: QAIHMModelReleaseAssets, failure_reasons: ModelDisableReasonsMapping
) -> QAIHMModelReleaseAssets:
    """
    Drop all device + runtime + precision pairs from the assets YAML for which a failure reason exists.

    Parameters
    ----------
    assets
        The assets YAML to modify.

    failure_reasons
        The failure reasons to consider.

    Returns
    -------
    QAIHMModelReleaseAssets
        New pre_release_assets.yaml with failing device + runtime + precisions pairs removed.
    """
    precisions: dict[Precision, QAIHMModelReleaseAssets.PrecisionDetails] = {}

    for precision, precision_details in assets.precisions.items():
        chipset_assets: dict[
            str,
            dict[ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails],
        ] = {}
        for chipset, asset_by_path in precision_details.chipset_assets.items():
            runtimes_to_strip = _runtimes_to_strip_for_chipset(
                failure_reasons, precision, chipset
            )
            path_assets = {
                path: copy.deepcopy(asset_by_path[path])
                for path in asset_by_path
                if path.runtime not in runtimes_to_strip
            }
            if path_assets:
                chipset_assets[chipset] = path_assets

        runtimes_to_strip = _runtimes_to_strip_for_chipset(
            failure_reasons, precision, None
        )
        universal_assets = {
            path: copy.deepcopy(precision_details.universal_assets[path])
            for path in precision_details.universal_assets
            if path.runtime not in runtimes_to_strip
        }
        if chipset_assets or universal_assets:
            precisions[precision] = QAIHMModelReleaseAssets.PrecisionDetails(
                universal_assets=universal_assets,
                chipset_assets=chipset_assets,
            )

    return QAIHMModelReleaseAssets(precisions=precisions)
