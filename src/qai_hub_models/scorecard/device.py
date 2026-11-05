# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
ScorecardDevice: scorecard-specific device wrapper.

Adds on top of the plain ``RegisteredDevice`` identity:
- A reference vs. execution device name split (metadata vs. Hub-job target).
- Compile / profile path selection.
- Env-var-driven enablement (``QAIHM_ENABLED_DEVICES``, etc.).
- Per-device disabled-model lists and ``include_in_all`` filtering.
- The scorecard-only "universal" device concept.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache, cached_property
from typing import Any

import qai_hub as hub
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from typing_extensions import assert_never

import qai_hub_models.utils.device as registered_device
from qai_hub_models import InferenceEngine, Precision, TargetRuntime
from qai_hub_models.scorecard.envvars import (
    EnabledDevicesEnvvar,
    SpecialDeviceSetting,
)
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.device import (
    CANARY_DEVICES,
    FormFactor,
    HubDeviceAttributes,
    RegisteredDevice,
    _get_cached_device,
)

# -----------------------------------------------------------------------------
# Chipset helpers (scorecard-specific)
# -----------------------------------------------------------------------------

UNIVERSAL_DEVICE_NAME = "universal"


COMPUTE_PEER_CHIPSETS = frozenset(
    {"qualcomm-snapdragon-x-elite", "qualcomm-snapdragon-x-plus-8-core"}
)


def compute_peer_chipsets(chipset: str) -> set[str]:
    """
    Compute chipsets that are interchangeable with ``chipset``, including itself.

    X Plus 8-Core is the same NPU as X Elite (htp 73, soc_model 60), but only two
    units exist in the QDC pool, so we measure X Elite and credit both.
    """
    if chipset in COMPUTE_PEER_CHIPSETS:
        return set(COMPUTE_PEER_CHIPSETS)
    return {chipset}


_FRAMEWORK_ATTR_PREFIX = "framework"


class ScorecardDevice(HubDeviceAttributes):
    """
    Scorecard device with reference / execution device split and scorecard
    behavior (compile/profile paths, env-var filtering, disabled-model lists).

    ScorecardDevice does not subclass RegisteredDevice: RegisteredDevice has a
    single name, while ScorecardDevice keeps three (registry ``name`` slug,
    ``reference_device_name`` for metadata, ``execution_device_name`` for
    Hub-job scheduling).

    Chipset-attribute properties (``chipset``, ``os``, ``form_factor``, etc.)
    come from ``HubDeviceAttributes``, which reads them off the reference
    device.
    """

    _registry: dict[str, ScorecardDevice] = {}

    @classmethod
    def get(
        cls, device_name: str, return_unregistered: bool = False
    ) -> ScorecardDevice:
        """
        Look up a scorecard device by its registry name, reference device
        name, or execution device name.

        Parameters
        ----------
        device_name
            The scorecard-registry slug (e.g. ``cs_8_gen_1``), the reference
            device name, or the execution device name of a registered
            scorecard device. The literal string ``"default"`` returns the
            device flagged ``is_default=True``.
        return_unregistered
            If True and no match is found, return a new unregistered
            ScorecardDevice with all three names set to ``device_name``.

        Returns
        -------
        ScorecardDevice
        """
        if device_name == "default":
            return cls.get_default()

        if device_name in cls._registry:
            return cls._registry[device_name]

        for device in cls.all_devices(check_available_in_hub=False):
            if device_name in {
                device.reference_device_name,
                device.execution_device_name,
            }:
                return device

        if return_unregistered:
            return cls(
                name=device_name,
                reference_device_name=device_name,
                register=False,
            )

        raise ValueError(f"Unknown Scorecard Device {device_name}")

    @classmethod
    def get_default(cls) -> ScorecardDevice:
        """Return the registered scorecard device with ``is_default=True``."""
        for device in cls._registry.values():
            if device.is_default:
                return device
        raise ValueError("No default scorecard device found.")

    @classmethod
    def parse(cls, obj: str | ScorecardDevice) -> ScorecardDevice:
        if isinstance(obj, str):
            return cls.get(obj, return_unregistered=True)
        if isinstance(obj, ScorecardDevice):
            return obj
        raise ValueError(f"Can't parse type {type(obj)} as ScorecardDevice")

    @classmethod
    def all_devices(
        cls,
        enabled: bool | None = None,
        npu_supports_precision: Precision | None = None,
        supports_compile_path: ScorecardCompilePath | None = None,
        supports_profile_path: ScorecardProfilePath | None = None,
        form_factors: list[FormFactor] | None = None,
        is_mirror: bool | None = None,
        include_universal: bool = True,
        check_available_in_hub: bool = True,
    ) -> list[ScorecardDevice]:
        """
        Get all devices that match the given attributes.
        If an attribute is None, it is ignored when filtering devices.
        """
        return [
            device
            for device in cls._registry.values()
            if (
                (enabled is None or enabled == device.enabled)
                and (include_universal or device.name != UNIVERSAL_DEVICE_NAME)
                and (
                    not check_available_in_hub
                    # Ignore availability check if AI Hub Workbench is not accessible
                    or not can_access_qualcomm_ai_hub()
                    or device.available_in_hub
                )
                and (
                    npu_supports_precision is None
                    or device.npu_supports_precision(npu_supports_precision)
                )
                and (
                    supports_compile_path is None
                    or supports_compile_path in device.compile_paths
                )
                and (
                    supports_profile_path is None
                    or supports_profile_path in device.profile_paths
                )
                and (form_factors is None or device.form_factor in form_factors)
            )
        ]

    @staticmethod
    @cache
    def canary_devices() -> set[ScorecardDevice]:
        """Get 'canary' devices used in for continuous integration testing."""
        return {ScorecardDevice.get(device_name) for device_name in CANARY_DEVICES}

    def __init__(
        self,
        name: str,
        reference_device_name: str,
        execution_device_name: str | None = None,
        npu_count: int | None = None,
        is_default: bool = False,
        disabled_models: list[str] | None = None,
        compile_paths: list[ScorecardCompilePath] | None = None,
        profile_paths: list[ScorecardProfilePath] | None = None,
        register: bool = True,
        include_in_all: bool = True,
        qdc_enabled: bool = True,
        devicefarm_backend: str = "qdc",
    ) -> None:
        """
        Parameters
        ----------
        name
            Programmatic slug (e.g. ``cs_8_gen_1``). Registry key.
        reference_device_name
            Hub device name to use for metadata lookups (chipset, OS, form
            factor, etc.). Distinct from ``execution_device_name`` so a
            specific model can be used for metadata while jobs run against a
            family pool.
        execution_device_name
            Hub device name to use when submitting jobs. Defaults to
            ``reference_device_name`` (i.e. metadata device == job device).
        npu_count
            How many NPUs this device has. If undefined, defaults to 1.
        is_default
            Whether this device represents the user's default choice.
        disabled_models
            List of model IDs that should be disabled for this device.
        compile_paths
            The set of compile paths valid for this device. If unset, defaults
            based on this device's form factor.
        profile_paths
            The set of profile paths valid for this device. If unset, defaults
            based on this device's form factor.
        register
            Whether to register this device in the scorecard registry.
        include_in_all
            Whether this device is enabled when ``SpecialDeviceSetting.ALL``
            is selected. If False, the device only runs when its name is
            explicitly listed in ``EnabledDevicesEnvvar``.
        qdc_enabled
            Whether LLM perf tests are allowed to submit jobs (QDC or AWS,
            per ``devicefarm_backend``) against this device. When False, the
            device still participates in LLM compile parametrization but is
            filtered out of ``get_llm_perf_parametrization``.
        devicefarm_backend
            Which device-cloud backend runs Genie/GenieX LLM perf/accuracy
            jobs for this device: ``"qdc"`` (default) or ``"aws"`` (AWS
            Device Farm; Android only).
        """
        if register and name in ScorecardDevice._registry:
            raise ValueError(f"Device {name} already registered.")

        self.name = name
        self.reference_device_name = reference_device_name
        self.execution_device_name = execution_device_name
        self._npu_count = npu_count
        self.is_default = is_default
        self.disabled_models = disabled_models or []
        self._compile_paths = compile_paths
        self._profile_paths = profile_paths
        self.include_in_all = include_in_all
        self.qdc_enabled = qdc_enabled
        self.devicefarm_backend = devicefarm_backend

        if register:
            ScorecardDevice._registry[name] = self

    @classmethod
    def from_registered(
        cls,
        base: RegisteredDevice,
        name: str,
        reference_device_name: str | None = None,
        disabled_models: list[str] | None = None,
        compile_paths: list[ScorecardCompilePath] | None = None,
        profile_paths: list[ScorecardProfilePath] | None = None,
        register: bool = True,
        include_in_all: bool = True,
        qdc_enabled: bool = True,
        devicefarm_backend: str = "qdc",
    ) -> ScorecardDevice:
        """
        Build a ScorecardDevice from an existing ``RegisteredDevice``.

        The execution device name is always set to ``base.device_name`` -- i.e.
        Hub jobs run against the same target the RegisteredDevice was
        instantiated with (typically a Family alias).

        ``reference_device_name`` defaults to ``base.device_name``; callers
        override it when metadata should come from a specific device (e.g.
        pass "Samsung Galaxy S22 5G" while jobs still schedule on the
        S22 Family pool).

        ``is_default`` and ``npu_count`` are inherited from ``base``.
        """
        return cls(
            name=name,
            reference_device_name=reference_device_name or base.device_name,
            execution_device_name=base.device_name,
            npu_count=base._npu_count,
            is_default=base.is_default,
            disabled_models=disabled_models,
            compile_paths=compile_paths,
            profile_paths=profile_paths,
            register=register,
            include_in_all=include_in_all,
            qdc_enabled=qdc_enabled,
            devicefarm_backend=devicefarm_backend,
        )

    def __str__(self) -> str:
        return self.reference_device_name

    def __repr__(self) -> str:
        return self.name.lower()

    @property
    def device_name(self) -> str:
        """The Hub device name this scorecard device runs on (matches RegisteredDevice.device_name)."""
        return self.execution_device_name or self.reference_device_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScorecardDevice):
            return False
        return (
            self.name == other.name
            and self.reference_device_name == other.reference_device_name
            and self.execution_device_name == other.execution_device_name
        )

    def __hash__(self) -> int:
        return (
            hash(self.name)
            + hash(self.reference_device_name)
            + hash(self.execution_device_name)
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            lambda obj, _: cls.parse(obj),
            handler(Any),
            serialization=core_schema.plain_serializer_function_ser_schema(
                ScorecardDevice.__str__, when_used="json"
            ),
        )

    # ------------------------------------------------------------------
    # Chipset-attribute properties (against reference or execution device)
    # ------------------------------------------------------------------

    @cached_property
    def reference_device(self) -> hub.Device:
        """
        Get the "reference" device used for metadata.
        This is not used by any actual Hub jobs.
        """
        device = _get_cached_device(self.reference_device_name)
        if not device:
            raise ValueError(f"Device {self.reference_device_name} not found on Hub.")
        return device

    @cached_property
    def execution_device(self) -> hub.Device:
        """Get the device used for Hub job submission."""
        if self.execution_device_name is not None:
            device = _get_cached_device(self.execution_device_name)
            if not device:
                raise ValueError(
                    f"Device {self.execution_device_name} not found on Hub."
                )
            return device
        return self.reference_device

    @cached_property
    def available_in_hub(self) -> bool:
        """Returns true if this device is available in AI Hub Workbench."""
        return _get_cached_device(self.reference_device_name) is not None and (
            self.execution_device_name is None
            or _get_cached_device(self.execution_device_name) is not None
        )

    @property
    def _hub_device(self) -> hub.Device:
        """Metadata source for ``HubDeviceAttributes``: the reference device."""
        return self.reference_device

    def _display_name(self) -> str:
        return self.name

    # ------------------------------------------------------------------
    # Scorecard-specific behavior
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """
        Whether the scorecard should include this scorecard device.
        This applies both to submitted jobs and analyses applied to an existing scorecard job yaml.
        """
        valid_test_devices = EnabledDevicesEnvvar.get()
        if self.name in valid_test_devices and not self.available_in_hub:
            raise ValueError(
                f"Device {self.name} is not available in AI Hub Workbench."
            )

        return self.name in ScorecardDevice._registry and (
            (SpecialDeviceSetting.ALL in valid_test_devices and self.include_in_all)
            or self.name == UNIVERSAL_DEVICE_NAME
            or self.name in valid_test_devices
            or (
                SpecialDeviceSetting.CANARY in valid_test_devices
                and self in ScorecardDevice.canary_devices()
            )
        )

    @cached_property
    def supported_runtimes(self) -> list[TargetRuntime]:
        """All runtimes supported by this device."""
        supports_qnn = False
        runtimes: list[TargetRuntime] = []
        for attr in self.reference_device.attributes:
            if attr.startswith(_FRAMEWORK_ATTR_PREFIX):
                fw_name = attr[len(_FRAMEWORK_ATTR_PREFIX) + 1 :].lower()
                runtimes.extend(
                    [x for x in TargetRuntime if x.inference_engine.value == fw_name]
                )
                supports_qnn = supports_qnn or fw_name == InferenceEngine.QNN.value

        # GENIE is built on top of QAIRT/QNN (compiles to --target_runtime qnn_dlc),
        # so any device with QNN support can run GENIE.
        if supports_qnn:
            if TargetRuntime.GENIE not in runtimes:
                runtimes.append(TargetRuntime.GENIE)
            if TargetRuntime.GENIEX_QAIRT not in runtimes:
                runtimes.append(TargetRuntime.GENIEX_QAIRT)

        if not supports_qnn:
            # No QNN support == QAIRT converters can't be used
            runtimes = [
                x
                for x in runtimes
                if not x.is_aot_compiled and x.inference_engine != InferenceEngine.QNN
            ]

        return runtimes

    @cached_property
    def profile_paths(self) -> list[ScorecardProfilePath]:
        """
        All profile paths supported by this device.

        Note that we exclude some paths that are "supported" by Hub devices
        because we don't want to test them in scorecard. For example, we don't
        run ONNX on auto devices even though this is supported by AI Hub Workbench.
        """
        if self._profile_paths is not None:
            return self._profile_paths

        paths_to_test: list[ScorecardProfilePath] = []
        inference_engines_to_test: list[InferenceEngine] = []
        if (
            self.form_factor == FormFactor.PHONE  # noqa: PLR1714
            or self.form_factor == FormFactor.TABLET
            or self.form_factor == FormFactor.IOT
        ):
            inference_engines_to_test = list(InferenceEngine)
        elif self.form_factor == FormFactor.AUTO:
            inference_engines_to_test = [
                InferenceEngine.QNN,
                InferenceEngine.TFLITE,
            ]
            # GenieX is not published for automotive devices.
            paths_to_test.extend([ScorecardProfilePath.GENIE])
        elif self.form_factor == FormFactor.XR:
            inference_engines_to_test = [InferenceEngine.QNN, InferenceEngine.TFLITE]
        elif self.form_factor == FormFactor.COMPUTE:
            inference_engines_to_test = [
                InferenceEngine.QNN,
                InferenceEngine.ONNX,
            ]
            paths_to_test.extend(
                [ScorecardProfilePath.GENIE, ScorecardProfilePath.GENIEX_QAIRT]
            )
        else:
            assert_never(self.form_factor)

        out = paths_to_test + [
            path
            for path in ScorecardProfilePath
            if path.runtime in self.supported_runtimes
            and path.runtime.inference_engine in inference_engines_to_test
        ]

        # If running "all" devices, only run tfl_converter on 6490 and the
        # default device. Any explicitly set device will also run.
        restricted_paths = [
            ScorecardProfilePath.TFL_CONVERTER,
        ]
        if (
            not self.is_default
            and self != cs_6490
            and self.name not in EnabledDevicesEnvvar.get()
        ):
            out = [x for x in out if x not in restricted_paths or not x.enabled]

        return out

    @cached_property
    def compile_paths(self) -> list[ScorecardCompilePath]:
        """All compile paths supported by this device."""
        if self._compile_paths is not None:
            return self._compile_paths

        return [
            path.compile_path
            for path in self.profile_paths
            if path.runtime in self.supported_runtimes
            # Universal compile paths are disabled by default,
            # since we need to compile only once and the universal
            # device will do that.
            and not path.compile_path.is_universal
        ]

    @cached_property
    def extended_supported_chipsets(self) -> set[str]:
        """
        If this device can run a model, get a set of all chipsets that should also be supported.
        This device's chipset will be included in the list.

        Assumes the model is compiled just in time, so the mobile proxy list
        applies. Callers that know the runtimes involved should prefer
        ``supported_chipsets_for_paths``, which drops the proxies for assets
        compiled ahead of time.
        """
        return self._extended_supported_chipsets(extend_chipsets=True)

    def supported_chipsets_for_paths(
        self, paths: Iterable[ScorecardProfilePath]
    ) -> set[str]:
        """
        Chipsets to claim support for when this device passes on ``paths``.

        An ahead-of-time compiled asset is a context binary built for one
        chipset; it cannot run on an older chipset the way a JIT asset can, so
        the mobile proxy list only applies when some path is compiled on device.
        """
        is_jit_asset = any(not path.runtime.is_aot_compiled for path in paths)
        return self._extended_supported_chipsets(extend_chipsets=is_jit_asset)

    def _extended_supported_chipsets(self, extend_chipsets: bool) -> set[str]:
        if self.form_factor in [
            FormFactor.PHONE,
            FormFactor.TABLET,
        ]:
            mobile_chips = [
                "qualcomm-snapdragon-8-elite-extreme-gen6",
                "qualcomm-snapdragon-8-elite-gen6",
                "qualcomm-snapdragon-8-elite-gen5-for-galaxy",
                "qualcomm-snapdragon-8-elite-for-galaxy",
                "qualcomm-snapdragon-8gen3",
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
            if extend_chipsets and self.chipset in mobile_chips:
                idx = mobile_chips.index(self.chipset)
                # Return this chipset and all older chipsets as proxies —
                # we don't run older devices in the scorecard.
                return {self.chipset} | set(mobile_chips[idx + 1 :])
        if self.form_factor == FormFactor.COMPUTE:
            return compute_peer_chipsets(self.chipset)
        return {self.chipset}


# ----------------------
# SCORECARD DEVICE DEFINITIONS
#
# Wrap each RegisteredDevice from utils.device with scorecard-specific
# augmentations.
# ----------------------

# Mobile chipsets. S22-S26 override reference_device_name so metadata comes
# from the specific device even though Hub jobs schedule on the family pool.
cs_8_gen_1 = ScorecardDevice.from_registered(
    registered_device.cs_8_gen_1,
    name="cs_8_gen_1",
    reference_device_name="Samsung Galaxy S22 5G",
)

cs_8_gen_2 = ScorecardDevice.from_registered(
    registered_device.cs_8_gen_2,
    name="cs_8_gen_2",
    reference_device_name="Samsung Galaxy S23",
    include_in_all=False,
)

cs_8_gen_3 = ScorecardDevice.from_registered(
    registered_device.cs_8_gen_3,
    name="cs_8_gen_3",
    reference_device_name="Samsung Galaxy S24",
    include_in_all=False,
)

cs_8_elite = ScorecardDevice.from_registered(
    registered_device.cs_8_elite,
    name="cs_8_elite",
    reference_device_name="Samsung Galaxy S25",
    devicefarm_backend="aws",
)

cs_7_gen_4 = ScorecardDevice.from_registered(
    registered_device.cs_7_gen_4,
    name="cs_7_gen_4",
)

cs_8_elite_gen_5 = ScorecardDevice.from_registered(
    registered_device.cs_8_elite_gen_5,
    name="cs_8_elite_gen_5",
    reference_device_name="Samsung Galaxy S26",
    devicefarm_backend="aws",
)

cs_8_elite_gen_6 = ScorecardDevice.from_registered(
    registered_device.cs_8_elite_gen_6,
    name="cs_8_elite_gen_6",
)

cs_8_elite_extreme_gen_6 = ScorecardDevice.from_registered(
    registered_device.cs_8_elite_extreme_gen_6,
    name="cs_8_elite_extreme_gen_6",
)

# Compute chipsets
cs_x_elite = ScorecardDevice.from_registered(
    registered_device.cs_x_elite,
    name="cs_x_elite",
)

cs_x_plus_8_core = ScorecardDevice.from_registered(
    registered_device.cs_x_plus_8_core,
    name="cs_x_plus_8_core",
    compile_paths=[
        ScorecardCompilePath.GENIE,
        ScorecardCompilePath.GENIEX_QAIRT,
    ],
    profile_paths=[
        ScorecardProfilePath.GENIE,
        ScorecardProfilePath.GENIEX_QAIRT,
    ],
)

cs_x2_elite = ScorecardDevice.from_registered(
    registered_device.cs_x2_elite,
    name="cs_x2_elite",
)

# Auto chipsets
cs_auto_monaco_7255 = ScorecardDevice.from_registered(
    registered_device.cs_auto_monaco_7255,
    name="cs_auto_monaco_7255",
)

cs_auto_makena_8295 = ScorecardDevice.from_registered(
    registered_device.cs_auto_makena_8295,
    name="cs_auto_makena_8295",
)

cs_auto_lemans_8775 = ScorecardDevice.from_registered(
    registered_device.cs_auto_lemans_8775,
    name="cs_auto_lemans_8775",
)

# IoT chipsets
cs_6490 = ScorecardDevice.from_registered(
    registered_device.cs_6490,
    name="cs_6490",
)

cs_6690 = ScorecardDevice.from_registered(
    registered_device.cs_6690,
    name="cs_6690",
)

cs_8550 = ScorecardDevice.from_registered(
    registered_device.cs_8550,
    name="cs_8550",
)

cs_ventuno_q = ScorecardDevice.from_registered(
    registered_device.cs_ventuno_q,
    name="cs_ventuno_q",
)

cs_9075 = ScorecardDevice.from_registered(
    registered_device.cs_9075,
    name="cs_9075",
)

# Universal device: compile-only, borrows the default scorecard device's
# reference so metadata lookups still resolve.
cs_universal = ScorecardDevice(
    name=UNIVERSAL_DEVICE_NAME,
    reference_device_name=ScorecardDevice.get_default().reference_device_name,
    compile_paths=[path for path in ScorecardCompilePath if path.is_universal],
    profile_paths=[],
)


DEFAULT_SCORECARD_DEVICE = ScorecardDevice.get_default()


# Devices LLM models compile against by default (any precision).
# The first entry is the canonical default device for LLMs (DEFAULT_QDC_DEVICE).
LLM_COMPILE_DEVICES = [
    cs_x_elite,
    cs_9075,
    cs_ventuno_q,
    cs_auto_lemans_8775,
    cs_x2_elite,
    cs_8_elite,
    cs_8_elite_gen_5,
]

# Default device for LLM (QDC / Genie) tests and asset uploads.
DEFAULT_QDC_DEVICE = LLM_COMPILE_DEVICES[0]

# Extra devices LLM models compile against for w4 precision only,
# in addition to LLM_COMPILE_DEVICES.
LLM_W4FP16_COMPILE_DEVICES = [
    cs_auto_makena_8295,
]
