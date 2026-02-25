# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum, unique
from functools import cache, cached_property
from typing import Any

import qai_hub as hub
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from typing_extensions import assert_never

from qai_hub_models.models.common import InferenceEngine, Precision, TargetRuntime
from qai_hub_models.scorecard.envvars import (
    EnabledDevicesEnvvar,
    SpecialDeviceSetting,
)
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub

_FRAMEWORK_ATTR_PREFIX = "framework"
_DEVICE_CACHE: dict[str, hub.Device | None] = {}
UNIVERSAL_DEVICE_SCORECARD_NAME = "universal"


def _get_cached_device(device_name: str) -> hub.Device | None:
    # Gets a device with attributes & OS. This only comes from hub.get_devices()
    device = _DEVICE_CACHE.get(device_name)
    if not device:
        devices = hub.get_devices(device_name)
        device = devices[0] if devices else None
        _DEVICE_CACHE[device_name] = device
    return device


class ScorecardDevice:
    _registry: dict[str, ScorecardDevice] = {}

    @classmethod
    def get(
        cls, device_name: str, return_unregistered: bool = False
    ) -> ScorecardDevice:
        if device_name == "default":
            for device in cls._registry.values():
                if device.is_default:
                    return device
            raise ValueError("No default device found.")
        # If the name is a device name in the registry, return that device
        if device_name in ScorecardDevice._registry:
            return ScorecardDevice._registry[device_name]

        # Check for universal device
        if device_name == cs_universal.reference_device_name:
            # Sanity check in case universal device changes
            assert (
                cs_universal.reference_device_name == cs_8_gen_3.reference_device_name
            )

            # Don't return cs_universal for a specific device name.
            # Always return the specific device instead
            return cs_8_gen_3

        # Return any device with a matching reference device name
        if out := [
            x
            for x in ScorecardDevice.all_devices(check_available_in_hub=False)
            if device_name in {x.reference_device_name, x.execution_device_name}
        ]:
            return out[0]

        # Return a new unregistered device
        if return_unregistered:
            return ScorecardDevice(device_name, device_name, register=False)

        raise ValueError(f"Unknown Scorecard Device {device_name}")

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
        form_factors: list[ScorecardDevice.FormFactor] | None = None,
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
                and (include_universal or device != cs_universal)
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
                and (is_mirror is None or bool(device.mirror_device) == is_mirror)
            )
        ]

    @staticmethod
    @cache
    def canary_devices() -> set[ScorecardDevice]:
        """Get 'canary' devices used in for continuous integration testing."""
        return {ScorecardDevice.get(x) for x in CANARY_DEVICES}

    @unique
    class FormFactor(Enum):
        PHONE = "Phone"
        TABLET = "Tablet"
        AUTO = "Auto"
        XR = "XR"
        COMPUTE = "Compute"
        IOT = "IoT"

    @unique
    class OperatingSystemType(Enum):
        ANDROID = "Android"
        WINDOWS = "Windows"
        LINUX = "Linux"
        QC_LINUX = "Qualcomm Linux"

    class OperatingSystem(BaseQAIHMConfig):
        ostype: ScorecardDevice.OperatingSystemType
        version: str

        def __str__(self) -> str:
            return f"{self.ostype.name} {self.version}"

    def __init__(
        self,
        name: str,
        reference_device_name: str,
        execution_device_name: str | None = None,
        disabled_models: list[str] | None = None,
        compile_paths: list[ScorecardCompilePath] | None = None,
        profile_paths: list[ScorecardProfilePath] | None = None,
        mirror_device: ScorecardDevice | None = None,
        npu_count: int | None = None,
        public: bool = True,
        register: bool = True,
        is_default: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        name
            Name of this device for scorecard use.
        reference_device_name
            The name of the "reference" device used by the scorecard for metadata when collating results.
        execution_device_name
            The name of the device to be used by associated Hub jobs.
            If not provided, jobs will be submitted with the chipset of the reference device.
            Hub will decide what device to use depending on availability.
        disabled_models
            List of model IDs that should be disabled for this device.
        compile_paths
            The set of compile paths valid for this device. If unset, will use the default set of paths for this device's form factor.
        profile_paths
            The set of profile paths valid for this device. If unset, will use the default set of paths for this device's form factor.
        mirror_device
            If set, jobs are not run on this device. Instead, results for this will "mirror" of the given device.
        npu_count
            How many NPUs this device has. If undefined, uses the NPU count of the mirror device or defaults to 1.
        public
            Whether this device is publicly available on AI Hub Workbench.
            NOTE: Private devices are not included when "all" devices are selected. They must be explicitly included in the list of devices to test.
        register
            Whether to register this device in the list of all devices.
        is_default
            Whether this device represents the user choosing the default device.
        """
        if register and name in ScorecardDevice._registry:
            raise ValueError("Device " + name + "already registered.")

        if mirror_device:
            assert not compile_paths, (
                "Compile paths should not be set, mirror devices will use the mirror device settings."
            )
            assert not profile_paths, (
                "Profile paths should not be set, mirror devices will use the mirror device settings."
            )
            assert not disabled_models, (
                "Disabled models should not be set, mirror devices will use the mirror device settings."
            )
            assert not execution_device_name, (
                "Execution device is not applicable when mirroring results of a different device."
            )

        self.name = name
        self.disabled_models: list[str] = (
            mirror_device.disabled_models if mirror_device else (disabled_models or [])
        )
        self.reference_device_name = reference_device_name
        self.execution_device_name = execution_device_name
        self._compile_paths = compile_paths
        self._profile_paths = profile_paths
        self.mirror_device: ScorecardDevice | None = mirror_device
        self._npu_count = npu_count
        self.public = public
        self.is_default = is_default

        if register:
            ScorecardDevice._registry[name] = self

    def __str__(self) -> str:
        return self.reference_device_name

    def __repr__(self) -> str:
        return self.name.lower()

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
            (self.public and SpecialDeviceSetting.ALL in valid_test_devices)
            or self.name == UNIVERSAL_DEVICE_SCORECARD_NAME
            or self.name in valid_test_devices
            or (
                SpecialDeviceSetting.CANARY in valid_test_devices
                and self in ScorecardDevice.canary_devices()
            )
        )

    @cached_property
    def reference_device(self) -> hub.Device:
        """
        Get the "reference" device used by the scorecard for metadata when collating results.
        This is not used by any actual scorecard jobs.
        """
        device = _get_cached_device(self.reference_device_name)
        if not device:
            raise ValueError(f"Device {self.reference_device_name} not found on Hub.")
        return device

    @cached_property
    def execution_device(self) -> hub.Device:
        """Get the device used by the scorecard for job submission."""
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

    @cached_property
    def chipset(self) -> str:
        """The chipset used by this device."""
        device = (
            self.execution_device
            if self.execution_device_name
            else self.reference_device
        )
        for attr in device.attributes:
            if attr.startswith("chipset:"):
                return attr[8:]
        raise ValueError(f"Chipset not found for device: {self.name}")

    @cached_property
    def chipset_aliases(self) -> list[str]:
        """The aliases for the chipset used by this device."""
        device = (
            self.execution_device
            if self.execution_device_name
            else self.reference_device
        )

        return [attr[8:] for attr in device.attributes if attr.startswith("chipset:")]

    @cached_property
    def npu_count(self) -> int:
        """Returns the number of NPUs on this device."""
        if self._npu_count is not None:
            return self._npu_count
        if self.mirror_device:
            return self.mirror_device.npu_count
        return 1

    @cached_property
    def extended_supported_chipsets(self) -> set[str]:
        """
        If this device can run a model, get a set of all chipsets that should also be supported.
        This device's chipset will be included in the list.
        """
        if self.form_factor in [
            ScorecardDevice.FormFactor.PHONE,
            ScorecardDevice.FormFactor.TABLET,
        ]:
            mobile_chips = [
                "qualcomm-snapdragon-8-elite",
                "qualcomm-snapdragon-8gen3",
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
            if self.chipset in mobile_chips:
                # Return this chipset and all older chipsets.
                # We don't run older devices in the scorecard, so this is a proxy.
                return set(mobile_chips[mobile_chips.index(self.chipset) :])
        if self.form_factor == ScorecardDevice.FormFactor.COMPUTE:
            # If either compute chip works, both work
            compute_chips = {
                "qualcomm-snapdragon-x-elite",
                "qualcomm-snapdragon-x-plus-8-core",
            }
            if self.chipset in compute_chips:
                return compute_chips
        return {self.chipset}

    @cached_property
    def os(self) -> OperatingSystem:
        """The operating system used by this device."""
        for attr in self.reference_device.attributes:
            if attr.startswith("os:"):
                return ScorecardDevice.OperatingSystem(
                    ostype=ScorecardDevice.OperatingSystemType[
                        attr.split(":")[-1].upper()
                    ],
                    version=self.reference_device.os,
                )
        raise ValueError(f"OS not found for device: {self.name}")

    @cached_property
    def vendor(self) -> str:
        """The vendor that manufactures this device."""
        for attr in self.reference_device.attributes:
            if attr.startswith("vendor:"):
                return attr.split(":")[-1]
        raise ValueError(f"Vendor not found for device: {self.name}")

    @cached_property
    def form_factor(self) -> FormFactor:
        """The device form factor (eg. Auto, IoT, Mobile, ...)"""
        for attr in self.reference_device.attributes:
            if attr.startswith("format:"):
                return ScorecardDevice.FormFactor[attr.split(":")[-1].upper()]
        raise ValueError(f"Format not found for device: {self.name}")

    @cached_property
    def hexagon_version(self) -> int:
        """The chipset hexagon version number"""
        for attr in self.reference_device.attributes:
            if attr.startswith("hexagon:v"):
                return int(attr[9:])
        raise ValueError(f"Hexagon version not found for device: {self.name}")

    @cached_property
    def supports_fp16_npu(self) -> bool:
        """Whether this device's NPU supports FP16 inference."""
        if self.mirror_device:
            return self.mirror_device.supports_fp16_npu

        return "htp-supports-fp16:true" in self.reference_device.attributes

    def npu_supports_precision(self, precision: Precision) -> bool:
        """Whether this device's NPU supports the given quantization spec."""
        if self.mirror_device:
            return self.mirror_device.npu_supports_precision(precision)

        return not precision.has_float_activations or self.supports_fp16_npu

    @cached_property
    def supported_runtimes(self) -> list[TargetRuntime]:
        """All runtimes supported by this device."""
        if self.mirror_device:
            return self.mirror_device.supported_runtimes

        supports_qnn = False
        runtimes: list[TargetRuntime] = []
        for attr in self.reference_device.attributes:
            if attr.startswith(_FRAMEWORK_ATTR_PREFIX):
                fw_name = attr[len(_FRAMEWORK_ATTR_PREFIX) + 1 :].lower()
                runtimes.extend(
                    [x for x in TargetRuntime if x.inference_engine.value == fw_name]
                )
                supports_qnn = supports_qnn or fw_name == InferenceEngine.QNN.value

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
        if self.mirror_device:
            return self.mirror_device.profile_paths

        if self._profile_paths is not None:
            return self._profile_paths

        inference_engines_to_test: list[InferenceEngine] = []
        if (
            self.form_factor == ScorecardDevice.FormFactor.PHONE  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self.form_factor == ScorecardDevice.FormFactor.TABLET
            or self.form_factor == ScorecardDevice.FormFactor.IOT
        ):
            inference_engines_to_test = list(InferenceEngine)
        elif (
            self.form_factor == ScorecardDevice.FormFactor.AUTO  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self.form_factor == ScorecardDevice.FormFactor.XR
        ):
            inference_engines_to_test = [InferenceEngine.QNN, InferenceEngine.TFLITE]
        elif self.form_factor == ScorecardDevice.FormFactor.COMPUTE:
            inference_engines_to_test = [InferenceEngine.QNN, InferenceEngine.ONNX]
        else:
            assert_never(self.form_factor)

        return [
            path
            for path in ScorecardProfilePath
            if path.runtime in self.supported_runtimes
            and path.runtime.inference_engine in inference_engines_to_test
        ]

    @cached_property
    def compile_paths(self) -> list[ScorecardCompilePath]:
        """All compile paths supported by this device."""
        if self.mirror_device:
            return self.mirror_device.compile_paths

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


# ----------------------
# DEVICE DEFINITIONS
#
# This list is the set of devices we use by default when benchmarking models.
#
# Typically we define one device per chipset, and devices are named after that chipset:
# cs_8_gen_3 == device representative of the 8 gen 3 chipset
# ----------------------

##
# Universal Chipset
#
# A placeholder for compiling universal assets (that are applicable to any device)
#
# .tflite, .onnx, and .dlc are always universal, so they are compiled once for this device
# and used for inference on all other devices.
#
##
cs_universal = ScorecardDevice(
    name=UNIVERSAL_DEVICE_SCORECARD_NAME,
    reference_device_name="Samsung Galaxy S24",
    compile_paths=[path for path in ScorecardCompilePath if path.is_universal],
    profile_paths=[],
)


##
# Mobile Chipsets (cs)
##
cs_8_gen_3 = ScorecardDevice(
    name="cs_8_gen_3",
    reference_device_name="Samsung Galaxy S24",
    execution_device_name="Samsung Galaxy S24 (Family)",
)

cs_8_elite = ScorecardDevice(
    name="cs_8_elite",
    reference_device_name="Samsung Galaxy S25",
    execution_device_name="Samsung Galaxy S25 (Family)",
    is_default=True,
)

cs_7_gen_4 = ScorecardDevice(
    name="cs_7_gen_4",
    reference_device_name="Snapdragon 7 Gen 4 QRD",
)

cs_8_elite_gen_5 = ScorecardDevice(
    name="cs_8_elite_gen_5", reference_device_name="Snapdragon 8 Elite Gen 5 QRD"
)


##
# Compute Chipsets (cs)
##
cs_x_elite = ScorecardDevice(
    name="cs_x_elite",
    reference_device_name="Snapdragon X Elite CRD",
)

cs_x2_elite = ScorecardDevice(
    name="cs_x2_elite",
    reference_device_name="Snapdragon X2 Elite CRD",
)


##
# Auto Chipsets (cs)
##
cs_auto_monaco_7255 = ScorecardDevice(
    name="cs_auto_monaco_7255",
    reference_device_name="SA7255P ADP",
)

cs_auto_makena_8295 = ScorecardDevice(
    name="cs_auto_makena_8295",
    reference_device_name="SA8295P ADP",
)

cs_auto_lemans_8775 = ScorecardDevice(
    name="cs_auto_lemans_8775",
    reference_device_name="SA8775P ADP",
    npu_count=2,
)


##
# IoT Chipsets (cs)
##
cs_6490 = ScorecardDevice(
    name="cs_6490",
    reference_device_name="Dragonwing RB3 Gen 2 Vision Kit",
)

cs_6690 = ScorecardDevice(
    name="cs_6690",
    reference_device_name="Dragonwing Q-6690 MTP",
)

cs_8275 = ScorecardDevice(
    name="cs_8275",
    reference_device_name="QCS8275 (Proxy)",
    mirror_device=cs_auto_monaco_7255,
)

cs_8550 = ScorecardDevice(name="cs_8550", reference_device_name="QCS8550 (Proxy)")

cs_9075 = ScorecardDevice(
    name="cs_9075",
    reference_device_name="Dragonwing IQ-9075 EVK",
    npu_count=2,
)


##
# XR Chipsets (cs)
##
cs_xr_8450 = ScorecardDevice(name="cs_xr_8450", reference_device_name="QCS8450 (Proxy)")


DEFAULT_SCORECARD_DEVICE = ScorecardDevice.get("default")
DEFAULT_EXPORT_DEVICE = (
    ScorecardDevice.get("default").execution_device_name
    or ScorecardDevice.get("default").reference_device_name
)
CANARY_DEVICES = {
    DEFAULT_EXPORT_DEVICE,
    "Snapdragon X Elite CRD",
}
