# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Device identity, registry, and metadata.

Defines:

- ``HubDeviceAttributes``: mixin that reads chipset / OS / form-factor / etc.
  from a ``hub.Device``. Consumers implement ``_hub_device`` to point at their
  underlying Hub device.
- ``RegisteredDevice``: single-name device with a global registry, mixes in
  ``HubDeviceAttributes`` over its own Hub device.

Pure-data types (``FormFactor``, ``OperatingSystem``) also live here so config
schemas can import device-related types without pulling in heavier
dependencies.
"""

from __future__ import annotations

from enum import Enum, unique
from functools import cached_property
from typing import Any

import qai_hub as hub
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from qai_hub_models import Precision
from qai_hub_models.utils.base_config import BaseQAIHMConfig

# -----------------------------------------------------------------------------
# Pure-data types (device metadata)
# -----------------------------------------------------------------------------


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
    UBUNTU = "Ubuntu"


class OperatingSystem(BaseQAIHMConfig):
    ostype: OperatingSystemType
    version: str

    def __str__(self) -> str:
        return f"{self.ostype.name} {self.version}"


# -----------------------------------------------------------------------------
# Device cache
# -----------------------------------------------------------------------------

_DEVICE_CACHE: dict[str, hub.Device | None] = {}


def _get_cached_device(device_name: str) -> hub.Device | None:
    """Get a hub.Device by name, with caching."""
    device = _DEVICE_CACHE.get(device_name)
    if not device:
        devices = hub.get_devices(device_name)
        device = devices[0] if devices else None
        _DEVICE_CACHE[device_name] = device
    return device


def resolve_hub_device(name: str = "", chipset: str = "", os: str = "") -> hub.Device:
    """Resolve a CLI ``--device`` / ``--chipset`` to a fully-populated hub.Device.

    Callers must pass at least one of ``name`` or ``chipset``. The returned
    ``hub.Device`` is guaranteed to carry Hub's full attribute list, so
    downstream consumers can rely on ``device.attributes`` without worrying
    about stubs.

    When ``os`` is set, the caches are bypassed on both read and write so that
    OS-scoped requests always go to Hub. That path is rare, and skipping the
    cache is simpler than keying it on ``(name, os)`` / ``(chipset, os)``.
    """
    if not name and not chipset:
        raise ValueError(
            "resolve_hub_device requires either a device name or a chipset."
        )
    if not os:
        if name and (cached := _get_cached_device(name)) is not None:
            return cached
        if chipset and not name:
            for rd in RegisteredDevice._registry.values():
                if chipset == rd.chipset or chipset in rd.chipset_aliases:
                    return rd.hub_device
    devices = hub.get_devices(
        name=name, os=os, attributes=[f"chipset:{chipset}"] if chipset else []
    )
    if not devices:
        raise ValueError(
            f"No hub device found for name={name!r} chipset={chipset!r}. "
            "Use `qai-hub list-devices` to see valid devices."
        )
    device = devices[0]
    if not os:
        _DEVICE_CACHE[device.name] = device
    return device


# -----------------------------------------------------------------------------
# HubDeviceAttributes mixin
# -----------------------------------------------------------------------------


class HubDeviceAttributes:
    """
    Mixin that reads chipset / OS / form-factor / etc. from a ``hub.Device``.

    Subclasses implement ``_hub_device`` to point at the Hub device that
    metadata should be read from.
    """

    # ``_npu_count`` is read from an override on the concrete class when
    # provided; otherwise falls back to 1 (see ``npu_count`` below).
    _npu_count: int | None

    @property
    def _hub_device(self) -> hub.Device:
        """The ``hub.Device`` backing all attribute lookups on this device."""
        raise NotImplementedError

    def _display_name(self) -> str:
        """Identifier used in error messages when an attribute is missing."""
        raise NotImplementedError

    @cached_property
    def chipset(self) -> str:
        """The chipset used by this device."""
        for attr in self._hub_device.attributes:
            if attr.startswith("chipset:"):
                return attr[8:]
        raise ValueError(f"Chipset not found for device: {self._display_name()}")

    @cached_property
    def chipset_aliases(self) -> list[str]:
        """The aliases for the chipset used by this device."""
        return [
            attr[8:]
            for attr in self._hub_device.attributes
            if attr.startswith("chipset:")
        ]

    @cached_property
    def npu_count(self) -> int:
        """Returns the number of NPUs on this device."""
        if self._npu_count is not None:
            return self._npu_count
        return 1

    @cached_property
    def os(self) -> OperatingSystem:
        """The operating system used by this device."""
        for attr in self._hub_device.attributes:
            if attr.startswith("os:"):
                return OperatingSystem(
                    ostype=OperatingSystemType[attr.split(":")[-1].upper()],
                    version=self._hub_device.os,
                )
        raise ValueError(f"OS not found for device: {self._display_name()}")

    @cached_property
    def vendor(self) -> str:
        """The vendor that manufactures this device."""
        for attr in self._hub_device.attributes:
            if attr.startswith("vendor:"):
                return attr.split(":")[-1]
        raise ValueError(f"Vendor not found for device: {self._display_name()}")

    @cached_property
    def form_factor(self) -> FormFactor:
        """The device form factor (eg. Auto, IoT, Mobile, ...)"""
        for attr in self._hub_device.attributes:
            if attr.startswith("format:"):
                return FormFactor[attr.split(":")[-1].upper()]
        raise ValueError(f"Format not found for device: {self._display_name()}")

    @cached_property
    def hexagon_version(self) -> int:
        """The chipset hexagon version number"""
        for attr in self._hub_device.attributes:
            if attr.startswith("hexagon:v"):
                return int(attr[len("hexagon:v") :])
        raise ValueError(
            f"Hexagon version not found for device: {self._display_name()}"
        )

    @cached_property
    def soc_model(self) -> int:
        for attr in self._hub_device.attributes:
            if attr.startswith("soc-model:"):
                return int(attr[len("soc-model:") :])
        raise ValueError(f"SoC model not found for device: {self._display_name()}")

    @cached_property
    def supports_fp16_npu(self) -> bool:
        """Whether this device's NPU supports FP16 inference."""
        return "htp-supports-fp16:true" in self._hub_device.attributes

    @cached_property
    def supports_weight_sharing(self) -> bool:
        """Whether this device's NPU supports weight sharing."""
        return "htp-supports-weight-sharing:true" in self._hub_device.attributes

    def npu_supports_precision(self, precision: Precision) -> bool:
        """Whether this device's NPU supports the given quantization spec."""
        return not precision.has_float_activations or self.supports_fp16_npu


# -----------------------------------------------------------------------------
# RegisteredDevice class
# -----------------------------------------------------------------------------


class RegisteredDevice(HubDeviceAttributes):
    """
    A registered Qualcomm device with a Hub device name and chipset metadata.

    RegisteredDevice owns identity plus chipset-attribute lookups against Hub.
    ``get()`` accepts only registered Hub device names -- use ``get_default()``
    to fetch the device flagged as the default choice.
    """

    _registry: dict[str, RegisteredDevice] = {}

    @classmethod
    def get(
        cls, device_name: str, return_unregistered: bool = False
    ) -> RegisteredDevice:
        """
        Get a device by its registered device_name.

        Parameters
        ----------
        device_name
            Registered Hub device name.
        return_unregistered
            If True and the device is not in the registry, return a new
            unregistered device with the given name.

        Returns
        -------
        RegisteredDevice
            The requested device.

        Raises
        ------
        ValueError
            If the device is not found and ``return_unregistered`` is False.
        """
        if device_name in cls._registry:
            return cls._registry[device_name]

        if return_unregistered:
            return cls(device_name, register=False)

        raise ValueError(f"Unknown device: {device_name}")

    @classmethod
    def get_default(cls) -> RegisteredDevice:
        """Return the registered device with ``is_default=True``."""
        for device in cls._registry.values():
            if device.is_default:
                return device
        raise ValueError("No default device found.")

    @classmethod
    def parse(cls, obj: str | RegisteredDevice) -> RegisteredDevice:
        """Pydantic-friendly parser for device names or instances."""
        if isinstance(obj, str):
            return cls.get(obj, return_unregistered=True)
        if isinstance(obj, RegisteredDevice):
            return obj
        raise ValueError(f"Can't parse type {type(obj)} as RegisteredDevice")

    def __init__(
        self,
        device_name: str,
        npu_count: int | None = None,
        register: bool = True,
        is_default: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        device_name
            Hub device name. Used both as the registry key and as the Hub
            device name for attribute lookups. Devices with a "family" alias
            can pass the family name here so Hub scheduling picks any member.
        npu_count
            How many NPUs this device has. If undefined, defaults to 1.
        register
            Whether to register this device in the global registry.
        is_default
            Whether this device represents the user's default choice.
        """
        if register and device_name in self.__class__._registry:
            raise ValueError(f"Device {device_name} already registered.")

        self.device_name = device_name
        self._npu_count = npu_count
        self.is_default = is_default

        if register:
            self.__class__._registry[device_name] = self

    def __str__(self) -> str:
        return self.device_name

    def __repr__(self) -> str:
        return self.device_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisteredDevice):
            return False
        return self.device_name == other.device_name

    def __hash__(self) -> int:
        return hash(self.device_name)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            lambda obj, _: cls.parse(obj),
            handler(Any),
            serialization=core_schema.plain_serializer_function_ser_schema(
                RegisteredDevice.__str__, when_used="json"
            ),
        )

    @cached_property
    def hub_device(self) -> hub.Device:
        """The underlying ``hub.Device`` used for chipset lookups."""
        device = _get_cached_device(self.device_name)
        if not device:
            raise ValueError(f"Device {self.device_name} not found on Hub.")
        return device

    @cached_property
    def available_in_hub(self) -> bool:
        """Returns true if this device is available in AI Hub Workbench."""
        return _get_cached_device(self.device_name) is not None

    @property
    def _hub_device(self) -> hub.Device:
        return self.hub_device

    def _display_name(self) -> str:
        return self.device_name


# -----------------------------------------------------------------------------
# Device instances
# -----------------------------------------------------------------------------

# Mobile chipsets. For S22-S26 the ``device_name`` is the Family alias so Hub
# job scheduling can pick any member of the family.
cs_8_gen_1 = RegisteredDevice(device_name="Samsung Galaxy S22 (Family)")
cs_8_gen_2 = RegisteredDevice(device_name="Samsung Galaxy S23 (Family)")
cs_8_gen_3 = RegisteredDevice(device_name="Samsung Galaxy S24 (Family)")
cs_8_elite = RegisteredDevice(
    device_name="Samsung Galaxy S25 (Family)",
    is_default=True,
)
cs_8_elite_qrd = RegisteredDevice(device_name="Snapdragon 8 Elite QRD")
cs_7_gen_4 = RegisteredDevice(device_name="Snapdragon 7 Gen 4 QRD")
cs_8_elite_gen_5 = RegisteredDevice(device_name="Samsung Galaxy S26 (Family)")
cs_8_elite_gen_5_qrd = RegisteredDevice(device_name="Snapdragon 8 Elite Gen 5 QRD")

# Compute chipsets
cs_x_elite = RegisteredDevice(device_name="Snapdragon X Elite CRD")
cs_x_plus_8_core = RegisteredDevice(device_name="Snapdragon X Plus 8-Core CRD")
cs_x2_elite = RegisteredDevice(device_name="Snapdragon X2 Elite CRD")

# Auto chipsets
cs_auto_monaco_7255 = RegisteredDevice(device_name="SA7255P ADP")
cs_auto_makena_8295 = RegisteredDevice(device_name="SA8295P ADP")
cs_auto_lemans_8775 = RegisteredDevice(device_name="SA8775P ADP", npu_count=2)

# IoT chipsets
cs_6490 = RegisteredDevice(device_name="Dragonwing RB3 Gen 2 Vision Kit")
cs_6690 = RegisteredDevice(device_name="Dragonwing Q-6690 MTP")
cs_8550 = RegisteredDevice(device_name="QCS8550 (Proxy)")
cs_9075 = RegisteredDevice(device_name="Dragonwing IQ-9075 EVK", npu_count=2)


# -----------------------------------------------------------------------------
# Registry constants
# -----------------------------------------------------------------------------

DEFAULT_REGISTERED_DEVICE = RegisteredDevice.get_default()
DEFAULT_EXPORT_DEVICE = DEFAULT_REGISTERED_DEVICE.device_name
CANARY_DEVICES = {
    DEFAULT_EXPORT_DEVICE,
    "Snapdragon X Elite CRD",
    "Dragonwing IQ-9075 EVK",
}
