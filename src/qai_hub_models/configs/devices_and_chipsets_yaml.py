# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import re
from enum import Enum, unique
from functools import cache

import qai_hub as hub
from pydantic import Field
from qai_hub_models_cli.proto import platform_pb2
from typing_extensions import assert_never

from qai_hub_models.configs.proto_helpers import form_factor_to_proto, runtime_to_proto
from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    get_canonical_chipset_name,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT
from qai_hub_models.utils.qai_hub_helpers import get_device_and_chipset_name

SCORECARD_DEVICE_YAML_PATH = QAIHM_PACKAGE_ROOT / "devices_and_chipsets.yaml"
SIMILAR_DEVICES_YAML_PATH = QAIHM_PACKAGE_ROOT / "similar_devices.yaml"


@unique
class WebsiteWorld(Enum):
    Mobile = "Mobile"
    Compute = "Compute"
    Automotive = "Automotive"
    IoT = "IoT"
    XR = "XR"

    @staticmethod
    def from_form_factor(form_factor: ScorecardDevice.FormFactor) -> WebsiteWorld:
        if (
            form_factor == ScorecardDevice.FormFactor.PHONE  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or form_factor == ScorecardDevice.FormFactor.TABLET
        ):
            return WebsiteWorld.Mobile
        if form_factor == ScorecardDevice.FormFactor.XR:
            return WebsiteWorld.XR
        if form_factor == ScorecardDevice.FormFactor.COMPUTE:
            return WebsiteWorld.Compute
        if form_factor == ScorecardDevice.FormFactor.IOT:
            return WebsiteWorld.IoT
        if form_factor == ScorecardDevice.FormFactor.AUTO:
            return WebsiteWorld.Automotive
        assert_never(form_factor)


@unique
class WebsiteIcon(Enum):
    Car = "Car"
    IoT_Chip = "IoT_Chip"
    IoT_Drone = "IoT_Drone"
    Laptop_Generic = "Laptop_Generic"
    Laptop_X_Elite = "Laptop_X_Elite"
    Phone_S21 = "Phone_S21"
    Phone_S22 = "Phone_S22"
    Phone_S23 = "Phone_S23"
    Phone_S23_Ultra = "Phone_S23_Ultra"
    Phone_S24 = "Phone_S24"
    Phone_S24_Ultra = "Phone_S24_Ultra"
    Tablet_Android = "Tablet_Android"
    XR_Headset = "XR_Headset"

    @staticmethod
    def from_device(device: ScorecardDevice) -> WebsiteIcon:
        if device.form_factor == ScorecardDevice.FormFactor.PHONE:
            if device.chipset == "qualcomm-snapdragon-888":
                return WebsiteIcon.Phone_S21
            if device.chipset == "qualcomm-snapdragon-8gen1":
                return WebsiteIcon.Phone_S22
            if device.chipset == "qualcomm-snapdragon-8gen2":
                if "Ultra" in device.reference_device_name:
                    return WebsiteIcon.Phone_S23_Ultra
                return WebsiteIcon.Phone_S23
            if device.chipset == "qualcomm-snapdragon-8gen3":
                if "Ultra" in device.reference_device_name:
                    return WebsiteIcon.Phone_S24_Ultra
                return WebsiteIcon.Phone_S24
            return WebsiteIcon.Phone_S21
        if device.form_factor == ScorecardDevice.FormFactor.COMPUTE:
            if device.chipset in [
                "qualcomm-snapdragon-8cxgen3",
                "qualcomm-snapdragon-x-plus-8-core",
                "qualcomm-snapdragon-x-elite",
            ]:
                return WebsiteIcon.Laptop_X_Elite
            return WebsiteIcon.Laptop_Generic
        if device.form_factor == ScorecardDevice.FormFactor.TABLET:
            return WebsiteIcon.Tablet_Android
        if device.form_factor == ScorecardDevice.FormFactor.XR:
            return WebsiteIcon.XR_Headset
        if device.form_factor == ScorecardDevice.FormFactor.IOT:
            if device.chipset in [
                "qualcomm-qcs6490-proxy",
                "qualcomm-qcs8250-proxy",
                "qualcomm-qcs8275-proxy",
                "qualcomm-qcs9075-proxy",
            ] and device.reference_device_name not in [
                "RB3 Gen 2 (Proxy)",
                "RB5 (Proxy)",
            ]:
                return WebsiteIcon.IoT_Chip
            return WebsiteIcon.IoT_Drone
        if device.form_factor == ScorecardDevice.FormFactor.AUTO:
            return WebsiteIcon.Car
        assert_never(device.form_factor)


_WEBSITE_WORLD_TO_PROTO: dict[str, int] = {
    "Mobile": platform_pb2.WEBSITE_WORLD_MOBILE,
    "Compute": platform_pb2.WEBSITE_WORLD_COMPUTE,
    "Automotive": platform_pb2.WEBSITE_WORLD_AUTOMOTIVE,
    "IoT": platform_pb2.WEBSITE_WORLD_IOT,
    "XR": platform_pb2.WEBSITE_WORLD_XR,
}

_WEBSITE_ICON_TO_PROTO: dict[str, int] = {
    "Car": platform_pb2.WEBSITE_ICON_CAR,
    "IoT_Chip": platform_pb2.WEBSITE_ICON_IOT_CHIP,
    "IoT_Drone": platform_pb2.WEBSITE_ICON_IOT_DRONE,
    "Laptop_Generic": platform_pb2.WEBSITE_ICON_LAPTOP_GENERIC,
    "Laptop_X_Elite": platform_pb2.WEBSITE_ICON_LAPTOP_X_ELITE,
    "Phone_S21": platform_pb2.WEBSITE_ICON_PHONE_S21,
    "Phone_S22": platform_pb2.WEBSITE_ICON_PHONE_S22,
    "Phone_S23": platform_pb2.WEBSITE_ICON_PHONE_S23,
    "Phone_S23_Ultra": platform_pb2.WEBSITE_ICON_PHONE_S23_ULTRA,
    "Phone_S24": platform_pb2.WEBSITE_ICON_PHONE_S24,
    "Phone_S24_Ultra": platform_pb2.WEBSITE_ICON_PHONE_S24_ULTRA,
    "Tablet_Android": platform_pb2.WEBSITE_ICON_TABLET_ANDROID,
    "XR_Headset": platform_pb2.WEBSITE_ICON_XR_HEADSET,
}

_OS_TYPE_TO_PROTO: dict[str, int] = {
    "Android": platform_pb2.OPERATING_SYSTEM_TYPE_ANDROID,
    "Windows": platform_pb2.OPERATING_SYSTEM_TYPE_WINDOWS,
    "Linux": platform_pb2.OPERATING_SYSTEM_TYPE_LINUX,
    "Qualcomm Linux": platform_pb2.OPERATING_SYSTEM_TYPE_QC_LINUX,
}


class FormFactorYaml(BaseQAIHMConfig):
    display_name: str
    world: WebsiteWorld

    @staticmethod
    def from_form_factor(form_factor: ScorecardDevice.FormFactor) -> FormFactorYaml:
        return FormFactorYaml(
            display_name=FormFactorYaml._form_factor_to_display_name(form_factor),
            world=WebsiteWorld.from_form_factor(form_factor),
        )

    def to_proto(
        self, form_factor: ScorecardDevice.FormFactor
    ) -> platform_pb2.FormFactorInfo:
        return platform_pb2.FormFactorInfo(
            form_factor=form_factor_to_proto(form_factor),
            display_name=self.display_name,
            world=_WEBSITE_WORLD_TO_PROTO[self.world.value],
        )

    @staticmethod
    def _form_factor_to_display_name(ff: ScorecardDevice.FormFactor) -> str:
        if ff == ScorecardDevice.FormFactor.AUTO:
            return "Automotive"
        return ff.value


class DeviceDetailsYaml(BaseQAIHMConfig):
    chipset: str
    # Only set in similar_devices.yaml: the chipset whose perf numbers are
    # duplicated onto this device. The device's `chipset` is what gets added
    # to `supported_chipsets` in perf.yaml; `reference_chipset` is the lookup
    # key used to find a workbench device with results to copy.
    reference_chipset: str | None = None
    os: ScorecardDevice.OperatingSystem
    form_factor: ScorecardDevice.FormFactor
    vendor: str
    icon: WebsiteIcon
    npu_count: int = 1
    enabled_in_scorecard: bool = False
    available_in_workbench: bool = True

    @staticmethod
    def from_device(device: ScorecardDevice) -> DeviceDetailsYaml:
        return DeviceDetailsYaml(
            chipset=device.chipset,
            os=device.os,
            form_factor=device.form_factor,
            vendor=device.vendor,
            icon=WebsiteIcon.from_device(device),
            npu_count=device.npu_count,
            enabled_in_scorecard=device in ScorecardDevice._registry.values(),
        )

    def to_proto(self, name: str) -> platform_pb2.DeviceInfo:
        return platform_pb2.DeviceInfo(
            name=name,
            chipset=self.chipset,
            reference_chipset=self.reference_chipset or "",
            npu_count=self.npu_count,
            os=platform_pb2.OperatingSystem(
                ostype=_OS_TYPE_TO_PROTO[self.os.ostype.value],
                version=self.os.version,
            ),
            form_factor=form_factor_to_proto(self.form_factor),
            vendor=self.vendor,
            icon=_WEBSITE_ICON_TO_PROTO[self.icon.value],
            enabled_in_scorecard=self.enabled_in_scorecard,
            available_in_workbench=self.available_in_workbench,
        )


# By default, similar devices are stripped from the platform protobuf that we publish with releases.
# This is an exception list. (Similar devices in this list are not stripped.)
ALLOWED_SIMILAR_DEVICES = frozenset(
    {
        "Dragonwing IQ-8275 EVK"  # We hardcoded release assets for this, so it needs to be a valid device in the CLI.
    }
)


@cache
def _load_similar_devices_raw() -> DevicesAndChipsetsYaml:
    """Load the similar devices YAML as typed DeviceDetailsYaml entries."""
    return DevicesAndChipsetsYaml.from_yaml(SIMILAR_DEVICES_YAML_PATH)


@cache
def load_similar_devices() -> dict[str, tuple[str, list[str]]]:
    """
    Load the similar devices mapping, resolving reference chipsets to device names.

    For each entry, the lookup uses `reference_chipset` (or `chipset` if unset),
    plus any chipset that normalizes to the same value via get_canonical_chipset_name
    (e.g. 8-elite-for-galaxy -> 8-elite).

    Returns unsupported_device_name -> (real_chipset, [reference_device_names]).
    The real_chipset (the device's own `chipset` field) gets added to perf.yaml's
    `supported_chipsets` list when perf numbers are copied from a reference device.
    """
    raw = _load_similar_devices_raw()

    similar_names = set(raw.devices.keys())

    dc = DevicesAndChipsetsYaml.load()

    sanitized_to_devices: dict[str, list[str]] = {}
    for name, details in dc.devices.items():
        if name in similar_names:
            continue
        key = get_canonical_chipset_name(details.chipset)
        if key not in sanitized_to_devices:
            sanitized_to_devices[key] = []
        sanitized_to_devices[key].append(name)

    resolved: dict[str, tuple[str, list[str]]] = {}
    for unsupported_name, entry in raw.devices.items():
        lookup_chipset = entry.reference_chipset or entry.chipset
        key = get_canonical_chipset_name(lookup_chipset)
        resolved[unsupported_name] = (
            entry.chipset,
            sanitized_to_devices.get(key, []),
        )
    return resolved


class ChipsetYaml(BaseQAIHMConfig):
    aliases: list[str]
    marketing_name: str
    world: WebsiteWorld
    supports_fp16: bool = False
    htp_version: int
    soc_model: int
    reference_device: str
    supports_weight_sharing: bool = False

    @staticmethod
    def from_device(device: ScorecardDevice) -> ChipsetYaml:
        world = WebsiteWorld.from_form_factor(device.form_factor)
        return ChipsetYaml(
            aliases=device.chipset_aliases,
            marketing_name=ChipsetYaml.chipset_marketing_name(device.chipset, world),
            world=world,
            supports_fp16=device.supports_fp16_npu,
            htp_version=device.hexagon_version,
            soc_model=device.soc_model,
            reference_device=device.reference_device_name,
            supports_weight_sharing=device.supports_weight_sharing,
        )

    def to_proto(self, name: str) -> platform_pb2.ChipsetInfo:
        return platform_pb2.ChipsetInfo(
            name=name,
            aliases=self.aliases,
            marketing_name=self.marketing_name,
            world=_WEBSITE_WORLD_TO_PROTO[self.world.value],
            supports_fp16=self.supports_fp16,
            htp_version=self.htp_version,
            soc_model=self.soc_model,
            reference_device=self.reference_device,
        )

    @staticmethod
    def chipset_marketing_name(chipset: str, world: WebsiteWorld | None = None) -> str:
        """Sanitize chip name to match marketing."""
        chipset = get_canonical_chipset_name(chipset)
        chip = " ".join([word.capitalize() for word in chipset.split("-")])
        chip = chip.replace(
            "Qualcomm Snapdragon", "Snapdragon®"
        )  # Marketing name for Qualcomm Snapdragon is Snapdragon®
        chip = chip.replace(
            "Qualcomm", "Qualcomm®"
        )  # All other Qualcomm brand names should include a registered trademark

        chip = chip.replace("Proxy", "(Proxy)")

        # 8cxgen2 -> 8cx Gen 2
        # 8gen2 -> 8 Gen 2
        # Gen5 -> Gen 5
        chip = re.sub(
            r"(\w*)[g|G]en(\d+)",
            lambda m: f"{m.group(1)} Gen {m.group(2)}".strip(),
            chip,
        )

        # 8 Core -> 8-Core
        chip = re.sub(r"(\d+) Core", r"\g<1>-Core", chip)

        # qcs6490 -> QCS6490
        # sa8775p -> SA8775P
        chip = re.sub(
            r"(Qcs|Qcm|Sa)\s*(\w+)",
            lambda m: f"{m.group(1).upper()}{m.group(2).upper()}",
            chip,
        )

        return chip + (f" {world.value}" if world == WebsiteWorld.Mobile else "")


class DevicesAndChipsetsYaml(BaseQAIHMConfig):
    """
    Storage for device and chipset metadata.

    This class stores definitions / attributes of valid:
        * devices
        * chipsets
        * form factors
        * scorecard paths

    That the website reads from AI Hub Models perf.yaml files
    to create model card webpages.
    """

    scorecard_path_to_website_runtime: dict[ScorecardProfilePath, str] = Field(
        default_factory=dict
    )
    scorecard_path_assets_require_chipset: dict[ScorecardProfilePath, bool] = Field(
        default_factory=dict
    )
    scorecard_path_extensions: dict[ScorecardProfilePath, str] = Field(
        default_factory=dict
    )
    form_factors: dict[ScorecardDevice.FormFactor, FormFactorYaml] = Field(
        default_factory=dict
    )
    devices: dict[str, DeviceDetailsYaml] = Field(default_factory=dict)
    chipsets: dict[str, ChipsetYaml] = Field(default_factory=dict)

    @staticmethod
    def from_all_runtimes_and_devices() -> DevicesAndChipsetsYaml:
        """
        Re-generate a DevicesAndChipsetsYaml configuration from the current
        set of devices / runtimes that are valid in AI Hub Models perf.yaml files.
        """
        out = DevicesAndChipsetsYaml()
        out.form_factors = {
            ff: FormFactorYaml.from_form_factor(ff) for ff in ScorecardDevice.FormFactor
        }

        for profile_path in ScorecardProfilePath:
            if profile_path.is_published:
                out.scorecard_path_to_website_runtime[profile_path] = (
                    profile_path.website_runtime_name
                )
                out.scorecard_path_extensions[profile_path] = (
                    f".{profile_path.runtime.file_extension}"
                )
                out.scorecard_path_assets_require_chipset[profile_path] = (
                    profile_path.runtime.is_aot_compiled
                )

        # For each hub device...
        for hub_device in hub.get_devices():
            if "(Family)" in hub_device.name:
                # Exclude "Family" devices
                continue
            if hub_device.name in out.devices:
                # Exclude multiple devices with the same name
                # (eg different OS)
                continue

            device = ScorecardDevice.get(hub_device.name, return_unregistered=True)

            if "qualcomm" not in device.chipset:
                # Exclude non-qualcomm devices
                continue

            out.devices[device.reference_device_name] = DeviceDetailsYaml.from_device(
                device
            )
            if device.chipset not in out.chipsets:
                out.chipsets[device.chipset] = ChipsetYaml.from_device(device)
            out.chipsets[device.chipset].supports_fp16 |= (
                "htp-supports-fp16:true" in hub_device.attributes
            )

        # Use the scorecard device for the reference device, if one exists.
        for device in ScorecardDevice.all_devices():
            out.chipsets[device.chipset].reference_device = device.reference_device.name

        # Add similar (unsupported) devices with their own explicit metadata.
        sd = _load_similar_devices_raw()

        chips_in_hub_and_sd = set(sd.chipsets.keys()).intersection(out.chipsets.keys())
        assert not chips_in_hub_and_sd, (
            f"Similar devices yaml contains chipsets that are available on workbench: {chips_in_hub_and_sd}"
        )
        out.chipsets.update(sd.chipsets)

        dev_in_hub_and_sd = set(sd.devices.keys()).intersection(out.devices.keys())
        assert not dev_in_hub_and_sd, (
            f"Similar devices yaml contains devices that are available on workbench: {dev_in_hub_and_sd}"
        )
        for device_name, device_entry in sd.devices.items():
            # Set this so we manually don't have to set this for every model in similar devices yaml.
            device_entry.available_in_workbench = False
            assert device_entry.chipset in out.chipsets, (
                f"Unknown chipset for device {device_name} in similar-devices.yaml: {device_entry.chipset}"
            )
            if device_entry.reference_chipset is not None:
                assert device_entry.reference_chipset in out.chipsets, (
                    f"Unknown reference_chipset for device {device_name} in similar-devices.yaml: {device_entry.reference_chipset}"
                )
        out.devices.update(sd.devices)

        for chipset_name, chipset_entry in sd.chipsets.items():
            assert chipset_entry.reference_device in out.devices, (
                f"Unknown reference device for chipset {chipset_name} in similar-devices.yaml: {chipset_entry.reference_device}"
            )

        return out

    def to_proto(
        self,
        aihm_version: str,
        exclude_similar_devices: bool = True,
    ) -> platform_pb2.PlatformInfo:
        """Serialize to a ``PlatformInfo`` proto.

        When *exclude_similar_devices* is True (default), "similar" devices
        (those with a ``reference_chipset``, whose perf is borrowed rather than
        measured) are dropped, along with any chipset only those devices use.
        Devices in :data:`ALLOWED_SIMILAR_DEVICES` are kept regardless.
        """
        runtimes = [
            platform_pb2.RuntimeInfo(
                runtime=runtime_to_proto(path.runtime),
                website_runtime=self.scorecard_path_to_website_runtime[path],
                file_extension=self.scorecard_path_extensions[path],
                is_aot_compiled=self.scorecard_path_assets_require_chipset[path],
                display_name=path.runtime.display_name,
                description=path.runtime.description,
                documentation_url=path.runtime.documentation_url,
            )
            for path in self.scorecard_path_to_website_runtime
        ]
        form_factors = [
            ff_yaml.to_proto(ff) for ff, ff_yaml in self.form_factors.items()
        ]

        devices = self.devices
        chipsets = self.chipsets
        if exclude_similar_devices:
            devices = {
                name: d
                for name, d in self.devices.items()
                if d.reference_chipset is None or name in ALLOWED_SIMILAR_DEVICES
            }
            # Keep only chipsets still used by a retained device.
            used = {d.chipset for d in devices.values()}
            chipsets = {name: c for name, c in self.chipsets.items() if name in used}

        device_protos = [details.to_proto(name) for name, details in devices.items()]
        chipset_protos = [
            chipset_yaml.to_proto(name) for name, chipset_yaml in chipsets.items()
        ]
        return platform_pb2.PlatformInfo(
            aihm_version=aihm_version,
            runtimes=runtimes,
            form_factors=form_factors,
            devices=device_protos,
            chipsets=chipset_protos,
        )

    @staticmethod
    def load() -> DevicesAndChipsetsYaml:
        """Load this configuration from its standard YAML location in the AI Hub Models python package."""
        return DevicesAndChipsetsYaml.from_yaml(SCORECARD_DEVICE_YAML_PATH)

    def save(self) -> None:
        """Save this configuration to its standard YAML location in the AI Hub Models python package."""
        self.to_yaml(SCORECARD_DEVICE_YAML_PATH)

    def get_device_details_without_aihub(
        self, device: hub.Device
    ) -> tuple[str, DeviceDetailsYaml]:
        """
        Uses the devices defined in the YAML to convert a hub device to a device name and device details.
        This allows us to get device information without using the AI Hub API.

        Parameters
        ----------
        device
            Device for which to get details.

        Returns
        -------
        device_name : str
            Device Name

        device_details : DeviceDetailsYaml
            Device Details
        """
        device_name, chipset = get_device_and_chipset_name(device)
        # Device families aren't included in the YAML. Replace with the original device name.
        device_name = device_name.replace(" (Family)", "") if device_name else None

        if device_name is not None:
            if device_details := self.devices.get(device_name):
                return (device_name, device_details)

        elif chipset is not None:
            # Prefer to match scorecard-enabled devices first.
            for name, details in self.devices.items():
                if details.enabled_in_scorecard and details.chipset == chipset:
                    return (name, details)

            # Otherwise check all devices.
            for name, details in self.devices.items():
                if details.chipset == chipset:
                    return (name, details)

        raise ValueError(f"Unknown device: {device}.")
