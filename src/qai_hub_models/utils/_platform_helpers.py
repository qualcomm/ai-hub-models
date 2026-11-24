# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
The AI Hub platform registry (devices, chipsets, form factors, runtimes).

``platform.yaml`` and ``similar_devices.yaml`` are the YAML form of a
``PlatformInfo`` proto. The loaders here return that proto directly; the result is
cached, so treat it as read-only.
"""

from __future__ import annotations

import re
from functools import cache

import qai_hub as hub
from qai_hub_models_cli.proto import platform_pb2
from qai_hub_models_cli.proto_helpers._common import read_proto_yaml, write_proto_yaml
from typing_extensions import assert_never

from qai_hub_models.configs.chipset_yaml import WebsiteWorld, website_world_to_proto
from qai_hub_models.configs.proto_helpers import form_factor_to_proto, runtime_to_proto
from qai_hub_models.core.configs import get_platform
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.utils.device import WEBSITE_EXCLUDED_DEVICES, FormFactor
from qai_hub_models.utils.path_helpers import (
    PLATFORM_YAML_PATH,
    SIMILAR_DEVICES_YAML_PATH,
)

_OS_TYPE_TO_PROTO: dict[str, int] = {
    "Android": platform_pb2.OPERATING_SYSTEM_TYPE_ANDROID,
    "Windows": platform_pb2.OPERATING_SYSTEM_TYPE_WINDOWS,
    "Linux": platform_pb2.OPERATING_SYSTEM_TYPE_LINUX,
    "Qualcomm Linux": platform_pb2.OPERATING_SYSTEM_TYPE_QC_LINUX,
    "Ubuntu": platform_pb2.OPERATING_SYSTEM_TYPE_UBUNTU,
}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


@cache
def load_similar_devices_raw() -> platform_pb2.PlatformInfo:
    """Load similar_devices.yaml (devices and chipsets only). Cached; do not mutate."""
    return read_proto_yaml(SIMILAR_DEVICES_YAML_PATH, platform_pb2.PlatformInfo)


@cache
def platform_devices(
    include_similar_devices: bool = False,
) -> dict[str, platform_pb2.DeviceInfo]:
    """Devices from the platform, keyed by name."""
    return {
        d.name: d
        for d in get_platform(_include_similar_devices=include_similar_devices).devices
    }


@cache
def platform_chipsets(
    include_similar_devices: bool = False,
) -> dict[str, platform_pb2.ChipsetInfo]:
    """Chipsets from the platform, keyed by name."""
    return {
        c.name: c
        for c in get_platform(_include_similar_devices=include_similar_devices).chipsets
    }


@cache
def device_names_with_form_factor(form_factor: FormFactor) -> frozenset[str]:
    """Names of all platform devices (similar devices included) with this form factor."""
    proto_form_factor = form_factor_to_proto(form_factor)
    return frozenset(
        name
        for name, device in platform_devices(include_similar_devices=True).items()
        if device.form_factor == proto_form_factor
    )


@cache
def load_similar_devices() -> dict[str, tuple[str, list[str]]]:
    """
    Load the similar devices mapping, resolving reference chipsets to device names.

    For each entry, the lookup uses `reference_chipset` (or `chipset` if unset).

    Returns unsupported_device_name -> (real_chipset, [reference_device_names]).
    The real_chipset (the device's own `chipset` field) gets added to perf.yaml's
    `supported_chipsets` list when perf numbers are copied from a reference device.
    """
    chipset_to_devices: dict[str, list[str]] = {}
    for device in get_platform().devices:
        chipset_to_devices.setdefault(device.chipset, []).append(device.name)

    return {
        entry.name: (
            entry.chipset,
            chipset_to_devices.get(entry.reference_chipset or entry.chipset, []),
        )
        for entry in load_similar_devices_raw().devices
    }


# -----------------------------------------------------------------------------
# Generation (from AI Hub Workbench + ScorecardDevice)
# -----------------------------------------------------------------------------


def chipset_marketing_name(
    chipset: str,
    aliases: list[str] | None = None,
    world: WebsiteWorld | None = None,
) -> str:
    """Sanitize chip name to match marketing.

    If ``aliases`` contains a ``qualcomm-dragonwing-*`` entry, the marketing
    name is derived from that alias so IoT chipsets pick up the Dragonwing™
    brand automatically (e.g. ``qualcomm-dragonwing-qcs8550-proxy`` ->
    ``Qualcomm® Dragonwing™ QCS8550 (Proxy)``).
    """
    for alias in aliases or ():
        if not alias.startswith("qualcomm-dragonwing-"):
            continue
        parts = alias.removeprefix("qualcomm-dragonwing-").split("-")
        # QCS/QCM/SA families fuse the prefix with the digits (QCS8550).
        # Short prefixes (Q, IQ) keep the dash to the model number (Q-6690, IQ-8275).
        if parts[0].startswith(("qcs", "qcm", "sa")):
            product = parts[0].upper()
            extras = parts[1:]
        else:
            # Enurues Q- and IQ- are capitalized
            product = "-".join(p.upper() for p in parts[:2])
            extras = parts[2:]
        out = f"Qualcomm® Dragonwing™ {product}"
        for extra in extras:
            out += " (Proxy)" if extra == "proxy" else f" {extra.upper()}"
        return out
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


def create_chipset_info(device: ScorecardDevice) -> platform_pb2.ChipsetInfo:
    """Create a ChipsetInfo from a ScorecardDevice with scorecard-specific formatting."""
    world = WebsiteWorld.from_form_factor(device.form_factor)
    aliases = device.chipset_aliases
    return platform_pb2.ChipsetInfo(
        name=device.chipset,
        aliases=aliases,
        marketing_name=chipset_marketing_name(device.chipset, aliases, world),
        world=website_world_to_proto(world),
        supports_fp16=device.supports_fp16_npu,
        htp_version=device.hexagon_version,
        soc_model=device.soc_model,
        reference_device=device.reference_device_name,
        supports_weight_sharing=device.supports_weight_sharing,
    )


def website_icon_for_device(device: ScorecardDevice) -> int:
    """The ``WebsiteIcon`` proto value to show for this device."""
    if device.form_factor == FormFactor.PHONE:
        if device.chipset == "qualcomm-snapdragon-888":
            return platform_pb2.WEBSITE_ICON_PHONE_S21
        if device.chipset == "qualcomm-snapdragon-8gen1":
            return platform_pb2.WEBSITE_ICON_PHONE_S22
        if device.chipset == "qualcomm-snapdragon-8gen2":
            if "Ultra" in device.reference_device_name:
                return platform_pb2.WEBSITE_ICON_PHONE_S23_ULTRA
            return platform_pb2.WEBSITE_ICON_PHONE_S23
        if device.chipset == "qualcomm-snapdragon-8gen3":
            if "Ultra" in device.reference_device_name:
                return platform_pb2.WEBSITE_ICON_PHONE_S24_ULTRA
            return platform_pb2.WEBSITE_ICON_PHONE_S24
        return platform_pb2.WEBSITE_ICON_PHONE_S21
    if device.form_factor == FormFactor.COMPUTE:
        if device.chipset in [
            "qualcomm-snapdragon-8cxgen3",
            "qualcomm-snapdragon-x-plus-8-core",
            "qualcomm-snapdragon-x-elite",
        ]:
            return platform_pb2.WEBSITE_ICON_LAPTOP_X_ELITE
        return platform_pb2.WEBSITE_ICON_LAPTOP_GENERIC
    if device.form_factor == FormFactor.TABLET:
        return platform_pb2.WEBSITE_ICON_TABLET_ANDROID
    if device.form_factor == FormFactor.XR:
        return platform_pb2.WEBSITE_ICON_XR_HEADSET
    if device.form_factor == FormFactor.IOT:
        if device.chipset in [
            "qualcomm-qcs6490-proxy",
            "qualcomm-qcs8250-proxy",
            "qualcomm-qcs8275-proxy",
            "qualcomm-qcs9075-proxy",
        ] and device.reference_device_name not in [
            "RB3 Gen 2 (Proxy)",
            "RB5 (Proxy)",
        ]:
            return platform_pb2.WEBSITE_ICON_IOT_CHIP
        return platform_pb2.WEBSITE_ICON_IOT_DRONE
    if device.form_factor == FormFactor.AUTO:
        return platform_pb2.WEBSITE_ICON_CAR
    assert_never(device.form_factor)


def create_device_info(device: ScorecardDevice) -> platform_pb2.DeviceInfo:
    """Create a DeviceInfo from a ScorecardDevice."""
    return platform_pb2.DeviceInfo(
        name=device.reference_device_name,
        chipset=device.chipset,
        npu_count=device.npu_count,
        os=platform_pb2.OperatingSystem(
            ostype=_OS_TYPE_TO_PROTO[device.os.ostype.value],
            version=device.os.version,
        ),
        form_factor=form_factor_to_proto(device.form_factor),
        vendor=device.vendor,
        icon=website_icon_for_device(device),
        enabled_in_scorecard=device in ScorecardDevice._registry.values(),
        available_in_workbench=True,
    )


def _form_factor_display_name(ff: FormFactor) -> str:
    if ff == FormFactor.AUTO:
        return "Automotive"
    return ff.value


def _validate_similar_devices(device_names: set[str], chipset_names: set[str]) -> None:
    """
    Check similar_devices.yaml is consistent with the generated platform.

    Similar (unsupported) devices live in similar_devices.yaml and are never saved
    to platform.yaml, but they must not overlap with it and must reference known
    chipsets and devices.

    Parameters
    ----------
    device_names
        Names of the devices generated for platform.yaml.
    chipset_names
        Names of the chipsets generated for platform.yaml.
    """
    similar = load_similar_devices_raw()
    similar_chipsets = {c.name: c for c in similar.chipsets}
    similar_devices = {d.name: d for d in similar.devices}

    chips_in_hub_and_sd = set(similar_chipsets) & chipset_names
    assert not chips_in_hub_and_sd, (
        f"Similar devices yaml contains chipsets that are available on workbench: {chips_in_hub_and_sd}"
    )
    dev_in_hub_and_sd = set(similar_devices) & device_names
    assert not dev_in_hub_and_sd, (
        f"Similar devices yaml contains devices that are available on workbench: {dev_in_hub_and_sd}"
    )

    known_chipsets = chipset_names | set(similar_chipsets)
    known_devices = device_names | set(similar_devices)
    for name, entry in similar_devices.items():
        assert entry.chipset in known_chipsets, (
            f"Unknown chipset for device {name} in similar-devices.yaml: {entry.chipset}"
        )
        if entry.reference_chipset:
            assert entry.reference_chipset in known_chipsets, (
                f"Unknown reference_chipset for device {name} in similar-devices.yaml: {entry.reference_chipset}"
            )
    for name, chipset in similar_chipsets.items():
        assert chipset.reference_device in known_devices, (
            f"Unknown reference device for chipset {name} in similar-devices.yaml: {chipset.reference_device}"
        )


def generate_platform_info() -> platform_pb2.PlatformInfo:
    """
    Re-generate the platform registry from the current set of devices / runtimes
    that are valid in AI Hub Models perf.yaml files.
    """
    out = platform_pb2.PlatformInfo()

    for ff in sorted(FormFactor, key=lambda ff: ff.value):
        out.form_factors.add(
            form_factor=form_factor_to_proto(ff),
            display_name=_form_factor_display_name(ff),
            world=website_world_to_proto(WebsiteWorld.from_form_factor(ff)),
        )

    for path in sorted(
        (p for p in ScorecardInferencePath if p.is_published), key=lambda p: p.value
    ):
        out.runtimes.add(
            runtime=runtime_to_proto(path.runtime),
            website_runtime=path.website_runtime_name,
            file_extension=f".{path.runtime.file_extension}",
            is_aot_compiled=path.runtime.is_aot_compiled,
            display_name=path.runtime.display_name,
            description=path.runtime.description,
            documentation_url=path.runtime.documentation_url,
        )

    devices: dict[str, platform_pb2.DeviceInfo] = {}
    chipsets: dict[str, platform_pb2.ChipsetInfo] = {}

    # For each hub device...
    for hub_device in hub.get_devices():
        if "(Family)" in hub_device.name:
            # Exclude "Family" devices
            continue
        if hub_device.name in WEBSITE_EXCLUDED_DEVICES:
            continue
        if hub_device.name in devices:
            # Exclude multiple devices with the same name
            # (eg different OS)
            continue

        device = ScorecardDevice.get(hub_device.name, return_unregistered=True)

        if "qualcomm" not in device.chipset:
            # Exclude non-qualcomm devices
            continue

        devices[device.reference_device_name] = create_device_info(device)
        if device.chipset not in chipsets:
            chipsets[device.chipset] = create_chipset_info(device)
        if "htp-supports-fp16:true" in hub_device.attributes:
            chipsets[device.chipset].supports_fp16 = True

    # Use the scorecard device for the reference device, if one exists.
    # Skip devices whose chipset was filtered above (non-qualcomm) so their
    # similar_devices.yaml entry stays authoritative.
    for device in ScorecardDevice.all_devices():
        if device.chipset not in chipsets:
            continue
        chipsets[device.chipset].reference_device = device.reference_device.name

    _validate_similar_devices(set(devices), set(chipsets))

    out.devices.extend(devices[name] for name in sorted(devices))
    out.chipsets.extend(chipsets[name] for name in sorted(chipsets))
    return out


def save_platform(platform: platform_pb2.PlatformInfo) -> None:
    """Write the platform registry to its standard YAML location."""
    write_proto_yaml(platform, PLATFORM_YAML_PATH)
