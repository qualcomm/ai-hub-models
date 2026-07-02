# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Conversions between platform proto enums and their human-readable strings."""

from __future__ import annotations

from collections.abc import Iterable

from qai_hub_models_cli.proto.info_pb2 import (
    ModelDomain,
    ModelLicense,
    ModelTag,
    ModelUseCase,
)
from qai_hub_models_cli.proto.platform_pb2 import (
    FormFactor,
    OperatingSystem,
    OperatingSystemType,
    RuntimeInfo,
    WebsiteWorld,
)
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime


def normalize_label(s: str) -> str:
    """Lowercase and unify separators so filter values match display labels."""
    return s.lower().replace("_", " ").replace("-", " ")


# Display overrides for enum keys that should not be naively title-cased.
_PLATFORM_DISPLAY_OVERRIDES: dict[str, str] = {
    "XR": "XR",
    "IOT": "IoT",
    "AUTOMOTIVE": "Auto",
}


def precision_proto_to_str(precision: Precision.ValueType) -> str:
    """
    Convert a Precision proto enum value to its lowercase string name.

    Parameters
    ----------
    precision
        ``Precision`` enum value (e.g. ``PRECISION_FLOAT``).

    Returns
    -------
    str
        Lowercase name without the ``PRECISION_`` prefix (e.g. ``"float"``).

    Raises
    ------
    KeyError
        If *precision* is not a valid enum value.
    """
    name = Precision.Name(precision)
    if not name.startswith("PRECISION_"):
        raise KeyError(f"Unknown precision value: {precision!r}")
    return name.removeprefix("PRECISION_").lower()


def precision_str_to_proto(precision: str | Precision.ValueType) -> Precision.ValueType:
    """
    Convert a precision string to its proto enum value.

    Parameters
    ----------
    precision
        Precision name (e.g. ``"float"``, ``"w8a8"``, ``"PRECISION_MXFP4"``).
        Case-insensitive. The ``PRECISION_`` prefix is optional.

    Returns
    -------
    Precision.ValueType
        Corresponding ``Precision`` enum value.

    Raises
    ------
    KeyError
        If *precision* does not match any known precision.
    """
    if not isinstance(precision, str):
        return precision

    key = precision.upper()
    if not key.startswith("PRECISION_"):
        key = f"PRECISION_{key}"
    try:
        return Precision.Value(key)
    except ValueError:
        valid = ", ".join(
            name.removeprefix("PRECISION_").lower()
            for name in Precision.DESCRIPTOR.values_by_name
            if name != "PRECISION_UNSPECIFIED"
        )
        raise KeyError(
            f"Unknown precision: {precision!r}. Valid precisions: {valid}"
        ) from None


def runtime_proto_to_str(
    runtime: Runtime.ValueType,
    runtimes: Iterable[RuntimeInfo] | None = None,
    display_name: bool = False,
) -> str:
    """
    Convert a Runtime proto enum value to a string.

    Parameters
    ----------
    runtime
        ``Runtime`` enum value (e.g. ``RUNTIME_TFLITE``).
    runtimes
        Optional runtime registry (``platform.runtimes``) supplying display
        names. Required when *display_name* is True.
    display_name
        If True, return the human display name (e.g. ``"TensorFlow Lite"``) from
        *runtimes* instead of the lowercase token. Falls back to the token if
        the runtime has no display name in *runtimes*.

    Returns
    -------
    str
        The lowercase token (e.g. ``"tflite"``), or the display name when
        *display_name* is True.

    Raises
    ------
    KeyError
        If *runtime* is not a valid enum value.
    """
    if display_name and runtimes is not None:
        for rt in runtimes:
            if rt.runtime == runtime and rt.display_name:
                return rt.display_name

    name = Runtime.Name(runtime)
    if not name.startswith("RUNTIME_"):
        raise KeyError(f"Unknown runtime value: {runtime!r}")
    return name.removeprefix("RUNTIME_").lower()


def runtimes_str_to_proto_set(
    values: str | Runtime.ValueType | list[str | Runtime.ValueType] | None,
    runtimes: Iterable[RuntimeInfo] | None = None,
) -> set[Runtime.ValueType] | None:
    """
    Resolve a single runtime or a list of runtimes to a set of enum values.

    Parameters
    ----------
    values
        A single runtime or a list of them. See :func:`runtime_str_to_proto`
        for accepted value forms. ``None`` means no filter.
    runtimes
        Optional runtime registry (``platform.runtimes``) supplying display
        names.

    Returns
    -------
    set[Runtime.ValueType] | None
        The resolved enum values, or ``None`` when *values* is ``None``.
    """
    if values is None:
        return None
    if isinstance(values, (str, int)):
        values = [values]
    return {runtime_str_to_proto(r, runtimes) for r in values}


def precisions_str_to_proto_set(
    precisions: str | Precision.ValueType | list[str | Precision.ValueType] | None,
) -> set[Precision.ValueType] | None:
    """
    Resolve a single precision or a list of precisions to a set of enum values.

    Parameters
    ----------
    precisions
        A single precision or a list of them. See :func:`precision_str_to_proto`
        for accepted value forms. ``None`` means no filter.

    Returns
    -------
    set[Precision.ValueType] | None
        The resolved enum values, or ``None`` when *precisions* is ``None``.
    """
    if precisions is None:
        return None
    if isinstance(precisions, (str, int)):
        precisions = [precisions]
    return {precision_str_to_proto(p) for p in precisions}


def runtime_str_to_proto(
    runtime: str | Runtime.ValueType,
    runtimes: Iterable[RuntimeInfo] | None = None,
) -> Runtime.ValueType:
    """
    Convert a runtime string to its proto enum value.

    Parameters
    ----------
    runtime
        Runtime name (e.g. ``"tflite"``, ``"qnn_dlc"``, ``"RUNTIME_ONNX"``).
        Case-insensitive; the ``RUNTIME_`` prefix is optional. If *runtimes* is
        given, the human display name (e.g. ``"TensorFlow Lite"``) is also
        accepted, matched against ``RuntimeInfo.display_name`` ignoring case,
        spaces, and punctuation.
    runtimes
        Optional runtime registry (``platform.runtimes``) supplying display
        names.

    Returns
    -------
    Runtime.ValueType
        Corresponding ``Runtime`` enum value.

    Raises
    ------
    KeyError
        If *runtime* does not match any known runtime.
    """
    if not isinstance(runtime, str):
        return runtime

    key = runtime.upper()
    if not key.startswith("RUNTIME_"):
        key = f"RUNTIME_{key}"
    try:
        return Runtime.Value(key)
    except ValueError:
        pass

    # Fall back to matching the human display name from the runtime registry.
    runtimes = list(runtimes) if runtimes is not None else None
    if runtimes is not None:
        target = "".join(c for c in runtime if c.isalnum()).lower()
        for rt in runtimes:
            display = rt.display_name
            if display and "".join(c for c in display if c.isalnum()).lower() == target:
                return rt.runtime

    # When the registry is available, list valid runtimes as "Display Name (token)";
    # otherwise fall back to the bare tokens from the proto enum.
    if runtimes is not None:
        valid = ", ".join(
            f"{rt.display_name} ({runtime_proto_to_str(rt.runtime)})"
            if rt.display_name
            else runtime_proto_to_str(rt.runtime)
            for rt in runtimes
        )
    else:
        valid = ", ".join(
            name.removeprefix("RUNTIME_").lower()
            for name in Runtime.DESCRIPTOR.values_by_name
            if name != "RUNTIME_UNSPECIFIED"
        )
    raise KeyError(f"Unknown runtime: {runtime!r}. Valid runtimes: {valid}") from None


def form_factor_proto_to_str(form_factor: int) -> str:
    """Convert a FormFactor enum value to a human-readable string."""
    name = FormFactor.Name(form_factor)  # type: ignore[arg-type]
    key = name.removeprefix("FORM_FACTOR_")
    return _PLATFORM_DISPLAY_OVERRIDES.get(key, key.replace("_", " ").title())


def form_factor_str_to_proto(form_factor: str) -> FormFactor.ValueType:
    """
    Convert a form-factor display string (e.g. ``"phone"``, ``"IoT"``) to its
    proto enum value. Case- and separator-insensitive.

    Parameters
    ----------
    form_factor
        Form-factor display name.

    Returns
    -------
    FormFactor.ValueType
        Corresponding ``FormFactor`` enum value.

    Raises
    ------
    KeyError
        If *form_factor* does not match any known form factor.
    """
    target = normalize_label(form_factor)
    for ff in FormFactor.values():
        if ff != FormFactor.FORM_FACTOR_UNSPECIFIED and (
            normalize_label(form_factor_proto_to_str(ff)) == target
        ):
            return ff
    valid = ", ".join(
        form_factor_proto_to_str(ff)
        for ff in FormFactor.values()
        if ff != FormFactor.FORM_FACTOR_UNSPECIFIED
    )
    raise KeyError(f"Unknown device type: {form_factor!r}. Valid types: {valid}")


def world_proto_to_str(world: int) -> str:
    """Convert a WebsiteWorld enum value to a human-readable string."""
    name = WebsiteWorld.Name(world)  # type: ignore[arg-type]
    key = name.removeprefix("WEBSITE_WORLD_")
    return _PLATFORM_DISPLAY_OVERRIDES.get(key, key.replace("_", " ").title())


def world_str_to_proto(world: str) -> WebsiteWorld.ValueType:
    """
    Convert a world display string (e.g. ``"mobile"``, ``"auto"``) to its proto
    enum value. Case- and separator-insensitive.

    Parameters
    ----------
    world
        World (chipset type) display name.

    Returns
    -------
    WebsiteWorld.ValueType
        Corresponding ``WebsiteWorld`` enum value.

    Raises
    ------
    KeyError
        If *world* does not match any known chipset type.
    """
    target = normalize_label(world)
    for w in WebsiteWorld.values():
        if w != WebsiteWorld.WEBSITE_WORLD_UNSPECIFIED and (
            normalize_label(world_proto_to_str(w)) == target
        ):
            return w
    valid = ", ".join(
        world_proto_to_str(w)
        for w in WebsiteWorld.values()
        if w != WebsiteWorld.WEBSITE_WORLD_UNSPECIFIED
    )
    raise KeyError(f"Unknown chipset type: {world!r}. Valid types: {valid}")


def os_type_proto_to_str(os_type: int) -> str:
    """Convert an OperatingSystemType enum value to a human-readable string."""
    name = OperatingSystemType.Name(os_type)  # type: ignore[arg-type]
    key = name.removeprefix("OPERATING_SYSTEM_TYPE_")
    return {"QC_LINUX": "QC Linux"}.get(key, key.title())


def os_str_to_proto(os: str) -> OperatingSystem:
    """
    Parse an OS filter string into an ``OperatingSystem``.

    Accepts a bare OS name (e.g. ``"android"``) or a name with a version
    (e.g. ``"android 14"``). Case- and separator-insensitive on the name.

    Parameters
    ----------
    os
        OS filter string (name, optionally followed by a version).

    Returns
    -------
    OperatingSystem
        Parsed OS with its type and (optional) version.

    Raises
    ------
    KeyError
        If the OS name does not match any known operating system.
    """
    name, _, version = os.strip().partition(" ")
    target = normalize_label(name)
    for ot in OperatingSystemType.values():
        if ot != OperatingSystemType.OPERATING_SYSTEM_TYPE_UNSPECIFIED and (
            normalize_label(os_type_proto_to_str(ot)) == target
        ):
            return OperatingSystem(ostype=ot, version=version.strip())
    valid = ", ".join(
        os_type_proto_to_str(ot)
        for ot in OperatingSystemType.values()
        if ot != OperatingSystemType.OPERATING_SYSTEM_TYPE_UNSPECIFIED
    )
    raise KeyError(f"Unknown OS: {os!r}. Valid operating systems: {valid}")


def os_proto_to_str(os: OperatingSystem) -> str:
    """Format an OperatingSystem message as ``Type Version`` (e.g. ``Android 14``)."""
    if os.ostype == OperatingSystemType.OPERATING_SYSTEM_TYPE_UNSPECIFIED:
        return ""
    name = os_type_proto_to_str(os.ostype)
    return f"{name} {os.version}".strip() if os.version else name


def domain_proto_to_str(domain: int) -> str:
    """Convert a ModelDomain enum value to a human-readable string."""
    name = ModelDomain.Name(domain)  # type: ignore[arg-type]
    return (
        name.removeprefix("MODEL_DOMAIN_").replace("_", " ").title().replace("Ai", "AI")
    )


def use_case_proto_to_str(use_case: int) -> str:
    """Convert a ModelUseCase enum value to a human-readable string.

    Returns the empty string for ``MODEL_USE_CASE_UNSPECIFIED`` so callers don't
    have to guard against an "Unspecified" label leaking into output.
    """
    if use_case == ModelUseCase.MODEL_USE_CASE_UNSPECIFIED:
        return ""
    name = ModelUseCase.Name(use_case)  # type: ignore[arg-type]
    return (
        name.removeprefix("MODEL_USE_CASE_")
        .replace("_", " ")
        .title()
        .replace("Ai", "AI")
    )


def use_case_str_to_proto(use_case: str) -> ModelUseCase.ValueType:
    """
    Convert a use-case display string (e.g. ``"image classification"``) to its
    proto enum value. Case- and separator-insensitive.

    Parameters
    ----------
    use_case
        Use-case display name.

    Returns
    -------
    ModelUseCase.ValueType
        Corresponding ``ModelUseCase`` enum value.

    Raises
    ------
    KeyError
        If *use_case* does not match any known use case.
    """
    target = normalize_label(use_case)
    for u in ModelUseCase.values():
        if u != ModelUseCase.MODEL_USE_CASE_UNSPECIFIED and (
            normalize_label(use_case_proto_to_str(u)) == target
        ):
            return u
    valid = ", ".join(
        use_case_proto_to_str(u)
        for u in ModelUseCase.values()
        if u != ModelUseCase.MODEL_USE_CASE_UNSPECIFIED
    )
    raise KeyError(f"Unknown use case: {use_case!r}. Valid use cases: {valid}")


_TAG_DISPLAY_NAMES: dict[str, str] = {
    "BACKBONE": "Backbone",
    "REAL_TIME": "Real-Time",
    "FOUNDATION": "Foundation",
    "LLM": "LLM",
    "GENERATIVE_AI": "Generative AI",
    "BU_IOT": "BU IoT",
    "BU_AUTO": "BU Auto",
    "BU_COMPUTE": "BU Compute",
    "MOE": "MoE",
    "VLM": "VLM",
}


def tag_proto_to_str(tag: int) -> str:
    """Convert a ModelTag enum value to a human-readable string."""
    name = ModelTag.Name(tag)  # type: ignore[arg-type]
    key = name.removeprefix("MODEL_TAG_")
    return _TAG_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def tag_str_to_proto(tag: str) -> ModelTag.ValueType:
    """
    Convert a tag display string (e.g. ``"llm"``, ``"real time"``) to its proto
    enum value. Case- and separator-insensitive.

    Parameters
    ----------
    tag
        Tag display name.

    Returns
    -------
    ModelTag.ValueType
        Corresponding ``ModelTag`` enum value.

    Raises
    ------
    KeyError
        If *tag* does not match any known tag.
    """
    target = normalize_label(tag)
    for t in ModelTag.values():
        if t != ModelTag.MODEL_TAG_UNSPECIFIED and (
            normalize_label(tag_proto_to_str(t)) == target
        ):
            return t
    valid = ", ".join(
        tag_proto_to_str(t)
        for t in ModelTag.values()
        if t != ModelTag.MODEL_TAG_UNSPECIFIED
    )
    raise KeyError(f"Unknown tag: {tag!r}. Valid tags: {valid}")


_LICENSE_DISPLAY_NAMES: dict[str, str] = {
    "UNLICENSED": "Unlicensed",
    "COMMERCIAL": "Commercial",
    "AI_HUB_MODELS_LICENSE": "AI Hub Models License",
    "APACHE_2_0": "Apache-2.0",
    "MIT": "MIT",
    "BSD_3_CLAUSE": "BSD-3-Clause",
    "CC_BY_4_0": "CC-BY-4.0",
    "AGPL_3_0": "AGPL-3.0",
    "GPL_3_0": "GPL-3.0",
    "CREATIVEML_OPENRAIL_M": "CreativeML OpenRAIL-M",
    "CC_BY_NON_COMMERCIAL_4_0": "CC-BY-NC-4.0",
    "OTHER_NON_COMMERCIAL": "Other (Non-Commercial)",
    "LLAMA2": "Llama 2",
    "LLAMA3": "Llama 3",
    "TAIDE": "TAIDE",
    "FALCON3": "Falcon 3",
    "GEMMA": "Gemma",
    "LFM1_0": "LFM-1.0",
    "AIMET_MODEL_ZOO": "AIMET Model Zoo",
    "SAM3": "SAM3",
}


def license_proto_to_str(license_val: int) -> str:
    """Convert a ModelLicense enum value to a human-readable string."""
    name = ModelLicense.Name(license_val)  # type: ignore[arg-type]
    key = name.removeprefix("MODEL_LICENSE_")
    return _LICENSE_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
