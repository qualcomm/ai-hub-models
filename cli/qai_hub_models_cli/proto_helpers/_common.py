# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import io
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, TypeVar

import yaml
from google.protobuf import json_format
from google.protobuf.message import Message
from packaging.version import Version

from qai_hub_models_cli._internal.utils import use_internal_releases
from qai_hub_models_cli.common import CACHE_DIR, is_heavy_package_installed
from qai_hub_models_cli.envvars import FORCE_MANIFEST_ROOT_ENVVAR
from qai_hub_models_cli.utils import download
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    verify_not_dev_release,
    verify_version_supported,
)

try:
    from yaml import CSafeLoader as _BaseLoader
except ImportError:
    from yaml import SafeLoader as _BaseLoader  # type: ignore[assignment]

_M = TypeVar("_M", bound=Message)


def get_release_cache_dir(version: Version, internal: bool | None = None) -> Path:
    if internal is None:
        internal = use_internal_releases()
    if path := os.environ.get(FORCE_MANIFEST_ROOT_ENVVAR):
        return Path(path)
    return CACHE_DIR / ("internal_releases" if internal else "releases") / f"v{version}"


def read_proto(path: Path, proto_type: type[_M]) -> _M:
    """Parse a protobuf file from a local path."""
    msg = proto_type()
    msg.ParseFromString(path.read_bytes())
    return msg


def write_proto(path: Path, msg: Message) -> None:
    """Serialize a protobuf message to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msg.SerializeToString())


def _sorted_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sorted_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_keys(v) for v in value]
    return value


def proto_to_dict(msg: Message) -> dict[str, Any]:
    """Convert a protobuf message to a plain dict using proto field names, with sorted keys."""
    data = json_format.MessageToDict(msg, preserving_proto_field_name=True)
    return _sorted_keys(data)


def proto_from_dict(
    data: dict[str, Any] | None,
    proto_type: type[_M],
    ignore_unknown_fields: bool = False,
) -> _M:
    """Build a protobuf message from a dict produced by ``proto_to_dict``."""
    msg = proto_type()
    json_format.ParseDict(data or {}, msg, ignore_unknown_fields=ignore_unknown_fields)
    return msg


def read_proto_json(path: str | os.PathLike, proto_type: type[_M]) -> _M:
    """Read a JSON file written by ``write_proto_json`` into a protobuf message."""
    with open(path) as f:
        content = f.read()
    return json_format.Parse(content, proto_type())


def write_proto_json(msg: Message, path: str | os.PathLike) -> None:
    """Write a protobuf message to a JSON file (proto field names, sorted keys)."""
    json = json_format.MessageToJson(msg, preserving_proto_field_name=True)
    with open(path, "w") as f:
        f.write(json)


class _YamlLoader(_BaseLoader):
    """Safe loader that reads only true/false as booleans, so bare yes/no/on/off stay strings."""


_YamlLoader.yaml_implicit_resolvers = {
    first_char: [r for r in resolvers if r[0] != "tag:yaml.org,2002:bool"]
    for first_char, resolvers in _BaseLoader.yaml_implicit_resolvers.items()
}
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def safe_load_yaml(stream: str | IO[str]) -> Any:
    """Parse YAML from a string or text stream into plain Python objects."""
    return yaml.load(stream, Loader=_YamlLoader)


def read_proto_yaml(path: str | os.PathLike, proto_type: type[_M]) -> _M:
    """Read a YAML file written by ``write_proto_yaml`` into a protobuf message."""
    with open(path) as f:
        data = safe_load_yaml(f)
    return proto_from_dict(data, proto_type)


class _Dumper(yaml.SafeDumper):
    def choose_scalar_style(self) -> str | None:
        style: str | None = super().choose_scalar_style()
        # Double-quote strings containing an apostrophe rather than doubling it.
        if style == "'" and "'" in self.analysis.scalar:
            return '"'
        return style


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", data, style="|" if "\n" in data else None
    )


def _represent_flow_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_Dumper.add_representer(str, _represent_str)


# add_representer copies the registry per class, so this does not affect _Dumper.
class _FlowListDumper(_Dumper):
    pass


_FlowListDumper.add_representer(list, _represent_flow_list)


def dump_yaml(data: Any, stream: IO[str], flow_lists: bool = False) -> None:
    """Write ``data`` to ``stream`` the way every QAIHM YAML file is written."""
    yaml.dump(
        data,
        stream,
        Dumper=_FlowListDumper if flow_lists else _Dumper,
        default_flow_style=False,
        sort_keys=False,
        # Wrapping long values (e.g. S3 keys) leaves trailing whitespace.
        width=4096,
        allow_unicode=True,
    )


def dump_yaml_str(data: Any, flow_lists: bool = False) -> str:
    """Like ``dump_yaml``, returning the YAML as a string."""
    stream = io.StringIO()
    dump_yaml(data, stream, flow_lists)
    return stream.getvalue()


def write_proto_yaml(msg: Message, path: str | os.PathLike) -> None:
    """Write a protobuf message to a YAML file (proto field names, sorted keys)."""
    proto_dict = proto_to_dict(msg)
    with open(path, "w") as f:
        dump_yaml(proto_dict, f)


def fetch_proto(url: str, cache_path: Path, proto_type: type[_M]) -> _M:
    """Download a protobuf file (from URL or S3 key) and cache it on disk."""
    if not cache_path.exists():
        download(url, cache_path, quiet=True)
    return read_proto(cache_path, proto_type)


def model_cache_path(
    version: Version, model_id: str, filename: str, internal: bool | None = None
) -> Path:
    if internal is None:
        internal = use_internal_releases()
    return get_release_cache_dir(version, internal) / "models" / model_id / filename


def use_aihm_source(version: Version) -> bool:
    """Check if the qai_hub_models source package should be used as the source of truth, rather than prebuilt manifest files stored on S3."""
    return (
        version == CURRENT_VERSION
        and CURRENT_VERSION.is_devrelease
        and is_heavy_package_installed()
    )


def fetch_release_proto(
    version: Version,
    proto_type: type[_M],
    cache_filename: str,
    source_getter: str,
    url: str | None = None,
    local_path: Path | None = None,
) -> _M:
    """
    Shared fetch logic for release-level proto helpers (manifest, platform).

    Parameters
    ----------
    version
        AI Hub Models release version.
    proto_type
        The protobuf message class to parse into.
    cache_filename
        Filename for the local cache (e.g. "manifest.pb").
    source_getter
        Function name on qai_hub_models.core.configs module (e.g. "get_release_manifest").
    url
        URL to fetch from when not using source. Required unless local_path is set.
    local_path
        If provided, read directly from this path instead of fetching.

    Returns
    -------
    _M
        Parsed protobuf message.
    """
    if local_path is not None:
        return read_proto(local_path, proto_type)

    if not use_aihm_source(version):
        verify_not_dev_release(version)
        verify_version_supported(version, verify_manifest_supported=True)
    else:
        from qai_hub_models.core import configs as models_configs

        getter: Callable[[], _M] = getattr(models_configs, source_getter)
        return getter()

    assert url is not None
    cache_path = get_release_cache_dir(version) / cache_filename
    return fetch_proto(url, cache_path, proto_type)


def fetch_model_proto(
    model: str,
    version: Version,
    proto_type: type[_M],
    cache_filename: str,
    manifest_url_field: str,
    source_getter: str,
    local_path: Path | None = None,
) -> _M:
    """
    Shared fetch logic for per-model proto helpers (info, perf, numerics).

    Parameters
    ----------
    model
        Model ID or display name.
    version
        AI Hub Models release version.
    proto_type
        The protobuf message class to parse into.
    cache_filename
        Filename for the local cache (e.g. "info.pb").
    manifest_url_field
        Attribute name on ReleaseManifestModelEntry.manifest_urls (e.g. "manifest").
    source_getter
        Function name on qai_hub_models.core.configs module (e.g. "get_model_manifest").
    local_path
        If provided, read directly from this path instead of fetching.

    Returns
    -------
    _M
        Parsed protobuf message, or an empty instance if unavailable.
    """
    if local_path is not None:
        return read_proto(local_path, proto_type)

    if not use_aihm_source(version):
        verify_not_dev_release(version)
        verify_version_supported(version, verify_manifest_supported=True)

    # Avoids circular import.
    from qai_hub_models_cli.proto_helpers.release_manifest import (
        get_release_manifest_entry,
        resolve_model_id,
    )

    if use_aihm_source(version):
        from qai_hub_models.core import configs as models_configs

        getter: Callable[[str], _M] = getattr(models_configs, source_getter)
        proto = getter(resolve_model_id(model, version))
        if proto is None:
            return proto_type()
        return proto

    entry = get_release_manifest_entry(model, version)
    url = getattr(entry.manifest_urls, manifest_url_field)
    if not url:
        return proto_type()
    cache_path = model_cache_path(version, entry.id, cache_filename)
    return fetch_proto(url, cache_path, proto_type)
