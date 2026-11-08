# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Generic

import ruamel.yaml
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import core_schema
from pydantic_yaml import parse_yaml_file_as, to_yaml_str
from ruamel.yaml.representer import RoundTripRepresenter
from typing_extensions import Self, TypeVar

# Size of an 'empty' YAML serialization. When a BaseQAIHMConfig's values are all defaults, it will serialize to the below string.
EMPTY_SERIALIZED_YAML_SIZE = len(b"{}\n")


def sorted_mappings(value: Any) -> Any:
    """Return ``value`` with every dict's keys sorted alphabetically (recursively).

    Used by :meth:`BaseQAIHMConfig.to_yaml` so on-disk YAML files have a stable
    key order regardless of insertion order. The actual order doesn't matter,
    only that it's deterministic.
    """
    if isinstance(value, dict):
        return {k: sorted_mappings(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list):
        return [sorted_mappings(v) for v in value]
    return value


def _yaml_writer(flow_lists: bool = False) -> ruamel.yaml.YAML:
    """A ruamel writer configured the way every QAIHM YAML file is written."""
    yaml = ruamel.yaml.YAML()

    # add_representer is a classmethod, so registering on yaml.representer
    # would reconfigure every RoundTripRepresenter in the process, not just
    # this dump. A throwaway subclass gets its own registry (ruamel copies on
    # first write), so register new representers on _Representer, not on
    # yaml.representer.
    class _Representer(RoundTripRepresenter):
        pass

    yaml.Representer = _Representer

    # build_and_test.py uses simplistic YAML readers that can't read strings on
    # multiple lines. Make sure strings aren't dumped to multiple lines.
    yaml.width = 4096

    # Allow strings with newlines to dump as newlines rather than \n
    def _yaml_repr_str(dumper: RoundTripRepresenter, data: str) -> Any:
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _Representer.add_representer(str, _yaml_repr_str)

    if flow_lists:

        def _yaml_repr_list(dumper: RoundTripRepresenter, data: list) -> Any:
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        _Representer.add_representer(list, _yaml_repr_list)

    return yaml


def dump_yaml_dict(data: Any, path: str | Path, flow_lists: bool = False) -> None:
    """Write a plain dict to YAML with every mapping's keys sorted alphabetically.

    Split out of to_yaml so callers that merge several configs into one file (the
    website's info.yaml) get byte-identical output to a single config's dump.
    """
    with open(path, "w") as f:
        _yaml_writer(flow_lists).dump(sorted_mappings(data), f)


class BaseQAIHMConfig(BaseModel):
    """
    A base class for all AI Hub Models configs.
    Config fields are defined as typed dataclass fields.

    This class is capable of loading a YAML file (via .from_yaml()) or
    an arbirary python dict (via .from_dict()) into an instance of itself.

    The class instance is also capable of dumping itself to a dictionary
    or to yaml (via .to_yaml() or .to_dict()).
    """

    # Default behavior should be to forbid unknown keys in parsed YAML / JSON files.
    model_config = ConfigDict(extra="forbid")

    def __str__(self) -> str:
        return to_yaml_str(
            self,
            exclude_defaults=True,
            exclude_none=True,
        )

    def to_yaml_dict(self, exclude_defaults: bool = True, **kwargs: Any) -> Any:
        """This config as a plain dict, exactly as to_yaml would write it.

        Rendered via pydantic_yaml first (which handles JSON-mode serializers and
        custom types) and re-parsed, so the result contains only YAML primitives.
        """
        rendered = to_yaml_str(
            self,
            custom_yaml_writer=_yaml_writer(),
            exclude_defaults=exclude_defaults,
            exclude_none=True,
            **kwargs,
        )
        return ruamel.yaml.YAML(typ="safe", pure=True).load(rendered)

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = False,
        delete_if_empty: bool = True,
        flow_lists: bool = False,
        exclude_defaults: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Converts this class to a dict and saves that dict to a YAML file.

        Every dict in the output (including the top-level mapping) has its keys
        sorted alphabetically. The actual order is irrelevant; the goal is just
        that regenerated YAML files have stable diffs regardless of which order
        the scorecard / codegen happened to populate fields in.

        Parameters
        ----------
        path
            Path to save the file.
        write_if_empty
            If False, the YAML file will not be written to disk if the dictionary to be saved is empty.
        delete_if_empty
            If True, an existing YAML file at the given path will be deleted if the dictionary to be saved is empty.
        flow_lists
            If True, lists will be formatted in flow style (e.g., [1, 2, 3]) instead of block style.
        exclude_defaults
            If True, fields set to their default values will be excluded from the YAML output.
        **kwargs
            Additional args (used by overrides).

        Returns
        -------
        written : bool
            True if the file was written, False if it was empty and not written/deleted.

        Notes
        -----
        Generally, the dictionary to be saved to YAML is empty only if:
         * all dataclass fields have default values
         * every field in this dataclass instance is set to its default value
        """
        data = self.to_yaml_dict(exclude_defaults=exclude_defaults, **kwargs)
        dump_yaml_dict(data, path, flow_lists=flow_lists)

        # Remove file if empty
        if (not write_if_empty or delete_if_empty) and os.path.getsize(path) in (
            0,
            EMPTY_SERIALIZED_YAML_SIZE,
        ):
            os.remove(path)
            return False

        return True

    def to_json(
        self,
        path: str | os.PathLike,
        exclude_defaults: bool = True,
        exclude_none: bool = True,
        indent: int | None = 4,
    ) -> None:
        """
        Converts this class to a dict and saves that dict to a JSON file.

        Parameters
        ----------
        path
            Path to save the file.
        exclude_defaults
            Whether to exclude fields with default values.
        exclude_none
            Whether to exclude fields with None values.
        indent
            Number of spaces for indentation. None produces compact (single-line) output. Defaults to 4.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                self.model_dump_json(
                    exclude_defaults=exclude_defaults,
                    exclude_none=exclude_none,
                    indent=indent,
                )
            )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> Self:
        """Reads the JSON file at the given path and loads it into an instance of this class."""
        with open(path) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        create_empty_if_no_file: bool = False,
    ) -> Self:
        """Reads the yaml file at the given path and loads it into an instance of this class."""
        if create_empty_if_no_file and (
            not os.path.exists(path) or os.path.getsize(path) == 0
        ):
            return cls()
        return parse_yaml_file_as(cls, path)


BaseQAIHMConfigTypeVar = TypeVar("BaseQAIHMConfigTypeVar", bound=BaseQAIHMConfig)


EnumT = TypeVar("EnumT", bound=Enum)


class EnumListWithParseableAll(list[EnumT], Generic[EnumT]):
    """
    Helper list class that can parse an enum list to / from "all".
    If "all" is in the list, then all enum elements are returned.
    """

    # Subclasses should set this to the EnumT class.
    EnumType: type[Enum]
    ALL: list[EnumT] | None = None

    @classmethod
    def default(
        cls,
    ) -> Self:
        if cls.ALL is not None:
            return cls(cls.ALL)
        return cls(list(cls.EnumType))

    @classmethod
    def parse(cls, obj: Any) -> Self:
        if isinstance(obj, list):
            out: Self = cls()
            for x in obj:
                if x == "all":
                    out = cls.default()
                else:
                    out.append(cls.EnumType(x))  # type: ignore[arg-type]
            return out
        raise ValueError(f"Unsupported type {type(obj)} for parsing to {cls}")

    @classmethod
    def serialize(cls, enum_list: list[EnumT]) -> list[str]:
        if len(set(enum_list)) == len(cls.EnumType):
            return ["all"]
        return [x.value for x in enum_list]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            lambda obj, _: cls.parse(obj),
            handler(Any),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize, when_used="json"
            ),
        )


EnumListWithParseableAllTypeVar = TypeVar(
    "EnumListWithParseableAllTypeVar", bound=EnumListWithParseableAll
)
