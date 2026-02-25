# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import _io
import os
import struct
from collections.abc import (
    Hashable,
    Iterable,
)

import pandas as pd
from pandas.core.series import Series

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

# uc_node_map = {}

table_header = [
    "code",
    "name",
    "general_category",
    "ccc",
    "bd_cat",
    "comp_mapping",
    "decimal_digit_value",
    "digit_value",
    "numeric_value",
    "mirrored",
    "misc1",
    "misc2",
    "uppercase",
    "lowercase",
    "titlecase",
]

UNICODE_DATA_ASSET = CachedWebModelAsset(
    url="https://www.unicode.org/Public/3.0-Update/UnicodeData-3.0.0.txt",
    model_id="melotts_shared",
    model_asset_version=1,
    filename="UnicodeData-3.0.0.txt",
)


def read_unicode_input(input_path: str) -> pd.DataFrame:
    """Read unicode data."""
    return pd.read_csv(input_path, delimiter=";", names=table_header)
    # add CJK unicode values to the table


# Struct format
#   int     unicode_value
#   char    general_category[2]
#   int     num_children
#   int     decomp_offset


def create_unicode_node(uc: int, gc: str, mem_idx: int = 0) -> dict:
    """
    Create a unicode node containing the unicode value and its properties.

    Parameters
    ----------
    uc
        unicode value (converted from hex)
    gc
        unicode category code
    mem_idx
        flat memory index in the binary

    Returns
    -------
    dict
        node dictionary
    """
    return {
        "unicode_value": uc,
        "general_category": gc,
        "flat_mem_idx": mem_idx,
        "decomp_offset": 0,
        "num_children": 0,
        "children": [],
    }


def insert_node_to_dag(current: dict, uc: int, gc: str, debug: bool = False) -> dict:
    """
    Create a node and insert it into the parent node.

    Parameters
    ----------
    current
        parent node
    uc
        unicode value of the new node
    gc
        category code
    debug
        flag to enable/disable debug logs

    Returns
    -------
    dict
        new child node
    """
    # check if node is already added
    for child_node in current["children"]:
        if child_node["unicode_value"] == uc:
            return child_node
    # cur_node = uc_node_map.get(uc, None)
    # if cur_node is None:
    cur_node = create_unicode_node(uc, gc)
    # uc_node_map[uc] = cur_node
    current["children"].append(cur_node)
    current["num_children"] += 1
    # update uc_node_map
    # uc_node_map[current['unicode_value']] = current

    if debug is True:
        print(cur_node)
    return cur_node


def depth_wise_node_insertion(
    node: dict,
    uc_rows: Iterable[tuple[Hashable, Series]],
    full_uc_data: pd.DataFrame,
    debug: bool = False,
) -> None:
    """
    Recursively insert nodes to the DAG.

    Parameters
    ----------
    node
        current root node
    uc_rows
        sliced rows from the dataframe with the child unicode values
    full_uc_data
        complete unicode dataframe
    debug
        enable/disable debug logs
    """
    for _, row in uc_rows:
        if debug:
            print(f"row['code'], {row['code']}")
            print(f"row['general_category'], {row['general_category']}")
        new_node = insert_node_to_dag(
            node, row["code"], row["general_category"], debug=False
        )
        if str(row["comp_mapping"]) == "nan" or str(row["comp_mapping"]).startswith(
            "<"
        ):
            continue
        decomp_codes = [int(x, 16) for x in str(row["comp_mapping"]).split(" ")]
        filtered_rows = full_uc_data[full_uc_data["code"].isin(decomp_codes)]
        if debug:
            print(f"Filtered rows {filtered_rows}")
        depth_wise_node_insertion(
            new_node, filtered_rows.iterrows(), full_uc_data, debug=debug
        )


def generate_unicode_dag(uc_data: pd.DataFrame) -> dict:
    """Generate the unicode DAG."""
    flat_mem_idx = 0
    root = create_unicode_node(-1, "", flat_mem_idx)
    # uc_node_map[-1] = root

    depth_wise_node_insertion(root, uc_data.iterrows(), uc_data, debug=False)
    return root


def assign_mem_idx(root: dict) -> None:
    """Assign flattened memory index for each node."""
    node_stack = [root]
    mem_idx = 0
    while len(node_stack) > 0:
        curr_node = node_stack.pop(0)
        curr_node["flat_mem_idx"] = mem_idx
        mem_idx += 1
        node_stack.extend(curr_node["children"])


def dump_node(
    node: dict, flat_indices_array: list, debug: bool = False
) -> tuple[bytes, bytes, int, int]:
    """
    Pack a node from the DAG into a byte object.

    Parameters
    ----------
    node
        current node to be packed
    flat_indices_array
        list to store flat indices
    debug
        enable/disable debug logs

    Returns
    -------
    packed_fixed : bytes
        Bytes object containing fixed component of the DAG node
        (unicode value, category, num_children, mem offset to the child nodes)
    packed_variable : bytes
        Bytes object containing variable component of the DAG node
        (mem indices of the child nodes)
    size_fixed : int
        Size of the fixed component
    size_variable : int
        Size of the variable component
    """
    fixed_format = "i 3s I I"
    variable_format = "{}I".format(node["num_children"])
    size_variable = struct.calcsize(variable_format)
    size_fixed = struct.calcsize(fixed_format)

    children = []
    for child in node["children"]:
        children.append(child["flat_mem_idx"])  # noqa: PERF401
    flat_indices_array.extend(children)

    if debug is True:
        print(f"Children: {children}")
        print(
            node["unicode_value"],
            node["general_category"].encode(),
            node["num_children"],
            node["decomp_offset"],
        )

    packed_fixed = struct.pack(
        fixed_format,
        node["unicode_value"],
        node["general_category"].encode(),
        node["num_children"],
        node["decomp_offset"],
    )
    packed_variable = struct.pack(variable_format, *children)

    return packed_fixed, packed_variable, size_fixed, size_variable


def dump_bin(root: dict, bin_file: _io.BufferedWriter) -> int:
    """
    Dump the DAG into a bin file.

    Parameters
    ----------
    root
        root node of the DAG
    bin_file
        file handle to the output bin file

    Returns
    -------
    int
        offset to the start of the variable data section
    """
    num_entries = 0
    size_data_fixed = 0
    size_data_variable = 0
    prev_mem_idx = 0
    packed_data_fixed = b""
    packed_data_variable = b""

    curr_node = root
    node_stack = [root]
    flat_indices_array: list[int] = []
    while len(node_stack) > 0:
        prev_node = curr_node
        curr_node = node_stack.pop(0)
        mem_idx = curr_node["flat_mem_idx"]
        if curr_node != prev_node:
            if mem_idx != prev_mem_idx + 1:
                print(prev_mem_idx, mem_idx)
                raise RuntimeError("Error in assigning mem_idx")
            curr_node["decomp_offset"] = prev_node["decomp_offset"] + (
                prev_node["num_children"] * 4
            )
        else:
            curr_node["decomp_offset"] = 0

        prev_mem_idx = mem_idx
        # pack into struct
        packed_fixed, packed_variable, size_fixed, size_variable = dump_node(
            curr_node, flat_indices_array, debug=False
        )
        packed_data_fixed += packed_fixed
        packed_data_variable += packed_variable
        size_data_fixed += size_fixed
        size_data_variable += size_variable

        num_entries += 1
        for child in curr_node["children"]:
            node_stack.append(child)  # noqa: PERF402

    # Write the binary data to a file
    print("\n  Dumping to bin file...")
    bin_file.write(struct.pack("I", num_entries))
    bin_file.write(struct.pack("I", size_data_fixed))
    bin_file.write(struct.pack("I", size_data_variable))
    bin_file.write(packed_data_fixed)
    bin_file.write(packed_data_variable)
    return 12 + size_data_fixed


def verify_offsets(
    root: dict, base_offset: int, bin_file: _io.BufferedReader, debug: bool = False
) -> None:
    """
    Verify the memory offsets for each node in the bin file.

    Parameters
    ----------
    root
        root node of the DAG
    base_offset
        offset to the start of the mem indices array
    bin_file
        file handle to the output bin file
    debug
        enable/disable debug logs
    """
    node_stack = [root]
    integer_size = struct.calcsize("i")
    while len(node_stack) > 0:
        curr_node = node_stack.pop(0)
        # Verify the data stored at curr_node['offset'] with what is stored at curr_node
        num_children = curr_node["num_children"]
        offset = curr_node["decomp_offset"]
        bin_file.seek(base_offset + offset)
        child_ids = []
        for _ in range(num_children):
            data = bin_file.read(integer_size)
            if debug:
                print(struct.unpack("i", data)[0])
            if not data:
                print("Unable to read data from file")
                break
            child_ids.append(struct.unpack("i", data)[0])

        if child_ids != [x["flat_mem_idx"] for x in curr_node["children"]]:
            raise RuntimeError(
                "Child ID mismatch at:{}, in-node:{}".format(
                    curr_node["flat_mem_idx"], curr_node["unicode_value"]
                )
            )

        children_node = []
        for child in curr_node["children"]:
            node_stack.append(child)
            children_node.append(child["flat_mem_idx"])


def generate_unicode_binary(bin_path: str | os.PathLike) -> None:
    """Generate a unicode binary file from unicode data."""
    unicode_data = read_unicode_input(UNICODE_DATA_ASSET.fetch())
    # convert unicode hex to int
    unicode_data["code"] = unicode_data["code"].apply(lambda x: int(x, 16))

    root = generate_unicode_dag(unicode_data)
    assign_mem_idx(root)

    with open(bin_path, "wb") as bin_file:
        base_offset = dump_bin(root, bin_file)

    # # verify offsets
    with open(bin_path, "rb") as bin_file_:
        verify_offsets(root, base_offset, bin_file_)
        print("\n Verified all offsets to variable data")
