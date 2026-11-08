# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

import argparse
import sys


def main(args: argparse.Namespace | None = None) -> None:
    args_str = f" {' '.join(sys.argv[1:])}" if len(sys.argv) > 1 else ""
    print(
        "\n-------------------------------------\n\n"
        f"Please use `qai-hub-models export parakeet_tdt{args_str}`\n\n"
        "Use of `python -m` to invoke export is no longer supported.\n\n"
        "-------------------------------------\n\n"
        "This message will be removed in a future release."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
