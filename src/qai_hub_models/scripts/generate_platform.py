# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import sys

from qai_hub_models.utils._platform_helpers import (
    generate_platform_info,
    save_platform,
)
from qai_hub_models.utils.envvars import IsOnCIEnvvar

if __name__ == "__main__":
    """
    Re-generate and save the platform.yaml platform registry from the current
    set of devices / runtimes that are valid in AI Hub Models perf.yaml files.
    """
    if not IsOnCIEnvvar.get():
        print(
            "This script must be run on PROD WORKBENCH with a PUBLIC USER (no access to un-published devices). We recommend using qaihm_bot@qti.qualcomm.com as the user."
        )
        if input("Are you set up accordingly? (y/n) --> ").lower() not in ["y", "yes"]:
            print("exiting")
            sys.exit(0)

    save_platform(generate_platform_info())
