#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# shellcheck source=/dev/null # we are statically sourcing a script.
# This can be sourced and hence does not specify an interpreter.

orig_flags=$-

set -e

# Path to the virtual environment, relative to the repository root.
ENV_PATH="qaihm-dev"

SYNC=1

PYTHON="python3.10"

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --venv=*)            ENV_PATH=${1##--venv=} ;;
    --no-sync)           SYNC=0 ;;
    --python=*)          PYTHON=${1##--python=} ;;
    *) echo "Bad opt $1." && exit 1;;
  esac
  shift
done

ACTIVATE_SUBDIR="bin"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ACTIVATE_SUBDIR="Scripts" ;;
esac

# A venv active in PowerShell leaks a ';'-separated _OLD_VIRTUAL_PATH that
# activate's `deactivate` would restore as PATH, dropping uname, git, etc.
unset _OLD_VIRTUAL_PATH _OLD_VIRTUAL_PYTHONHOME

if [ ! -d "$ENV_PATH" ]; then
  mkdir -p "$(dirname "$ENV_PATH")"

  echo "Creating virtual env $ENV_PATH."
  if command -v uv > /dev/null; then
    uv venv --seed --python "$PYTHON" "$ENV_PATH"
  else
    $PYTHON -m venv "$ENV_PATH"
  fi

  echo "Activating virtual env."
  source "$ENV_PATH/$ACTIVATE_SUBDIR/activate"
else
  source "$ENV_PATH/$ACTIVATE_SUBDIR/activate"
  echo "Env created already. Skipping creation."
fi

if [ $SYNC -eq 1 ]; then
    source scripts/util/env_sync.sh --venv="$ENV_PATH"
fi

# Unset -e so our shell doesn't close the next time something exits with
# non-zero status.
if [[ ! "${orig_flags}" =~ e ]]; then
  set +e
fi
