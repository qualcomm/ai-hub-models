#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Common utilities

# shellcheck disable=SC2034 # various definitions appear unused in this included source.

REPO_ROOT=$(git rev-parse --show-toplevel)

COLOR_GREEN='\033[0;32m'
COLOR_GREY='\033[0;37m'
COLOR_RED='\033[0;31m'
COLOR_RED_BOLD='\033[0;1;31m'
COLOR_RED_REVERSED_VIDEO='\033[0;7;31m'
COLOR_YELLOW='\033[0;33m'
COLOR_YELLOW_BOLD='\033[0;1;33m'
COLOR_OFF='\033[0m'

FORMAT_BOLD='\033[0;1m'
FORMAT_UNDERLINED='\033[0;4m'
FORMAT_BLINKING='\033[0;5m'
FORMAT_REVERSE_VIDEO='\033[0;7m'

#
# Emit a message to stderr.
#
function log_err() {
    echo -e "${COLOR_RED_REVERSED_VIDEO}$*${COLOR_OFF}" 1>&2
}

#
# Emit a message to stderr.
#
function log_warn() {
    echo -e "${COLOR_YELLOW_BOLD}$*${COLOR_OFF}" 1>&2
}

#
# Emit a message to stderr.
#
function log_info() {
    echo -e "${FORMAT_BOLD}$*${COLOR_OFF}" 1>&2
}

#
# Emit a message to stderr.
#
function log_debug() {
    echo -e "${COLOR_GREY}$*${COLOR_OFF}" 1>&2
}

#
# Emit a log message and exit with non-zero return.
#
function die() {
    log_err "$*"
    exit 1
}

#
# Enable trace logging via set -x
# Args
#  1: [Default: $QAIHM_BUILD_XTRACE] If set to non-empty, enable tracing
#
# shellcheck disable=SC2120
function set_xtrace() {
    local enable="${1:-${QAIHM_BUILD_XTRACE:-}}"
    if [ -n "${enable}" ]; then
        set -x
    fi
}

#
# Enable bash strict mode and conditionally set -x.
# @see http://redsymbol.net/articles/unofficial-bash-strict-mode/
# @see set_xtrace
#
function set_strict_mode() {
    set -euo pipefail
    set_xtrace
}

function pretty_print_arr() {
  arr=("$@")

  echo "[$(echo "${arr[@]}" | tr ' ' ',')]"
}

function run_as_root()
{
    # Don't use sudo if user is root already (e.g., in docker)
    if [ "${EUID}" -eq 0 ]; then
        log_debug "We're already root; running ${*} without sudo."
        "${@}"
    else
        log_debug "We're ${EUID}; running ${*} via sudo."
        sudo "${@}"
    fi
}
