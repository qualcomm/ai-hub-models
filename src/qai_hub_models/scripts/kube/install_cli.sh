#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

if ! command -v jq &>/dev/null; then
  run_sudo apt-get update -qq
  run_sudo apt-get install -y -qq jq
fi

if ! command -v argo &>/dev/null; then
  curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 30 \
    "https://github.com/argoproj/argo-workflows/releases/download/v3.5.5/argo-linux-amd64.gz" -o /tmp/argo.gz
  gunzip -f /tmp/argo.gz
  run_sudo install /tmp/argo /usr/local/bin/argo
  rm /tmp/argo
fi

if ! command -v kubectl &>/dev/null; then
  curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 30 \
    "https://dl.k8s.io/release/v1.29.0/bin/linux/amd64/kubectl" -o /tmp/kubectl
  run_sudo install /tmp/kubectl /usr/local/bin/kubectl
  rm /tmp/kubectl
fi
