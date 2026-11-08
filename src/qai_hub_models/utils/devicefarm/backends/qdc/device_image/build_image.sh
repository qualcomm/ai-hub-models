#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Build the linux/arm64 runtime image for run_linux_container.sh on any host
# (x86_64 builds go through qemu) and save it as a tarball for a device bundle.
#
# Usage: build_image.sh <image-tag> <output.tar> [requirements-file]
#
# Without a requirements file the image only gets the base packages; the checked-in
# requirements.txt is empty so a plain `docker buildx build` of this folder also works.
#
# Needs docker with buildx and, on x86_64, arm64 binfmt registered:
#   docker run --privileged --rm tonistiigi/binfmt --install arm64
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <image-tag> <output.tar> [requirements-file]" >&2
    exit 2
fi

TAG="$1"
OUT="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The build context is a temp dir holding the Dockerfile and the requirements file.
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
cp "${3:-$HERE/requirements.txt}" "$CTX/requirements.txt"

if [ "$(uname -m)" != "aarch64" ] && [ ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
    echo "arm64 binfmt is not registered; run: docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
    exit 1
fi

docker buildx build --platform linux/arm64 --load -t "$TAG" "$CTX"
docker save -o "$OUT" "$TAG"
ls -lh "$OUT"
