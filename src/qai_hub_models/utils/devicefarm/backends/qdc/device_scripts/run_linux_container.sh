#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Generic "load an image, run one command in it on the NPU" job for Linux IoT
# devices (e.g. Dragonwing IQ-9075 EVK) through QDC. Model-agnostic: everything
# model-specific arrives through the placeholders below, filled in by
# utils/devicefarm/device_run.py.
#
#   {DEVICE_LOGS_DIR}  directory name QDC harvests logs from
#   {IMAGE_TAR}        docker image tarball name under TestContent
#   {IMAGE_TAG}        tag the tarball provides
#   {PYTHON_ARGS}      arguments for the image's `python`, already shell-quoted
#
# The bundle root is mounted at /work and the log directory at /out.

set +e

TC=/data/local/tmp/TestContent
LOG=/data/local/tmp/{DEVICE_LOGS_DIR}
mkdir -p "$LOG"
# File-only redirect: a tee process substitution can stall QDC's SSH channel.
exec > "$LOG/script.log" 2>&1

date -u
uname -a
nproc
free -m
df -h /data/local/tmp

docker load -i "$TC/{IMAGE_TAR}" || { echo "FATAL: docker load failed"; exit 1; }

# Node names differ across SoCs (secure node, second CDSP), so pass whichever exist.
dev_args=()
for d in /dev/fastrpc-* /dev/dma_heap/system; do
    [ -e "$d" ] && dev_args+=(--device "$d")
done
# QC Linux keeps the libs in /usr/lib, Ubuntu (e.g. VENTUNO Q) in the multiarch dir.
find_lib() {
    for dir in /usr/lib /usr/lib/aarch64-linux-gnu; do
        [ -e "$dir/$1" ] && { readlink -f "$dir/$1"; return; }
    done
}
rpc_lib=$(find_lib libcdsprpc.so.1)
heap_lib=$(find_lib libdmabufheap.so.0)
echo "devices: ${dev_args[*]}"
echo "libs: rpc=$rpc_lib heap=$heap_lib"
[ -n "$rpc_lib" ] || { echo "FATAL: libcdsprpc not found"; exit 1; }

# The container has no FastRPC client of its own: bind-mount the device's
# libcdsprpc and the libdmabufheap it links against (where present). Leave
# ADSP_LIBRARY_PATH unset so the QNN package's own Hexagon skel libraries are used.
lib_args=(
    -v "$rpc_lib":/usr/lib/aarch64-linux-gnu/libcdsprpc.so.1:ro
    -v "$rpc_lib":/usr/lib/aarch64-linux-gnu/libcdsprpc.so:ro
)
[ -n "$heap_lib" ] && lib_args+=(-v "$heap_lib":/usr/lib/aarch64-linux-gnu/libdmabufheap.so.0:ro)
docker run --rm \
    "${dev_args[@]}" \
    "${lib_args[@]}" \
    -v "$TC":/work \
    -v "$LOG":/out \
    "{IMAGE_TAG}" python {PYTHON_ARGS}
run_rc=$?
echo "docker run rc=$run_rc"

docker rmi "{IMAGE_TAG}" > /dev/null 2>&1

# QDC marks the job Unsuccessful on a non-zero exit of this bash script (unlike the
# Windows entry script, where only a thrown error counts), so pass the code through.
exit "$run_rc"
