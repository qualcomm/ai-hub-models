#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

set +e
umask 022

LOG=/data/local/tmp/{DEVICE_LOGS_DIR}
OUT=$LOG/results
MM_CACHE=/data/local/tmp/geniex-cache
TC=/data/local/tmp/TestContent

# Drop per-job state on exit (dedicated-pool devices are reused across jobs;
# leftover extracted bundles / caches / matrix TSVs would leak into the next
# tenant's run).
cleanup_device() {
    rm -rf "$TC"/geniex-bench-linux-arm64-* 2>/dev/null || true
    rm -f "$TC"/geniex-bench.tar.gz 2>/dev/null || true
    rm -rf "$TC"/qairt "$TC"/qairt.zip 2>/dev/null || true
    rm -rf "$MM_CACHE" 2>/dev/null || true
    rm -f /data/local/tmp/matrix-*.tsv 2>/dev/null || true
}
trap cleanup_device EXIT
# A job QDC killed mid-run skips the EXIT trap; drop its bundle up front so the
# glob below can't pick a stale geniex-bench version.
rm -rf "$TC"/geniex-bench-linux-arm64-* 2>/dev/null || true

# QDC reuses device cells across jobs; wipe stale cell JSONs from a prior
# run so compute_metrics doesn't ingest another model/plugin's results.
rm -rf "$OUT"
mkdir -p "$LOG" "$OUT" "$MM_CACHE"
# Keep stdout off the SSH channel: `exec > >(tee ...) 2>&1` lets the process
# substitution stall the channel until its 4 KiB stdout buffer fills, then
# QDC's pyshell-bash-ssh runner trips sshd's ClientAliveInterval=15/CountMax=4
# (~73s) and tears the session down mid-cell. File-only redirect avoids it.
exec > "$LOG/script.log" 2>&1
date -u
uname -a

TGZ=$TC/geniex-bench.tar.gz
URL="{LINUX_BENCH_URL}"
curl -fSL --retry 3 --retry-delay 5 -o "$TGZ" "$URL"
tar -xzf "$TGZ" -C "$TC"
rm -f "$TGZ"

BUNDLE=$(ls -d "$TC"/geniex-bench-linux-arm64-* 2>/dev/null | head -n1)
[ -n "$BUNDLE" ] || { echo "FATAL: extracted bundle dir missing under $TC"; exit 1; }

[ -x "$BUNDLE/bin/geniex-bench" ] || chmod +x "$BUNDLE/bin/geniex-bench" 2>/dev/null
[ -x "$BUNDLE/bin/geniex-bench" ] || { echo "FATAL: $BUNDLE/bin/geniex-bench missing"; exit 1; }

cd "$BUNDLE"

# Run the qairt plugin against QAIRT_VERSION's SDK via --qairt-lib; empty
# QAIRT_VERSION keeps the QAIRT bundled with geniex-bench.
PLUGIN="{PLUGIN}"
QAIRT_VERSION="{QAIRT_VERSION}"
QAIRT_LIB_ARGS=()
if [ "$PLUGIN" = "qairt" ] && [ -n "$QAIRT_VERSION" ]; then
  echo "=== Using QAIRT $QAIRT_VERSION ==="
  QZIP=$TC/qairt.zip
  # The SDK zip's top-level entry is qairt/<version>/; extract into its parent.
  QROOT="$TC/qairt"
  QURL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$QAIRT_VERSION/v$QAIRT_VERSION.zip"
  curl -fSL --max-time 3600 --retry 3 --retry-delay 5 -o "$QZIP" "$QURL" || { echo "FATAL: QAIRT SDK download failed: $QURL"; exit 1; }
  rm -rf "$QROOT"
  # unzip can exit nonzero on a successful extract; verify the dir instead.
  unzip -q "$QZIP" -d "$TC" || true
  rm -f "$QZIP"
  SDK="$QROOT/$QAIRT_VERSION"
  [ -d "$SDK/lib" ] || { echo "FATAL: QAIRT SDK not extracted to $SDK"; exit 1; }
  QAIRT_LIB_ARGS=(--qairt-lib "$SDK")
  # GenieX joins skel dirs with ':' but FastRPC splits on ';' (IQ9 crash); expose one dir.
  ARCH=$(grep -rhom1 '"dsp_arch": *"v[0-9]*"' "$TC"/qairt_bundles 2>/dev/null | grep -o 'v[0-9]*' | head -n1)
  if [ -d "$SDK/lib/hexagon-$ARCH" ]; then
    SEL="$QROOT/sel"
    mkdir -p "$SEL/lib"
    ln -s "$SDK"/lib/aarch64-* "$SDK/lib/hexagon-$ARCH" "$SEL/lib/"
    QAIRT_LIB_ARGS=(--qairt-lib "$SEL")
  fi
fi

export LD_LIBRARY_PATH="$BUNDLE/lib:$BUNDLE/lib/llama_cpp:$BUNDLE/lib/qairt:$LD_LIBRARY_PATH"
export GENIEX_PLUGIN_PATH="$BUNDLE/lib"

# geniex-bench fails randomly on QDC devices; give each invocation one
# retry before letting the failure propagate.
geniex_retry() {
    "$@" || {
        echo "geniex_retry: command failed, retrying once: $*" >&2
        "$@"
    }
}

IMG=$TC/test.png

RUN_PERF={RUN_PERF}
RUN_EVAL={RUN_EVAL}
# {CTX_LIST} is a comma-separated placeholder (e.g. "512,1024,4096"); turn the
# commas into spaces so the shell for-loops below word-split it.
CTX_LIST="{CTX_LIST}"
CTX_LIST="${CTX_LIST//,/ }"

# Perf records failures but defers the fatal exit to the end so a perf failure
# never skips the accuracy eval below.
failed_ctxs=""
if [ "$RUN_PERF" = "1" ]; then
  # One matrix (and one geniex-bench process) per ctx+backend: a fatal error
  # in one backend (compute unit) could otherwise prevent subsequent ones.
  declare -A TSV
  ctx_devs=""

  while IFS='|' read -r name plugin devs model_id vlm image; do
    [ -z "$name" ] && continue
    echo "=== plan $name id=$model_id ==="
    imgpath=""
    [ "$image" = "1" ] && imgpath="$IMG"
    # $devs is comma-separated; split on commas, but keep ctx word-splitting on
    # the default IFS (space) since CTX_LIST is space-separated.
    IFS=',' read -ra dev_arr <<< "$devs"
    for d in "${dev_arr[@]}"; do
      for ctx in $CTX_LIST; do
        key="$ctx-$d"
        if [ -z "${TSV[$key]}" ]; then
          TSV[$key]=/data/local/tmp/matrix-$key.tsv
          : > "${TSV[$key]}"
          ctx_devs="$ctx_devs $key"
        fi
        printf '%s-%s-%s-c%s\t%s\t%s\t%s\t\t\t%s\t%s\n' \
          "$name" "$plugin" "$d" "$ctx" "$plugin" "$d" "$model_id" "$imgpath" "$vlm" \
          >> "${TSV[$key]}"
      done
    done
  done <<'EOF'
{MODELS}
EOF

  for key in $ctx_devs; do
    tsv="${TSV[$key]}"
    ctx="${key%%-*}"
    echo "=== matrix ctx=$ctx device=${key#*-} ==="
    cat "$tsv"
    geniex_retry ./bin/geniex-bench --matrix-file "$tsv" --output-json-dir "$OUT" -r 3 \
      "${QAIRT_LIB_ARGS[@]}" {BENCH_SIZE_FLAGS} \
      --mm-data-dir "$MM_CACHE" --chipset "{CHIPSET}"
    bench_rc=$?
    echo "rc=$bench_rc  ($(ls "$OUT" | wc -l) cell json files so far)"
    [ "$bench_rc" -ne 0 ] && failed_ctxs="$failed_ctxs $key"
  done
else
  echo "=== perf sweep disabled (RUN_PERF=0); eval only ==="
fi

# Accuracy eval: runs `geniex-bench --accuracy` once per staged prompt, marking
# each invocation's stdout with ===EVAL_IDX_NNN=== for the host to split.
PROMPT_DIR=$TC/prompts
EVAL_OUT=$LOG/geniex_eval_outputs.txt
if [ "$RUN_EVAL" = "1" ]; then
  # Eval requested, so the host must have staged prompts; a missing dir is a
  # real failure, not a quiet skip.
  if [ ! -d "$PROMPT_DIR" ]; then
    echo "FATAL: eval requested but $PROMPT_DIR missing"
    exit 1
  fi
  if [ ! -f "$PROMPT_DIR/system_prompt.txt" ]; then
    echo "FATAL: eval requested but $PROMPT_DIR/system_prompt.txt missing"
    exit 1
  fi
  SYSTEM_PROMPT=$(cat "$PROMPT_DIR/system_prompt.txt")
  echo "=== accuracy eval ==="
  # The eval model/plugin/device come from the first matrix row (single-model
  # geniex-bench jobs submit one row). col4 is the -m model ref (a device-side
  # qairt bundle path after rewrite, or a llama_cpp model-manager id).
  first_row=$(printf '%s\n' "{MODELS}" | grep -v '^[[:space:]]*$' | head -n1)
  IFS='|' read -r _e_name e_plugin _e_devs e_model _e_vlm _e_img <<< "$first_row"
  # Accuracy eval always runs on the NPU (the on-device target).
  e_dev=npu
  EVAL_ERR=$LOG/geniex_eval_stderr.txt

  # Retry a random device flake once, but not a "Context length exceeded"
  # overflow or a geniex-bench self-abort -- those are legit long/diverging
  # generations, so keep the partial [gen] output and move on. Each attempt writes to fresh temp files so the accepted
  # attempt's output goes under the ===EVAL_IDX=== marker exactly once; a
  # first-attempt failure's [gen] lines are not concatenated with the retry.
  # $1 = per-prompt stdout tmp; $2 = per-prompt stderr tmp; $3.. = the command.
  geniex_eval_retry() {
    local out="$1"; local err="$2"; shift 2
    "$@" >"$out" 2>"$err" && return 0
    if grep -qE "Context length exceeded|self-abort: reason=" "$out" "$err"; then
      echo "geniex_eval_retry: context overflow or self-abort, skipping retry: $*" >&2
      return 0
    fi
    echo "geniex_eval_retry: command failed, retrying once: $*" >&2
    "$@" >"$out" 2>"$err"
  }

  : > "$EVAL_OUT"
  : > "$EVAL_ERR"
  eval_ran=0
  for prompt_file in "$PROMPT_DIR"/prompt_*.txt; do
    [ -e "$prompt_file" ] || continue
    idx=$(basename "$prompt_file" | sed 's/prompt_\([0-9]*\)\.txt/\1/')
    pout=$(mktemp)
    perr=$(mktemp)
    # timeout kills a crashed/hung DSP run so the loop advances (SIGTERM lets
    # geniex-bench self-abort cleanly, SIGKILL after 5s); sleep lets the DSP
    # release before the next attach.
    if geniex_eval_retry "$pout" "$perr" \
      timeout -s TERM -k 5 {EVAL_TIMEOUT_S} ./bin/geniex-bench --plugin "$e_plugin" --device "$e_dev" \
        -m "$e_model" "${QAIRT_LIB_ARGS[@]}" --accuracy --prompt-file "$prompt_file" \
        --system-prompt "$SYSTEM_PROMPT" --no-think \
        -c {EVAL_CTX} -n {EVAL_N_GEN} --seed {EVAL_SEED} --mm-data-dir "$MM_CACHE" \
        --max-gen-time-s {EVAL_MAX_GEN_TIME_S} \
        --no-progress-timeout-s {EVAL_NO_PROGRESS_TIMEOUT_S} \
        --repetition-max-repeats {EVAL_REPETITION_MAX_REPEATS} \
        --ngram-max-repeats {EVAL_NGRAM_MAX_REPEATS} \
        --chipset "{CHIPSET}"; then
      eval_ran=$((eval_ran + 1))
    fi
    # Emit the marker + accepted attempt's output only once so retried prompts
    # don't return concatenated generations to the grader.
    echo "===EVAL_IDX_${idx}===" >> "$EVAL_OUT"
    echo "===EVAL_IDX_${idx}===" >> "$EVAL_ERR"
    cat "$pout" >> "$EVAL_OUT"
    cat "$perr" >> "$EVAL_ERR"
    rm -f "$pout" "$perr"
    sleep {EVAL_SLEEP_S}
  done
  echo "eval done ($(grep -c '===EVAL_IDX_' "$EVAL_OUT") prompts, $eval_ran ran)"
  # Every prompt failed => surface it instead of a false SUCCESS.
  if [ "$eval_ran" -eq 0 ]; then
    echo "FATAL: accuracy eval produced no output"
    exit 1
  fi
fi

echo "=== done ==="
if [ -n "$failed_ctxs" ]; then
  echo "FATAL: geniex-bench failed for ctx-device:$failed_ctxs"
  exit 1
fi
exit 0
