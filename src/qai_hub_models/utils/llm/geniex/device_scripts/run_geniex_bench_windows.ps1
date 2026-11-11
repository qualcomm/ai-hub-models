[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$ErrorActionPreference = "Continue"
# Expand-Archive of the QAIRT SDK emits ~12k progress records; left on, they
# overflow the host's output and crash the next Write-Host.
$ProgressPreference = "SilentlyContinue"

$LOG = "C:\Temp\QDC_Logs"
$OUT = "$LOG\results"
$MM_CACHE = "C:\Temp\geniex-cache"
$MATRIX_DIR = "C:\Temp\geniex-matrix"
$TC = "C:\Temp\TestContent"

# A job QDC killed mid-run skips the finally below and can leave geniex-bench
# running with script.log open, which makes Start-Transcript fail outright.
Stop-Process -Name geniex-bench, genie-t2t-run -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# QDC reuses device cells across jobs; wipe stale cell JSONs from a prior
# run so compute_metrics doesn't ingest another model/plugin's results.
Remove-Item -Recurse -Force $OUT, $MM_CACHE, $MATRIX_DIR -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $LOG, $OUT, $MM_CACHE, $MATRIX_DIR | Out-Null
Start-Transcript -Path "$LOG\script.log" -Force | Out-Null

try {

# A stale bundle would otherwise win the 'Select-Object -First 1' pick below.
Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
if (Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' -ErrorAction SilentlyContinue) {
    throw "Stale geniex-bench bundle under $TC could not be removed; device needs manual cleanup."
}
# Best-effort sweep of SDKs left by killed jobs; a locked one is skipped, not fatal.
Remove-Item -Recurse -Force "$TC\geniex-bench.zip", "$TC\qairt", "$TC\sdk_*", "$TC\qairt.zip" -ErrorAction SilentlyContinue

# geniex-bench.exe writes informational lines to stderr even on success.
# Called via the bare `&` operator, every such line becomes a NativeCommandError
# ErrorRecord that QDC's parser flags as Unsuccessful — same trap that bit
# genie's run_windows.ps1. Start-Process redirects at the OS-process level so
# stderr bypasses PowerShell's error stream entirely. One retry mirrors
# Invoke-GenieRetry. Returns $true on success, $false on double-failure so the
# caller can defer the fatal exit until after the accuracy eval.
function Invoke-GenieXBenchRetry {
    param([Parameter(Mandatory = $true)][string[]]$BenchArgs)
    foreach ($attempt in 1, 2) {
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process -FilePath "$BUNDLE\bin\geniex-bench.exe" `
                -ArgumentList $BenchArgs `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
            $exitCode = $proc.ExitCode
            $stdout = Get-Content $stdoutFile -Raw -Encoding UTF8
            $stderr = Get-Content $stderrFile -Raw -Encoding UTF8
        } finally {
            Remove-Item $stdoutFile -ErrorAction SilentlyContinue
            Remove-Item $stderrFile -ErrorAction SilentlyContinue
        }
        $captured = @($stdout) + @($stderr)
        $captured | Out-String | Write-Host
        if ($exitCode -eq 0) { return $true }
        Write-Host "Invoke-GenieXBenchRetry: geniex-bench.exe failed (exit $exitCode)"
    }
    return $false
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

# A 1-byte ranged GET works for presigned S3 URLs, where HEAD is rejected.
function Get-RemoteSize {
    param([Parameter(Mandatory = $true)][string]$Url)
    $hdr = (& curl.exe -sSL -D - -o NUL -r 0-0 --max-time 30 $Url 2>$null) -join "`n"
    $m = [regex]::Matches($hdr, '(?im)^content-range:\s*bytes\s+\S+/(\d+)')
    if ($m.Count -gt 0) { return [int64]$m[$m.Count - 1].Groups[1].Value }
    return $null
}

function Test-ZipReadable {
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        [System.IO.Compression.ZipFile]::OpenRead($Path).Dispose()
        return $true
    } catch {
        return $false
    }
}

# Resumes a short file with curl -C -; accepts a download only when its size
# matches the server's total and the zip's central directory is readable.
function Save-ZipWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Dest,
        [int]$MaxTime = 3600
    )
    $expected = Get-RemoteSize -Url $Url
    Write-Host "Expected size of ${Dest}: $expected"
    Remove-Item $Dest -Force -ErrorAction SilentlyContinue
    $attempts = 5
    for ($attempt = 1; $attempt -le $attempts; $attempt++) {
        $resume = if (Test-Path $Dest) { @("-C", "-") } else { @() }
        & curl.exe -sSL --fail --max-time $MaxTime @resume -o $Dest $Url 2>$null
        $rc = $LASTEXITCODE
        $size = if (Test-Path $Dest) { (Get-Item $Dest).Length } else { 0 }
        $valid = (($null -eq $expected) -or ($size -eq $expected)) -and (Test-Path $Dest) -and (Test-ZipReadable -Path $Dest)
        Write-Host "Download attempt ${attempt}: curl rc=$rc, size=$size, valid=$valid"
        if ($valid) { return }
        if (($null -eq $expected) -or ($size -ge $expected)) {
            Remove-Item $Dest -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds (5 * $attempt)
    }
    throw "Download of $Url failed after $attempts attempts (expected size $expected)"
}

$ZIP = "$TC\geniex-bench.zip"
$URL = "{WINDOWS_BENCH_URL}"
Save-ZipWithRetry -Url $URL -Dest $ZIP -MaxTime 1800
Expand-Archive -Path $ZIP -DestinationPath $TC -Force
Remove-Item $ZIP

$BUNDLE = (Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' | Select-Object -First 1).FullName
if (-not $BUNDLE) { throw "extracted bundle dir missing under $TC" }
if (-not (Test-Path "$BUNDLE\bin\geniex-bench.exe")) {
    throw "geniex-bench.exe not found at $BUNDLE\bin"
}

Set-Location $BUNDLE

# Run the qairt plugin against QAIRT_VERSION's SDK via --qairt-lib; empty
# QAIRT_VERSION keeps the QAIRT bundled with geniex-bench.
$PLUGIN = "{PLUGIN}"
$QAIRT_VERSION = "{QAIRT_VERSION}"
$QAIRT_LIB_ARGS = @()
if ($PLUGIN -eq "qairt" -and $QAIRT_VERSION) {
    Write-Output "=== Using QAIRT $QAIRT_VERSION ==="
    $QZIP = "$TC\qairt.zip"
    # Extract into a per-run dir so a stale/locked SDK from a killed job can't
    # block or corrupt this run. The zip's top-level entry is qairt\<version>\.
    $SDK_DIR = "$TC\sdk_$PID"
    $QROOT = "$SDK_DIR\qairt"
    $QURL = "{QAIRT_SDK_URL}"
    Save-ZipWithRetry -Url $QURL -Dest $QZIP -MaxTime 3600
    Expand-Archive -Path $QZIP -DestinationPath $SDK_DIR -Force
    Remove-Item $QZIP
    $SDK = "$QROOT\$QAIRT_VERSION"
    if (-not (Test-Path "$SDK\lib")) { throw "QAIRT SDK not extracted to $SDK" }
    $QAIRT_LIB_ARGS = @("--qairt-lib", $SDK)
}

$env:GENIEX_PLUGIN_PATH = "$BUNDLE\lib"
$env:PATH = "$BUNDLE\lib;$BUNDLE\lib\llama_cpp;$BUNDLE\lib\qairt;$BUNDLE\lib\qairt\htp-files;$env:PATH"

$rows = @'
{MODELS}
'@ -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }

$IMG = "$TC/test.png" -replace '\\', '/'

$RUN_PERF = {RUN_PERF}
$RUN_EVAL = {RUN_EVAL}
$ctxList = @({CTX_LIST})
# Perf records failures but defers the fatal exit to the end so a perf failure
# never skips the accuracy eval below.
$failed_ctxs = ""

if ($RUN_PERF) {
    $tsvByCtx = @{}
    foreach ($ctx in $ctxList) {
        $tsvByCtx[$ctx] = "$MATRIX_DIR\matrix-$ctx.tsv"
    }

    foreach ($row in $rows) {
        $name, $plugin, $devs, $model_id, $vlm, $image = $row -split '\|'
        Write-Output "=== plan $name id=$model_id ==="
        $imgpath = if ($image -eq "1") { $IMG } else { "" }
        foreach ($d in $devs -split ',') {
            foreach ($ctx in $ctxList) {
                "{0}-{1}-{2}-c{3}`t{1}`t{2}`t{4}`t`t`t{5}`t{6}" -f `
                    $name, $plugin, $d, $ctx, $model_id, $imgpath, $vlm `
                    | Add-Content $tsvByCtx[$ctx]
            }
        }
    }

    foreach ($ctx in $ctxList) {
        $tsv = $tsvByCtx[$ctx]
        Write-Output "=== matrix ctx=$ctx ==="
        if (Test-Path $tsv) { Get-Content $tsv }
        $ok = Invoke-GenieXBenchRetry -BenchArgs (@(
            "--matrix-file", $tsv, "--output-json-dir", $OUT, "-r", "3",
            {BENCH_SIZE_FLAGS_ARGS}
            "--mm-data-dir", $MM_CACHE, "--chipset", "{CHIPSET}"
        ) + $QAIRT_LIB_ARGS)
        if (-not $ok) { $failed_ctxs += " $ctx" }
        Write-Output "$((Get-ChildItem $OUT).Count) cell json files so far"
    }
} else {
    Write-Output "=== perf sweep disabled (RUN_PERF=`$false); eval only ==="
}

# Accuracy eval: runs `geniex-bench --accuracy` once per staged prompt, marking
# each invocation's stdout with ===EVAL_IDX_NNN=== for the host to split.
$PROMPT_DIR = "$TC\prompts"
$EVAL_OUT = "$LOG\geniex_eval_outputs.txt"
if ($RUN_EVAL) {
    if (-not (Test-Path $PROMPT_DIR)) {
        throw "eval requested but $PROMPT_DIR missing"
    }
    $systemPromptFile = "$PROMPT_DIR\system_prompt.txt"
    if (-not (Test-Path $systemPromptFile)) {
        throw "eval requested but $systemPromptFile missing"
    }
    $SYSTEM_PROMPT = Get-Content $systemPromptFile -Raw -Encoding UTF8
    # -ArgumentList doesn't quote elements itself; escape so this survives as one arg.
    $SYSTEM_PROMPT_ARG = '"' + ($SYSTEM_PROMPT -replace '"', '\"') + '"'
    Write-Output "=== accuracy eval ==="
    # Eval model/plugin/device come from the first matrix row (single-model job).
    # @($rows) forces array semantics: with a single MODELS line the pipeline
    # that builds $rows unwraps to a scalar string, so a bare $rows[0] would
    # index the first character instead of the first row.
    $e_name, $e_plugin, $e_devs, $e_model = (@($rows)[0] -split '\|')[0..3]
    $e_dev = "npu"
    $EVAL_ERR = "$LOG\geniex_eval_stderr.txt"
    Remove-Item $EVAL_OUT, $EVAL_ERR -ErrorAction SilentlyContinue
    $eval_ran = 0
    foreach ($pf in Get-ChildItem "$PROMPT_DIR\prompt_*.txt" | Sort-Object Name) {
        $idx = $pf.BaseName -replace 'prompt_', ''
        $tmpOut = $EVAL_OUT.Replace('.txt', '.tmp')
        $tmpErr = $EVAL_ERR.Replace('.txt', '.tmp')
        # Retry a random device flake once, but not a "Context length exceeded"
        # overflow -- that's a legit long generation, so keep the partial output
        # and stop. Each attempt writes to fresh temp files; the accepted
        # attempt's stdout is appended under the ===EVAL_IDX=== marker exactly
        # once so retries do not concatenate two generations at the grader.
        $exitCode = 1
        $stderrText = ""
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            # Wipe any leftover output from the previous attempt so this loop
            # iteration only ever contributes one attempt's stdout/stderr.
            Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
            try {
                # Start-Process redirects at the OS-process level so geniex-bench's
                # stderr info lines don't become PowerShell NativeCommandErrors (QDC
                # flags those Unsuccessful).
                $proc = Start-Process -FilePath "$BUNDLE\bin\geniex-bench.exe" -ArgumentList (@(
                    "--plugin", $e_plugin, "--device", $e_dev, "-m", $e_model,
                    "--accuracy", "--prompt-file", $pf.FullName,
                    "--system-prompt", $SYSTEM_PROMPT_ARG, "--no-think",
                    "-c", "{EVAL_CTX}", "-n", "{EVAL_N_GEN}", "--seed", "{EVAL_SEED}",
                    "--max-gen-time-s", "{EVAL_MAX_GEN_TIME_S}",
                    "--no-progress-timeout-s", "{EVAL_NO_PROGRESS_TIMEOUT_S}",
                    "--repetition-max-repeats", "{EVAL_REPETITION_MAX_REPEATS}",
                    "--ngram-max-repeats", "{EVAL_NGRAM_MAX_REPEATS}",
                    "--mm-data-dir", $MM_CACHE, "--chipset", "{CHIPSET}"
                ) + $QAIRT_LIB_ARGS) -NoNewWindow -PassThru -RedirectStandardOutput $tmpOut `
                    -RedirectStandardError $tmpErr
                # Without -Wait, PS 5.1 hands back a Process with no cached native
                # handle, so .ExitCode reads $null after exit; touching .Handle keeps
                # one. Non-fatal: an exe that dies instantly can outrace the open.
                try { $null = $proc.Handle } catch { }
                # timeout kills a crashed/hung DSP run so the loop advances.
                if (-not $proc.WaitForExit({EVAL_TIMEOUT_S} * 1000)) {
                    Write-Host "eval idx $idx timed out after {EVAL_TIMEOUT_S}s; killing"
                    try { $proc.Kill($true) } catch { $proc.Kill() }
                    $proc.WaitForExit()
                }
                $exitCode = $proc.ExitCode
                $stderrText = (Get-Content $tmpErr -Raw -Encoding UTF8) + (Get-Content $tmpOut -Raw -Encoding UTF8)
            } catch {
                Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
                throw
            }
            if ($exitCode -eq 0 -or ($stderrText -match 'Context length exceeded|self-abort: reason=')) { break }
            if ($attempt -lt 2) {
                Write-Host "eval idx $idx failed (exit $exitCode); retrying once"
            }
        }
        # Emit the marker + accepted attempt's output only once, then clean up.
        # -Encoding utf8 on every write: Add-Content defaults to the ANSI
        # codepage on PS 5.1, which mangles non-ASCII generations into CP1252.
        try {
            Add-Content -Path $EVAL_OUT -Value "===EVAL_IDX_$idx===" -Encoding utf8
            Add-Content -Path $EVAL_ERR -Value "===EVAL_IDX_$idx===" -Encoding utf8
            if (Test-Path $tmpOut) { Get-Content $tmpOut -Encoding UTF8 | Add-Content $EVAL_OUT -Encoding utf8 }
            if (Test-Path $tmpErr) { Get-Content $tmpErr -Encoding UTF8 | Add-Content $EVAL_ERR -Encoding utf8 }
        } finally {
            Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
        }
        # A context-length overflow counts as run (keeps its partial output).
        if ($exitCode -eq 0 -or ($stderrText -match 'Context length exceeded|self-abort: reason=')) {
            $eval_ran++
        } else {
            Write-Host "eval idx $idx failed (exit $exitCode)"
        }
        Start-Sleep -Seconds {EVAL_SLEEP_S}
    }
    $n = (Select-String -Path $EVAL_OUT -Pattern '===EVAL_IDX_').Count
    Write-Output "eval done ($n prompts, $eval_ran ran)"
    if ($eval_ran -eq 0) {
        throw "accuracy eval produced no output"
    }
}

Write-Output "=== done ==="

# Deferred perf failure: raise inside the try so the finally cleanup still runs.
if ($failed_ctxs) { throw "geniex-bench failed for context lengths:$failed_ctxs" }

}
finally {
    # Drop per-job state on exit (dedicated-pool devices are reused across jobs;
    # leftover extracted bundles / caches / matrix TSVs would leak into the next
    # tenant's run).
    Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Force "$TC\geniex-bench.zip" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$TC\qairt", "$TC\sdk_*", "$TC\qairt.zip" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$TC\qairt_bundles", "$TC\prompts", "$TC\perf_prompts" -ErrorAction SilentlyContinue
    Remove-Item -Force "$TC\test.png", $PSCommandPath -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $MM_CACHE, $MATRIX_DIR -ErrorAction SilentlyContinue
    Stop-Transcript | Out-Null
}
exit 0
