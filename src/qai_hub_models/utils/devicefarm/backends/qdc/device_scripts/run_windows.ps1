# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Generic "install Python deps, run one command" job for Windows-on-Snapdragon
# devices (e.g. Snapdragon X2 Elite) through QDC. Model-agnostic: everything
# model-specific arrives through the placeholders below, filled in by
# utils/devicefarm/backends/qdc/device_run.py.
#
#   {PYTHON_EMBED_URL}     embeddable Python zip to download
#   {PYTHON_EMBED_SHA256}  SHA256 the zip must match
#   {GET_PIP_URL}          pinned get-pip.py to download
#   {GET_PIP_SHA256}       SHA256 get-pip.py must match
#   {REQUIREMENTS_FILE}    file name under TestContent, one pip requirement per
#                          line; empty for none
#   {COMMAND_FILE}         file name under TestContent with the arguments for
#                          python.exe, one per line (device_run.py checks they
#                          hold no double quote or %)
#
# The x64 embeddable Python is used on purpose: it runs emulated on the ARM64
# device and has wheels for packages (pyarrow, opencv, onnxsim, ...) that have
# no win_arm64 build.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = "C:\Temp\TestContent"
# QDC only harvests files written under this directory name.
$LogDir = "C:\Temp\QDC_Logs"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $LogDir
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path "$LogDir\script.log" -Force | Out-Null

function Get-Verified($Url, $Path, $Sha256) {
    (New-Object System.Net.WebClient).DownloadFile($Url, $Path)
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) { throw "FATAL: $Url has SHA256 $actual, expected $Sha256" }
}

# Send a native command's output to a file so its stderr is not reported by
# PowerShell as a NativeCommandError (QDC marks such a run unsuccessful).
function Invoke-Logged($Name, $Exe, $ArgString) {
    cmd /c "`"$Exe`" $ArgString > `"$LogDir\$Name.log`" 2>&1"
    Write-Host "=== $Name rc=$LASTEXITCODE"
    return $LASTEXITCODE
}

try {
    Get-Date -Format u
    systeminfo | Select-String "OS Name|System Type|Total Physical" | Write-Host

    $PyDir = "$Root\python"
    New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
    Get-Verified "{PYTHON_EMBED_URL}" "$Root\python-embed.zip" "{PYTHON_EMBED_SHA256}"
    Expand-Archive -Path "$Root\python-embed.zip" -DestinationPath $PyDir -Force
    $Py = "$PyDir\python.exe"

    # The embeddable distribution disables site-packages via its ._pth file, so
    # pip itself and every package it installs would be unimportable.
    $Pth = Get-ChildItem -Path $PyDir -Filter "python*._pth" | Select-Object -First 1
    (Get-Content $Pth.FullName) -replace '^#\s*import site', 'import site' |
        Set-Content $Pth.FullName
    Get-Verified "{GET_PIP_URL}" "$PyDir\get-pip.py" "{GET_PIP_SHA256}"
    $rc = Invoke-Logged "get_pip" $Py "`"$PyDir\get-pip.py`" --no-warn-script-location"
    if ($rc -ne 0) { throw "FATAL: get-pip failed" }

    $ReqFile = "{REQUIREMENTS_FILE}"
    if ($ReqFile -ne "") {
        # One resolve over the whole file keeps the packages consistent with each other.
        $rc = Invoke-Logged "pip_all" $Py "-m pip install --only-binary=:all: --no-warn-script-location -r `"$Root\$ReqFile`""
        if ($rc -ne 0) {
            # Install one at a time only to find which requirements fail (one log
            # each); the run fails either way.
            $failed = @()
            $index = 0
            foreach ($line in (Get-Content "$Root\$ReqFile" | ForEach-Object { ($_ -replace '\s#.*$', '').Trim() } | Where-Object { $_ -and -not $_.StartsWith("#") })) {
                $index++
                $rc = Invoke-Logged ("pip_{0:D3}" -f $index) $Py "-m pip install --only-binary=:all: --no-warn-script-location `"$line`""
                if ($rc -ne 0) { $failed += $line }
            }
            if ($failed.Count -gt 0) { throw ("FATAL: pip install failed for: " + ($failed -join "; ")) }
            throw "FATAL: pip install -r failed although each requirement installs alone; they conflict (see pip_all.log)"
        }
    }

    $PyArgs = (Get-Content -Encoding UTF8 "$Root\{COMMAND_FILE}" |
        ForEach-Object { "`"$_`"" }) -join " "
    $runRc = Invoke-Logged "run" $Py $PyArgs
    Get-Content "$LogDir\run.log" -Tail 60 | ForEach-Object {
        if ($_.Length -gt 300) { $_.Substring(0, 300) } else { $_ }
    } | Write-Host
    # QDC ignores a non-zero `exit` code but marks the job Unsuccessful when the
    # script throws, so a failed command has to throw (after the log tail above).
    if ($runRc -ne 0) { throw "FATAL: command failed with exit code $runRc" }
}
finally {
    Stop-Transcript | Out-Null
    # Dedicated-pool devices are reused: drop everything but this script.
    Get-ChildItem -Path $Root -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "run_windows.ps1" } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
