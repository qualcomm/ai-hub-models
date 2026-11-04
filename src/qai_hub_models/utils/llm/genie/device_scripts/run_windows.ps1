# genie-t2t-run.exe writes UTF-8 to stdout, and if we don't configure the
# powershell terminal correctly we'll end up capturing mojibake.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

# genie-t2t-run.exe fails randomly on QDC devices; give each invocation one
# retry before letting the failure abort the job. The .exe is launched via
# Start-Process -Wait -PassThru so the genuine process exit code (not Out-File's
# always-0 exit) drives the retry. PowerShell has no `set -e`, so we throw on a
# double-failure to fail the job (the captured output is also only committed to
# $OutFile on success, keeping a crashed attempt out of it).
function Invoke-GenieRetry {
    param(
        [Parameter(Mandatory = $true)][string[]]$GenieArgs,
        [string]$OutFile
    )
    foreach ($attempt in 1, 2) {
        # genie-t2t-run.exe writes informational lines (e.g. '[INFO] "Using
        # create From Binary"') to stderr even on success. When the .exe is run
        # via the call operator (&), Windows PowerShell wraps each native stderr
        # line as a NativeCommandError ErrorRecord -- even with `2>file`, the
        # rendered record (carrying "FullyQualifiedErrorId : NativeCommandError")
        # is what gets captured, and QDC's server-side parser flags the job
        # "Unsuccessful" on seeing that text despite exit code 0.
        #
        # Start-Process redirects stdout/stderr at the OS-process level, so the
        # .exe's streams go straight to files and never pass through PowerShell's
        # error stream -- no ErrorRecord, no FullyQualifiedErrorId.
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process -FilePath "genie-t2t-run.exe" -ArgumentList $GenieArgs `
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
        # Echo to the console as well as the log file, so progress is visible
        # even when a failed QDC job never makes the log files available.
        $captured | Out-String | Write-Host
        if ($exitCode -eq 0) {
            if ($OutFile) { $captured | Out-File -FilePath $OutFile -Append -Encoding utf8 }
            return
        }
        Write-Host "Invoke-GenieRetry: genie-t2t-run.exe failed (exit $exitCode)"
    }
    throw "genie-t2t-run.exe failed twice: genie-t2t-run.exe $($GenieArgs -join ' ')"
}

Set-Location C:\Temp\TestContent\

# A job QDC killed mid-run skips the finally below, leaving its SDK on disk and
# possibly an orphaned process holding it (and its logs) locked against deletion.
Stop-Process -Name genie-t2t-run, geniex-bench -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Drop stale logs from a prior job on this shared device.
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "C:\Temp\{DEVICE_LOGS_DIR}"
New-Item -ItemType Directory -Force -Path "C:\Temp\{DEVICE_LOGS_DIR}" | Out-Null

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "C:\Temp\TestContent\qairt"
if (Test-Path "C:\Temp\TestContent\qairt") {
    throw "Stale QAIRT SDK at C:\Temp\TestContent\qairt could not be removed; device needs manual cleanup."
}

try {

# Verify network connectivity before download
Write-Host "=== Pre-download connectivity check ==="
Write-Host "Pinging google.com before QAIRT SDK download..."
$prePing = Test-Connection -ComputerName google.com -Count 1 -Quiet
if ($prePing) { Write-Host "Pre-download ping: SUCCESS" } else { Write-Host "Pre-download ping: FAILED" }

# Always re-download (the stale SDK was removed above) so a partial extract
# from a previous job can't silently corrupt this run.
$qairtHome = "C:\Temp\TestContent\qairt\{QAIRT_VERSION}"
$source = "{QAIRT_SDK_URL}"
$output = "C:\Temp\TestContent\qairt.zip"
(New-Object System.Net.WebClient).DownloadFile($source, $output)

Write-Host "=== Post-download connectivity check ==="
Write-Host "Pinging google.com after QAIRT SDK download..."
$postPing = Test-Connection -ComputerName google.com -Count 1 -Quiet
if ($postPing) { Write-Host "Post-download ping: SUCCESS (WiFi active)" } else { Write-Host "Post-download ping: FAILED (WiFi down)" }
Expand-Archive -Path "C:\Temp\TestContent\qairt.zip" -DestinationPath "C:\Temp\TestContent\"
$env:QAIRT_HOME = $qairtHome
$env:Path = "$env:QAIRT_HOME\bin\aarch64-windows-msvc;" + $env:Path
$env:Path = "$env:QAIRT_HOME\lib\aarch64-windows-msvc;" + $env:Path
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-{HEXAGON_VERSION}\unsigned"

# Cap generation for perf so a diverging model can't stall profiling.
(Get-Content genie_config.json) -replace '"max-num-tokens": \d+', '"max-num-tokens": 128' | Set-Content genie_config.json
Invoke-GenieRetry -GenieArgs @("-c", "genie_config.json", "--prompt_file", "sample_prompt.txt") -OutFile "C:/Temp/{DEVICE_LOGS_DIR}/genie.log"

for ($i = 1; $i -le {NUM_TRIALS}; $i++) {
    $profileName = "profile$($i).json"
    $outputPath = "C:/Temp/{DEVICE_LOGS_DIR}/$profileName"
    (Get-Content genie_config.json) -replace '"seed": \d+', "`"seed`": $i" | Set-Content genie_config.json
    Invoke-GenieRetry -GenieArgs @("-c", "genie_config.json", "--prompt_file", "sample_prompt.txt", "--profile", $outputPath)
}

$PromptDir = "C:\Temp\TestContent\prompts"
$EvalOutputFile = "C:/Temp/{DEVICE_LOGS_DIR}/eval_outputs.txt"
if (Test-Path $PromptDir) {
    New-Item -ItemType Directory -Force -Path "C:/Temp/{DEVICE_LOGS_DIR}"
    # The perf sweep above left the seed at {NUM_TRIALS}; reset to the bundle's
    # nominal default so eval isn't silently run at a different seed.
    (Get-Content genie_config.json) -replace '"seed": \d+', '"seed": 42' | Set-Content genie_config.json
    (Get-Content genie_config.json) -replace '"max-num-tokens": \d+', '"max-num-tokens": 2048' | Set-Content genie_config.json
    # Switch to power_saver perf_profile: sustained burst thermal-throttles and kills the eval loop on QDC.
    (Get-Content htp_backend_ext_config.json) -replace '"perf_profile": "[^"]*"', '"perf_profile": "power_saver"' | Set-Content htp_backend_ext_config.json
    "" | Out-File -FilePath $EvalOutputFile -Encoding utf8
    $promptFiles = Get-ChildItem -Path $PromptDir -Filter "prompt_*.txt" | Sort-Object Name
    foreach ($promptFile in $promptFiles) {
        $idx = [regex]::Match($promptFile.Name, 'prompt_(\d+)\.txt').Groups[1].Value
        "===EVAL_IDX_${idx}===" | Out-File -FilePath $EvalOutputFile -Append -Encoding utf8
        # One prompt exhausting the context must not discard every other response;
        # the grader scores a missing response as empty.
        try {
            Invoke-GenieRetry -GenieArgs @("-c", "genie_config.json", "--prompt_file", $promptFile.FullName) -OutFile $EvalOutputFile
        } catch {
            "[EVAL_FAILED] $($_.Exception.Message)" | Out-File -FilePath $EvalOutputFile -Append -Encoding utf8
            Write-Host "Invoke-GenieRetry: eval prompt $idx failed; continuing with the rest."
        }
        # Short inter-prompt cooldown to keep the HTP from thermal-throttling.
        Start-Sleep -Seconds 3
    }
}

}
finally {
    # Drop everything on exit (dedicated-pool devices are reused).
    Get-ChildItem -Path "C:\Temp\TestContent" -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "run_windows.ps1" } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
