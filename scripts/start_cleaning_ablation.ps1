<#
.SYNOPSIS
Sole authorized launch path for the Phase 3 cleaning ablation.

.DESCRIPTION
Refuses to run without -ConfirmPreregisteredRun. On confirmation it writes the
hash-bound Phase 3 authorization artifact, queues the 2,425 frozen tasks, and
starts the serial worker in the foreground.

It never mutates configs/cleaning_ablation.yaml, never touches results/full_study,
and refuses if the paired-identity probe is not VERIFIED.
#>
param(
    [switch]$ConfirmPreregisteredRun,
    [switch]$Background
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ConfirmPreregisteredRun) {
    Write-Error @"
Refused: the Phase 3 cleaning ablation produces model outcomes.

This run is bounded by research/phase3-cleaning-ablation-preregistration.md:
  - 2,425 tasks, arms C and D only
  - affected-cell rates 0.15 and 0.50 only
  - one fixed cleaner, no tuning, no mechanism-specific variants
  - Phase 2 results are read-only comparison arms

Re-run with -ConfirmPreregisteredRun to authorize exactly that scope.
"@
    exit 1
}

. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

$driver = Join-Path $PSScriptRoot "phase3_run.py"
if (-not (Test-Path -LiteralPath $driver)) {
    throw "Missing Phase 3 driver: $driver"
}

Push-Location -LiteralPath $ProjectRoot
try {
    if ($Background) {
        $logPath = Join-Path $ProjectRoot "results/cleaning_ablation/worker.stdout.log"
        $errPath = Join-Path $ProjectRoot "results/cleaning_ablation/worker.stderr.log"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
        $process = Start-Process -FilePath $Python `
            -ArgumentList @("-B", $driver, "--authorize", "--initialize", "--run") `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $logPath `
            -RedirectStandardError $errPath `
            -PassThru -WindowStyle Hidden
        Write-Output "Phase 3 worker started in background. Launcher PID: $($process.Id)"
        Write-Output "Note: on Windows the launcher PID may differ from the Python interpreter PID."
        Write-Output "stdout: $logPath"
        Write-Output "stderr: $errPath"
        Write-Output "Monitor with: ./scripts/cleaning_ablation_status.ps1 -Watch"
    } else {
        & $Python -B $driver --authorize --initialize --run
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 3 run exited with code $LASTEXITCODE"
        }
    }
} finally {
    Pop-Location
}
