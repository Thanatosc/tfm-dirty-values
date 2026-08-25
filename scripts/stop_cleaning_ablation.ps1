<#
.SYNOPSIS
Request a graceful checkpoint of the Phase 3 worker after the current task.

.DESCRIPTION
Writes results/cleaning_ablation/control/stop.request. The worker finishes the
task it holds, commits it atomically, and exits. Resume by re-running
scripts/start_cleaning_ablation.ps1 -ConfirmPreregisteredRun, which requeues any
interrupted task without consuming a model-failure retry.

Creates no stop file when the ablation was never initialized, so a stray request
cannot silently halt a future run.
#>
param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$database = Join-Path $ProjectRoot "results/cleaning_ablation/state.sqlite3"
$controlDir = Join-Path $ProjectRoot "results/cleaning_ablation/control"
$stopPath = Join-Path $controlDir "stop.request"

if (-not (Test-Path -LiteralPath $database)) {
    Write-Output "Phase 3 is not initialized; no stop request created."
    exit 0
}

New-Item -ItemType Directory -Force -Path $controlDir | Out-Null
Set-Content -LiteralPath $stopPath -Encoding utf8 -Value (Get-Date -Format "o")
Write-Output "Stop requested: $stopPath"
Write-Output "The worker will finish the current task, commit it, then exit."

if ($Force) {
    $heartbeatPath = Join-Path $ProjectRoot "results/cleaning_ablation/heartbeat.json"
    if (-not (Test-Path -LiteralPath $heartbeatPath)) {
        Write-Output "No heartbeat file; nothing to terminate."
        exit 0
    }
    $heartbeat = Get-Content -Raw -Encoding utf8 $heartbeatPath | ConvertFrom-Json
    $processId = [int]$heartbeat.pid
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Output "Worker PID $processId is not running."
        exit 0
    }
    Write-Warning "Force-terminating worker PID $processId. The in-flight task will be requeued."
    Stop-Process -Id $processId -Force
    Write-Output "Terminated. Re-run the start script with -ConfirmPreregisteredRun to resume."
}
