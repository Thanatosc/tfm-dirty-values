[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ResultDir = Join-Path $ProjectRoot "results\kill_test"
$HeartbeatPath = Join-Path $ResultDir "heartbeat.json"
if (-not (Test-Path -LiteralPath $HeartbeatPath)) {
    Write-Output "No Kill Test heartbeat exists; no stop request was created."
    return
}
$Heartbeat = Get-Content -Raw -Encoding UTF8 -LiteralPath $HeartbeatPath | ConvertFrom-Json
$Worker = Get-Process -Id ([int]$Heartbeat.pid) -ErrorAction SilentlyContinue
if ($null -eq $Worker -or $Heartbeat.status -notin @("starting", "running")) {
    Write-Output "No active Kill Test worker was found; no stop request was created."
    return
}
$ControlDir = Join-Path $ResultDir "control"
New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null
Set-Content -Encoding ASCII -LiteralPath (Join-Path $ControlDir "stop.request") -Value ((Get-Date).ToString("o"))
Write-Output "Stop requested. The worker will finish its current task, persist it, and exit."
