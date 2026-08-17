[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ResultDir = Join-Path $ProjectRoot "results\full_study"
$HeartbeatPath = Join-Path $ResultDir "heartbeat.json"
if (-not (Test-Path -LiteralPath $HeartbeatPath)) {
    Write-Output "No full-study heartbeat exists; no stop request was created."
    return
}
$Heartbeat = Get-Content -Raw -Encoding UTF8 -LiteralPath $HeartbeatPath | ConvertFrom-Json
$Worker = Get-Process -Id ([int]$Heartbeat.pid) -ErrorAction SilentlyContinue
if ($null -eq $Worker -or $Heartbeat.status -notin @("starting", "running")) {
    Write-Output "No active full-study worker was found; no stop request was created."
    return
}
if ($Force) {
    Stop-Process -Id ([int]$Heartbeat.pid) -Force
    Write-Output "Worker force-stopped. Its running task will be requeued by the next confirmed resume."
    return
}
$ControlDir = Join-Path $ResultDir "control"
New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null
Set-Content -Encoding ASCII -LiteralPath (Join-Path $ControlDir "stop.request") -Value ((Get-Date).ToString("o"))
Write-Output "Stop requested. The worker will finish its current task, persist it, and exit."
