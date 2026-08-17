[CmdletBinding()]
param([switch]$ConfirmResume)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$Config = Join-Path $ProjectRoot "configs\full_study.yaml"
$ResultDir = Join-Path $ProjectRoot "results\full_study"
$ControlDir = Join-Path $ResultDir "control"
$LogDir = Join-Path $ProjectRoot "results\full_study\logs"
$Database = Join-Path $ResultDir "state.sqlite3"
$Authorization = Join-Path $ProjectRoot "research\phase2-full-study-authorization.json"

if (-not $ConfirmResume) {
    throw "Refusing to resume the outcome-producing full-study worker without -ConfirmResume."
}
if (-not (Test-Path -LiteralPath $Authorization)) {
    throw "Cannot resume before the Phase 2 authorization artifact exists."
}
if (-not (Test-Path -LiteralPath $Database)) {
    throw "Cannot resume before the Phase 2 task database is initialized."
}

Push-Location $ProjectRoot
try {
    $StatusJson = & $Python -B -m tfm_dirty_values.cli --config $Config status --json
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the existing full-study worker status."
    }
    $Status = ($StatusJson -join [Environment]::NewLine) | ConvertFrom-Json
    if ($Status.worker_status -in @("starting", "running")) {
        throw "A full-study worker is already active with PID=$($Status.heartbeat.pid)."
    }

    New-Item -ItemType Directory -Force -Path $ControlDir, $LogDir | Out-Null
    $StopRequest = Join-Path $ControlDir "stop.request"
    if (Test-Path -LiteralPath $StopRequest) {
        Remove-Item -LiteralPath $StopRequest -Force
    }

    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Stdout = Join-Path $LogDir "worker-resume-$Timestamp.out.log"
    $Stderr = Join-Path $LogDir "worker-resume-$Timestamp.err.log"
    $Arguments = @(
        "-B", "-m", "tfm_dirty_values.cli", "--config", $Config,
        "full-worker", "--stdout-log", $Stdout, "--stderr-log", $Stderr
    )
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    [ordered]@{
        pid = $Process.Id
        mode = "resume"
        started_at = (Get-Date).ToString("o")
        executable = $Python
        arguments = $Arguments
        working_directory = $ProjectRoot
        stdout = $Stdout
        stderr = $Stderr
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $ResultDir "launcher.json")

    $HeartbeatPath = Join-Path $ResultDir "heartbeat.json"
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        if ($Process.HasExited) {
            throw "Resumed worker exited before creating a fresh heartbeat. Inspect $Stderr"
        }
        if (Test-Path -LiteralPath $HeartbeatPath) {
            $Heartbeat = Get-Content -Raw -Encoding UTF8 -LiteralPath $HeartbeatPath | ConvertFrom-Json
            if ([int]$Heartbeat.pid -eq $Process.Id -and $Heartbeat.status -in @("starting", "running")) {
                $Ready = $true
                break
            }
        }
        Start-Sleep -Seconds 1
    }
    if (-not $Ready) {
        throw "Resumed worker failed to publish a fresh heartbeat within 30 seconds. Inspect $Stderr"
    }
    Write-Output "Full-study worker resumed. PID=$($Heartbeat.pid)"
    Write-Output "Status: powershell -ExecutionPolicy Bypass -File scripts\full_study_status.ps1 -Watch"
    Write-Output "Stop after current task: powershell -ExecutionPolicy Bypass -File scripts\stop_full_study.ps1"
} finally {
    Pop-Location
}
