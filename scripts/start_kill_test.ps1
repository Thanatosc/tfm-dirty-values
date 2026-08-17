[CmdletBinding()]
param([switch]$ConfirmPreregisteredRun)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$Config = Join-Path $ProjectRoot "configs\kill_test.yaml"
$ResultDir = Join-Path $ProjectRoot "results\kill_test"
$ControlDir = Join-Path $ResultDir "control"
$LogDir = Join-Path $ResultDir "logs"

if (-not $ConfirmPreregisteredRun) {
    throw "Refusing to start outcome-producing runs without -ConfirmPreregisteredRun."
}

Push-Location $ProjectRoot
try {
    # This writes an external, hash-bound authorization artifact without mutating the frozen config.
    & $Python -B -m tfm_dirty_values.cli --config $Config authorize --confirmed-by-project-owner
    if ($LASTEXITCODE -ne 0) {
        throw "Kill Test authorization was refused or failed with exit code $LASTEXITCODE."
    }
    # This command validates the external authorization before creating the task database.
    & $Python -B -m tfm_dirty_values.cli --config $Config init
    if ($LASTEXITCODE -ne 0) {
        throw "Task initialization was refused or failed with exit code $LASTEXITCODE."
    }

    New-Item -ItemType Directory -Force -Path $ControlDir, $LogDir | Out-Null
    $HeartbeatPath = Join-Path $ResultDir "heartbeat.json"
    if (Test-Path -LiteralPath $HeartbeatPath) {
        $Heartbeat = Get-Content -Raw -Encoding UTF8 -LiteralPath $HeartbeatPath | ConvertFrom-Json
        $Existing = Get-Process -Id ([int]$Heartbeat.pid) -ErrorAction SilentlyContinue
        if ($null -ne $Existing -and $Heartbeat.status -in @("starting", "running")) {
            throw "A Kill Test worker is already running with PID=$($Heartbeat.pid)."
        }
    }
    $StopRequest = Join-Path $ControlDir "stop.request"
    if (Test-Path -LiteralPath $StopRequest) {
        Remove-Item -LiteralPath $StopRequest -Force
    }

    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Stdout = Join-Path $LogDir "worker-$Timestamp.out.log"
    $Stderr = Join-Path $LogDir "worker-$Timestamp.err.log"
    $Arguments = @(
        "-B", "-m", "tfm_dirty_values.cli", "--config", $Config,
        "worker", "--stdout-log", $Stdout, "--stderr-log", $Stderr
    )
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    [ordered]@{
        pid = $Process.Id
        started_at = (Get-Date).ToString("o")
        executable = $Python
        arguments = $Arguments
        working_directory = $ProjectRoot
        stdout = $Stdout
        stderr = $Stderr
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $ResultDir "launcher.json")

    for ($Attempt = 0; $Attempt -lt 30 -and -not (Test-Path -LiteralPath $HeartbeatPath); $Attempt++) {
        Start-Sleep -Seconds 1
        if ($Process.HasExited) {
            throw "Worker exited before creating a heartbeat. Inspect $Stderr"
        }
    }
    if (-not (Test-Path -LiteralPath $HeartbeatPath)) {
        throw "Worker failed to create a heartbeat within 30 seconds. Inspect $Stderr"
    }
    $WorkerHeartbeat = Get-Content -Raw -Encoding UTF8 -LiteralPath $HeartbeatPath | ConvertFrom-Json
    Write-Output "Kill Test worker started. PID=$($WorkerHeartbeat.pid)"
    Write-Output "Status: powershell -ExecutionPolicy Bypass -File scripts\kill_test_status.ps1"
    Write-Output "Stop after current task: powershell -ExecutionPolicy Bypass -File scripts\stop_kill_test.ps1"
} finally {
    Pop-Location
}
