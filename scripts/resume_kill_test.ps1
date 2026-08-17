[CmdletBinding()]
param([switch]$ConfirmResume)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$Config = Join-Path $ProjectRoot "configs\kill_test.yaml"
$LogDir = Join-Path $ProjectRoot "results\kill_test\logs"

if (-not $ConfirmResume) {
    throw "Refusing to resume the outcome-producing worker without -ConfirmResume."
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Stdout = Join-Path $LogDir "worker-resume-$Timestamp.out.log"
$Stderr = Join-Path $LogDir "worker-resume-$Timestamp.err.log"

Push-Location $ProjectRoot
try {
    & $Python -B -m tfm_dirty_values.cli --config $Config `
        worker --stdout-log $Stdout --stderr-log $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Resumed Kill Test worker failed with exit code $LASTEXITCODE. Inspect $Stderr"
    }
} finally {
    Pop-Location
}
