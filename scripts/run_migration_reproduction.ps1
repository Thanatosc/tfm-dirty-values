[CmdletBinding()]
param([switch]$ConfirmOutcomeRun)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$Config = Join-Path $ProjectRoot "configs\kill_test.yaml"
$Output = Join-Path $ProjectRoot "artifacts\phase0\migration-reproduction.json"

$Arguments = @(
    "-B", "-m", "tfm_dirty_values.migration_reproduction",
    "--config", $Config,
    "--output", $Output
)
if ($ConfirmOutcomeRun) {
    $Arguments += @("--execute", "--confirm-outcome-run")
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Migration reproduction failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
