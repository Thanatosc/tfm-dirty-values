<#
.SYNOPSIS
Read-only Phase 3 progress monitor. Creates no authorization and no result state.
#>
param(
    [switch]$Json,
    [switch]$Watch,
    [int]$IntervalSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$driver = Join-Path $PSScriptRoot "phase3_status.py"

$arguments = @("-B", $driver)
if ($Json) { $arguments += "--json" }

Push-Location -LiteralPath $ProjectRoot
try {
    if ($Watch) {
        while ($true) {
            Clear-Host
            Write-Output ("--- " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ---")
            & $Python @arguments
            Start-Sleep -Seconds $IntervalSeconds
        }
    } else {
        & $Python @arguments
    }
} finally {
    Pop-Location
}
