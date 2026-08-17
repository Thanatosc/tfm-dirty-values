[CmdletBinding()]
param(
    [switch]$Watch,
    [ValidateRange(5, 3600)][int]$IntervalSeconds = 30,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython
$Config = Join-Path $ProjectRoot "configs\full_study.yaml"
$Database = Join-Path $ProjectRoot "results\full_study\state.sqlite3"

do {
    if (-not (Test-Path -LiteralPath $Database)) {
        if ($Json) {
            [ordered]@{ status = "not_initialized"; completed = 0; total_tasks = 0 } | ConvertTo-Json
        } else {
            Write-Output "Full study is not initialized. No outcome tasks exist."
        }
    } else {
        $Arguments = @("-B", "-m", "tfm_dirty_values.cli", "--config", $Config, "status")
        if ($Json) {
            $Arguments += "--json"
        }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Status command failed with exit code $LASTEXITCODE."
        }
    }
    if ($Watch) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Watch)
