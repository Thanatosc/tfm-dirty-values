[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArguments
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime.ps1")
$Python = Get-ProjectPython

Push-Location $ProjectRoot
try {
    & $Python -B -m pytest -p no:tmpdir -p no:cacheprovider @PytestArguments
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
