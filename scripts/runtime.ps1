$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
. (Join-Path $PSScriptRoot 'workspace_env.ps1')

function Get-ProjectPython {
    $Candidate = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Python 3.11 runtime not found. Run 'uv sync --extra dev' in $ProjectRoot."
    }
    $SourcePath = Join-Path $ProjectRoot 'src'
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourcePath
    } else {
        $env:PYTHONPATH = $SourcePath + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
    }
    return [System.IO.Path]::GetFullPath($Candidate)
}
