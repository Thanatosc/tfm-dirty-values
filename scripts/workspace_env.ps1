[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$AllowedPrefix = $ProjectRoot.TrimEnd("\") + "\"
$RuntimeRoot = Join-Path $ProjectRoot "artifacts\tmp\runtime"

function Assert-ProjectRuntimePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith($AllowedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Runtime path escapes the project root: $FullPath"
    }
    return $FullPath
}

$RuntimePaths = [ordered]@{
    TEMP = Join-Path $RuntimeRoot "system"
    TMP = Join-Path $RuntimeRoot "system"
    TMPDIR = Join-Path $RuntimeRoot "system"
    PIP_CACHE_DIR = Join-Path $RuntimeRoot "pip-cache"
    UV_CACHE_DIR = Join-Path $RuntimeRoot "uv-cache"
    HF_HOME = Join-Path $RuntimeRoot "hf-cache"
    TORCH_HOME = Join-Path $RuntimeRoot "torch-cache"
    XDG_CACHE_HOME = Join-Path $RuntimeRoot "xdg-cache"
    PYTHONPYCACHEPREFIX = Join-Path $RuntimeRoot "python-cache"
    MPLCONFIGDIR = Join-Path $RuntimeRoot "matplotlib"
    NUMBA_CACHE_DIR = Join-Path $RuntimeRoot "numba-cache"
    JOBLIB_TEMP_FOLDER = Join-Path $RuntimeRoot "joblib"
    CUDA_CACHE_PATH = Join-Path $RuntimeRoot "cuda-cache"
    TRITON_CACHE_DIR = Join-Path $RuntimeRoot "triton-cache"
}

foreach ($Entry in $RuntimePaths.GetEnumerator()) {
    $SafePath = Assert-ProjectRuntimePath -Path $Entry.Value
    New-Item -ItemType Directory -Force -Path $SafePath | Out-Null
    Set-Item -Path ("Env:" + $Entry.Key) -Value $SafePath
}
