$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Install uv first from https://docs.astral.sh/uv/getting-started/installation/"
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    uv venv --python 3.11 .venv
}
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

Write-Host "Environment ready for the SmallLM chat interface."
