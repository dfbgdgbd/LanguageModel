$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Install uv first from https://docs.astral.sh/uv/getting-started/installation/"
}

uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

Write-Host "Environment ready."
Write-Host "Double-click Start_SmallLM.bat to open the terminal interface."
Write-Host "Or run: .venv\Scripts\python.exe Main_Run_Program.py"
