@echo off
REM Copyright 2026 Xuebin Feng
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM =========================================================================
REM Portable Startup Script for SSN_Tools.py
REM =========================================================================
setlocal EnableDelayedExpansion

:: Move to the project root directory (two levels up from this script)
cd /d "%~dp0..\.."

:: 1. Locate uv executable using labels (no parentheses to avoid parsing bugs)
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "UV_EXE=uv"
    goto UV_FOUND
)

set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if exist "%UV_EXE%" goto UV_FOUND

echo uv package manager not found. Installing it automatically...
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

:UV_FOUND

:: 2. Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating isolated local virtual environment .venv...
    "!UV_EXE!" venv --python 3.12
)

:: 3. Detect GPU brand using python script
echo Detecting hardware configuration...
set "GPU_TYPE=CPU"
for /f "tokens=*" %%g in ('"!UV_EXE!" run --quiet python src\Detect_GPU.py') do (
    set "GPU_TYPE=%%g"
)
echo Target device class: !GPU_TYPE!

:: 4. Resolve dependencies based on GPU Type
echo.
if "!GPU_TYPE!"=="NVIDIA" (
    echo NVIDIA GPU detected. Syncing with PyTorch CUDA 13.0 support...
    "!UV_EXE!" pip install -r src\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
) else if "!GPU_TYPE!"=="INTEL" (
    echo Intel Arc/GPU detected. Syncing with PyTorch XPU support...
    "!UV_EXE!" pip install -r src\requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
) else if "!GPU_TYPE!"=="AMD" (
    echo AMD GPU detected. Syncing with DirectML support for Windows...
    "!UV_EXE!" pip install -r src\requirements.txt
    "!UV_EXE!" pip install torch-directml
) else (
    echo No dedicated GPU detected. Syncing with CPU-only PyTorch...
    "!UV_EXE!" pip install -r src\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
)
echo.

:: 5. Run the tools
echo Starting SSN_Tools...
"!UV_EXE!" run src\SSN_Tools.py

:: Keep window open on error or exit
if %ERRORLEVEL% neq 0 (
    echo Application exited with code %ERRORLEVEL%.
    pause
)
