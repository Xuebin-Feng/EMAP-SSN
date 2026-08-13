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
REM Portable Startup Script for SSN_Config.py (SSN Viewer Entrypoint)
REM =========================================================================
setlocal EnableDelayedExpansion
set "LAUNCH_MODE=%~1"

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

if /I "%LAUNCH_MODE%"=="--check-only" exit /b 10

echo uv package manager not found. Installing it automatically...
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

:UV_FOUND

:: 2. Create or repair the managed virtual environment
set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "!VENV_PYTHON!" goto CREATE_VENV
"!VENV_PYTHON!" -c "import sys" >nul 2>nul
if !ERRORLEVEL! equ 0 goto VENV_READY

:CREATE_VENV
if /I "%LAUNCH_MODE%"=="--check-only" exit /b 10
echo Creating isolated local virtual environment .venv...
"!UV_EXE!" venv --clear --python 3.12
if !ERRORLEVEL! neq 0 exit /b !ERRORLEVEL!

:VENV_READY

if /I "%LAUNCH_MODE%"=="--check-only" (
    "!VENV_PYTHON!" src\Install_Dependencies.py --check-only --uv-executable "!UV_EXE!" --venv .venv
    exit /b !ERRORLEVEL!
)

:: 3. Resolve base, ESM, and hardware-specific PyTorch dependencies
echo Detecting hardware and synchronizing dependencies...
"!VENV_PYTHON!" src\Install_Dependencies.py --uv-executable "!UV_EXE!" --venv .venv
set "INSTALL_ERROR=!ERRORLEVEL!"
if !INSTALL_ERROR! neq 0 (
    echo Dependency installation failed.
    pause
    exit /b !INSTALL_ERROR!
)
echo.

if /I "%LAUNCH_MODE%"=="--setup-only" exit /b 0

:: 4. Run the configuration tool
echo Starting SSN_Config...
"!VENV_PYTHON!" src\SSN_Config.py

:: Keep window open on error or exit
if %ERRORLEVEL% neq 0 (
    echo Application exited with code %ERRORLEVEL%.
    pause
)
