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
call :SANITIZE_MANAGED_ENVIRONMENT

:: Move to the project root directory (two levels up from this script)
cd /d "%~dp0..\.."

if "%LAUNCH_MODE%"=="" call :ACTIVATE_EXISTING_INSTANCE
if "%LAUNCH_MODE%"=="" if !ERRORLEVEL! equ 0 exit /b 0
if /I "%LAUNCH_MODE%"=="--run-only" call :ACTIVATE_EXISTING_INSTANCE
if /I "%LAUNCH_MODE%"=="--run-only" if !ERRORLEVEL! equ 0 exit /b 0

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

:: 2. Validate, create, or repair the managed virtual environment
set "VENV_PYTHON=.venv\Scripts\python.exe"
if /I "%LAUNCH_MODE%"=="--check-only" (
    call :ENVIRONMENT_READY
    exit /b !ERRORLEVEL!
)

if /I "%LAUNCH_MODE%"=="--run-only" (
    "%COMSPEC%" /d /c ""%~f0" --wait-for-setup"
    if !ERRORLEVEL! neq 0 exit /b 1
    if not exist "!VENV_PYTHON!" exit /b 10
    "!VENV_PYTHON!" -c "import sys" >nul 2>nul
    if !ERRORLEVEL! neq 0 exit /b 10
    goto RUN_APPLICATION
)

if /I "%LAUNCH_MODE%"=="--wait-for-setup" (
    call :ACQUIRE_SETUP_LOCK
    if !ERRORLEVEL! neq 0 exit /b 1
    call :RELEASE_SETUP_LOCK
    exit /b 0
)

:: Healthy direct launches take the same read-only fast path as desktop launches.
call :ENVIRONMENT_READY
if !ERRORLEVEL! equ 0 goto SETUP_COMPLETE

if /I not "%LAUNCH_MODE%"=="--locked-setup" (
    "%COMSPEC%" /d /c ""%~f0" --locked-setup"
    if !ERRORLEVEL! neq 0 exit /b !ERRORLEVEL!
    goto SETUP_COMPLETE
)

call :ACQUIRE_SETUP_LOCK
if !ERRORLEVEL! neq 0 exit /b 1

:: Another launcher may have completed setup while this process waited.
call :ENVIRONMENT_READY
if !ERRORLEVEL! equ 0 (
    echo Dependency setup was completed by another launcher.
    call :RELEASE_SETUP_LOCK
    exit /b 0
)

if not exist "!VENV_PYTHON!" goto CREATE_VENV
"!VENV_PYTHON!" -c "import sys" >nul 2>nul
if !ERRORLEVEL! equ 0 goto INSTALL_DEPENDENCIES

:CREATE_VENV
echo Creating isolated local virtual environment .venv...
"!UV_EXE!" venv --clear --python 3.12
if !ERRORLEVEL! neq 0 (
    set "INSTALL_ERROR=!ERRORLEVEL!"
    call :RELEASE_SETUP_LOCK
    exit /b !INSTALL_ERROR!
)

:INSTALL_DEPENDENCIES
:: 3. Resolve base, ESM, and hardware-specific PyTorch dependencies.
echo Detecting hardware and synchronizing dependencies...
"!VENV_PYTHON!" src\Install_Dependencies.py --uv-executable "!UV_EXE!" --venv .venv
set "INSTALL_ERROR=!ERRORLEVEL!"
if !INSTALL_ERROR! neq 0 (
    echo Dependency installation failed.
    call :RELEASE_SETUP_LOCK
    pause
    exit /b !INSTALL_ERROR!
)
echo.
call :RELEASE_SETUP_LOCK
if /I "%LAUNCH_MODE%"=="--locked-setup" exit /b 0

:SETUP_COMPLETE
if /I "%LAUNCH_MODE%"=="--setup-only" exit /b 0

:: 4. Run the configuration tool
:RUN_APPLICATION
call :ACTIVATE_EXISTING_INSTANCE
if !ERRORLEVEL! equ 0 exit /b 0
echo Starting SSN_Config...
"!VENV_PYTHON!" src\SSN_Config.py
set "APP_EXIT=!ERRORLEVEL!"

:: Keep window open on error or exit
if !APP_EXIT! neq 0 (
    echo Application exited with code !APP_EXIT!.
    pause
)
exit /b !APP_EXIT!

:ACTIVATE_EXISTING_INSTANCE
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" "src\utilities\Single_Instance_Probe.py" viewer >nul 2>nul
exit /b !ERRORLEVEL!

:SANITIZE_MANAGED_ENVIRONMENT
set "PYTHONHOME="
set "PYTHONPATH="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QML_IMPORT_PATH="
set "QML2_IMPORT_PATH="
exit /b 0

:ENVIRONMENT_READY
if not exist "!VENV_PYTHON!" exit /b 10
"!VENV_PYTHON!" -c "import sys" >nul 2>nul
if !ERRORLEVEL! neq 0 exit /b 10
"!VENV_PYTHON!" src\Install_Dependencies.py --check-only --uv-executable "!UV_EXE!" --venv .venv
exit /b !ERRORLEVEL!

:GET_CURRENT_PROCESS_ID
set "SSN_SETUP_OWNER_PID="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID); $process.ParentProcessId"`) do set "SSN_SETUP_OWNER_PID=%%P"
if not defined SSN_SETUP_OWNER_PID exit /b 1
exit /b 0

:ACQUIRE_SETUP_LOCK
set "SSN_SETUP_LOCK_ROOT=%CD%\temp"
set "SSN_SETUP_LOCK_DIR=!SSN_SETUP_LOCK_ROOT!\dependency_setup.lock"
set "SSN_SETUP_LOCK_OWNED=0"
set /a SSN_SETUP_WAIT_COUNT=0
set /a SSN_SETUP_MISSING_OWNER_COUNT=0
set /a SSN_SETUP_WAIT_REPORTED=0
if not exist "!SSN_SETUP_LOCK_ROOT!" mkdir "!SSN_SETUP_LOCK_ROOT!" >nul 2>nul
if not exist "!SSN_SETUP_LOCK_ROOT!" (
    echo Could not create launcher state directory: !SSN_SETUP_LOCK_ROOT!
    exit /b 1
)
call :GET_CURRENT_PROCESS_ID
if !ERRORLEVEL! neq 0 (
    echo Could not determine dependency setup lock owner process.
    exit /b 1
)

:TRY_SETUP_LOCK
mkdir "!SSN_SETUP_LOCK_DIR!" >nul 2>nul
if !ERRORLEVEL! equ 0 (
    >"!SSN_SETUP_LOCK_DIR!\.owner.!SSN_SETUP_OWNER_PID!.tmp" echo !SSN_SETUP_OWNER_PID!
    move /y "!SSN_SETUP_LOCK_DIR!\.owner.!SSN_SETUP_OWNER_PID!.tmp" "!SSN_SETUP_LOCK_DIR!\owner.pid" >nul
    if !ERRORLEVEL! neq 0 (
        del /q "!SSN_SETUP_LOCK_DIR!\.owner.!SSN_SETUP_OWNER_PID!.tmp" >nul 2>nul
        rmdir "!SSN_SETUP_LOCK_DIR!" >nul 2>nul
        echo Could not record dependency setup lock ownership at: !SSN_SETUP_LOCK_DIR!
        exit /b 1
    )
    set "SSN_SETUP_LOCK_OWNED=1"
    exit /b 0
)

set "SSN_EXISTING_LOCK_PID="
if exist "!SSN_SETUP_LOCK_DIR!\owner.pid" set /p SSN_EXISTING_LOCK_PID=<"!SSN_SETUP_LOCK_DIR!\owner.pid"
if not defined SSN_EXISTING_LOCK_PID (
    set /a SSN_SETUP_MISSING_OWNER_COUNT+=1
    if !SSN_SETUP_MISSING_OWNER_COUNT! geq 5 (
        del /q "!SSN_SETUP_LOCK_DIR!\owner.pid" "!SSN_SETUP_LOCK_DIR!\.owner.*.tmp" >nul 2>nul
        rmdir "!SSN_SETUP_LOCK_DIR!" >nul 2>nul
        if not exist "!SSN_SETUP_LOCK_DIR!" echo Recovered an incomplete dependency setup lock.
        set /a SSN_SETUP_MISSING_OWNER_COUNT=0
        goto TRY_SETUP_LOCK
    )
) else (
    set /a SSN_SETUP_MISSING_OWNER_COUNT=0
    set "SSN_LOCK_OWNER_PID=!SSN_EXISTING_LOCK_PID!"
    powershell -NoProfile -Command "$value = 0; if ([int]::TryParse($env:SSN_LOCK_OWNER_PID, [ref]$value) -and (Get-Process -Id $value -ErrorAction SilentlyContinue)) { exit 0 }; exit 1" >nul 2>nul
    if !ERRORLEVEL! neq 0 (
        del /q "!SSN_SETUP_LOCK_DIR!\owner.pid" "!SSN_SETUP_LOCK_DIR!\.owner.*.tmp" >nul 2>nul
        rmdir "!SSN_SETUP_LOCK_DIR!" >nul 2>nul
        if not exist "!SSN_SETUP_LOCK_DIR!" echo Recovered stale dependency setup lock from process !SSN_EXISTING_LOCK_PID!.
        goto TRY_SETUP_LOCK
    )
    if !SSN_SETUP_WAIT_REPORTED! equ 0 (
        echo Dependency setup is already running in process !SSN_EXISTING_LOCK_PID!; waiting...
        set /a SSN_SETUP_WAIT_REPORTED=1
    )
)

if !SSN_SETUP_WAIT_COUNT! geq 3600 (
    echo Timed out waiting for dependency setup owned by process !SSN_EXISTING_LOCK_PID!.
    echo Lock retained at: !SSN_SETUP_LOCK_DIR!
    exit /b 1
)
timeout /t 1 /nobreak >nul
set /a SSN_SETUP_WAIT_COUNT+=1
goto TRY_SETUP_LOCK

:RELEASE_SETUP_LOCK
if not "!SSN_SETUP_LOCK_OWNED!"=="1" exit /b 0
set "SSN_EXISTING_LOCK_PID="
if exist "!SSN_SETUP_LOCK_DIR!\owner.pid" set /p SSN_EXISTING_LOCK_PID=<"!SSN_SETUP_LOCK_DIR!\owner.pid"
if not defined SSN_EXISTING_LOCK_PID (
    rmdir "!SSN_SETUP_LOCK_DIR!" >nul 2>nul
) else if "!SSN_EXISTING_LOCK_PID!"=="!SSN_SETUP_OWNER_PID!" (
    del /q "!SSN_SETUP_LOCK_DIR!\owner.pid" >nul 2>nul
    rmdir "!SSN_SETUP_LOCK_DIR!" >nul 2>nul
)
set "SSN_SETUP_LOCK_OWNED=0"
exit /b 0
