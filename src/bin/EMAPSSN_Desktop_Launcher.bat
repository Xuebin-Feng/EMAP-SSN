@echo off
REM Visible desktop startup terminal for EMAP-SSN Configuration and EMAP-SSN Tools.
setlocal EnableDelayedExpansion
call :SANITIZE_MANAGED_ENVIRONMENT

set "APP_KIND=%~1"
if /I "%APP_KIND%"=="viewer" (
    set "APP_LABEL=EMAP-SSN Configuration"
    set "PORTABLE_LAUNCHER=%~dp0EMAPSSN.bat"
) else if /I "%APP_KIND%"=="tools" (
    set "APP_LABEL=EMAP-SSN Tools"
    set "PORTABLE_LAUNCHER=%~dp0EMAPSSN_Tools.bat"
) else (
    echo Usage: %~nx0 viewer^|tools
    pause
    exit /b 2
)

cd /d "%~dp0..\.."
call :ACTIVATE_EXISTING_INSTANCE
if !ERRORLEVEL! equ 0 exit /b 0

echo Starting !APP_LABEL!...
echo Detecting hardware and validating dependencies...
call "!PORTABLE_LAUNCHER!" --check-only
if !ERRORLEVEL! neq 0 (
    echo.
    echo Setup or repair is required. The terminal will remain visible.
    call "!PORTABLE_LAUNCHER!" --setup-only
    if !ERRORLEVEL! neq 0 (
        echo.
        echo SSN setup failed. Review the errors above.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Expected .venv\Scripts\python.exe after setup, but it was not found.
    pause
    exit /b 1
)

call :ACTIVATE_EXISTING_INSTANCE
if !ERRORLEVEL! equ 0 exit /b 0

for /f %%I in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('N')"') do set "LAUNCH_TOKEN=%%I"
set "STATE_DIR=%CD%\temp\%APP_KIND%_!LAUNCH_TOKEN!"
mkdir "!STATE_DIR!" >nul 2>nul
if not exist "!STATE_DIR!" (
    echo Could not create launcher state directory: !STATE_DIR!
    pause
    exit /b 1
)

echo Launching the Qt window...
".venv\Scripts\python.exe" -u "src\utilities\Desktop_Launcher_Monitor.py" --launch-and-wait "%APP_KIND%" "!STATE_DIR!"
set "LAUNCH_RESULT=!ERRORLEVEL!"
if !LAUNCH_RESULT! equ 0 goto GUI_READY
if !LAUNCH_RESULT! equ 20 goto GUI_EXITED
if !LAUNCH_RESULT! equ 21 goto GUI_TIMEOUT
echo.
if exist "!STATE_DIR!\application.log" type "!STATE_DIR!\application.log"
echo Failed to start the detached !APP_LABEL! monitor.
echo Launcher state retained at: !STATE_DIR!
pause
exit /b !LAUNCH_RESULT!

:GUI_READY
>"!STATE_DIR!\terminal.dismissed" echo dismissed
echo !APP_LABEL! is ready.
exit /b 0

:GUI_EXITED
set "APP_EXIT=1"
set /p APP_EXIT=<"!STATE_DIR!\application.exit"
if "!APP_EXIT!"=="0" (
    >"!STATE_DIR!\terminal.dismissed" echo dismissed
    exit /b 0
)
echo.
if exist "!STATE_DIR!\application.log" type "!STATE_DIR!\application.log"
echo.
echo !APP_LABEL! failed before its window became ready.
echo Log retained at: !STATE_DIR!\application.log
pause
exit /b !APP_EXIT!

:GUI_TIMEOUT
echo.
echo !APP_LABEL! did not report a ready window within 10 minutes.
echo The terminal will remain open. Diagnostic log:
echo !STATE_DIR!\application.log
pause
exit /b 1

:ACTIVATE_EXISTING_INSTANCE
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" "src\utilities\Single_Instance_Probe.py" "!APP_KIND!" >nul 2>nul
exit /b !ERRORLEVEL!

:SANITIZE_MANAGED_ENVIRONMENT
set "PYTHONHOME="
set "PYTHONPATH="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QML_IMPORT_PATH="
set "QML2_IMPORT_PATH="
exit /b 0
