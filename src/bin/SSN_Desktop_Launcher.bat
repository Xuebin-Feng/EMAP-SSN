@echo off
REM Visible desktop startup terminal for SSN Config and SSN Tools.
setlocal EnableDelayedExpansion
call :SANITIZE_MANAGED_ENVIRONMENT

set "APP_KIND=%~1"
if /I "%APP_KIND%"=="viewer" (
    set "APP_LABEL=SSN Config"
    set "PORTABLE_LAUNCHER=%~dp0SSN_Viewer.bat"
) else if /I "%APP_KIND%"=="tools" (
    set "APP_LABEL=SSN Tools"
    set "PORTABLE_LAUNCHER=%~dp0SSN_Tools.bat"
) else (
    echo Usage: %~nx0 viewer^|tools
    pause
    exit /b 2
)

cd /d "%~dp0..\.."
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

if not exist ".venv\Scripts\pythonw.exe" (
    echo Expected .venv\Scripts\pythonw.exe after setup, but it was not found.
    pause
    exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('N')"') do set "LAUNCH_TOKEN=%%I"
set "STATE_DIR=%CD%\Cache_Files\Launcher_State\%APP_KIND%_!LAUNCH_TOKEN!"
mkdir "!STATE_DIR!" >nul 2>nul
if not exist "!STATE_DIR!" (
    echo Could not create launcher state directory: !STATE_DIR!
    pause
    exit /b 1
)

echo Launching the Qt window...
start "" /b ".venv\Scripts\pythonw.exe" "src\utilities\Desktop_Launcher_Monitor.py" "%APP_KIND%" "!STATE_DIR!"

set /a WAIT_COUNT=0
:WAIT_FOR_GUI
if exist "!STATE_DIR!\gui.ready" goto GUI_READY
if exist "!STATE_DIR!\application.exit" goto GUI_EXITED
if !WAIT_COUNT! geq 600 goto GUI_TIMEOUT
set /a WAIT_COUNT+=1
timeout /t 1 /nobreak >nul
goto WAIT_FOR_GUI

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

:SANITIZE_MANAGED_ENVIRONMENT
set "PYTHONHOME="
set "PYTHONPATH="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QML_IMPORT_PATH="
set "QML2_IMPORT_PATH="
exit /b 0
