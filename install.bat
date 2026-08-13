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
REM Installation and Shortcut Generation Script for SSN Viewer & Tools
REM =========================================================================
setlocal EnableDelayedExpansion

:: Move to the directory containing this batch script (project root)
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

echo Setting up shortcuts for SSN Viewer and SSN Tools...
echo Project root: !PROJECT_ROOT!

:: 1. Auto-generate .ico files from .png if they are missing
if not exist "src\bin\logos\viewer_logo_large.ico" (
    echo Icon files are missing. Attempting to generate them...
    set "PY_EXE=.venv\Scripts\python.exe"
    if exist "!PY_EXE!" (
        "!PY_EXE!" -c "import sys, os; from PySide6.QtWidgets import QApplication; from PySide6.QtGui import QPixmap; app = QApplication(sys.argv); logo_dir = r'src/bin/logos'; [QPixmap(os.path.join(logo_dir, f)).save(os.path.join(logo_dir, os.path.splitext(f)[0] + '.ico'), 'ICO') for f in os.listdir(logo_dir) if f.endswith('.png')]"
        echo [OK] Generated .ico files from png files.
    ) else (
        echo [INFO] Python virtual environment not found. Please run SSN_Viewer.bat or SSN_Tools.bat once first to set up the environment, or ensure the .ico files are synced.
    )
)

:: 2. Define icon paths
set "VIEWER_ICON=!PROJECT_ROOT!\src\bin\logos\viewer_logo_large.ico"
set "TOOL_ICON=!PROJECT_ROOT!\src\bin\logos\tool_logo_large.ico"

:: 2. Create a visible-startup Windows shortcut for SSN_Config.py
echo Creating shortcut for SSN_Viewer...
powershell -ExecutionPolicy Bypass -Command "$q = [char]34; $WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut($env:PROJECT_ROOT + '\SSN_Viewer.lnk'); $Shortcut.TargetPath = $env:WINDIR + '\System32\cmd.exe'; $Shortcut.Arguments = '/d /c ' + $q + $q + $env:PROJECT_ROOT + '\src\bin\SSN_Desktop_Launcher.bat' + $q + ' viewer' + $q; $Shortcut.WorkingDirectory = $env:PROJECT_ROOT; $Shortcut.IconLocation = $env:VIEWER_ICON; $Shortcut.Save();"
if exist "SSN_Viewer.lnk" (
    echo [OK] Created SSN_Viewer.lnk in project root.
)

:: 3. Create a visible-startup Windows shortcut for SSN_Tools.py
echo Creating shortcut for SSN_Tools...
powershell -ExecutionPolicy Bypass -Command "$q = [char]34; $WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut($env:PROJECT_ROOT + '\SSN_Tools.lnk'); $Shortcut.TargetPath = $env:WINDIR + '\System32\cmd.exe'; $Shortcut.Arguments = '/d /c ' + $q + $q + $env:PROJECT_ROOT + '\src\bin\SSN_Desktop_Launcher.bat' + $q + ' tools' + $q; $Shortcut.WorkingDirectory = $env:PROJECT_ROOT; $Shortcut.IconLocation = $env:TOOL_ICON; $Shortcut.Save();"
if exist "SSN_Tools.lnk" (
    echo [OK] Created SSN_Tools.lnk in project root.
)

:: 4. Optional: Copy shortcuts to Desktop
echo.
choice /M "Would you like to copy these shortcuts to your Desktop"
if %ERRORLEVEL% equ 1 (
    for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('CommonDesktopDirectory')"`) do set "COMMON_DESKTOP=%%i"
    for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "USER_DESKTOP=%%i"
    
    echo Copying shortcuts to Public Desktop at !COMMON_DESKTOP!...
    copy "!PROJECT_ROOT!\SSN_Viewer.lnk" "!COMMON_DESKTOP!\SSN_Viewer.lnk" /Y >nul 2>nul
    
    if !ERRORLEVEL! neq 0 (
        echo [INFO] Writing to Public Desktop requires Administrator privileges - Access Denied.
        echo Falling back: Copying shortcuts to your personal Desktop at !USER_DESKTOP!...
        copy "!PROJECT_ROOT!\SSN_Viewer.lnk" "!USER_DESKTOP!\SSN_Viewer.lnk" /Y >nul
        copy "!PROJECT_ROOT!\SSN_Tools.lnk" "!USER_DESKTOP!\SSN_Tools.lnk" /Y >nul
        echo [OK] Shortcuts successfully copied to your personal Desktop!
    ) else (
        copy "!PROJECT_ROOT!\SSN_Tools.lnk" "!COMMON_DESKTOP!\SSN_Tools.lnk" /Y >nul 2>nul
        echo [OK] Shortcuts successfully copied to the Public Desktop!
    )
)

echo.
echo Setup Complete! You can now run SSN Viewer and Tools using the root-level shortcuts.
pause
