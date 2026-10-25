@echo off
REM Copyright 2026 Xuebin Feng
REM Author affiliation: University of Toronto
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
REM
REM Builds EMAP-SSN.exe in the project folder: the launcher the shortcuts start, a few kilobytes with no console
REM window, which shows the splash within a third of a second (src\bin\Launcher.cs). The C# compiler that ships
REM with Windows builds it in a fraction of a second and nothing is downloaded.
REM
REM   Build_Launcher.bat              build it, replacing the one there (install.bat does)
REM   Build_Launcher.bat --if-needed  build it only when it is missing or older than Launcher.cs, as one built
REM                                   before a new kind of window was added is (install_vr.bat does)
REM
REM Exit status: 0 when EMAP-SSN.exe is there and not older than Launcher.cs, 1 when it is not (the compiler is
REM missing or failed): the caller then keeps the shortcuts that run the batch files in a console window.
setlocal EnableDelayedExpansion

:: The project folder is two levels up from src\bin\. (No cd: this is called from installers that stay where they are.)
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "LAUNCHER_EXE=!PROJECT_ROOT!\EMAP-SSN.exe"
set "LAUNCHER_SOURCE=!PROJECT_ROOT!\src\bin\Launcher.cs"
set "LAUNCHER_ICON=!PROJECT_ROOT!\src\bin\logos\viewer_logo_large.ico"

if /i "%~1"=="--if-needed" (
    call :LAUNCHER_IS_CURRENT
    if !ERRORLEVEL! equ 0 (
        echo [OK] EMAP-SSN.exe is up to date.
        exit /b 0
    )
)

set "CSC_EXE="
for %%C in ("%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" "%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe") do (
    if exist "%%~C" if not defined CSC_EXE set "CSC_EXE=%%~C"
)
if defined CSC_EXE (
    del /q "!LAUNCHER_EXE!" >nul 2>nul
    "!CSC_EXE!" /nologo /target:winexe /optimize+ /out:"!LAUNCHER_EXE!" /win32icon:"!LAUNCHER_ICON!" "!LAUNCHER_SOURCE!" >nul 2>nul
)
:: A launcher that could not be replaced (it was running) and is older than the source is as good as none.
call :LAUNCHER_IS_CURRENT
if !ERRORLEVEL! equ 0 (
    echo [OK] Built EMAP-SSN.exe, the launcher with no console window.
    exit /b 0
)
echo [INFO] Could not build EMAP-SSN.exe ^(the C# compiler of the .NET Framework is missing or failed^).
exit /b 1

:LAUNCHER_IS_CURRENT
if not exist "!LAUNCHER_EXE!" exit /b 1
powershell -NoProfile -Command "if ((Get-Item -LiteralPath $env:LAUNCHER_EXE).LastWriteTimeUtc -ge (Get-Item -LiteralPath $env:LAUNCHER_SOURCE).LastWriteTimeUtc) { exit 0 } else { exit 1 }"
exit /b !ERRORLEVEL!
