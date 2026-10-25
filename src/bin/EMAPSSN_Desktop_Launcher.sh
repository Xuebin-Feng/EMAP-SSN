#!/bin/bash
# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Desktop start for EMAP-SSN Configuration and EMAP-SSN Tools: what the .desktop entries run.
#
#   EMAPSSN_Desktop_Launcher.sh viewer|tools                      start the window with a start-up screen, no terminal
#   EMAPSSN_Desktop_Launcher.sh viewer|tools --console            the old start, in a terminal that shows what it does
#   EMAPSSN_Desktop_Launcher.sh viewer|tools --terminal-session   what runs inside that terminal
#
# The default mode runs the Python launcher (src/desktop/Desktop_Launcher.py), which shows the start-up screen,
# checks the environment and starts the window. It opens a terminal only when it has to: the managed Python
# is not there yet (the first start), the launcher cannot run or asks for the console (exit code 3: the
# environment needs setup or repair), or it fails. --console (and its older name --open-terminal) always opens one.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 1
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd) || exit 1
APP_KIND="${1:-}"
LAUNCH_MODE="${2:-}"
TERMINAL_LAUNCHER="$SCRIPT_DIR/EMAPSSN_Terminal_Launcher.sh"

if [ ! -r "$PROJECT_ROOT/install.sh" ]; then
    printf 'Could not load launcher support: %s\n' "$PROJECT_ROOT/install.sh" >&2
    exit 1
fi
# shellcheck source=../../install.sh
. "$PROJECT_ROOT/install.sh"
ssn_sanitize_managed_environment

if [ ! -r "$TERMINAL_LAUNCHER" ]; then
    printf 'Could not load terminal launcher: %s\n' "$TERMINAL_LAUNCHER" >&2
    exit 1
fi
# shellcheck source=EMAPSSN_Terminal_Launcher.sh
. "$TERMINAL_LAUNCHER"

case "$APP_KIND" in
    viewer)
        PORTABLE_LAUNCHER="$SCRIPT_DIR/EMAPSSN.sh"
        APP_LABEL="EMAP-SSN Configuration"
        ;;
    tools)
        PORTABLE_LAUNCHER="$SCRIPT_DIR/EMAPSSN_Tools.sh"
        APP_LABEL="EMAP-SSN Tools"
        ;;
    *)
        printf 'Usage: %s viewer|tools [--console|--terminal-session]\n' "$0" >&2
        exit 2
        ;;
esac

activate_existing_instance() {
    local venv_python="$PROJECT_ROOT/.venv/bin/python"
    [ -x "$venv_python" ] || return 1
    "$venv_python" "$PROJECT_ROOT/src/desktop/Single_Instance_Probe.py" \
        "$APP_KIND" >/dev/null 2>&1
}

open_startup_terminal() {
    ssn_launch_in_terminal \
        --cwd "$PROJECT_ROOT" \
        --title "$APP_LABEL" \
        -- "$0" "$APP_KIND" --terminal-session
}

# The start without a terminal. Its exit code is the Windows stub's contract (src/bin/Launcher.cs): 0 means the
# launcher handled the start (the window opened, or its start-up screen showed what went wrong), anything else
# that nothing was shown, and the terminal start below says what is wrong or does the setup.
start_without_terminal() {
    local venv_python="$PROJECT_ROOT/.venv/bin/python"
    local launcher="$PROJECT_ROOT/src/desktop/Desktop_Launcher.py"
    local launcher_result

    [ -x "$venv_python" ] && [ -f "$launcher" ] || return 1
    cd "$PROJECT_ROOT" || return 1
    "$venv_python" -u "$launcher" "$APP_KIND"
    launcher_result=$?
    if [ "$launcher_result" -ne 0 ]; then
        printf 'The start-up screen could not start %s (exit code %s); starting it in a terminal.\n' \
            "$APP_LABEL" "$launcher_result" >&2
    fi
    return "$launcher_result"
}

if [ -z "$LAUNCH_MODE" ] && start_without_terminal; then
    exit 0
fi
if [ -z "$LAUNCH_MODE" ] || [ "$LAUNCH_MODE" = "--console" ] || [ "$LAUNCH_MODE" = "--open-terminal" ]; then
    if activate_existing_instance; then
        exit 0
    fi
    open_startup_terminal
    exit $?
fi
if [ "$LAUNCH_MODE" != "--terminal-session" ]; then
    printf 'Unknown desktop launcher mode: %s\n' "$LAUNCH_MODE" >&2
    exit 2
fi

cd "$PROJECT_ROOT" || exit 1
if activate_existing_instance; then
    exit 0
fi
printf 'Starting %s...\n' "$APP_LABEL"
printf 'Detecting hardware and validating dependencies...\n'
if ! "$PORTABLE_LAUNCHER" --check-only; then
    printf '\nSetup or repair is required. The terminal will remain visible.\n'
    if ! "$PORTABLE_LAUNCHER" --setup-only; then
        printf '\nSSN setup failed. Review the errors above.\n'
        read -r -p 'Press Enter to close...' _
        exit 1
    fi
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    printf 'Expected %s after setup, but it was not found.\n' "$VENV_PYTHON" >&2
    read -r -p 'Press Enter to close...' _
    exit 1
fi

if activate_existing_instance; then
    exit 0
fi

STATE_ROOT="$PROJECT_ROOT/temp"
mkdir -p "$STATE_ROOT" || exit 1
STATE_DIR=$(mktemp -d "$STATE_ROOT/${APP_KIND}.XXXXXX") || exit 1
printf 'Launching the Qt window...\n'
"$VENV_PYTHON" -u "$PROJECT_ROOT/src/desktop/Desktop_Launcher_Monitor.py" \
    --launch-and-wait "$APP_KIND" "$STATE_DIR"
launch_result=$?
case "$launch_result" in
    0)
        printf 'dismissed\n' >"$STATE_DIR/.terminal.dismissed.tmp"
        mv -f "$STATE_DIR/.terminal.dismissed.tmp" "$STATE_DIR/terminal.dismissed"
        printf '%s is ready.\n' "$APP_LABEL"
        exit 0
        ;;
    20)
        app_exit=$(head -n 1 "$STATE_DIR/application.exit" 2>/dev/null || printf '1')
        if [ "$app_exit" = "0" ]; then
            printf 'dismissed\n' >"$STATE_DIR/.terminal.dismissed.tmp"
            mv -f "$STATE_DIR/.terminal.dismissed.tmp" "$STATE_DIR/terminal.dismissed"
            exit 0
        fi
        printf '\n'
        [ ! -f "$STATE_DIR/application.log" ] || cat "$STATE_DIR/application.log"
        printf '\n%s failed before its window became ready.\n' "$APP_LABEL" >&2
        printf 'Log retained at: %s\n' "$STATE_DIR/application.log" >&2
        read -r -p 'Press Enter to close...' _
        exit "$app_exit"
        ;;
    21)
        printf '\n%s did not report a ready window within 10 minutes.\n' "$APP_LABEL" >&2
        printf 'The terminal will remain open. Diagnostic log:\n%s\n' \
            "$STATE_DIR/application.log" >&2
        read -r -p 'Press Enter to close...' _
        exit 1
        ;;
    *)
        printf '\n' >&2
        [ ! -f "$STATE_DIR/application.log" ] || cat "$STATE_DIR/application.log"
        printf 'Failed to start the detached %s monitor.\n' "$APP_LABEL" >&2
        printf 'Launcher state retained at: %s\n' "$STATE_DIR" >&2
        read -r -p 'Press Enter to close...' _
        exit "$launch_result"
        ;;
esac
