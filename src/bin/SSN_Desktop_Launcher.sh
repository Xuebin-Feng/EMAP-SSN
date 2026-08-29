#!/bin/bash
# Visible desktop startup terminal for SSN Config and SSN Tools.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 1
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd) || exit 1
APP_KIND="${1:-}"
LAUNCH_MODE="${2:---open-terminal}"
TERMINAL_LAUNCHER="$SCRIPT_DIR/SSN_Terminal_Launcher.sh"

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
# shellcheck source=SSN_Terminal_Launcher.sh
. "$TERMINAL_LAUNCHER"

case "$APP_KIND" in
    viewer)
        PORTABLE_LAUNCHER="$SCRIPT_DIR/SSN_Viewer.sh"
        APP_LABEL="SSN Config"
        ;;
    tools)
        PORTABLE_LAUNCHER="$SCRIPT_DIR/SSN_Tools.sh"
        APP_LABEL="SSN Tools"
        ;;
    *)
        printf 'Usage: %s viewer|tools [--open-terminal|--terminal-session]\n' "$0" >&2
        exit 2
        ;;
esac

activate_existing_instance() {
    local venv_python="$PROJECT_ROOT/.venv/bin/python"
    [ -x "$venv_python" ] || return 1
    "$venv_python" "$PROJECT_ROOT/src/utilities/Single_Instance_Probe.py" \
        "$APP_KIND" >/dev/null 2>&1
}

open_startup_terminal() {
    ssn_launch_in_terminal \
        --cwd "$PROJECT_ROOT" \
        --title "$APP_LABEL" \
        -- "$0" "$APP_KIND" --terminal-session
}

if [ "$LAUNCH_MODE" = "--open-terminal" ]; then
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

STATE_ROOT="$PROJECT_ROOT/Cache_Files/Launcher_State"
mkdir -p "$STATE_ROOT" || exit 1
STATE_DIR=$(mktemp -d "$STATE_ROOT/${APP_KIND}.XXXXXX") || exit 1
printf 'Launching the Qt window...\n'
"$VENV_PYTHON" -u "$PROJECT_ROOT/src/utilities/Desktop_Launcher_Monitor.py" \
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
