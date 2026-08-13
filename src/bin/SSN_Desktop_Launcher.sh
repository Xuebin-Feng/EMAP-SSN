#!/bin/bash
# Visible desktop startup terminal for SSN Config and SSN Tools.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 1
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd) || exit 1
APP_KIND="${1:-}"
LAUNCH_MODE="${2:---open-terminal}"

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

terminal_command() {
    printf 'cd %q && %q %q --terminal-session' \
        "$PROJECT_ROOT" "$0" "$APP_KIND"
}

open_startup_terminal() {
    local command_text terminal escaped
    command_text=$(terminal_command)

    if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
        escaped=${command_text//\\/\\\\}
        escaped=${escaped//\"/\\\"}
        osascript \
            -e 'tell application "Terminal"' \
            -e 'activate' \
            -e 'set launchTab to do script ""' \
            -e 'set launchWindow to front window' \
            -e "do script \"$escaped\" in launchTab" \
            -e 'repeat while busy of launchTab' \
            -e 'delay 0.2' \
            -e 'end repeat' \
            -e 'try' \
            -e 'close launchWindow' \
            -e 'end try' \
            -e 'end tell' >/dev/null
        return $?
    fi

    for terminal in gnome-terminal konsole xfce4-terminal mate-terminal lxterminal kitty alacritty xterm x-terminal-emulator; do
        command -v "$terminal" >/dev/null 2>&1 || continue
        case "$terminal" in
            gnome-terminal|kitty|alacritty)
                "$terminal" -- bash -c "$command_text" >/dev/null 2>&1 &
                ;;
            konsole)
                "$terminal" -e bash -c "$command_text" >/dev/null 2>&1 &
                ;;
            *)
                "$terminal" -e bash -c "$command_text" >/dev/null 2>&1 &
                ;;
        esac
        return 0
    done

    printf 'Could not locate a supported terminal emulator for %s.\n' "$APP_LABEL" >&2
    return 1
}

if [ "$LAUNCH_MODE" = "--open-terminal" ]; then
    open_startup_terminal
    exit $?
fi
if [ "$LAUNCH_MODE" != "--terminal-session" ]; then
    printf 'Unknown desktop launcher mode: %s\n' "$LAUNCH_MODE" >&2
    exit 2
fi

cd "$PROJECT_ROOT" || exit 1
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

STATE_ROOT="$PROJECT_ROOT/Cache_Files/Launcher_State"
mkdir -p "$STATE_ROOT" || exit 1
STATE_DIR=$(mktemp -d "$STATE_ROOT/${APP_KIND}.XXXXXX") || exit 1
printf 'Launching the Qt window...\n'
nohup "$VENV_PYTHON" "$PROJECT_ROOT/src/utilities/Desktop_Launcher_Monitor.py" \
    "$APP_KIND" "$STATE_DIR" >/dev/null 2>&1 </dev/null &

wait_count=0
while [ "$wait_count" -lt 600 ]; do
    if [ -f "$STATE_DIR/gui.ready" ]; then
        printf 'dismissed\n' >"$STATE_DIR/.terminal.dismissed.tmp"
        mv -f "$STATE_DIR/.terminal.dismissed.tmp" "$STATE_DIR/terminal.dismissed"
        printf '%s is ready.\n' "$APP_LABEL"
        exit 0
    fi
    if [ -f "$STATE_DIR/application.exit" ]; then
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
    fi
    sleep 1
    wait_count=$((wait_count + 1))
done

printf '\n%s did not report a ready window within 10 minutes.\n' "$APP_LABEL" >&2
printf 'The terminal will remain open. Diagnostic log:\n%s\n' \
    "$STATE_DIR/application.log" >&2
read -r -p 'Press Enter to close...' _
exit 1
