#!/bin/bash
# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

# Bootstrap-safe terminal selection for desktop launchers. This file must not
# depend on Python because it runs before the managed virtual environment exists.

ssn_terminal_supported_names() {
    printf '%s' 'xdg-terminal-exec, x-terminal-emulator, ptyxis, gnome-terminal, kgx, konsole, xfce4-terminal, mate-terminal, kitty, alacritty, wezterm, foot, footclient, tilix, terminator, qterminal, lxterminal, xterm, urxvt, rxvt, st'
}

ssn_launch_in_terminal() {
    local working_directory title terminal command_text escaped
    local -a child
    working_directory="$PWD"
    title=""

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --cwd)
                [ "$#" -ge 2 ] || {
                    printf '%s\n' 'SSN terminal launcher: --cwd requires a value.' >&2
                    return 2
                }
                working_directory=$2
                shift 2
                ;;
            --title)
                [ "$#" -ge 2 ] || {
                    printf '%s\n' 'SSN terminal launcher: --title requires a value.' >&2
                    return 2
                }
                title=$2
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                printf 'SSN terminal launcher: unknown option: %s\n' "$1" >&2
                return 2
                ;;
        esac
    done

    [ "$#" -gt 0 ] || {
        printf '%s\n' 'SSN terminal launcher: a command is required after --.' >&2
        return 2
    }

    child=(
        bash -c
        'cd "$1" || exit; title=$2; shift 2; if [ -n "$title" ]; then printf "\033]0;%s\007" "$title"; fi; exec "$@"'
        ssn-terminal "$working_directory" "$title" "$@"
    )

    if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
        printf -v command_text '%q ' "${child[@]}"
        # Terminal's `do script` is asynchronous.  Its `busy` property can
        # still be false immediately after the command is submitted, which
        # previously made us close the new window before the command started.
        # Keep the tab busy long enough for AppleScript to observe the state,
        # and explicitly wait for that transition before waiting for exit.
        command_text="/bin/sleep 1; $command_text"
        escaped=${command_text//\\/\\\\}
        escaped=${escaped//\"/\\\"}
        osascript \
            -e 'tell application "Terminal"' \
            -e 'activate' \
            -e 'set launchTab to do script ""' \
            -e 'set launchWindow to front window' \
            -e "do script \"$escaped\" in launchTab" \
            -e 'set launchStarted to false' \
            -e 'repeat with launchAttempt from 1 to 100' \
            -e 'if busy of launchTab then' \
            -e 'set launchStarted to true' \
            -e 'exit repeat' \
            -e 'end if' \
            -e 'delay 0.05' \
            -e 'end repeat' \
            -e 'if not launchStarted then error "Terminal did not start the SSN command." number 1' \
            -e 'repeat while busy of launchTab' \
            -e 'delay 0.05' \
            -e 'end repeat' \
            -e 'try' \
            -e 'close launchWindow' \
            -e 'end try' \
            -e 'end tell' >/dev/null
        return $?
    fi

    for terminal in \
        xdg-terminal-exec x-terminal-emulator ptyxis gnome-terminal kgx \
        konsole xfce4-terminal mate-terminal kitty alacritty wezterm foot \
        footclient tilix terminator qterminal lxterminal xterm urxvt rxvt st
    do
        command -v "$terminal" >/dev/null 2>&1 || continue
        case "$terminal" in
            xdg-terminal-exec|ptyxis|gnome-terminal|kgx|kitty)
                "$terminal" -- "${child[@]}" >/dev/null 2>&1 &
                ;;
            xfce4-terminal|mate-terminal|terminator)
                "$terminal" -x "${child[@]}" >/dev/null 2>&1 &
                ;;
            wezterm)
                "$terminal" start -- "${child[@]}" >/dev/null 2>&1 &
                ;;
            qterminal|lxterminal)
                printf -v command_text '%q ' "${child[@]}"
                "$terminal" -e "$command_text" >/dev/null 2>&1 &
                ;;
            *)
                "$terminal" -e "${child[@]}" >/dev/null 2>&1 &
                ;;
        esac
        return 0
    done

    printf 'No supported terminal emulator was found. Install or configure one of: %s.\n' \
        "$(ssn_terminal_supported_names)" >&2
    return 127
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    ssn_launch_in_terminal "$@"
fi
