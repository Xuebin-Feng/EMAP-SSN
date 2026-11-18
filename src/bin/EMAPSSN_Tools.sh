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

# =========================================================================
# Portable Startup Script for EMAPSSN_Tools.py (Linux/macOS)
# =========================================================================

# Load dependency checks and desktop-launch failure handling from the installer.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$SCRIPT_DIR/../../install.sh" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
elif [ -f "$SCRIPT_DIR/install.sh" ]; then
    # Invoked through the project-root emapssn_tools symlink.
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "Could not locate install.sh from $SCRIPT_DIR." >&2
    exit 1
fi
# shellcheck source=../../install.sh
. "$PROJECT_ROOT/install.sh"
LAUNCH_MODE="${1:-}"
ssn_sanitize_managed_environment
if [ "$LAUNCH_MODE" != "--check-only" ]; then
    ssn_enable_desktop_failure_pause
fi

# Move to the project root directory.
cd "$PROJECT_ROOT"

activate_existing_instance() {
    [ -x ".venv/bin/python" ] || return 1
    ".venv/bin/python" src/desktop/Single_Instance_Probe.py tools \
        2>/dev/null
}

if { [ -z "$LAUNCH_MODE" ] || [ "$LAUNCH_MODE" = "--run-only" ]; } &&
        activate_existing_instance; then
    exit 0
fi

# 0. Prefer XWayland on Wayland sessions. The vispy OpenGL canvas hosted inside
# Qt6 is unreliable on the native Wayland platform plugin; xcb is the known-good
# path. Only applied when the user has not chosen a platform themselves, so
# `QT_QPA_PLATFORM=wayland ./emapssn_tools` still overrides this.
if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] && [ -z "${QT_QPA_PLATFORM:-}" ]; then
    export QT_QPA_PLATFORM=xcb
fi

# EMAP-SSN Tools also embeds QtWebEngine, so its preflight includes the Chromium
# runtime libraries in addition to the shared xcb dependencies.
if ! ssn_require_linux_gui_dependencies tools; then
    exit 1
fi

# 1. Locate uv. It may be missing: a ready environment starts without it, so nothing is downloaded
# until the check says setup is needed.
UV_EXE=$(ssn_find_uv) || UV_EXE=""

# 2. Validate, create, or repair the managed virtual environment
VENV_PYTHON=".venv/bin/python"
# The managed environment must run Python 3.13. A .venv on another version,
# such as one kept by an in-place upgrade, is recreated instead of reused.
VENV_PROBE='import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)'
if [ "$LAUNCH_MODE" = "--check-only" ]; then
    if [ ! -x "$VENV_PYTHON" ]; then
        echo "Setup is needed: the Python environment (.venv) is missing." >&2
        exit 10
    fi
    if ! "$VENV_PYTHON" -c "$VENV_PROBE" >/dev/null 2>&1; then
        echo "Setup is needed: the Python environment (.venv) does not run Python 3.13." >&2
        exit 10
    fi
    ssn_check_environment "$VENV_PYTHON" "$UV_EXE"
    exit $?
fi

if [ "$LAUNCH_MODE" = "--run-only" ]; then
    ssn_wait_for_dependency_setup "$PROJECT_ROOT" || exit 1
    if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "$VENV_PROBE" >/dev/null 2>&1; then
        printf 'Managed Python is unavailable; run setup before --run-only.\n' >&2
        exit 10
    fi
    # On macOS the window runs through a link named for it, which the Dock shows (ssn_window_python, install.sh).
    exec "$(ssn_window_python "$PROJECT_ROOT" tools)" src/EMAPSSN_Tools.py
fi

# Healthy direct launches take the same read-only fast path as desktop launches.
# --setup-only callers have just run --check-only, so it is not repeated.
environment_ready=0
if [ "$LAUNCH_MODE" != "--setup-only" ] &&
        [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "$VENV_PROBE" >/dev/null 2>&1; then
    ssn_check_environment "$VENV_PYTHON" "$UV_EXE"
    check_status=$?
    if [ "$check_status" -eq 0 ]; then
        environment_ready=1
    elif [ "$check_status" -eq 11 ]; then
        # macOS hid the environment's files. Setup cannot repair that, and the
        # check has printed what to do.
        exit 1
    fi
fi

if [ "$environment_ready" -ne 1 ]; then
    ssn_acquire_dependency_setup_lock "$PROJECT_ROOT" || exit 1

    # Another launcher may have completed setup while this process waited. The
    # earlier check already printed why setup is needed, and the installer
    # prints its own detection, so this re-check stays quiet.
    if [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "$VENV_PROBE" >/dev/null 2>&1 &&
            ssn_check_environment "$VENV_PYTHON" "$UV_EXE" >/dev/null 2>&1; then
        printf 'Dependency setup was completed by another launcher.\n'
    else
        # Only setup needs uv: this is the one place it is downloaded.
        if [ -z "$UV_EXE" ]; then
            ssn_install_uv || exit 1
        fi
        if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "$VENV_PROBE" >/dev/null 2>&1; then
            echo "Creating isolated local virtual environment (.venv)..."
            # uv's own Python first, so that updating or removing another Python
            # (Anaconda, Homebrew) cannot break the environment.
            if ! UV_PYTHON_PREFERENCE=only-managed "$UV_EXE" venv --clear --python 3.13; then
                echo "uv's own Python 3.13 is not available; using a Python 3.13 found on this computer."
                "$UV_EXE" venv --clear --python 3.13 || exit 1
            fi
        fi

        # 3. Resolve base, ESM, and hardware-specific PyTorch dependencies.
        echo "Detecting hardware and synchronizing dependencies..."
        if ! "$VENV_PYTHON" src/Install_Dependencies.py --uv-executable "$UV_EXE" --venv .venv; then
            echo "Dependency installation failed."
            exit 1
        fi
        echo ""
    fi
    ssn_release_dependency_setup_lock
    # A new or rebuilt environment has no window links yet (macOS only; elsewhere this prints plain python).
    ssn_window_python "$PROJECT_ROOT" viewer >/dev/null
    ssn_window_python "$PROJECT_ROOT" tools >/dev/null
fi

if [ "$LAUNCH_MODE" = "--setup-only" ]; then
    exit 0
fi

# 4. Run the tools
if activate_existing_instance; then
    exit 0
fi
echo "Starting EMAPSSN_Tools..."
"$(ssn_window_python "$PROJECT_ROOT" tools)" src/EMAPSSN_Tools.py
