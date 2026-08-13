#!/bin/bash
# Copyright 2026 Xuebin Feng
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
# Portable Startup Script for SSN_Tools.py (Linux/macOS)
# =========================================================================

# Load dependency checks and desktop-launch failure handling from the installer.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$SCRIPT_DIR/../../install.sh" ]; then
    PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
elif [ -f "$SCRIPT_DIR/install.sh" ]; then
    # Invoked through the project-root SSN_Tools symlink.
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "Could not locate install.sh from $SCRIPT_DIR." >&2
    exit 1
fi
# shellcheck source=../../install.sh
. "$PROJECT_ROOT/install.sh"
LAUNCH_MODE="${1:-}"
if [ "$LAUNCH_MODE" != "--check-only" ]; then
    ssn_enable_desktop_failure_pause
fi

# Move to the project root directory.
cd "$PROJECT_ROOT"

# 0. Prefer XWayland on Wayland sessions. The vispy OpenGL canvas hosted inside
# Qt6 is unreliable on the native Wayland platform plugin; xcb is the known-good
# path. Only applied when the user has not chosen a platform themselves, so
# `QT_QPA_PLATFORM=wayland ./SSN_Tools` still overrides this.
if [ "$XDG_SESSION_TYPE" = "wayland" ] && [ -z "$QT_QPA_PLATFORM" ]; then
    export QT_QPA_PLATFORM=xcb
fi

# SSN Tools also embeds QtWebEngine, so its preflight includes the Chromium
# runtime libraries in addition to the shared xcb dependencies.
if ! ssn_require_linux_gui_dependencies tools; then
    exit 1
fi

# 1. Locate uv executable
if command -v uv &> /dev/null; then
    UV_EXE="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_EXE="$HOME/.local/bin/uv"
else
    if [ "$LAUNCH_MODE" = "--check-only" ]; then
        exit 10
    fi
    echo "uv package manager not found. Installing it automatically..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_EXE="$HOME/.local/bin/uv"
fi

# 2. Create or repair the managed virtual environment
VENV_PYTHON=".venv/bin/python"
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "import sys" >/dev/null 2>&1; then
    if [ "$LAUNCH_MODE" = "--check-only" ]; then
        exit 10
    fi
    echo "Creating isolated local virtual environment (.venv)..."
    "$UV_EXE" venv --clear --python 3.12 || exit 1
fi

if [ "$LAUNCH_MODE" = "--check-only" ]; then
    "$VENV_PYTHON" src/Install_Dependencies.py --check-only --uv-executable "$UV_EXE" --venv .venv
    exit $?
fi

if [ "$LAUNCH_MODE" = "--run-only" ]; then
    exec "$VENV_PYTHON" src/SSN_Tools.py
fi

# 3. Resolve base, ESM, and hardware-specific PyTorch dependencies
echo "Detecting hardware and synchronizing dependencies..."
if ! "$VENV_PYTHON" src/Install_Dependencies.py --uv-executable "$UV_EXE" --venv .venv; then
    echo "Dependency installation failed."
    exit 1
fi
echo ""

if [ "$LAUNCH_MODE" = "--setup-only" ]; then
    exit 0
fi

# 4. Run the tools
echo "Starting SSN_Tools..."
"$VENV_PYTHON" src/SSN_Tools.py
