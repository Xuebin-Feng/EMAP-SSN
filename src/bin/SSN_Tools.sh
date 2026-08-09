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

# Move to the project root directory (two levels up from this script)
cd "$(dirname "$0")/../.."

# 0. Prefer XWayland on Wayland sessions. The vispy OpenGL canvas hosted inside
# Qt6 is unreliable on the native Wayland platform plugin; xcb is the known-good
# path. Only applied when the user has not chosen a platform themselves, so
# `QT_QPA_PLATFORM=wayland ./SSN_Tools` still overrides this.
if [ "$XDG_SESSION_TYPE" = "wayland" ] && [ -z "$QT_QPA_PLATFORM" ]; then
    export QT_QPA_PLATFORM=xcb
fi

# 1. Locate uv executable
if command -v uv &> /dev/null; then
    UV_EXE="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_EXE="$HOME/.local/bin/uv"
else
    echo "uv package manager not found. Installing it automatically..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_EXE="$HOME/.local/bin/uv"
fi

# 2. Create or repair the managed virtual environment
VENV_PYTHON=".venv/bin/python"
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "import sys" >/dev/null 2>&1; then
    echo "Creating isolated local virtual environment (.venv)..."
    "$UV_EXE" venv --clear --python 3.12 || exit 1
fi

# 3. Resolve base, ESM, and hardware-specific PyTorch dependencies
echo "Detecting hardware and synchronizing dependencies..."
if ! "$VENV_PYTHON" src/Install_Dependencies.py --uv-executable "$UV_EXE" --venv .venv; then
    echo "Dependency installation failed."
    exit 1
fi
echo ""

# 4. Run the tools
echo "Starting SSN_Tools..."
"$VENV_PYTHON" src/SSN_Tools.py
