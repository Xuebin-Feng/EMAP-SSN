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
# Portable Startup Script for SSN_Config.py (Linux/macOS)
# =========================================================================

# Move to the project root directory (two levels up from this script)
cd "$(dirname "$0")/../.."

# 0. Prefer XWayland on Wayland sessions. The vispy OpenGL canvas hosted inside
# Qt6 is unreliable on the native Wayland platform plugin; xcb is the known-good
# path. Only applied when the user has not chosen a platform themselves, so
# `QT_QPA_PLATFORM=wayland ./SSN_Viewer` still overrides this.
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

# 2. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating isolated local virtual environment (.venv)..."
    "$UV_EXE" venv --python 3.12
fi

# 3. Detect GPU type using python script
echo "Detecting hardware configuration..."
GPU_TYPE=$("$UV_EXE" run --quiet python src/Detect_GPU.py)

# 4. Resolve dependencies based on GPU Type
echo "Detected platform/GPU type: $GPU_TYPE"
echo ""

if [ "$GPU_TYPE" = "NVIDIA" ]; then
    echo "NVIDIA GPU detected. Syncing with PyTorch CUDA 13.0 support..."
    "$UV_EXE" pip install -r src/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
elif [ "$GPU_TYPE" = "INTEL" ]; then
    echo "Intel Arc/GPU detected. Syncing with PyTorch XPU (oneAPI/SYCL) support..."
    "$UV_EXE" pip install -r src/requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
elif [ "$GPU_TYPE" = "AMD" ]; then
    echo "AMD GPU detected on Linux. Syncing with PyTorch ROCm 6.1 support..."
    "$UV_EXE" pip install -r src/requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.1 --index-strategy unsafe-best-match
elif [ "$GPU_TYPE" = "MPS" ]; then
    echo "Apple Silicon detected. Syncing with macOS MPS support..."
    "$UV_EXE" pip install -r src/requirements.txt
else
    echo "No dedicated GPU detected. Syncing with CPU-only PyTorch..."
    "$UV_EXE" pip install -r src/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
fi
echo ""

# 5. Run the configuration tool
echo "Starting SSN_Config..."
"$UV_EXE" run src/SSN_Config.py
