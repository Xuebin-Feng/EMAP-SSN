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

import subprocess
import sys
import platform

def detect_gpu():
    # 1. First, check for NVIDIA via nvidia-smi (fastest and standard for CUDA)
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
        if res.returncode == 0:
            return "NVIDIA"
    except Exception:
        pass

    system = platform.system().lower()

    # 2. Windows specific controller search
    if system == "windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                gpu_names = res.stdout.lower()
                if "nvidia" in gpu_names:
                    return "NVIDIA"
                elif "intel" in gpu_names:
                    return "INTEL"
                elif "amd" in gpu_names or "radeon" in gpu_names:
                    return "AMD"
        except Exception:
            pass

    # 3. Linux specific controller search
    elif system == "linux":
        try:
            res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                gpu_names = res.stdout.lower()
                if "nvidia" in gpu_names:
                    return "NVIDIA"
                elif "intel" in gpu_names:
                    return "INTEL"
                elif "amd" in gpu_names or "radeon" in gpu_names:
                    return "AMD"
        except Exception:
            pass

    # 4. macOS specific check
    elif system == "darwin":
        if platform.machine() == "arm64":
            return "MPS"

    return "CPU"

if __name__ == "__main__":
    print(detect_gpu())
