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
# Installation and Launcher Generation Script for SSN Viewer & Tools (Linux)
# =========================================================================

# Move to the directory containing this script (project root)
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo "Setting up launchers for SSN Viewer & SSN Tools on Linux..."
echo "Project root: $PROJECT_ROOT"

# 1. Make sure all executables in src/bin have execution permissions
chmod +x src/bin/*.sh
echo "[OK] Configured execution permissions for scripts in src/bin/"

# 2. Create symbolic links in the project root pointing to the actual launchers
ln -sf src/bin/SSN_Viewer.sh SSN_Viewer
ln -sf src/bin/SSN_Tools.sh SSN_Tools
chmod +x SSN_Viewer SSN_Tools
echo "[OK] Created SSN_Viewer and SSN_Tools executables in project root."

# 3. Generate .desktop entry launchers
echo ""
echo "Generating .desktop entry launchers..."

# Use viewer_logo_large.png and tool_logo_large.png as icons
VIEWER_ICON="${PROJECT_ROOT}/src/bin/logos/viewer_logo_large.png"
TOOL_ICON="${PROJECT_ROOT}/src/bin/logos/tool_logo_large.png"

# Create SSN_Viewer.desktop
cat <<EOF > SSN_Viewer.desktop
[Desktop Entry]
Type=Application
Name=SSN Viewer
Comment=Sequence Similarity Network Viewer
Exec="${PROJECT_ROOT}/src/bin/SSN_Viewer.sh"
Path=${PROJECT_ROOT}
Icon=${VIEWER_ICON}
Terminal=true
Categories=Science;Biology;
EOF
chmod +x SSN_Viewer.desktop

# Create SSN_Tools.desktop
cat <<EOF > SSN_Tools.desktop
[Desktop Entry]
Type=Application
Name=SSN Tools
Comment=Sequence Similarity Network Utilities
Exec="${PROJECT_ROOT}/src/bin/SSN_Tools.sh"
Path=${PROJECT_ROOT}
Icon=${TOOL_ICON}
Terminal=true
Categories=Science;Biology;
EOF
chmod +x SSN_Tools.desktop

echo "[OK] Created SSN_Viewer.desktop and SSN_Tools.desktop in project root."

# Ask if they want to copy desktop files to Applications menu
read -p "Would you like to install these launchers to your system application menu? (y/n): " install_menu
if [[ "$install_menu" =~ ^[Yy]$ ]]; then
    mkdir -p ~/.local/share/applications
    cp SSN_Viewer.desktop ~/.local/share/applications/
    cp SSN_Tools.desktop ~/.local/share/applications/
    echo "[OK] Launchers successfully added to your system applications menu!"
fi

# Ask if they want to copy desktop files to Desktop
if [ -d "$HOME/Desktop" ]; then
    read -p "Would you like to copy these launchers to your Desktop? (y/n): " install_desktop
    if [[ "$install_desktop" =~ ^[Yy]$ ]]; then
        cp SSN_Viewer.desktop "$HOME/Desktop/"
        cp SSN_Tools.desktop "$HOME/Desktop/"
        chmod +x "$HOME/Desktop/SSN_Viewer.desktop" "$HOME/Desktop/SSN_Tools.desktop"
        echo "[OK] Launchers successfully copied to your Desktop!"
    fi
fi

echo ""
echo "Setup Complete! You can now run SSN Viewer and Tools using the launchers in the project root."
