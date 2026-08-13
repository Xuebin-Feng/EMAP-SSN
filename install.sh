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
# Linux Installation, Dependency Checks, and Launcher Support
# =========================================================================

# The launcher scripts source this file for the shared dependency checks and
# desktop-terminal failure handling. The installer itself runs only through the
# guarded ssn_install call at the end of the file.

SSN_LAUNCH_PACKAGES=(
    curl
)

SSN_XCB_PACKAGES=(
    libx11-xcb1
    libxcb1
    libxcb-cursor0
    libxcb-glx0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-randr0
    libxcb-render0
    libxcb-render-util0
    libxcb-shape0
    libxcb-shm0
    libxcb-sync1
    libxcb-util1
    libxcb-xfixes0
    libxcb-xinerama0
    libxcb-xkb1
    libxkbcommon-x11-0
)

SSN_WEBENGINE_PACKAGES=(
    libnss3
    libnspr4
    libxcomposite1
    libxdamage1
    libxrandr2
    libxtst6
    libgbm1
    libegl1
    libxslt1.1
)

SSN_MISSING_PACKAGES=()

ssn_is_debian_family_linux() {
    [ "$(uname -s 2>/dev/null)" = "Linux" ] &&
        command -v dpkg-query >/dev/null 2>&1 &&
        command -v apt >/dev/null 2>&1
}

ssn_package_is_installed() {
    dpkg-query -W -f='${db:Status-Abbrev}' "$1" 2>/dev/null | grep -q '^ii'
}

ssn_ubuntu_uses_t64_packages() {
    local ID="" VERSION_ID=""

    [ -r /etc/os-release ] || return 1
    # shellcheck disable=SC1091
    . /etc/os-release
    [ "$ID" = "ubuntu" ] &&
        command -v dpkg >/dev/null 2>&1 &&
        dpkg --compare-versions "${VERSION_ID:-0}" ge 24.04
}

ssn_collect_missing_gui_packages() {
    local launcher_kind="${1:-viewer}"
    local package
    local packages=("${SSN_LAUNCH_PACKAGES[@]}" "${SSN_XCB_PACKAGES[@]}")

    SSN_MISSING_PACKAGES=()
    ssn_is_debian_family_linux || return 0

    if [ "$launcher_kind" = "tools" ] || [ "$launcher_kind" = "all" ]; then
        packages+=("${SSN_WEBENGINE_PACKAGES[@]}")
        if ssn_ubuntu_uses_t64_packages; then
            packages+=(libasound2t64 libcups2t64)
        else
            packages+=(libasound2 libcups2)
        fi
    fi

    for package in "${packages[@]}"; do
        if ! ssn_package_is_installed "$package"; then
            SSN_MISSING_PACKAGES+=("$package")
        fi
    done
}

ssn_print_missing_gui_packages() {
    local package

    printf '\nMissing Linux GUI system dependencies:\n' >&2
    printf '  ' >&2
    printf '%s ' "${SSN_MISSING_PACKAGES[@]}" >&2
    printf '\n\nInstall them with:\n  sudo apt install' >&2
    for package in "${SSN_MISSING_PACKAGES[@]}"; do
        printf ' %s' "$package" >&2
    done
    printf '\n\nThese libraries are required by the Qt xcb/QtWebEngine plugins and cannot be installed with pip or uv.\n' >&2
}

ssn_require_linux_gui_dependencies() {
    ssn_collect_missing_gui_packages "${1:-viewer}"
    if [ "${#SSN_MISSING_PACKAGES[@]}" -eq 0 ]; then
        return 0
    fi

    ssn_print_missing_gui_packages
    return 1
}

ssn_pause_after_desktop_failure() {
    local status=$?

    if [ "$status" -ne 0 ] && [ "${SSN_LAUNCHED_FROM_DESKTOP:-0}" = "1" ]; then
        printf '\nSSN did not start successfully. The error and any required install command are shown above.\n' >&2
        if [ -t 0 ]; then
            read -r -p "Press Enter to close this terminal..." _
        elif [ -r /dev/tty ]; then
            read -r -p "Press Enter to close this terminal..." _ </dev/tty || true
        fi
    fi

    return "$status"
}

ssn_enable_desktop_failure_pause() {
    trap ssn_pause_after_desktop_failure EXIT
}

ssn_install() {
    local PROJECT_ROOT
    local VIEWER_ICON TOOL_ICON
    local install_dependencies install_menu install_desktop

    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 1
    cd "$PROJECT_ROOT" || return 1

    echo "Setting up launchers for SSN Viewer & SSN Tools on Linux..."
    echo "Project root: $PROJECT_ROOT"

    # Qt's xcb and QtWebEngine plugins use Linux system libraries that pip/uv
    # cannot provide. Detect missing Ubuntu/Debian packages while the installer
    # is already open and offer to install them before first launch.
    if ! ssn_require_linux_gui_dependencies tools; then
        read -r -p "Would you like to install these packages now with sudo apt? (y/n): " install_dependencies
        if [[ "$install_dependencies" =~ ^[Yy]$ ]]; then
            if sudo apt install "${SSN_MISSING_PACKAGES[@]}"; then
                echo "[OK] Installed Linux GUI system dependencies."
            else
                echo "[WARNING] The apt installation failed. You can rerun the command shown above later." >&2
            fi
        else
            echo "[WARNING] System dependencies were not installed. The launchers will show the required command again." >&2
        fi
        echo ""
    fi

    # 1. Make sure all executables in src/bin have execution permissions
    chmod +x src/bin/*.sh
    echo "[OK] Configured execution permissions for scripts in src/bin/"

    # 2. Create symbolic links in the project root pointing to the launchers
    ln -sf src/bin/SSN_Viewer.sh SSN_Viewer
    ln -sf src/bin/SSN_Tools.sh SSN_Tools
    chmod +x SSN_Viewer SSN_Tools
    echo "[OK] Created SSN_Viewer and SSN_Tools executables in project root."

    # 3. Generate .desktop entry launchers
    echo ""
    echo "Generating .desktop entry launchers..."

    VIEWER_ICON="${PROJECT_ROOT}/src/bin/logos/viewer_logo_large.png"
    TOOL_ICON="${PROJECT_ROOT}/src/bin/logos/tool_logo_large.png"

    cat <<EOF > SSN_Viewer.desktop
[Desktop Entry]
Type=Application
Name=SSN Viewer
Comment=Sequence Similarity Network Viewer
Exec="${PROJECT_ROOT}/src/bin/SSN_Desktop_Launcher.sh" viewer
Path=${PROJECT_ROOT}
Icon=${VIEWER_ICON}
Terminal=false
Categories=Science;Biology;
EOF
    chmod +x SSN_Viewer.desktop

    cat <<EOF > SSN_Tools.desktop
[Desktop Entry]
Type=Application
Name=SSN Tools
Comment=Sequence Similarity Network Utilities
Exec="${PROJECT_ROOT}/src/bin/SSN_Desktop_Launcher.sh" tools
Path=${PROJECT_ROOT}
Icon=${TOOL_ICON}
Terminal=false
Categories=Science;Biology;
EOF
    chmod +x SSN_Tools.desktop

    echo "[OK] Created SSN_Viewer.desktop and SSN_Tools.desktop in project root."

    read -r -p "Would you like to install these launchers to your system application menu? (y/n): " install_menu
    if [[ "$install_menu" =~ ^[Yy]$ ]]; then
        mkdir -p ~/.local/share/applications
        cp SSN_Viewer.desktop ~/.local/share/applications/
        cp SSN_Tools.desktop ~/.local/share/applications/
        echo "[OK] Launchers successfully added to your system applications menu!"
    fi

    if [ -d "$HOME/Desktop" ]; then
        read -r -p "Would you like to copy these launchers to your Desktop? (y/n): " install_desktop
        if [[ "$install_desktop" =~ ^[Yy]$ ]]; then
            cp SSN_Viewer.desktop "$HOME/Desktop/"
            cp SSN_Tools.desktop "$HOME/Desktop/"
            chmod +x "$HOME/Desktop/SSN_Viewer.desktop" "$HOME/Desktop/SSN_Tools.desktop"
            echo "[OK] Launchers successfully copied to your Desktop!"
        fi
    fi

    echo ""
    echo "Setup Complete! You can now run SSN Viewer and Tools using the launchers in the project root."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    ssn_install "$@"
fi
