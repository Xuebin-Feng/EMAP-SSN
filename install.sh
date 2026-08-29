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
# Linux/macOS Installation, Dependency Checks, and Launcher Support
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
SSN_SETUP_LOCK_OWNED=0
SSN_SETUP_LOCK_DIR=""

ssn_sanitize_managed_environment() {
    # The managed interpreter must discover its own Python and Qt resources.
    # Preserve PATH, Conda metadata, accelerator variables, display variables,
    # and QT_QPA_PLATFORM, which is an intentional user-facing override.
    unset PYTHONHOME PYTHONPATH
    unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
    unset QML_IMPORT_PATH QML2_IMPORT_PATH
    if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
        unset DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH
    fi
}

ssn_setup_lock_path() {
    printf '%s\n' "$1/temp/dependency_setup.lock"
}

ssn_release_dependency_setup_lock() {
    local owner_pid=""

    [ "${SSN_SETUP_LOCK_OWNED:-0}" = "1" ] || return 0
    [ -n "${SSN_SETUP_LOCK_DIR:-}" ] || return 0
    if [ -r "$SSN_SETUP_LOCK_DIR/owner.pid" ]; then
        IFS= read -r owner_pid <"$SSN_SETUP_LOCK_DIR/owner.pid" || owner_pid=""
    fi
    if [ "$owner_pid" = "$$" ] || [ -z "$owner_pid" ]; then
        [ -z "$owner_pid" ] || rm -f -- "$SSN_SETUP_LOCK_DIR/owner.pid"
        rmdir -- "$SSN_SETUP_LOCK_DIR" 2>/dev/null || true
    fi
    SSN_SETUP_LOCK_OWNED=0
    SSN_SETUP_LOCK_DIR=""
}

ssn_acquire_dependency_setup_lock() {
    local project_root="$1"
    local lock_root lock_dir owner_pid owner_tmp
    local wait_count=0 missing_owner_count=0 waiting_reported=0
    local timeout_seconds="${SSN_SETUP_LOCK_TIMEOUT_SECONDS:-3600}"

    case "$timeout_seconds" in
        ''|*[!0-9]*) timeout_seconds=3600 ;;
    esac

    lock_root="$project_root/temp"
    lock_dir=$(ssn_setup_lock_path "$project_root") || return 1
    mkdir -p "$lock_root" || return 1

    while :; do
        if mkdir "$lock_dir" 2>/dev/null; then
            owner_tmp="$lock_dir/.owner.$$.tmp"
            if ! printf '%s\n' "$$" >"$owner_tmp" ||
                    ! mv -f -- "$owner_tmp" "$lock_dir/owner.pid"; then
                rm -f -- "$owner_tmp"
                rmdir -- "$lock_dir" 2>/dev/null || true
                printf 'Could not record dependency setup lock ownership at: %s\n' "$lock_dir" >&2
                return 1
            fi
            SSN_SETUP_LOCK_DIR="$lock_dir"
            SSN_SETUP_LOCK_OWNED=1
            return 0
        fi

        owner_pid=""
        if [ -r "$lock_dir/owner.pid" ]; then
            IFS= read -r owner_pid <"$lock_dir/owner.pid" || owner_pid=""
        fi
        case "$owner_pid" in
            ''|*[!0-9]*)
                missing_owner_count=$((missing_owner_count + 1))
                if [ "$missing_owner_count" -ge 5 ]; then
                    rm -f -- "$lock_dir/owner.pid" "$lock_dir"/.owner.*.tmp 2>/dev/null || true
                    if rmdir -- "$lock_dir" 2>/dev/null; then
                        printf 'Recovered an incomplete dependency setup lock.\n'
                    fi
                    missing_owner_count=0
                fi
                ;;
            *)
                missing_owner_count=0
                if ! kill -0 "$owner_pid" 2>/dev/null; then
                    rm -f -- "$lock_dir/owner.pid" "$lock_dir"/.owner.*.tmp 2>/dev/null || true
                    if rmdir -- "$lock_dir" 2>/dev/null; then
                        printf 'Recovered stale dependency setup lock from process %s.\n' "$owner_pid"
                    fi
                    continue
                fi
                if [ "$waiting_reported" -eq 0 ]; then
                    printf 'Dependency setup is already running in process %s; waiting...\n' "$owner_pid"
                    waiting_reported=1
                fi
                ;;
        esac

        if [ "$wait_count" -ge "$timeout_seconds" ]; then
            printf 'Timed out waiting for dependency setup owned by process %s.\n' "${owner_pid:-unknown}" >&2
            printf 'Lock retained at: %s\n' "$lock_dir" >&2
            return 1
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done
}

ssn_wait_for_dependency_setup() {
    ssn_acquire_dependency_setup_lock "$1" || return 1
    ssn_release_dependency_setup_lock
}

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

    ssn_release_dependency_setup_lock

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

ssn_create_macos_app_launcher() {
    local project_root="$1"
    local app_name="$2"
    local app_kind="$3"
    local bundle_id="$4"
    local app_dir="$project_root/$app_name.app"
    local executable_dir="$app_dir/Contents/MacOS"
    local resources_dir="$app_dir/Contents/Resources"

    rm -rf "$app_dir"
    mkdir -p "$executable_dir" "$resources_dir"
    cat > "$executable_dir/launcher" <<EOF
#!/bin/bash
APP_ROOT=\$(cd "\$(dirname "\$0")/../.." && pwd)
exec /usr/bin/open -a Terminal "\$APP_ROOT/Contents/Resources/start.command"
EOF
    chmod +x "$executable_dir/launcher"

    # Opening a .command document through Launch Services does not require the
    # launcher bundle to automate Terminal with Apple Events. Terminal owns the
    # resulting session and runs the normal terminal-session entry point.
    cat > "$resources_dir/start.command" <<EOF
#!/bin/bash
PROJECT_ROOT=\$(cd "\$(dirname "\$0")/../../.." && pwd)
exec "\$PROJECT_ROOT/src/bin/SSN_Desktop_Launcher.sh" "$app_kind" --terminal-session
EOF
    chmod +x "$resources_dir/start.command"

    cat > "$app_dir/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundleIdentifier</key><string>$bundle_id</string>
    <key>CFBundleName</key><string>$app_name</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
</dict>
</plist>
EOF
}

ssn_install_macos() {
    local PROJECT_ROOT

    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 1
    cd "$PROJECT_ROOT" || return 1

    echo "Setting up launchers for SSN Viewer & SSN Tools on macOS..."
    echo "Project root: $PROJECT_ROOT"

    chmod +x src/bin/*.sh
    echo "[OK] Configured execution permissions for scripts in src/bin/"

    ssn_create_macos_app_launcher \
        "$PROJECT_ROOT" "SSN Viewer" "viewer" "ca.utoronto.ssn.viewer"
    ssn_create_macos_app_launcher \
        "$PROJECT_ROOT" "SSN Tools" "tools" "ca.utoronto.ssn.tools"

    # Remove generated legacy launchers; direct shell scripts remain available
    # under src/bin, while Finder users launch the generated application bundles.
    rm -f SSN_Viewer.command SSN_Tools.command SSN_Viewer SSN_Tools
    echo "[OK] Created SSN Viewer.app and SSN Tools.app launchers with visible startup terminals."

    echo ""
    echo "To set a custom icon on macOS:"
    echo "  1. Right-click on 'SSN Viewer.app' or 'SSN Tools.app' in Finder and select 'Get Info'."
    echo "  2. Open the corresponding large logo in Preview (for example, 'src/bin/logos/viewer_logo_large.png' or 'src/bin/logos/tool_logo_large.png'), press Cmd+A, then Cmd+C to copy it."
    echo "  3. Click the file icon thumbnail at the top-left of the 'Get Info' window and press Cmd+V to paste."

    echo ""
    echo "Setup Complete! You can now run SSN Viewer and Tools using the .app launchers in the project root."
}

ssn_install_linux() {
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
StartupWMClass=SSN_Viewer
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
StartupWMClass=SSN_Tools
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

ssn_install() {
    case "$(uname -s 2>/dev/null)" in
        Darwin)
            ssn_install_macos "$@"
            ;;
        Linux)
            ssn_install_linux "$@"
            ;;
        *)
            printf 'Unsupported operating system for install.sh: %s\n' \
                "$(uname -s 2>/dev/null || printf 'unknown')" >&2
            return 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    ssn_install "$@"
fi
