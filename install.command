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
# Installation and Launcher Generation Script for SSN Viewer & Tools (macOS)
# =========================================================================

# Move to the directory containing this script (project root)
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo "Setting up launchers for SSN Viewer & SSN Tools on macOS..."
echo "Project root: $PROJECT_ROOT"

# 1. Make sure all executables in src/bin have execution permissions
chmod +x src/bin/*.sh
echo "[OK] Configured execution permissions for scripts in src/bin/"

# 2. Create Finder-native .app bundles with a temporary startup Terminal
create_app_launcher() {
    local app_name="$1"
    local app_kind="$2"
    local bundle_id="$3"
    local app_dir="$PROJECT_ROOT/${app_name}.app"
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
    <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
EOF
}

create_app_launcher "SSN Viewer" "viewer" "ca.utoronto.ssn.viewer"
create_app_launcher "SSN Tools" "tools" "ca.utoronto.ssn.tools"

# Remove generated legacy launchers; direct shell scripts remain available in src/bin.
rm -f SSN_Viewer.command SSN_Tools.command SSN_Viewer SSN_Tools
echo "[OK] Created SSN Viewer.app and SSN Tools.app launchers with visible startup terminals."

echo ""
echo "To set a custom icon on macOS:"
echo "  1. Right-click on 'SSN Viewer.app' or 'SSN Tools.app' in Finder and select 'Get Info'."
echo "  2. Open the corresponding large logo in Preview (e.g. 'src/bin/logos/viewer_logo_large.png' or 'src/bin/logos/tool_logo_large.png'), press Cmd+A, then Cmd+C to copy it."
echo "  3. Click on the file icon thumbnail at the top-left of the 'Get Info' window and press Cmd+V to paste."

echo ""
echo "Setup Complete! You can now run SSN Viewer and Tools using the .app launchers in the project root."
