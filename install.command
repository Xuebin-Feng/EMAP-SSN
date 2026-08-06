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

# 2. Create double-clickable .command wrapper scripts in the project root
cat <<EOF > SSN_Viewer.command
#!/bin/bash
cd "\$(dirname "\$0")"
exec ./src/bin/SSN_Viewer.sh "\$@"
EOF

cat <<EOF > SSN_Tools.command
#!/bin/bash
cd "\$(dirname "\$0")"
exec ./src/bin/SSN_Tools.sh "\$@"
EOF

chmod +x SSN_Viewer.command SSN_Tools.command

# Remove any old extensionless links if they exist
rm -f SSN_Viewer SSN_Tools
echo "[OK] Created double-clickable SSN_Viewer.command and SSN_Tools.command launchers in project root."

echo ""
echo "To set a custom icon on macOS:"
echo "  1. Right-click on the 'SSN_Viewer.command' or 'SSN_Tools.command' launcher in Finder and select 'Get Info'."
echo "  2. Open the corresponding large logo in Preview (e.g. 'src/bin/logos/viewer_logo_large.png' or 'src/bin/logos/tool_logo_large.png'), press Cmd+A, then Cmd+C to copy it."
echo "  3. Click on the file icon thumbnail at the top-left of the 'Get Info' window and press Cmd+V to paste."

echo ""
echo "Setup Complete! You can now run SSN Viewer and Tools using the launchers in the project root."
