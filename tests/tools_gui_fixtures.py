# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Point the Tools window (EMAPSSN_Tools) at a temporary project.

ToolsGUI reads <EMAPSSN_Tools._PROJECT_ROOT>/tools_settings.json for its
directories and saved tool values, lists and validates the inputs those
directories hold, and its save actions write that file. isolated_tools_project()
patches _PROJECT_ROOT to a temporary folder whose tools_settings.json selects
absolute temporary directories, so a window never sees the developer's files.

Not collected by unittest; import with ``from tests.tools_gui_fixtures import ...``.
"""
import json
import pathlib
import tempfile
from unittest import mock


def isolated_tools_project(test_case, sections=None):
    """Patch EMAPSSN_Tools._PROJECT_ROOT for test_case; return the temporary root.

    sections: extra top-level entries (tool sections) for tools_settings.json.
    """
    import EMAPSSN_Tools
    from tools.tool_helpers.Tool_Pipeline import DEFAULT_DIRECTORY_PATHS

    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    root = pathlib.Path(directory.name)
    document = {
        "DIRECTORIES": {
            key: str(root / relative_path)
            for key, relative_path in DEFAULT_DIRECTORY_PATHS.items()
        }
    }
    document.update(sections or {})
    (root / "tools_settings.json").write_text(json.dumps(document, indent=4), encoding="utf-8")
    patcher = mock.patch.object(EMAPSSN_Tools, "_PROJECT_ROOT", str(root))
    patcher.start()
    test_case.addCleanup(patcher.stop)
    return root
