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


def open_tools_window(
    test_case,
    *,
    sections=None,
    show=True,
    size=None,
    carried=None,
    isolate=True,
    patch_browser=True,
    **replacements,
):
    """Build a ToolsGUI over an isolated project with standard test cleanup.

    sections: extra top-level entries for tools_settings.json (when isolate=True).
    show: show the window and flush Qt events before returning.
    size: optional (width, height) passed to window.resize() before showing.
    carried: optional language_carry_over() payload passed to ToolsGUI(carried=...).
    isolate: call isolated_tools_project(test_case, sections=sections) first.
        Pass False when the test already called isolated_tools_project() to
        populate files or settings before constructing ToolsGUI.
    patch_browser: replace EMAPSSN_Tools.ResponsiveTextBrowser with QTextBrowser
        unless ResponsiveTextBrowser is supplied in replacements.
    **replacements: attributes patched on EMAPSSN_Tools via mock.patch.object
        for the lifetime of test_case.
    """
    from PySide6.QtWidgets import QApplication, QTextBrowser
    import EMAPSSN_Tools

    if isolate:
        isolated_tools_project(test_case, sections=sections)

    patches = dict(replacements)
    if patch_browser and "ResponsiveTextBrowser" not in patches:
        patches["ResponsiveTextBrowser"] = QTextBrowser
    for attribute, value in patches.items():
        patcher = mock.patch.object(EMAPSSN_Tools, attribute, value)
        patcher.start()
        test_case.addCleanup(patcher.stop)

    app = QApplication.instance() or QApplication([])
    window = EMAPSSN_Tools.ToolsGUI(carried=carried)

    def close_window():
        window.close()
        window.deleteLater()
        for _ in range(4):
            app.processEvents()

    test_case.addCleanup(close_window)
    if size is not None:
        window.resize(*size)
    if show:
        window.show()
        for _ in range(4):
            app.processEvents()
    return window
