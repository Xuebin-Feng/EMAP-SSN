# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

WEB_PLUGIN = {
    "api_version": 1,
    "id": "tree",
    "backend": "web_ui.tree_backend",
    "register": "register_backend",
    "pages": [
        {
            "path": "/tree.html",
            "label": "Tree View UI",
            "client": "tree",
            "themed": True,
            "initial_state": False,
            "i18n": {
                "context": "TreePage",
                "files": [
                    "web_ui/tree.html", "web_ui/tree_session.js", "web_ui/tree_nav.js", "web_ui/tree_open.js",
                    "web_ui/tree_read.js", "web_ui/tree_read_xml.js", "web_ui/tree_read_json.js", "web_ui/tree_legend.js",
                ],
            },
        },
    ],
}
