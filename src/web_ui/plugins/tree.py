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
            "i18n": {
                "context": "TreePage",
                "files": ["web_ui/tree.html", "web_ui/tree_session.js", "web_ui/tree_nav.js"],
            },
        },
    ],
}
