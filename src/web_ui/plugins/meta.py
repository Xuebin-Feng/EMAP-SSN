# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

WEB_PLUGIN = {
    "api_version": 1,
    "id": "meta",
    "backend": "web_ui.meta_backend",
    "register": "register_backend",
    "pages": [
        {
            "path": "/meta.html",
            "label": "Metadata UI",
            "client": "meta",
            "themed": True,
            "i18n": {
                "context": "MetadataPage",
                "files": ["web_ui/meta.html"],
            },
        },
    ],
}
