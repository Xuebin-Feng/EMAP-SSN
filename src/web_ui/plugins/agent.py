# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

WEB_PLUGIN = {
    "api_version": 1,
    "id": "agent",
    "backend": "web_ui.agent_backend",
    "register": "register_backend",
    "pages": [
        {
            "path": "/agent.html",
            "label": "Agent UI",
            "client": "agent",
            "themed": True,
            "i18n": {
                "context": "AgentPage",
                "files": ["web_ui/agent.html", "resources/agent/attachments.js"],
            },
        },
    ],
}
