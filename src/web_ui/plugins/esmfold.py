# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

WEB_PLUGIN = {
    "api_version": 1,
    "id": "esmfold",
    "backend": "web_ui.esmfold_backend",
    "register": "register_backend",
    "pages": [
        {
            "path": "/esmfold.html",
            "label": "ESMFold Mol* UI",
            "client": "esmfold",
            "themed": True,
            # Mol* compiles code while it loads (embind's new Function, and the
            # WebAssembly of its encoder, which comes from a data: URL) and
            # reaches public servers when asked to download structures or maps;
            # tests/test_esmfold_shell.py, ScriptPolicyTests, pins that
            # 'unsafe-eval' is still needed. The Viewer allows only these
            # relaxations (Plugin_Manager.ALLOWED_CSP_RELAXATIONS).
            "csp": {
                "script-src": ["'unsafe-eval'"],
                "connect-src": ["https:", "data:"],
            },
            "i18n": {
                "context": "EsmfoldPage",
                "files": ["web_ui/esmfold.html", "web_ui/esmfold_*.js"],
            },
        },
    ],
}
