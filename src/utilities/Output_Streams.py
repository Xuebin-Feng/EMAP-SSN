# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Let an application print any text on the streams it was started with.

On Windows, Python opens a pipe or a file in the ANSI code page (cp1252 on
Western systems), while every reader of those streams here decodes UTF-8: the
desktop launcher monitor's application.log and its error terminal, the MCP
launch and job logs, and Viewer_Terminal's console copy. Text cp1252 can
encode, such as "é", came back as U+FFFD, and printing a path such as
"α-amylase" raised UnicodeEncodeError. In a Qt slot that closed the window,
because the applications exit on an uncaught exception.

Imports nothing beyond ``sys``, so entry points can call it before anything
else, including the imports that load Qt or torch.
"""

import sys


def configure_output_streams(streams=None):
    """Switch pipes and files to UTF-8 and escape what a stream cannot encode.

    A terminal keeps its own encoding (the Windows console is always UTF-8);
    only its error handler changes, so a character it cannot show is escaped
    instead of raised. Call this first in an entry point's ``__main__`` block,
    never at import: modules such as EMAPSSN_Config are also imported by other
    processes and by tests, whose streams are not theirs to change.
    """
    for stream in (sys.stdout, sys.stderr) if streams is None else streams:
        if not hasattr(stream, "reconfigure"):
            continue  # None under pythonw, or a replacement stream left as given
        try:
            if stream.isatty():
                stream.reconfigure(errors="backslashreplace")
            else:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass  # a closed stream
