"""Run an MCP Viewer in a real terminal while retaining both output streams."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import threading

import psutil


def copy_output(source, log, terminal):
    """Copy bytes, including partial lines and native-library output, immediately."""
    while chunk := source.read(8192):
        log.write(chunk)
        if terminal is not None:
            try:
                terminal.write(chunk)
                terminal.flush()
            except OSError:
                terminal = None  # Logging must survive a lost display.


def main():
    directory = Path(sys.argv[1])
    identity = directory / "terminal-process.json"
    partial = identity.with_suffix(".partial")
    partial.write_text(json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()}), encoding="utf-8")
    partial.replace(identity)
    with (directory / "stdout.log").open("ab", buffering=0) as out, (directory / "stderr.log").open("ab", buffering=0) as err:
        process = subprocess.Popen(sys.argv[2:], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   stdin=None, bufsize=0, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        readers = []
        for source, log, terminal in ((process.stdout, out, sys.stdout), (process.stderr, err, sys.stderr)):
            thread = threading.Thread(target=copy_output,
                args=(source, log, getattr(terminal, "buffer", None)), daemon=True)
            thread.start()
            readers.append(thread)
        code = process.wait()
        for thread in readers:
            thread.join()
        return code


if __name__ == "__main__":
    raise SystemExit(main())
