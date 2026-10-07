"""Run an MCP Viewer in a real terminal while retaining both output streams."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import threading

import psutil


def _character_boundary(data):
    """Return the length of DATA's whole UTF-8 characters.

    The rest, at most three bytes, starts a character that a pipe read cut in
    two. Invalid bytes count as whole, as Windows' console writer counts them.
    """
    for count in range(1, min(len(data), 3) + 1):
        byte = data[-count]
        if byte < 0x80:
            break
        if byte >= 0xC0:
            needed = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4 if byte < 0xF8 else 1
            if count < needed:
                return len(data) - count
            break
    return len(data)


def _write_terminal(terminal, data):
    """Write DATA, finishing short raw writes; return None once the display is lost."""
    try:
        while data and (written := terminal.write(data)):
            data = data[written:]
        terminal.flush()
    except OSError:
        return None  # Logging must survive a lost display.
    return terminal


def copy_output(source, log, terminal):
    """Copy bytes, including partial lines and native-library output, immediately.

    The log gets every read as it arrives; the terminal gets whole UTF-8
    characters. Under -u, a Windows console's sys.stdout.buffer is the raw
    console writer: given a read that ends inside a character, it writes the
    bytes before that character and returns the shorter count, so the cut-off
    bytes wait for the next read instead of being lost.
    """
    pending = b""
    while chunk := source.read(8192):
        log.write(chunk)
        if terminal is not None:
            data = pending + chunk
            cut = _character_boundary(data)
            pending = data[cut:]
            terminal = _write_terminal(terminal, data[:cut])
    if terminal is not None and pending:
        _write_terminal(terminal, pending)


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
