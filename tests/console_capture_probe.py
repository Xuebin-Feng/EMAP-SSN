"""Child side of tests/test_console_capture.py: install the capture, then write through every path a window's process uses.

Not collected as a test (the name has no test_ prefix). Usage:

    console_capture_probe.py <result.json> <log_dir> <auto|on|off> [crash] [qt] [qtportal] [ansi] [close] [burst=N]

Every source writes one marker line, "SRC <name>", and the probe waits until the ring holds all of them
before it writes its result (what the ring and the log hold, and what each stream was before and after).
"""

import ctypes
import json
import os
import subprocess
import sys
import time
import warnings

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

result_path, log_dir, forward_mode = sys.argv[1], sys.argv[2], sys.argv[3]
options = set(sys.argv[4:])
burst = next((int(option.split("=")[1]) for option in options if option.startswith("burst=")), 0)
forward = {"auto": None, "on": True, "off": False}[forward_mode]

from utilities import Console_Capture  # noqa: E402

state = {"stdout_before": repr(sys.stdout), "stderr_before": repr(sys.stderr)}
capture = Console_Capture.install("probe", log_dir, forward)
state["forwarding"] = dict(capture.forwarding)
expected = []


def source(name, writer):
    expected.append(name)
    writer(f"SRC {name}")


source("python stdout", lambda text: print(text))
source("python stderr", lambda text: print(text, file=sys.stderr))
source("unicode", lambda text: print(text + " α-amylase ✓ 蛋白质 é"))
print("progress 10%\rprogress 50%\rSRC progress 100% done")
expected.append("progress 100% done")
source("os.write stdout", lambda text: os.write(1, (text + "\n").encode()))
source("os.write stderr", lambda text: os.write(2, (text + "\n").encode()))

if sys.platform == "win32":
    crt = ctypes.CDLL("ucrtbase")
    expected.append("c puts")
    crt.puts(b"SRC c puts")
    crt.fflush(None)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    for handle_id, text in ((-11, b"SRC win32 stdout handle\n"), (-12, b"SRC win32 stderr handle\n")):
        expected.append(text.decode().strip()[4:])
        written = ctypes.c_uint32()
        kernel32.WriteFile(kernel32.GetStdHandle(handle_id), text, len(text), ctypes.byref(written), None)
    child_python = sys.executable.replace("pythonw.exe", "python.exe")
    expected += ["child cmd stdout", "child cmd stderr"]
    subprocess.run(["cmd.exe", "/d", "/c", "echo SRC child cmd stdout & echo SRC child cmd stderr 1>&2"],
                   stdin=subprocess.DEVNULL, creationflags=0x08000000)
else:
    libc = ctypes.CDLL(None)
    expected.append("c puts")
    libc.puts(b"SRC c puts")
    libc.fflush(None)
    expected += ["child sh stdout", "child sh stderr"]
    subprocess.run(["sh", "-c", "echo SRC child sh stdout; echo SRC child sh stderr 1>&2"], stdin=subprocess.DEVNULL)
    child_python = sys.executable
expected += ["child python stdout", "child python stderr"]
subprocess.run([child_python, "-c", "import sys; print('SRC child python stdout'); print('SRC child python stderr', file=sys.stderr)"],
               stdin=subprocess.DEVNULL, **({"creationflags": 0x08000000} if sys.platform == "win32" else {}))

with warnings.catch_warnings():
    warnings.simplefilter("always")
    expected.append("python warning")
    warnings.warn("SRC python warning", RuntimeWarning)
try:
    1 / 0
except ZeroDivisionError:
    import traceback
    expected.append("ZeroDivisionError")
    traceback.print_exc()

if "qt" in options:
    from PySide6 import QtCore
    Console_Capture.install_qt_message_handler()
    expected += ["qt warning", "qt critical"]
    QtCore.qWarning("SRC qt warning")
    QtCore.qCritical("SRC qt critical")

if "ansi" in options:
    # A command's colours: the question it asks, then the sequences, one of them cut across two writes.
    import contextlib
    import io
    from utilities.Terminal_Colors import colours_enabled
    esc = chr(27)
    state["colours_enabled"] = colours_enabled()
    with contextlib.redirect_stdout(io.StringIO()):
        state["colours_enabled_in_a_buffer"] = colours_enabled()
    expected += ["ansi marker", "ansi split", "ansi stderr"]
    print(f"{esc}[31mSRC ansi marker{esc}[0m {esc}[38;2;1;2;3mdot{esc}[0m")
    print(f"{esc}[1mSRC ansi stderr{esc}[0m", file=sys.stderr)
    time.sleep(0.4)  # the other stream's line is out before this one is cut: a terminal shows both in one place
    os.write(1, f"SRC ansi split {esc}[3".encode())
    time.sleep(0.2)
    os.write(1, f"1mcoloured{esc}[0m\n".encode())

if "qtportal" in options:
    # What Qt prints on GNOME once setDesktopFileName names the app, and a neighbour that is not that.
    from PySide6 import QtCore
    Console_Capture.install_qt_message_handler()
    services = QtCore.QLoggingCategory("qt.qpa.services")
    expected.append("SRC portal neighbour")
    QtCore.qCWarning(services, "Failed to register with host portal QDBusError("
                     "\"org.freedesktop.portal.Error.Failed\", \"Could not register app ID: "
                     "Connection already associated with an application ID\")")
    QtCore.qCWarning(services, "Failed to register with host portal QDBusError("
                     "\"org.freedesktop.portal.Error.Failed\", \"App info not found for 'x'\") SRC portal neighbour")


def in_ring():
    texts = [line.text for line in capture.snapshot()]
    return [name for name in expected if not any(name in text for text in texts)]


deadline = time.time() + 10
while in_ring() and time.time() < deadline:
    time.sleep(0.02)
state["missing_in_ring"] = in_ring()
state["ring"] = [[line.stream, line.text] for line in capture.snapshot()]

if burst:
    started = time.perf_counter()
    for number in range(burst):
        print(f"burst line {number}")
    state["burst_seconds"] = round(time.perf_counter() - started, 3)
    deadline = time.time() + 20
    while time.time() < deadline and not any(line.text == f"burst line {burst - 1}" for line in capture.snapshot()[-50:]):
        time.sleep(0.02)
    state["burst_in_ring"] = sum(1 for line in capture.snapshot() if line.text.startswith("burst line"))
    state["ring_dropped"] = capture.dropped
    state["forward_dropped"] = capture.forward_dropped

if "crash" in options:
    print("about to crash natively")
    sys.stdout.flush()
    time.sleep(0.3)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)
    ctypes.string_at(0)  # an access violation, with no chance to clean up

state["log_path"] = capture.log_path
state["stdout_after_install"] = repr(sys.stdout)
with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(state, handle, ensure_ascii=False)
if "close" in options:
    print("SRC last line before close")
    capture.close()
    state["streams_restored"] = [sys.stdout is capture._python_streams[0], sys.stderr is capture._python_streams[1]]
    state["log_after_close"] = open(capture.log_path, "rb").read().decode("utf-8", "replace")
    os.write(1, b"SRC after close\n")  # straight to what the process started with
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)
