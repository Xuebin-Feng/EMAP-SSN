#!/usr/bin/env python
# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The desktop probe's command line: install it, turn it on, drive the windows, read what it saw.

Run it with the checkout's own interpreter (.venv/bin/python on Linux and macOS, .venv\\Scripts\\python.exe on Windows):

    probe.py install [CHECKOUT]       write the .pth (and its stub) into CHECKOUT/.venv's site-packages
    probe.py uninstall [CHECKOUT]     remove them again
    probe.py on [--out DIR] [--fresh] [--no-auto-capture] [--settle-ms N]
    probe.py off                      the probe goes quiet in every process within about a second
    probe.py status [--json]          installed? on? which processes? what step?
    probe.py procs                    the processes that have recorded something
    probe.py step "label"             stamp the events and captures that follow with this step
    probe.py wait --kind K [--title REGEX] [--timeout S] [--gone]
    probe.py do --kind K|--pid N [--timeout S] '{"action": "windows"}'     (or: windows, click widget=Save)
    probe.py timeline [--json]        per launch: time to the splash, the window, the splash's close
    probe.py sheet                    write contact-sheet.html next to the captures
    probe.py attach IMAGE.png         add a picture made another way (the Viewer's print) to the sheet

See README.md in this folder for what the probe is, what it records and every action `do` takes. It is
test-only: nothing in it is meant for a user's machine, and nothing in src/ knows it exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_package():
    spec = importlib.util.spec_from_file_location("emapssn_desktop_probe_boot", HERE / "boot.py")
    boot = importlib.util.module_from_spec(spec)
    sys.modules["emapssn_desktop_probe_boot"] = boot
    spec.loader.exec_module(boot)
    boot.load_package(str(HERE))


_load_package()
from emapssn_desktop_probe import sheet, store  # noqa: E402

STUB_NAME = "emapssn_desktop_probe_stub.py"
PTH_PREFIX = "emapssn_desktop_probe-"
DEFAULT_TIMEOUT = 20.0
POLL_SECONDS = 0.05
ACTING = ("click", "type", "key", "close", "set_value", "viewer_command", "hover")  # requests that may end the program


# ---------------------------------------------------------------------------------------------
# install and uninstall
# ---------------------------------------------------------------------------------------------

def site_packages(checkout) -> list[Path]:
    """The site-packages folders of the checkout's .venv (Lib/ on Windows, lib/python3.X/ elsewhere)."""
    venv = Path(checkout) / ".venv"
    found = [venv / "Lib" / "site-packages"] if (venv / "Lib" / "site-packages").is_dir() else []
    found += sorted(venv.glob("lib/python3*/site-packages"))
    return [path for path in found if path.is_dir()]


def checkout_tag(checkout) -> str:
    return hashlib.sha1(os.path.normcase(os.path.realpath(checkout)).encode("utf-8")).hexdigest()[:8]


def pth_line(checkout) -> str:
    """The one line of the .pth: look at the gate file, and only if it exists load the stub, which loads the probe."""
    checkout = os.path.realpath(checkout)
    gate = str(store.gate_file(checkout))
    return f"import os; os.path.exists({gate!r}) and __import__('emapssn_desktop_probe_stub').load({checkout!r})\n"


def install(checkout) -> list[Path]:
    checkout = Path(checkout).resolve()
    folders = site_packages(checkout)
    if not folders:
        raise SystemExit(f"No .venv/site-packages under {checkout}: make the environment first (install.sh / install.bat).")
    written = []
    for folder in folders:
        shutil.copyfile(HERE / "stub.py", folder / STUB_NAME)
        pth = folder / f"{PTH_PREFIX}{checkout_tag(checkout)}.pth"
        pth.write_text(pth_line(checkout), encoding="utf-8")
        written.append(pth)
    return written


def uninstall(checkout) -> list[Path]:
    """Remove this checkout's .pth; the stub goes when no other checkout sharing the venv still has one."""
    checkout = Path(checkout).resolve()
    removed = []
    for folder in site_packages(checkout):
        pth = folder / f"{PTH_PREFIX}{checkout_tag(checkout)}.pth"
        if pth.exists():
            pth.unlink()
            removed.append(pth)
        if not list(folder.glob(PTH_PREFIX + "*.pth")):
            (folder / STUB_NAME).unlink(missing_ok=True)
            for cached in (folder / "__pycache__").glob("emapssn_desktop_probe_stub*"):
                cached.unlink(missing_ok=True)
    return removed


def installed(checkout) -> list[Path]:
    return [folder / f"{PTH_PREFIX}{checkout_tag(checkout)}.pth" for folder in site_packages(checkout)
            if (folder / f"{PTH_PREFIX}{checkout_tag(checkout)}.pth").exists()]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def emit(value) -> None:
    print(json.dumps(value, indent=1, ensure_ascii=False))


def live_processes(out, kind=None, pid=None) -> list[dict]:
    processes = store.list_processes(out, live_only=True)
    return [process for process in processes
            if (kind is None or process["kind"] == kind) and (pid is None or process["pid"] == pid)]


def parse_request(words: list[str]) -> dict:
    """The request on the command line: a JSON object, @file, - (stdin), or ``action key=value ...``."""
    text = " ".join(words).strip()
    if text == "-":
        text = sys.stdin.read()
    elif text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8-sig")
    if text.startswith("{"):
        request = json.loads(text)
        if not isinstance(request, dict) or "action" not in request:
            raise SystemExit("The request needs an \"action\".")
        return request
    if not words or not re.fullmatch(r"[a-z_]+", words[0]):
        raise SystemExit("Give a JSON object, @file, - for stdin, or an action name followed by key=value pairs.")
    request = {"action": words[0]}
    for pair in words[1:]:
        key, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"{pair!r} is not key=value.")
        try:
            request[key] = json.loads(value)
        except ValueError:
            request[key] = value
    return request


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------

def command_install(args) -> int:
    for pth in install(args.checkout_arg or args.checkout):
        print(f"installed {pth}")
    print("The probe stays inert until `probe.py on`.")
    return 0


def command_uninstall(args) -> int:
    removed = uninstall(args.checkout_arg or args.checkout)
    for pth in removed:
        print(f"removed {pth}")
    if not removed:
        print("nothing to remove")
    return 0


def command_on(args) -> int:
    checkout = args.checkout
    out = Path(args.out).resolve() if args.out else store.default_out(checkout)
    if args.fresh and out.exists():
        if not (out / store.OUT_MARKER).exists():
            raise SystemExit(f"{out} was not made by the probe; --fresh will not empty it.")
        shutil.rmtree(out)
    store.ensure_out(out)
    gate = store.write_gate(checkout, out, auto_capture=not args.no_auto_capture, settle_ms=args.settle_ms)
    print(f"on: processes started from now on record into {gate['out']}")
    if not installed(checkout):
        print("note: the hook is not installed in this checkout's .venv; run `probe.py install` first.")
    return 0


def command_off(args) -> int:
    print("off" if store.remove_gate(args.checkout) else "it was already off")
    return 0


def command_status(args) -> int:
    out = store.out_for(args.checkout, args.out)
    processes = store.list_processes(out)
    gate = store.read_gate(args.checkout)
    status = {
        "installed": [str(path) for path in installed(args.checkout)], "on": gate is not None, "out": str(out),
        "step": store.read_step(out)["label"], "captures": len(list((out / "captures").glob("*.png"))) if out.exists() else 0,
        "processes": [{"pid": process["pid"], "kind": process["kind"], "alive": process["alive"],
                       "started": sheet.clock(process["boot_wall_ms"])} for process in processes],
        "errors": [str(Path(process["folder"]) / "errors.log") for process in processes
                   if (Path(process["folder"]) / "errors.log").exists()],
        "boot_errors": str(store.home_dir(args.checkout) / "boot-errors.log")
        if (store.home_dir(args.checkout) / "boot-errors.log").exists() else None,
    }
    if args.json:
        emit(status)
        return 0
    print(f"installed: {', '.join(status['installed']) or 'no'}")
    print(f"on:        {'yes' if status['on'] else 'no'}    out: {status['out']}")
    print(f"step:      {status['step'] or '(none yet)'}    captures: {status['captures']}")
    for process in status["processes"]:
        print(f"  {process['kind']:<8} pid {process['pid']:<8} {'running' if process['alive'] else 'ended  '} since {process['started']}")
    for path in status["errors"] + ([status["boot_errors"]] if status["boot_errors"] else []):
        print(f"ERRORS in {path}")
    return 0


def command_procs(args) -> int:
    out = store.out_for(args.checkout, args.out)
    for process in store.list_processes(out, live_only=args.live):
        print(f"{process['pid']:<8} {process['kind']:<8} {'running' if process['alive'] else 'ended  '} "
              f"{sheet.clock(process['boot_wall_ms'])}  {' '.join(process.get('argv', []))[:100]}")
    return 0


def command_step(args) -> int:
    out = store.out_for(args.checkout, args.out)
    step = store.write_step(out, args.label)
    print(f"step {step['seq']}: {step['label']}")
    return 0


def command_wait(args) -> int:
    out = store.out_for(args.checkout, args.out)
    pattern = re.compile(args.title, re.IGNORECASE) if args.title else None
    deadline = time.monotonic() + args.timeout
    while True:
        found = None
        for process in live_processes(out, args.kind, args.pid):
            listing = store.read_json(Path(process["folder"]) / "windows.json", {}) or {}
            for window in listing.get("windows", []):
                if window.get("visible") and (pattern is None or pattern.search(window.get("title") or "")
                                              or pattern.search(window.get("class") or "")):
                    found = {"pid": process["pid"], "kind": process["kind"], "window": window}
                    break
            if found:
                break
        if args.gone and not found:
            print("gone")
            return 0
        if not args.gone and found:
            emit(found)
            return 0
        if time.monotonic() >= deadline:
            print("timed out waiting for " + ("the window to go" if args.gone else "the window"), file=sys.stderr)
            return 1
        time.sleep(POLL_SECONDS * 4)


def command_do(args) -> int:
    out = store.out_for(args.checkout, args.out)
    request = parse_request(args.request)
    deadline = time.monotonic() + args.timeout
    process = None
    while process is None:  # the process may not have started yet: wait for it as `wait` would
        matches = live_processes(out, args.kind, args.pid)
        if matches:
            process = matches[-1]  # the newest of a kind
        elif time.monotonic() >= deadline:
            known = [f"{p['kind']} {p['pid']} ({'running' if p['alive'] else 'ended'})" for p in store.list_processes(out)]
            raise SystemExit(f"No running process matches (kind={args.kind}, pid={args.pid}). Known: {known or 'none'}")
        else:
            time.sleep(POLL_SECONDS * 4)
    request_id = f"{store.wall_ms()}-{uuid.uuid4().hex[:6]}"
    store.write_request(out, {"pid": process["pid"]}, request, request_id)
    answer = store.response_path(out, request_id)
    while True:
        response = store.read_json(answer)
        if response is not None:
            emit(response)
            return 0 if response.get("ok") else 1
        if not store.process_alive(process):
            if request["action"] in ACTING:  # a click on Exit, or a close, ends the program before it can answer
                emit({"ok": None, "action": request["action"], "pid": process["pid"], "kind": process["kind"],
                      "note": "the process ended while it carried out the request, which is what closing it does"})
                return 0
            raise SystemExit(f"The {process['kind']} (pid {process['pid']}) ended before it answered.")
        if time.monotonic() >= deadline:
            raise SystemExit(f"No answer from the {process['kind']} (pid {process['pid']}) within {args.timeout:g} s. "
                             "Is it busy, or is the probe off in it? See `probe.py status`.")
        time.sleep(POLL_SECONDS)


def command_timeline(args) -> int:
    out = store.out_for(args.checkout, args.out)
    if args.json:
        emit(sheet.launches(out))
    else:
        print(sheet.render_timeline(out))
    return 0


def command_sheet(args) -> int:
    out = store.out_for(args.checkout, args.out)
    path = Path(args.file) if args.file else out / "contact-sheet.html"
    path.write_text(sheet.contact_sheet(out), encoding="utf-8")
    print(path)
    return 0


def png_size(path) -> tuple[int, int]:
    """The width and height in a PNG file's header."""
    header = Path(path).read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} is not a PNG file.")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def command_attach(args) -> int:
    """Put a picture made another way (the Viewer's `print` of its canvas) on the contact sheet, under the current step."""
    out = store.ensure_out(store.out_for(args.checkout, args.out))
    source = Path(args.image)
    width, height = png_size(source)
    step = store.read_step(out)
    attached = store.read_events(out / "attached.jsonl")
    number = len(attached) + 1
    caption = args.caption or source.stem
    name = f"attached-{number:04d}-{store.slug(step['label'], 30)}-{store.slug(caption, 40)}.png"
    shutil.copyfile(source, out / "captures" / name)
    record = {"ev": "capture", "wall_ms": store.wall_ms(), "mono_ms": store.mono_ms(), "pid": 0, "kind": "attached",
              "step": step["label"], "file": f"captures/{name}", "path": str(out / "captures" / name), "width": width,
              "height": height, "dpr": 1.0, "logical": [width, height], "title": caption, "window": "", "reason": "attached",
              "number": number, "screen": {}, "flat_colour": None, "drawn_elsewhere": []}
    with open(out / "attached.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(store.redact_data(record), ensure_ascii=False) + "\n")
    print(f"attached captures/{name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", default=str(HERE.parents[1]), help="the checkout (default: the one this file is in)")
    parser.add_argument("--out", help="the output folder to read (default: the gate's, else the last one)")
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("install", help="write the .pth into the checkout's .venv")
    sub.add_argument("checkout_arg", nargs="?", metavar="CHECKOUT")
    sub.set_defaults(run=command_install)
    sub = commands.add_parser("uninstall", help="remove the .pth again")
    sub.add_argument("checkout_arg", nargs="?", metavar="CHECKOUT")
    sub.set_defaults(run=command_uninstall)
    sub = commands.add_parser("on", help="turn the probe on for processes started from now on")
    sub.add_argument("--fresh", action="store_true", help="empty the output folder first (only one the probe made)")
    sub.add_argument("--no-auto-capture", action="store_true", help="capture only on request")
    sub.add_argument("--settle-ms", type=int, default=store.SETTLE_MS, help="wait before a window's automatic capture")
    sub.set_defaults(run=command_on)
    commands.add_parser("off", help="turn the probe off").set_defaults(run=command_off)
    sub = commands.add_parser("status", help="what is installed, on and running")
    sub.add_argument("--json", action="store_true")
    sub.set_defaults(run=command_status)
    sub = commands.add_parser("procs", help="the processes that recorded something")
    sub.add_argument("--live", action="store_true")
    sub.set_defaults(run=command_procs)
    sub = commands.add_parser("step", help="label what follows")
    sub.add_argument("label")
    sub.set_defaults(run=command_step)
    sub = commands.add_parser("wait", help="wait for a window (or for it to go)")
    sub.add_argument("--kind")
    sub.add_argument("--pid", type=int)
    sub.add_argument("--title", help="a regular expression for the window's title (or class)")
    sub.add_argument("--timeout", type=float, default=30.0)
    sub.add_argument("--gone", action="store_true", help="wait until no such window is open")
    sub.set_defaults(run=command_wait)
    sub = commands.add_parser("do", help="send a request to a process and print its answer")
    sub.add_argument("--kind")
    sub.add_argument("--pid", type=int)
    sub.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    sub.add_argument("request", nargs="+", help="JSON, @file, - for stdin, or an action name with key=value pairs")
    sub.set_defaults(run=command_do)
    sub = commands.add_parser("timeline", help="the time to the splash, the window and the splash's close, per launch")
    sub.add_argument("--json", action="store_true")
    sub.set_defaults(run=command_timeline)
    sub = commands.add_parser("sheet", help="write the contact sheet")
    sub.add_argument("--file", help="where to write it (default: contact-sheet.html in the output folder)")
    sub.set_defaults(run=command_sheet)
    sub = commands.add_parser("attach", help="add a PNG made another way (the Viewer's print of its canvas) to the sheet")
    sub.add_argument("image")
    sub.add_argument("--caption")
    sub.set_defaults(run=command_attach)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "do" and not (args.kind or args.pid):
        raise SystemExit("do needs --kind or --pid.")
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
