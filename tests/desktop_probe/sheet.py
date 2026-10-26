# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What a person reads after a test round: the launch timeline and the contact sheet.

Both are built from the files the processes wrote (procs/*/events.jsonl and process.json, steps.jsonl), so they
work after every process has ended and need no Qt. The contact sheet is one self-contained HTML page that links
the capture images by relative path; it replaces taking screenshots by hand.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

from emapssn_desktop_probe import store

NOTABLE = ("dialog_shown", "desktop_open", "file_dialog", "probe_error", "probe_disabled", "capture_skipped",
           "capture_failed")


def clock(wall_ms) -> str:
    return time.strftime("%H:%M:%S", time.localtime(wall_ms / 1000)) + f".{int(wall_ms) % 1000:03d}"


def load_run(out) -> list[dict]:
    """Every process under ``out`` with its record, its events and its folder, oldest first."""
    processes = []
    for record in store.list_processes(out):
        record["events"] = store.read_events(Path(record["folder"]) / "events.jsonl")
        processes.append(record)
    return processes


def start_ms(process) -> int:
    """When the process began: the operating system's creation time when the probe could read it, else its boot time."""
    created = process.get("created")
    return int(created * 1000) if isinstance(created, (int, float)) else int(process["boot_wall_ms"])


def parent_of(process, by_pid):
    """The process in the same run that started this one: the nearest of its ancestors that has a record."""
    for pid in [*process.get("ancestors", []), process.get("ppid")]:
        if pid in by_pid and pid != process["pid"]:
            return by_pid[pid]
    return None


def first_event(process, name, predicate=lambda event: True):
    return next((event for event in process["events"] if event["ev"] == name and predicate(event)), None)


def window_story(process) -> list[dict]:
    """Each window of a process (by serial): when it was first shown, painted, hidden and closed, in ms since the process began."""
    windows: dict = {}
    for event in process["events"]:
        serial = event.get("serial")
        if serial is None:
            continue
        entry = windows.setdefault(serial, {"serial": serial, "kind": process["kind"], "pid": process["pid"]})
        window = event.get("window") or {}
        entry.setdefault("class", window.get("class"))
        entry["title"] = window.get("title") or entry.get("title")
        name = event["ev"]
        if name == "window_shown" and "shown_ms" not in entry:
            entry["shown_ms"] = event["wall_ms"]
            entry["shown_late"] = bool(event.get("late"))
        elif name == "window_first_painted" and "painted_ms" not in entry:
            entry["painted_ms"] = event["wall_ms"]
        elif name in ("window_hidden", "window_closed", "window_destroyed"):
            entry["gone_ms"] = event["wall_ms"]
            entry["gone_how"] = name.replace("window_", "")
    return [entry for entry in windows.values() if "shown_ms" in entry]


def launches(out) -> list[dict]:
    """The launches in a run: each root process (one nobody in the run started) with everything it started, in time order."""
    processes = load_run(out)
    by_pid = {process["pid"]: process for process in processes}
    children: dict = {}
    roots = []
    for process in processes:
        parent = parent_of(process, by_pid)
        if parent is None:
            roots.append(process)
        else:
            children.setdefault(parent["pid"], []).append(process)

    def tree(process):
        yield process
        for child in sorted(children.get(process["pid"], []), key=start_ms):
            yield from tree(child)

    result = []
    for root in sorted(roots, key=start_ms):
        members = list(tree(root))
        t0 = start_ms(root)
        entries = []
        for member in members:
            ready = first_event(member, "app_ready")
            exit_event = first_event(member, "exit")
            entries.append({
                "kind": member["kind"], "pid": member["pid"], "detail": member.get("detail"),
                "started_ms": start_ms(member) - t0,
                "app_ready_ms": ready["wall_ms"] - t0 if ready else None,
                "exit_ms": exit_event["wall_ms"] - t0 if exit_event else None,
                "windows": [{**window, **{key: window[key] - t0 for key in ("shown_ms", "painted_ms", "gone_ms") if key in window}}
                            for window in window_story(member)],
            })
        result.append({"root": f"{root['kind']}" + (f" {root['detail']}" if root.get("detail") else ""), "pid": root["pid"],
                       "started_wall_ms": t0, "processes": entries})
    return result


def summary_line(launch) -> str:
    """The numbers a round is read for: splash shown and painted, the window shown, the splash closed."""
    parts = []
    splash = next((entry for entry in launch["processes"] if entry["kind"] == "splash"), None)
    if splash and splash["windows"]:
        window = splash["windows"][0]
        parts.append(f"splash shown +{window['shown_ms']} ms" + (" (seen by a scan)" if window.get("shown_late") else ""))
        if "painted_ms" in window:
            parts.append(f"first painted +{window['painted_ms']} ms")
    for entry in launch["processes"]:
        if entry["kind"] in ("splash", "monitor") or not entry["windows"]:
            continue
        window = entry["windows"][0]
        parts.append(f"{entry['kind']} window shown +{window['shown_ms']} ms")
        if "painted_ms" in window:
            parts.append(f"first painted +{window['painted_ms']} ms")
        break
    if splash and splash["windows"] and "gone_ms" in splash["windows"][0]:
        parts.append(f"splash closed +{splash['windows'][0]['gone_ms']} ms")
    return ", ".join(parts) or "no window was recorded"


def render_timeline(out) -> str:
    runs = launches(out)
    if not runs:
        return "No process has recorded anything under " + str(out)
    lines = []
    for number, launch in enumerate(runs, 1):
        lines.append(f"Launch {number}: {launch['root']} (pid {launch['pid']}), started {clock(launch['started_wall_ms'])}")
        lines.append("  " + summary_line(launch))
        rows = []
        for entry in launch["processes"]:
            label = entry["kind"] + (f" {entry['detail']}" if entry.get("detail") else "")
            rows.append((entry["started_ms"], f"{label} process started (pid {entry['pid']})"))
            if entry["app_ready_ms"] is not None:
                rows.append((entry["app_ready_ms"], f"{label}: QApplication ready"))
            for window in entry["windows"]:
                title = f"{window.get('class')} {window['title']!r}" if window.get("title") else str(window.get("class"))
                rows.append((window["shown_ms"], f"{label}: shown {title}" + (" (late by up to one scan)" if window.get("shown_late") else "")))
                if "painted_ms" in window:
                    rows.append((window["painted_ms"], f"{label}: first painted {title}"))
                if "gone_ms" in window:
                    rows.append((window["gone_ms"], f"{label}: {window['gone_how']} {title}"))
            if entry["exit_ms"] is not None:
                rows.append((entry["exit_ms"], f"{label} process exited"))
        for offset, text in sorted(rows, key=lambda row: row[0]):
            lines.append(f"    {offset:>7} ms  {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The contact sheet
# ---------------------------------------------------------------------------------------------

STYLE = """
body { font: 14px/1.4 system-ui, sans-serif; margin: 0; background: #f5f5f7; color: #1c1c1e; }
header, section { padding: 12px 20px; }
header { background: #2b2b33; color: #fff; }
h1 { font-size: 18px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 0; padding-top: 6px; }
section { border-bottom: 1px solid #d0d0d6; }
.meta { color: #666; font-size: 12px; } .cards { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; }
figure { margin: 0; background: #fff; border: 1px solid #d0d0d6; border-radius: 6px; padding: 8px; max-width: 520px; }
figure img { max-width: 500px; max-height: 420px; display: block; border: 1px solid #e3e3e8; }
figcaption { font-size: 12px; margin-top: 6px; } .warn { color: #a15c00; } .bad { color: #b00020; }
ul.events { margin: 6px 0 0; padding-left: 18px; font-size: 12px; color: #333; }
table { border-collapse: collapse; font-size: 12px; } td, th { padding: 2px 10px 2px 0; text-align: left; }
@media (prefers-color-scheme: dark) { body { background: #1c1c1e; color: #eee; } figure { background: #2a2a2e; border-color: #444; }
  .meta { color: #aaa; } ul.events { color: #ddd; } }
"""


def event_note(event) -> str | None:
    """One line about an event worth reading on the sheet, or None."""
    name = event["ev"]
    who = f"{event['kind']} {event['pid']}"
    if name == "dialog_shown":
        dialog = event["dialog"]
        text = dialog.get("text") or " / ".join(dialog.get("labels", []))
        return f"{who}: dialog {dialog.get('class')} {dialog.get('title')!r}: {text[:160]!r}, buttons {dialog.get('buttons')}"
    if name == "desktop_open":
        return f"{who}: {event['function']}({event['target']!r}) -> {event.get('result')}"
    if name == "file_dialog":
        return f"{who}: file dialog {event['function']} {event.get('caption')!r} answered {event.get('answer')!r}"
    if name in ("probe_error", "probe_disabled", "capture_failed"):
        return f"{who}: {name}: {event.get('error') or event.get('reason')}"
    if name == "window_shown" and event.get("first") and not event.get("popup"):
        window = event["window"]
        return f"{who}: window shown: {window.get('class')} {window.get('title')!r}"
    if name in ("window_closed", "window_destroyed"):
        window = event.get("window") or {}
        return f"{who}: window {name.replace('window_', '')}: {window.get('class')} {window.get('title')!r}"
    return None


def command_line(process, limit=140) -> str:
    """The process's command line as a person reads it: the script's file name, then its arguments."""
    argv = process.get("argv") or []
    text = " ".join([Path(argv[0]).name, *argv[1:]]) if argv else ""
    return text if len(text) <= limit else text[:limit] + "..."


def contact_sheet(out) -> str:
    """The contact sheet of a run as HTML: every capture in order, grouped by step, with what else happened."""
    out = Path(out)
    processes = load_run(out)
    steps = store.read_steps(out)
    items = []
    for process in processes:
        for event in process["events"]:
            if event["ev"] == "capture":
                items.append(("capture", event["wall_ms"], event))
            elif event["ev"] in NOTABLE or event["ev"] in ("window_shown", "window_closed", "window_destroyed"):
                note = event_note(event)
                if note:
                    items.append(("note", event["wall_ms"], {"text": note, "step_seq": None, "step": event.get("step", ""),
                                                             "wall_ms": event["wall_ms"]}))
    for event in store.read_events(out / "attached.jsonl"):  # pictures `probe.py attach` added
        items.append(("capture", event["wall_ms"], event))
    items.sort(key=lambda item: item[1])
    # A step owns what happened from its own time until the next step's.
    boundaries = [(step["wall_ms"], step) for step in steps]

    def step_of(wall_ms):
        owner = None
        for when, step in boundaries:
            if when <= wall_ms:
                owner = step
        return owner

    groups: dict = {}
    order = []
    for kind, wall_ms, payload in items:
        step = step_of(wall_ms)
        key = step["seq"] if step else 0
        if key not in groups:
            groups[key] = {"step": step, "items": []}
            order.append(key)
        groups[key]["items"].append((kind, payload))
    for step in steps:  # a step with nothing recorded is still listed
        if step["seq"] not in groups:
            groups[step["seq"]] = {"step": step, "items": []}
            order.append(step["seq"])
    order.sort()

    body = []
    for key in order:
        group = groups[key]
        step = group["step"]
        title = html.escape(step["label"]) if step else "(before the first step)"
        stamp = f" <span class='meta'>step {step['seq']}, {clock(step['wall_ms'])}</span>" if step else ""
        captures = [payload for kind, payload in group["items"] if kind == "capture"]
        notes = [payload for kind, payload in group["items"] if kind == "note"]
        cards = "".join(card(capture) for capture in captures) or "<p class='meta'>No capture in this step.</p>"
        lines = "".join(f"<li><span class='meta'>{clock(note['wall_ms'])}</span> {html.escape(note['text'])}</li>" for note in notes)
        body.append(f"<section><h2>{title}{stamp}</h2><div class='cards'>{cards}</div>"
                    + (f"<ul class='events'>{lines}</ul>" if lines else "") + "</section>")
    rows = "".join(
        f"<tr><td>{html.escape(process['kind'])}</td><td>{process['pid']}</td><td>{clock(process['boot_wall_ms'])}</td>"
        f"<td>{'running' if process['alive'] else 'ended'}</td><td>{html.escape(command_line(process))}</td></tr>"
        for process in processes)
    count = sum(1 for kind, *_ in items if kind == "capture")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Desktop probe contact sheet</title><style>{STYLE}</style></head><body>"
        f"<header><h1>Desktop probe contact sheet</h1><div class='meta'>{count} captures, {len(processes)} processes, "
        f"made {clock(store.wall_ms())}</div></header>"
        f"<section><table><tr><th>kind</th><th>pid</th><th>started</th><th>state</th><th>command line</th></tr>{rows}</table></section>"
        + "".join(body) + "</body></html>"
    )


def card(capture) -> str:
    notes = []
    if capture.get("flat_colour"):
        notes.append(f"<span class='bad'>the whole picture is one colour ({capture['flat_colour']})</span>")
    for part in capture.get("drawn_elsewhere", []):
        colour = part.get("flat_colour")
        notes.append(f"<span class='{'bad' if colour else 'warn'}'>{html.escape(part['class'])} "
                     + (f"came out one colour ({colour})" if colour else "is in the picture") + "</span>")
    screen = capture.get("screen") or {}
    return (
        f"<figure><a href='{html.escape(capture['file'])}'><img src='{html.escape(capture['file'])}' alt='' loading='lazy'></a>"
        f"<figcaption><b>{html.escape(capture.get('title') or '')}</b><br>"
        f"<span class='meta'>{clock(capture['wall_ms'])} &middot; {html.escape(capture['kind'])} {capture['pid']} &middot; "
        f"{capture['width']}&times;{capture['height']} px, {capture['logical'][0]}&times;{capture['logical'][1]} logical, "
        f"DPR {capture['dpr']} &middot; screen {html.escape(str(screen.get('name') or 'offscreen'))} {screen.get('scale')}x &middot; {html.escape(capture['reason'])}</span>"
        + ("<br>" + "; ".join(notes) if notes else "") + "</figcaption></figure>"
    )
