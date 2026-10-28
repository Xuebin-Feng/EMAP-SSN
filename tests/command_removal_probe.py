"""Run the program with any one command taken away, in a process of its own (tests/test_command_removal.py).

usage: command_removal_probe.py SRC_DIR SCRATCH_DIR

Offscreen, and nothing is shown. First every central module is imported while every command is made
unimportable (sys.modules['commands.<name>'] = None), which shows that no central module needs a command at import
time. Then what depends on the commands is read from a copy of the commands folder that has them all (the baseline)
and from a copy that lacks one command, for each command in turn: what the modules declare for the dispatch, the
Console buttons and the Session card's pick buttons, the reset targets, the catalog, the Agent's review policy and its
instructions, the list that help prints, what typing the missing command reports, and the Viewer's own pieces that
read the declarations (the Session card, the Console buttons a session brings back, the Zoom box).

The last line of the output is one JSON object: {"imported": {module: error or null}, "baseline": {...},
"removed": {command: {...}}, "problems": {command: [what differs from the baseline, except by that command]},
"seconds": {...}}. A command whose module imports another cannot be imported without it: those commands go with the
removed one in what is read by importing the modules (the test checks the commands that do against the imports in the
source), and every other difference is a problem.
"""
import contextlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.dont_write_bytecode = True
SRC = Path(sys.argv[1]).resolve()
SCRATCH = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(SRC))

# Layout engines that load numba, umap or pacmap when they are imported take seconds and read no command.
SLOW_IMPORTS = {"Layout_Engine_UMAP", "Layout_Engine_PaCMAP"}
SOURCES = ("typed", "script", "portal")
ARGUMENTS = ([], ["stop"], ["off"], ["hello", "world"])
STARTED = time.perf_counter()
SECONDS = {}
NAMES = sorted(path.stem for path in (SRC / "commands").glob("*.py") if not path.stem.startswith("_"))


def central_modules():
    modules = []
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[0] in ("commands", "tools", "resources") or parts[-1] == "__init__" or parts[-1] in SLOW_IMPORTS:
            continue
        modules.append(".".join(parts))
    return modules


def forget_the_commands():
    for name in list(sys.modules):
        if name == "commands" or name.startswith("commands."):
            del sys.modules[name]
    importlib.invalidate_caches()


# --- 1. every central module imports while no command can be imported --------------------------------------------------
began = time.perf_counter()
import commands  # noqa: E402  (the package itself, from src)

for name in NAMES:
    sys.modules[f"commands.{name}"] = None  # `import commands.<name>` now raises ImportError
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])
IMPORTED = {}
for module_name in central_modules():
    try:
        importlib.import_module(module_name)
        IMPORTED[module_name] = None
    except BaseException as error:  # noqa: BLE001 - whatever the import raised is the finding
        IMPORTED[module_name] = f"{type(error).__name__}: {error}"
SECONDS["import with no command importable"] = round(time.perf_counter() - began, 2)
forget_the_commands()

import Command_Declarations  # noqa: E402
import Command_Engine  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from desktop import Command_Metadata, Viewer_Inspection, Viewer_Widgets  # noqa: E402
from web_ui import agent_prompt  # noqa: E402

TEMPLATE = (SRC / "resources" / "agent" / "system_prompt.md").read_text(encoding="utf-8")
REAL_CATALOG = Viewer_Inspection.command_catalog
PACKAGE = SCRATCH / "commands"
if PACKAGE.exists():
    shutil.rmtree(PACKAGE)
shutil.copytree(SRC / "commands", PACKAGE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def install_package(without=None):
    """Make ``commands`` the copy in SCRATCH, none of its modules imported yet, and ``without`` not importable."""
    forget_the_commands()
    spec = importlib.util.spec_from_file_location("commands", PACKAGE / "__init__.py",
                                                  submodule_search_locations=[str(PACKAGE)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["commands"] = package
    spec.loader.exec_module(package)
    if without is not None:
        sys.modules[f"commands.{without}"] = None


@contextlib.contextmanager
def lacking(name):
    """The copy of the commands folder without the file of command ``name``, for the length of the block."""
    hidden = SCRATCH / "hidden" / f"{name}.py"
    hidden.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(PACKAGE / f"{name}.py"), str(hidden))
    try:
        install_package(without=name)
        yield
    finally:
        shutil.move(str(hidden), str(PACKAGE / f"{name}.py"))


def quietly(function, *arguments):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*arguments)


# --- 2. what the folder gives -------------------------------------------------------------------------------------------
def numbered_items(text):
    """{command: the item's lines without its number} for the numbered items of the Agent's instructions."""
    items, current = {}, None
    for line in text.splitlines():
        found = re.match(r"^\d+\. `([a-z_][a-z_0-9]*)", line)
        if found:
            current = found.group(1)
            items[current] = [re.sub(r"^\d+\. ", "", line)]
        elif current is not None and (line.startswith(" ") or not line.strip()):
            items[current].append(line.strip())
        else:
            current = None
    return {name: "\n".join(lines).rstrip() for name, lines in items.items()}


def read_from_the_source():
    """What is read from the modules' source, without importing them: catalog, review policy, instructions."""
    names = [name for name, _path in Command_Metadata.command_paths()]
    plain, by_form = Command_Metadata.agent_policy_summary()
    return {
        "names": names,
        "catalog": {entry["command"]: entry for entry in Viewer_Inspection.command_catalog(commands_dir=PACKAGE)["commands"]},
        "policies": {name: Command_Metadata.agent_policy(name) for name in names},
        "unknown_policy": Command_Metadata.agent_policy("nosuchcommand"),
        "plain": plain,
        "by_form": [list(item) for item in by_form],
        "prompt_items": [item.name for item in Command_Metadata.agent_prompt_items()],
        "prompt": numbered_items(quietly(agent_prompt.render_system_prompt, TEMPLATE)),
        "declarations": {name: list(Command_Metadata.declarations(name)[1:]) for name in names},
    }


def read_by_importing():
    """What the modules declare, which needs them imported: dispatch, Console buttons, pick buttons, reset targets."""
    dispatch = {}
    for name in NAMES:
        module = Command_Declarations.command_module(name)
        dispatch[name] = None if module is None else {
            "queue": [[Command_Engine.queue_kind(name, list(args), source) for source in SOURCES] for args in ARGUMENTS],
            "model": [Command_Declarations.starts_model_request(name, list(args)) for args in ARGUMENTS],
            "jobs": [Command_Declarations.acts_on_jobs(name, list(args)) for args in ARGUMENTS],
            "console_button": Command_Declarations.registers_console_button(name),
        }
    taken = {"dispatch": dispatch, "pickers": quietly(Command_Declarations.session_file_pickers),
             "console_buttons": sorted(name for name in NAMES if Command_Declarations.registers_console_button(name)),
             "unimportable": sorted(name for name in NAMES if dispatch[name] is None)}
    try:
        reset = importlib.import_module("commands.reset")
    except ImportError:
        taken["reset"] = None
    else:
        owners = reset.declared_targets()
        usage = reset.usage().removeprefix("Usage: reset [").removesuffix("]").split(" | ")
        lines = [line.strip() for line in reset.targets_help().split("\n") if line.startswith("      ") and not line.startswith("          ")]
        taken["reset"] = {"targets": {name: [module.__name__.rpartition(".")[2] for module in modules]
                                      for name, modules in owners.items()},
                          "usage_words": usage, "help_lines": lines,
                          "word_targets": {word: reset.target_name(word, owners)
                                           for word in usage + [w for line in lines for w in line.split(", ")]}}
    return taken


def read_help():
    """What `help` lists, and what `help <a name that is no command>` says."""
    try:
        help_command = importlib.import_module("commands.help")
    except ImportError:
        return None
    refusals = []
    viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
    output = io.StringIO()

    def catalog(command=None, commands_dir=None):
        return REAL_CATALOG(command, commands_dir=PACKAGE)

    with mock.patch.object(Viewer_Inspection, "command_catalog", catalog), \
            mock.patch.object(Command_Engine, "command_succeeded", lambda *a, **k: None), \
            mock.patch.object(Command_Engine, "command_failed", lambda viewer, message=None, *a, **k: refusals.append(str(message))), \
            mock.patch.object(Command_Engine, "show_status", lambda *a, **k: None), \
            mock.patch.object(Command_Engine, "print_help", lambda *a, **k: None), \
            contextlib.redirect_stdout(output):
        help_command.run(viewer, [])
        listing = output.getvalue()
        help_command.run(viewer, ["nosuchcommand"])
    # A command's line starts with two spaces and its name; a summary that wraps goes on under the summary, further in.
    listed = [found.group(1) for found in (re.match(r"^  (?=\S)(\S+)", line) for line in listing.splitlines())
              if found and found.group(1) != "Type"]
    return {"listed": listed, "refusals": refusals}


def type_a_command(name):
    """What typing ``name`` reports as a failure, with nothing else of the Viewer there."""
    failures = []
    viewer = SimpleNamespace(command_history=[])
    with mock.patch.object(Command_Engine, "command_failed", lambda viewer, message=None, *a, **k: failures.append(str(message))), \
            mock.patch.object(Command_Engine, "show_status", lambda *a, **k: None), \
            contextlib.redirect_stdout(io.StringIO()):
        Command_Engine.execute_command(viewer, name, record_history=False, silent=True)
    return failures


def try_the_engines_reset():
    """What the engine's name for a reset raises when there is no reset command."""
    try:
        Command_Engine.execute_reset(SimpleNamespace(), ["colors"])
    except BaseException as error:  # noqa: BLE001
        return type(error).__name__
    return None


def build_the_viewers_pieces(console_button_commands):
    """Build, offscreen, what the Viewer builds from what commands declare; say what came of it."""
    host = QWidget()
    typed = []
    info = {"node_count": 3, "edge_count": 2, "directories": {}, "project_root": "",
            "inputs": {key: os.path.abspath(key + ".dat") for key, _label in Viewer_Widgets.SESSION_FILES}}
    card = Viewer_Widgets.ControlsCard("x", "[ENTER] Open the command line", parent=host, session=lambda: info,
                                       command=typed.append)
    built = {"pickers": dict(card.pickers), "pick_buttons": sorted(card.pick_buttons)}
    for key in sorted(card.pick_buttons):
        card.pick_buttons[key].click()
    built["typed_by_the_buttons"] = sorted(typed)
    card.close()
    card.deleteLater()

    # The Console buttons a session brings back: the commands it lists that declare one are run, nothing else.
    run_by_the_viewer = []
    stand_in = SimpleNamespace(sidebar_buttons_to_persist=[*console_button_commands, "select", "nosuchcommand"],
                               process_command=lambda line, **options: run_by_the_viewer.append(line))
    quietly(EMAPSSN_Viewer.MainViewer._register_persisted_console_buttons, stand_in)
    built["register_only"] = run_by_the_viewer
    built["kept"] = list(stand_in.sidebar_buttons_to_persist)

    # The Zoom box is the Viewer's own: it needs no command to be built.
    viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
    holder = QWidget()
    viewer.canvas = SimpleNamespace(native=holder, size=(1200, 800), update=lambda: None)
    viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 38.0}
    viewer._hud_timer = mock.Mock()
    viewer.min_threshold, viewer.max_threshold, viewer.current_slider_threshold = 0.0, 9.6, 3.74
    viewer.update_edges = mock.Mock()
    quietly(viewer._build_slider_overlay)
    built["zoom_box"] = viewer.zoom_box is not None
    holder.deleteLater()
    host.deleteLater()
    return built


def snapshot(missing=None):
    """Everything the installed folder of commands gives, as data. ``missing`` is the command that was taken away."""
    taken = read_from_the_source()
    taken.update(read_by_importing())
    taken["help"] = read_help()
    taken["viewer"] = build_the_viewers_pieces(taken["console_buttons"])
    if missing is not None:
        # Typing a command that cannot be imported runs nothing, so only those are typed.
        taken["typed"] = {name: type_a_command(name) for name in [missing, *taken["unimportable"]]}
        if missing == "reset":
            taken["engines_reset"] = try_the_engines_reset()
    return taken


# --- 3. what removing a command must not change ------------------------------------------------------------------------
def problems_of(baseline, taken, name):
    """What differs from the baseline other than by ``name`` and the commands that cannot be imported without it."""
    found = []
    gone = {name, *taken["unimportable"]}

    def expect(what, expected, actual):
        if expected != actual:
            found.append(f"{what}: expected {json.dumps(expected, sort_keys=True)[:500]}, got {json.dumps(actual, sort_keys=True)[:500]}")

    without = lambda mapping: {key: value for key, value in mapping.items() if key != name}  # noqa: E731
    # Read from the source: only the removed command goes.
    expect("the commands the folder lists", [n for n in baseline["names"] if n != name], taken["names"])
    expect("the catalog's entries", without(baseline["catalog"]), taken["catalog"])
    expect("the Agent's review policy", without(baseline["policies"]), taken["policies"])
    expect("the policy of a command that is not there", baseline["unknown_policy"], taken["unknown_policy"])
    expect("the commands that run at once, wait or are refused", {kind: [n for n in names if n != name]
                                                                 for kind, names in baseline["plain"].items()}, taken["plain"])
    expect("the commands reviewed by their first word", [item for item in baseline["by_form"] if item[0] != name], taken["by_form"])
    expect("the commands the Agent's instructions describe", [n for n in baseline["prompt_items"] if n != name], taken["prompt_items"])
    expect("the text of each item of the Agent's instructions", without(baseline["prompt"]), taken["prompt"])
    expect("what the modules declare for the catalog and the Agent", without(baseline["declarations"]), taken["declarations"])
    # Read by importing: the removed command goes, and the commands that import it.
    expect("the dispatch declarations", {n: v for n, v in baseline["dispatch"].items() if n not in gone},
           {n: v for n, v in taken["dispatch"].items() if n not in gone})
    expect("the Console buttons", [n for n in baseline["console_buttons"] if n not in gone], taken["console_buttons"])
    expect("the Session card's pick buttons", {key: n for key, n in baseline["pickers"].items() if n not in gone}, taken["pickers"])
    # reset
    if name == "reset":
        expect("reset's targets with no reset command", None, taken["reset"])
        expect("what the engine's name for a reset raises without it", "ModuleNotFoundError", taken["engines_reset"])
    else:
        before = baseline["reset"]
        targets = {target: [m for m in modules if m not in gone] for target, modules in before["targets"].items()}
        expect("the reset targets", {target: modules for target, modules in targets.items() if modules}, taken["reset"]["targets"])
        listed = lambda word: bool(targets[before["word_targets"][word]])  # noqa: E731
        expect("the words of reset's usage line", [word for word in before["usage_words"] if listed(word)], taken["reset"]["usage_words"])
        expect("the lines of reset's help", [line for line in before["help_lines"] if any(listed(word) for word in line.split(", "))],
               taken["reset"]["help_lines"])
    # help
    if name == "help":
        expect("help's own listing", None, taken["help"])
    else:
        expect("help's listing", [n for n in baseline["help"]["listed"] if n != name], taken["help"]["listed"])
        if len(taken["help"]["refusals"]) != 1 or "nosuchcommand" not in taken["help"]["refusals"][0]:
            found.append(f"help for a name that is no command said {taken['help']['refusals']}")
    # typing
    if taken["typed"][name] != [f"Unknown command: {name}"]:
        found.append(f"typing {name}, which is gone, reported {taken['typed'][name]}")
    for dependent in taken["unimportable"]:
        reported = taken["typed"][dependent]
        if len(reported) != 1 or not reported[0].startswith(("Command dependency unavailable", "Unknown command")):
            found.append(f"typing {dependent}, which needs {name}, reported {reported}")
    # the Viewer's pieces
    pieces = taken["viewer"]
    expect("the Session card's buttons", taken["pickers"], pieces["pickers"])
    expect("the files the card lets a button pick", sorted(taken["pickers"]), pieces["pick_buttons"])
    expect("what the card's buttons type", sorted(taken["pickers"].values()), pieces["typed_by_the_buttons"])
    expect("the Console buttons a session brings back", [f"{n} --register-only" for n in taken["console_buttons"]], pieces["register_only"])
    expect("the Console buttons a session keeps", taken["console_buttons"], pieces["kept"])
    expect("the Zoom box is built", True, pieces["zoom_box"])
    return found


# --- 4. the baseline, then each command in turn ------------------------------------------------------------------------
began = time.perf_counter()
install_package()
BASELINE = snapshot()
SECONDS["baseline"] = round(time.perf_counter() - began, 2)
REMOVED, PROBLEMS = {}, {}
for name in NAMES:
    began = time.perf_counter()
    with lacking(name):
        taken = snapshot(missing=name)
    PROBLEMS[name] = problems_of(BASELINE, taken, name)
    REMOVED[name] = {"seconds": round(time.perf_counter() - began, 2), "unimportable": taken["unimportable"],
                     "reset_targets": sorted(taken["reset"]["targets"]) if taken["reset"] else None,
                     "listed_by_help": taken["help"]["listed"] if taken["help"] else None,
                     "prompt_items": taken["prompt_items"], "pickers": taken["pickers"],
                     "console_buttons": taken["console_buttons"], "typed": taken["typed"]}
install_package()
SECONDS["total"] = round(time.perf_counter() - STARTED, 2)
print(json.dumps({"imported": IMPORTED,
                  "baseline": {"names": BASELINE["names"], "unimportable": BASELINE["unimportable"],
                               "pickers": BASELINE["pickers"], "console_buttons": BASELINE["console_buttons"],
                               "reset_targets": sorted(BASELINE["reset"]["targets"]), "help": BASELINE["help"]["listed"]},
                  "removed": REMOVED, "problems": PROBLEMS, "seconds": SECONDS}, sort_keys=True))
