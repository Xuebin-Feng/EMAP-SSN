# Desktop probe

A test-only hook for EMAP-SSN's own Python processes. It lets you, the test agent, **see, capture and operate the
Qt windows from inside the process**. It needs no desktop permission: no screenshot portal, no accessibility
bus, no synthetic X events. It works the same on GNOME Wayland (through XWayland or native), X11, Windows and
macOS, because everything it does is `QWidget.grab()` and Qt's own event delivery.

Nothing in `src/` knows it exists. It acts only while a gate file exists, only in this checkout's entry points,
and never in a test run.

## The round, in commands

Run every command with the **clone's own interpreter** (`.venv/bin/python`; on Windows `.venv\Scripts\python.exe`):

```bash
P=".venv/bin/python tests/desktop_probe/probe.py"

$P install                  # once per clone: writes a .pth + a stub into .venv/site-packages
$P on --fresh               # turn it on (the gate file), start an empty output folder
$P status                   # installed? on? which processes are recording?

# ... start EMAP-SSN the way the user does: the menu entry, the .desktop file, EMAPSSN_Desktop_Launcher.sh ...
# (nothing has to be set in the environment: the hook finds its gate file, so a launch from GNOME works)

$P step "1 launch Config"                       # label what follows; every event and capture carries it
$P wait --kind splash --title . --timeout 30    # wait for a window (prints it); --gone waits for it to close
$P do --kind splash click widget="Show details" # click it, as a user does
$P do --kind splash capture                     # a PNG of the splash, at the screen's scale
$P do --kind config windows                     # what is open, geometry, DPR, screen, focus
$P timeline                                     # per launch: process start -> splash shown/painted -> window shown -> splash closed
$P sheet                                        # writes contact-sheet.html: read THAT, no screenshots needed

$P off                      # quiet in every process within about three seconds
$P uninstall                # remove the .pth and the stub (nothing else was changed)
```

Every window is also captured **automatically ~300 ms after it is first shown** (`--settle-ms N` changes it,
`--no-auto-capture` turns it off), so the contact sheet fills itself as you go. `step` labels group them.

## What loads it, and when it stays out

`install` writes `emapssn_desktop_probe-<hash>.pth` and `emapssn_desktop_probe_stub.py` into the clone's
`.venv` site-packages. The `.pth` is one line: it checks that `<clone>/temp/desktop_probe/enabled` exists and
only then imports the stub. So:

* **Gate off** (`probe.py off`, or never turned on): an interpreter start costs one `stat`. Nothing is imported,
  no Qt, nothing is written.
* **Gate on**: the stub loads `tests/desktop_probe/boot.py` from the clone and asks one question: *is this
  process one of this clone's entry points?* The list is `ENTRY_POINTS` in `boot.py`: `src/EMAPSSN_Config.py`
  (kind `config`), `EMAPSSN_Tools.py` (`tools`), `EMAPSSN_Viewer.py` (`viewer`), `Layout_Cache_Generator.py`
  (`layout`), `EMAPSSN_MCP_Server.py` (`mcp`), `desktop/Desktop_Launcher.py` (`splash`),
  `desktop/Desktop_Launcher_Monitor.py` (`monitor`). `EMAPSSN_Config.py --headless launch-viewer`, which the MCP
  server uses, runs the Viewer in-process and is a `viewer`. Anything else (pip, uv, `python -c`, a tool script,
  `run_sections.py` and every child of a test run, which carry `EMAPSSN_TEST_ENVIRONMENT`) is left alone.
  `EMAPSSN_DESKTOP_PROBE=0` in a process's environment also keeps it out.
* It never changes what the program does: no window of its own, no focus change, no swallowed event, nothing
  kept alive. A failure inside the probe is written to `procs/<pid>-<kind>/errors.log` and the probe turns
  itself off **in that process**; a failure to load is in `<clone>/temp/desktop_probe/boot-errors.log`.

It works through every launch path: the `.pth` is read by the venv's interpreter however it is started
(`-u`, `-I`, an environment the launcher rebuilt, GNOME Shell starting the `.desktop` entry, the monitor,
Config's Save & Run, the Layout Generator, the MCP server and its Viewers). `-S` would skip it; nothing in the
launch chain uses `-S`.

## Where it writes

`probe.py on --out DIR` chooses the folder; the default is `<clone>/temp/desktop_probe/out/` (`temp/` is
git-ignored).

```
step.json, steps.jsonl             the step label, and the history of them
captures/<kind>-<pid>-<n>-<step>-<title>.png
procs/<pid>-<kind>/process.json    argv (secrets redacted), start time, parent chain
procs/<pid>-<kind>/events.jsonl    one JSON object per event, wall-clock ms and monotonic ms
procs/<pid>-<kind>/windows.json    the window inventory (what `wait` reads)
procs/<pid>-<kind>/errors.log      only if the probe failed in that process
requests/  responses/              the request protocol below
files/                             where {files} in a dialog answer points
contact-sheet.html                 what `sheet` writes
```

Events (all carry `wall_ms`, `mono_ms`, `pid`, `kind`, `step`): `process_start` (argv), `app_ready`
(platform, Qt, screens with scale and DPI), `window_shown` (class, objectName, title, geometry and minimum size
in logical px, devicePixelRatio, screen name and scale, active, focus widget and whether it draws focus),
`window_first_painted`, `window_changed`, `window_hidden`, `window_close_requested`, `window_closed`,
`window_destroyed`, `focus_changed`, `dialog_shown` (title, text, buttons), `file_dialog`, `desktop_open`
(`open_path` / `open_url` arguments; the call still goes through), `capture`, `request`, `exit`.
Everything written is passed through a redaction filter (key shapes, `--api-key X`, `"token": "..."`, long random
runs of letters and digits); it is a pattern filter, not a guarantee. It leaves a FASTA header alone, also one whose
spaces became underscores (`sp|Q5ZKK2|GEMI4_CHICK_Gem-associated_protein_4_OS=Gallus_gallus_OX=9031`): a long run of
words, numbers and short codes joined by `_` or `-` is a name, not a key, and a secret's name must end its word
(`client_secret=...` and `accessToken: "..."` are masked, `Secretogranin-2_OS=...` and `gamma-secretase` are not).

## Operating the program: `probe.py do`

```bash
$P do --kind viewer windows
$P do --pid 4242 click widget="Save & Run"            # key=value pairs, JSON values allowed
$P do --kind config '{"action": "click", "window": "Configuration", "widget": {"text": "Save & Run"}}'
$P do --kind config @request.json                       # or a file, or - for stdin
```

`--kind` picks the newest live process of that kind (`splash config tools viewer layout mcp monitor`); `--pid`
is exact. The answer is JSON; the exit code is 0 when `ok` is true. Under the hood it writes
`requests/pid-<n>--<id>.json` and reads `responses/<id>.json`; you may do that yourself (a process polls every
~100 ms; a request named `kind-<kind>--<id>.json` goes to whichever process of that kind claims it first).

**Order.** A process handles its requests one at a time, in the order they were written (the files' modification
times, then their names), so commands started together, or request files written one after the other, do what
their order says. A request is handled (its window and widget looked up, its answer made or its action started)
only once the one before it has been answered. So an `eval` written after a `click` or a `viewer_command` sees what
that did, once its settle time has passed, and a click on a window the click before it opens finds the window.
Three things to know: a request that is still running when its answer falls due (a modal dialog, `block_ms`) counts
as answered, so the request that clicks the dialog's button is not held up behind it; a request file that is not
yet valid JSON and younger than a second (an agent is still writing it) holds back the ones written after it; and
a `viewer_command` that only queued the command (`outcome` null) has not finished when the next request runs: ask
again with `eval` or `windows` to see it end.

**Targets.** `window` is a title regex, or `{"title", "class", "objectName"}`, or an index from `windows`; with
none, the active window. When the regex matches several windows (the Viewer and its Console share a title
prefix) the active one is used, else the newest, and the answer carries a `window_note` saying so: anchor the
regex (`"\\]$"`, `"Console$"`) before a `close`. `widget` is a string or an object. A string matches `objectName`, accessible name,
visible text or tooltip (exact first, then as a substring; "Save & Run" matches a button written `Save && Run`
and a label written `Run &it` matches "Run it"). An object combines `objectName`, `accessibleName`, `text`,
`toolTip`, `class`, `path` (the child indices `tree` prints, e.g. `"0/2/1"`), `index` (which of several
matches), and the parts that are not widgets: `{"tab": "Advanced"}`, `{"item": "gamma"}` (a row of a list, tree
or combo popup), `{"action": "Copy"}` (a menu entry). Several matches are an error that lists them, so a click
never lands on a guess.

| action | what it does | example |
|---|---|---|
| `windows` | inventory of the visible windows, focus widget, current step | `do --kind config windows` |
| `tree` | the widget tree of a window (class, objectName, accessibleName, text, toolTip, geometry, visible, enabled, path) | `do --kind config tree window=Configuration depth=3` |
| `capture` | PNG of a window or one widget; `all=true` for every window | `do --kind viewer capture window=Console` |
| `click` | a real mouse press and release through the window, at the widget's centre (focus follows as for a user); `button`, `double`, `modifiers`; `link="text or href"` for a link in a QLabel | `do --kind splash click widget="Show details"` |
| `hover` | move the pointer there (tooltips appear; `capture window={"class":"QTipLabel"}`) | `do --kind config hover widget="Compute Network Statistics"` |
| `type` | key events into the named field (clicked first) or the focused one; any characters; `clear=true` first | `do --kind config type widget='{"class":"QLineEdit","index":0}' text="héllo" clear=true` |
| `key` | key presses: `Return`, `Esc`, `Tab`, `Ctrl+A`, a list of them (shortcuts fire) | `do --kind viewer key keys='["Ctrl+A","Delete"]'` |
| `set_value` | combo (by text or index), spin box, slider, check box and toggle switch (clicked if different), line edit, tab bar | `do --kind config set_value 'widget={"class":"NoScrollComboBox","index":2}' value=UMAP` |
| `close` | like the title-bar x (`accepted` says whether the window let itself close) | `do --kind viewer close window=Console` |
| `resize` | to `width`/`height`, or `to_minimum=true` (Qt clamps to what the layout allows; the answer says what it is) | `do --kind config resize to_minimum=true` |
| `clipboard` | read the clipboard text (`set="..."` sets it) | `do --kind viewer clipboard` |
| `viewer_command` | a command line, exactly as typed in the Viewer's command line (the call its Enter makes); returns `outcome` (`succeeded`/`failed`, null while queued), `status_text`, the lines it printed | `do --kind viewer viewer_command command=help` |
| `dialog_answers` | fill the file-dialog queue (below) | see below |
| `ping` | pid, kind, platform, uptime | `do --kind tools ping` |
| `eval` | evaluate a Python expression in the process (`mode=exec` runs statements, set `result`); names: `app windows window widget viewer main QtCore QtGui QtWidgets`. **Test-only**: it runs arbitrary code in the program | `do --kind viewer eval expr="viewer.n_nodes"` |

Actions that act like a user (`click type key set_value close hover resize viewer_command`) run from a timer and
are answered ~150 ms after they return (`settle_ms` changes it, 400 ms for `viewer_command`); the answer says
which windows are open afterwards. `{"completed": false}` means the action is still running: see modal dialogs.

**Modal dialogs.** A `QMessageBox.exec()` or any dialog that runs its own event loop does not stop the probe:
its timer keeps firing inside that loop. The click that opened it is answered with `completed: false` after
1.5 s (`block_ms`); then `windows` shows the dialog (`modal: true`), `capture window="<its title>"` takes it,
and `click window="<its title>" widget="Yes"` answers it. `dialog_shown` events already carry its text and buttons.

**File dialogs.** On GNOME a native file dialog is another process and cannot be driven. While the probe is on,
`QFileDialog.getOpenFileName / getOpenFileNames / getSaveFileName / getExistingDirectory` (and the URL
variants) take their answers from a queue you fill **before** the click. `{files}` stands for `<out>/files`:

```bash
$P do --kind viewer dialog_answers 'answers=[{"method":"getSaveFileName","path":"{files}/console.txt"}]'
$P do --kind viewer click window=Console widget="Save the output shown to a file"
```

An answer is `{"path": ...}`, `{"paths": [...]}` or `{"cancel": true}`, optionally `"method"` (default: the
next call of any kind) and `"filter"`. With the queue empty a call answers like Cancel, so the program never
waits for a dialog nobody can see. Every call is recorded (`file_dialog`: caption, directory, filter, answer).

## Captures

`QWidget.grab()` at the widget's device pixel ratio: a 200 % screen gives 2x pixels, 166.67 % gives 1.6667x
(`dpr`, `width`, `height`, `logical` are in every capture and on the sheet). Windows made with
`WA_DontShowOnScreen` (the Viewer's hidden mode) are captured too.

* **The Viewer's GL canvas** (VisPy's `QOpenGLWidget`): on Windows, with a real OpenGL context (normal and
  hidden mode), it **is** in the grab. Offscreen platforms have no GL and the Viewer does not start. On
  XWayland with the NVIDIA driver it is **not verified**: every capture lists such widgets in `drawn_elsewhere`
  with `flat_colour` (a colour only if every pixel in that rectangle is the same), and the sheet marks it in
  red. If the canvas comes out flat, use the Viewer's own command and add the picture to the sheet:
  `do --kind viewer viewer_command command="print canvas"` writes `canvas.png` under the analysis results
  folder's `Saved_Images` (the answer's `status_text` and `printed` name it), then
  `$P attach <that file> --caption canvas`.
* Web views (`QWebEngineView`, the Agent and Fold View pages) are GPU-composited and may come out blank; the
  same `drawn_elsewhere` note says so.
* A window that looks like it holds a key or password (its title or objectName says token, key, password...)
  is not auto-captured and is refused unless the request says `"force": true`. Never capture the Agent chat
  settings or the Biohub token dialog with a real key typed in.

## Timeline and sheet

`probe.py timeline` joins the chain launcher -> splash -> monitor -> window process (by parent pid) and prints,
per launch, milliseconds from the **process's own start** to: splash shown, splash first painted, the window shown
and first painted, the splash closed. "Shown" is when the probe first looked at the window: a scan runs every
25 ms in the GUI thread, so a splash is stamped within one scan of its show (the line says `seen by a scan`),
and a window the program shows while its GUI thread is still busy is stamped when the loop first runs again.
"First painted" is the first event-loop turn after the platform exposed the window.

`probe.py sheet` writes `contact-sheet.html` into the output folder: one self-contained page with every capture in
order, grouped by step, each with process, time, title, size, DPR, screen, and under each step the dialogs shown,
`open_path`/`open_url` calls, file dialogs and windows that opened or closed. Open it in a browser and review that
instead of screenshots.

## What does not work, or is not known

* **Nothing the window system does** is exercised: real pointer, real keyboard focus, real stacking, the
  compositor's drawing, native menus and native file dialogs. `click` and `key` are Qt events delivered through
  the window (the way Qt's own tests do), so the program sees what it sees for a user, but a Wayland compositor
  never sees them. Window position (`geometry` x, y) is 0, 0 on native Wayland.
* `activate`/focus: a click does not raise or activate the window; `windows` says which is active.
* The probe cannot be loaded into a process started with `python -S`, and sees only this clone's venv.
* Cost, measured on Windows (a laptop with Defender running): gate off, one `stat` per interpreter start (no
  difference measurable). Gate on: about 10 ms later for an entry point's first line (the rest of the start-up
  work is in a thread), about 50 ms (2 %) longer for Config to be ready, and about 1 % of one core while a window
  sits idle (the 25 ms scan). Files are written under the output folder only.
* `eval` and the request folder are a way to run code in the program: use the probe only in a test clone, and
  turn it off (`probe.py off`) and remove it (`probe.py uninstall`) when the round is over.
