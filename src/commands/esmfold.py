# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
import weakref
import Command_Engine
import EMAPSSN_Config as cfg
import web_ui.esmfold_backend as esmfold_backend
from PySide6.QtWidgets import QApplication, QMessageBox
from desktop import Biohub_Token_Dialog as token_dialog
from desktop.Desktop_App import translate
from utilities import Residue_Data
from utilities.Localization import JoinedMessage, Message
from utilities.Terminal_Launcher import launch_without_window
from Viewer_Worker_Tracking import worker_directory
from web_ui.Browser_Page import is_headless
from desktop.Command_Metadata import argument, choice, entry

# --- What this command declares about itself ---
# Starts a worker process, and `large` spends Biohub credit.
AGENT_POLICY = 'approve'

CATALOG_ENTRY = entry('Open the structure viewer or submit selected sequences for ESM3 folding. With an alignment loaded, a folded node it holds also gets per-residue data (Information Content, Residue Agreement) saved beside its PDB as <pdb name>.residues.json; the result says what was saved and why not.',
    argument('options', 'Omit with no selection to open the structure viewer: nothing is folded then, and a node that was only clicked is not folded either. With nothing selected, multi, large and refold fail. Folding defaults to local ESM3 and exactly one selected node; multi permits multiple nodes, large uses Biohub. A selected node whose structure is saved already (the file this fold would write, not empty: a local and a large fold write different files) is not folded again: it is shown in the structure viewer from that file, no job is queued for it, and the outcome names it and says that refold folds it again. refold folds such nodes anyway and replaces the file. When every selected node is saved, no job is queued and the command succeeds with that note; with an alignment loaded, a shown structure also gets its per-residue data file written. large, multi and refold may be combined in any order; duplicates are invalid. Each fold is a job of the background scheduler, in its own folding line beside the line of the file jobs (label, logo, print): it is never refused because another fold is running, it waits behind the folds already queued, and one worker runs at a time so that only one model is in memory. A node that another fold is to write, or is writing, is left out and the outcome names the job that holds it; when none is left and none was shown from a saved file the command is refused. The worker runs without any window, with its output in a log file the outcome names. large needs your Biohub API token, saved in the Biohub API settings file or given as ESM_API_KEY: without one the Viewer asks for it in a dialog before queueing (the request shows awaiting_user_input meanwhile, a Viewer that shows no windows fails the command at once, and cancelling queues nothing), and if Biohub rejects the saved token while folding it asks for a replacement once and folds the structures not yet folded with it. The token is never accepted as a command argument.',
        choice('large'), choice('multi'), choice('refold'), choice('--register-only')),
    argument('stop', 'Optional word stop, given alone: ends the fold that is running, with every process it started, and takes the folds that wait out of the line; nothing else. Structures already saved are kept, a half-written one is deleted, and each fold\'s job is reported cancelled. With none running or waiting it succeeds and says so. A request holding only stop commands runs at once, even while the esmfold request it ends still waits for its worker. Closing a Viewer that shows no windows ends its worker too.', choice('stop')), writes_files=True, opens_interface=True)

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 240
AGENT_PROMPT = """
`esmfold [large] [multi] [refold] | esmfold stop`
- With no keyword and no selected node, adds the Fold View Console button and opens the browser Mol* structure viewer, and folds nothing, even if a node was clicked. With exactly one selected node, the default mode runs local ESM3 1.4B structure prediction; with several, the command fails unless `multi` is supplied.
- `large` routes prediction through the Biohub API and needs a selected node (a clicked node does not count). `multi` processes all selected nodes sequentially. `large`, `multi` and `refold` combine in any order.
- A selected node whose structure file is saved already (local and `large` folds write different files) is shown in Fold View, not folded again: no job is queued for it, and the outcome names it and says `refold` folds it again.
- Sequences come from the configured source FASTA; structures go to `Cache_Files/Predicted_Structures/` by default; `large` uses no local compute hardware.
- Each fold is a job of the background scheduler's folding line, beside the line of `label`, `logo` and `print`: it waits behind queued folds instead of being refused, one worker runs at a time, without any window. A node another fold is to write or is writing is left out, and the outcome names that job. `large` needs the user's Biohub API token: with none saved the Viewer asks for it in a dialog, which the user must answer (never ask for the token or put it in a command), and a Viewer that shows no windows fails the command.
- While an alignment is loaded, each structure folded or shown for a node the alignment holds also gets `<pdb name>.residues.json` beside its PDB: per-residue Information Content and Residue Agreement from the whole loaded alignment. The result says what was saved, and why not.
- `esmfold stop` ends the running fold with every process it started and takes the waiting folds out of the line; nothing else. Saved structures are kept, a half-written one is deleted, each fold's job is reported cancelled; with none it succeeds and says so.
"""


# How the Viewer dispatches this command (Command_Declarations).
# `esmfold --register-only` brings back the Fold View button of the Console when a session loads.
CONSOLE_BUTTON = True


def queue_kind(args, source):
    """`esmfold stop` ends the fold the line may be waiting for, so it runs at once; any other form waits its turn."""
    return 'instant' if args and args[0].lower() == 'stop' else 'ordered'


def acts_on_jobs(args):
    """The command portal runs `esmfold stop` ahead of a request that waits for the very jobs it ends."""
    return [word.lower() for word in args] == ['stop']


def print_help():
    print("""
    ESM3 3D Structure Prediction
    ============================
    Usage:
      esmfold
          With exactly 1 node selected, folds it using ESM3 1.4B (biohub/esm3-sm-open-v1).
          With no node selected, folds nothing, even if a node was clicked: it adds the "Fold View" button
          to the Console's button row and opens the Mol* viewer in the browser.
      esmfold multi
          Folds all currently selected nodes sequentially using ESM3 1.4B (biohub/esm3-sm-open-v1).
      esmfold large
          Folds exactly 1 selected node through the Biohub API using the ESM3 model configured in Biohub_API.json.
      esmfold large multi
      esmfold multi large
          Folds all selected nodes sequentially through the configured Biohub ESM3 API model.
      esmfold refold
          Folds the selected nodes even when their structures are saved already (see below).
          It may be combined with large and multi in any order, as in esmfold refold large multi.
      esmfold stop
          Ends the fold that is running, with every process it started, and takes the folds
          that wait out of the line. Structures already saved are kept, a half-written one
          is deleted, and each fold's job is reported cancelled. With none running or
          waiting it succeeds and says so.
      esmfold help
          Displays this help message.

    Saved structures: a selected node whose structure file is saved already (the file this
    fold would write, not empty; a local fold's and a Biohub fold's are different files) is
    not folded again. It is shown in the Fold View page from that file, no job is queued for
    it, and the outcome names it and says that refold folds it again. When every selected
    node is saved, no job is queued, no model starts, and the command succeeds with that note.

    Each esmfold is a job of the background scheduler's folding lane: it never refuses, it
    waits its turn behind the folds already queued, and one worker runs at a time so that
    only one model is in memory. The Viewer's Session card lists the folds (Background jobs,
    Folding), where a waiting fold can be moved or removed and the running one stopped; jobs
    lists and cancels them too. A node that another fold is to write, or is writing, is left
    out of the new one, and the outcome says which job holds it.

    Per-residue data: while an alignment is loaded, each structure folded or shown for a node the
    alignment holds gets a file beside its PDB, <pdb name>.residues.json, with two values
    for every residue from the whole loaded alignment: Information Content (the logo
    command's column height, as a percentage of log2(20) bits) and Residue Agreement (the
    share of aligned sequences with a standard amino acid in the column that have this
    protein's residue). The command's result says what was saved, and why not for a node
    when no alignment is loaded, the alignment lacks the node, or holds another sequence.

    The worker runs without any window, with its output in a log file that the outcome
    names. A headless Viewer (no screen) also opens no browser page, and the outcome gives
    the structure viewer's address. Closing the Viewer ends the worker.

    Biohub token: esmfold large needs your Biohub API token, saved in Biohub_API.json (or
    given as ESM_API_KEY). Without one, the Viewer asks for it in a dialog (typed hidden)
    before the fold is queued, and saves it; cancelling queues nothing. If Biohub rejects the
    saved token while folding, the Viewer asks for a replacement, and the structures not yet
    folded go on with it, once. A Viewer that shows no windows cannot ask: the command fails.
    The token is never taken from a command, and is shown and logged nowhere.
    """)

# A command's outcome names the selected nodes it skipped when there are this few.
MAX_LISTED_SKIPPED = 5


def sanitize_filename(name):
    import re
    # Replace any character that is not alphanumeric, a dash, dot, or underscore with '_'
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)


def _set_console_text(viewer, message):
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, message)


def _report_usage_error(viewer, message):
    """Report message, a Message, in English to the terminal and translated on the console line."""
    print(f"Error: {message}")
    Command_Engine.command_failed(viewer, f'Error: {message}')
    print("Usage: esmfold [large] [multi] [refold]")
    _set_console_text(viewer, Message("Error: {error}", error=message))


def _open_structure_viewer(viewer):
    """Open the Mol* page; return whether it opened.

    A command run for the MCP or the agent page cannot wait for someone to
    close the dialog that says the page is already open. A headless Viewer
    has no browser to open it in.
    """
    from Viewer_Command_Portal import CURRENT
    if is_headless(viewer):
        return False
    if CURRENT.get() is None:
        return esmfold_backend.open_esmfold_ui(viewer)
    return esmfold_backend.open_esmfold_ui(viewer, show_existing_dialog=False)


def _structure_viewer_connected(viewer):
    has_event_client = getattr(getattr(viewer, "web_server", None), "has_event_client", None)
    return bool(callable(has_event_client) and has_event_client("esmfold"))


def _structure_viewer_address(viewer):
    """The URL the Viewer serves the Mol* page at, or None while its web server is unavailable."""
    try:
        return viewer.get_web_url("/esmfold.html")
    except RuntimeError:
        return None


def _environment_is_headless():
    return os.environ.get('SSN_VIEWER_HEADLESS') == '1' or os.environ.get('QT_QPA_PLATFORM') == 'offscreen'


def _discard_worker_input(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _residue_data_inputs(viewer):
    """What the per-residue data providers read from the Viewer: its alignment, and each node's row in it."""
    alignment = getattr(viewer, "alignment", None)
    if alignment is None or getattr(alignment, "aln", None) is None:
        return Residue_Data.ViewerInputs()
    mapping, _ = Command_Engine.get_alignment_mapping(viewer)
    return Residue_Data.ViewerInputs(alignment, mapping, tuple(viewer.full_headers))


def _residue_data_note(kind, count):
    """The sentence for `count` structures that get no per-residue data, because of `kind`."""
    if kind == Residue_Data.NOT_IN_ALIGNMENT:
        return Message("No per-residue alignment data for %n structure(s): not in the alignment.", n=count)
    if kind == Residue_Data.SEQUENCE_DIFFERS:
        return Message(
            "No per-residue alignment data for %n structure(s): the sequence differs from the alignment's.",
            n=count,
        )
    return Message("No per-residue alignment data for %n structure(s): it could not be computed.", n=count)


def _worker_records(viewer, indices, nodes):
    """(the records of the worker's input file, what to say about their per-residue data).

    A record is [name, sequence], or, for a node with per-residue data from the
    loaded alignment (utilities.Residue_Data), [name, sequence, extra] where
    extra is {"header": the node's full header, "properties": {...}}. The worker
    writes the data beside the node's PDB. Without an alignment, for a node the
    alignment lacks or whose sequence it holds differently, no property exists
    and the record keeps its two items. The second value is a list of
    (Message, names) pairs: the names of the structures the sentence is about,
    for the outcome to list when there are few.
    """
    try:
        inputs = _residue_data_inputs(viewer)
        failure = None
    except Exception as error:  # Per-residue data must never stop a structure from being folded.
        inputs, failure = Residue_Data.ViewerInputs(), error
        print(f"Warning: Could not read the loaded alignment for per-residue data: {error}")
    records = []
    labels = {}
    with_data = []
    missing = {}
    no_alignment = False
    for index, (rec_id, sequence) in zip(indices, nodes):
        if failure is not None:
            records.append([rec_id, sequence])
            missing.setdefault(Residue_Data.FAILED, []).append(rec_id)
            continue
        result = Residue_Data.compute_residue_data(inputs, index, sequence)
        # Every property of a node the alignment lacks says the same: it is said once for the node.
        for kind, reason in dict.fromkeys((error.kind, error.reason) for error in result.unavailable.values()):
            if kind == Residue_Data.FAILED:
                print(f"Warning: Could not compute per-residue data for {rec_id}: {reason}")
            elif kind != Residue_Data.NO_ALIGNMENT:
                print(f"Note: No per-residue alignment data for {rec_id}: {reason}.")
        if result.properties:
            records.append([rec_id, sequence, {
                "header": viewer.full_headers[index],
                "properties": result.properties,
            }])
            with_data.append(rec_id)
            labels.update({data["label"]: None for data in result.properties.values()})
        else:
            records.append([rec_id, sequence])
            reasons = [error.kind for error in result.unavailable.values()]
            if Residue_Data.NO_ALIGNMENT in reasons:
                no_alignment = True
            elif reasons:
                missing.setdefault(reasons[0], []).append(rec_id)

    notes = []
    if with_data:
        msa_file = getattr(inputs.alignment, "msa_file", None)
        notes.append((Message(
            "Per-residue alignment data from {file} will be saved beside %n structure(s): {properties}.",
            n=len(with_data),
            file=os.path.basename(str(msa_file)) if msa_file else "the loaded alignment",
            properties=", ".join(labels),
        ), []))
    if no_alignment:
        notes.append((Message("No per-residue alignment data is saved: no alignment is loaded."), []))
    for kind, names in missing.items():
        notes.append((_residue_data_note(kind, len(names)), names))
    return records, notes


# The lane the folds go in, and the folds this Viewer started, by the id its worker echoes on every event.
FOLDING_LANE = "folding"
MAX_RUNS_KEPT = 20
RUNS = collections.OrderedDict()


class FoldRun:
    """One queued run of the folding worker: what to fold, fixed when the command was given, and what it needs.

    The nodes and their sequences (and the per-residue data computed from the
    alignment as it was then) are in ``records``, the worker's private input
    file; the scheduler job launches the worker with them and waits for it to
    end. The run keeps what its events and its end tell (the structures folded,
    the reasons some failed) and the job's handle, so that the worker's progress
    shows on the Session card.
    """

    def __init__(self, records, nodes, *, mode, device, structures_dir, action_url, portal, viewer=None):
        self.run_id = uuid.uuid4().hex[:12]
        try:
            self.viewer_ref = weakref.ref(viewer)
        except TypeError:  # A stand-in that cannot be weakly referenced.
            self.viewer_ref = lambda: viewer
        self.records = records
        self.nodes = list(nodes)
        self.mode = mode
        self.device = device
        self.structures_dir = structures_dir
        self.action_url = action_url
        self.skip_existing = False
        self.job_id = None
        self.input_path = None
        directory = worker_directory()
        self.log_path = str(directory / f"esmfold-{self.run_id}.log")
        # Only a command from the portal (MCP, the Agent page) follows its worker through a status file.
        self.status_path = str(directory / f"esmfold-{self.run_id}.json") if portal else None
        self.handle = None
        self.tracker = None
        self.process = None
        self.worker = None
        self.folded = set()
        self.failures = []
        self.reason = None
        # Whether a replacement token was asked for already, and the dialog open for one (to close it on a stop).
        self.replaced = False
        self._dialog = None
        # The scheduler's job, once it began (what its owner needs: the request that gave it).
        self.job = None
        self._lock = threading.Lock()

    # --- the worker's input ---

    def write_input(self):
        """Save the records to a private file, which the worker reads and deletes (--delete-input)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2)
            self.input_path = handle.name

    def discard_input(self):
        """Delete the input file if the worker did not: it never ran, or ended before it read it."""
        if self.input_path:
            _discard_worker_input(self.input_path)

    def command(self):
        """The worker's command line."""
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.abspath(os.path.join(src_dir, "resources", "esmfold", "esmfold_worker.py"))
        if self.mode == "large":
            command = [sys.executable, script, self.input_path, self.structures_dir, "--delete-input",
                       "--mode", "large", "--action-url", self.action_url]
        else:
            command = [sys.executable, script, self.input_path, self.structures_dir, self.device,
                       "--delete-input", "--action-url", self.action_url]
        if self.status_path:
            command += ["--portal-status", self.status_path]
        # No window, so nothing the worker could ask a person for: it never reads input.
        command += ["--noninteractive", "--run-id", self.run_id]
        if self.skip_existing:
            command += ["--skip-existing"]
        return command

    # --- the process and its tracker, which meet from two threads ---

    def attach(self, process, worker):
        """The scheduler's thread launched the worker."""
        with self._lock:
            self.process, self.worker = process, worker
            tracker = self.tracker
        self._link(tracker)

    def set_tracker(self, tracker):
        """The GUI thread made the tracker that reports this run to its portal request."""
        with self._lock:
            self.tracker = tracker
        self._link(tracker)

    def _link(self, tracker):
        with self._lock:
            process, worker = self.process, self.worker
        if tracker is None or worker is None:
            return
        worker.tracker = tracker
        tracker.worker = worker
        tracker.attach_hidden(process, self.log_path)

    # --- what the worker reports ---

    def note_event(self, message):
        """Take an esmfold_job event of this run: it is what the card shows after the job's time."""
        event = message.get("event")
        if event == "started":
            self.say(Message("Getting the model ready…"))
        elif event == "downloading":
            self.say(Message("Downloading the model… {percent}%", percent=message.get("percent", 0)))
        elif event == "progress":
            self.say(Message("Folding {index} of {total}: {name}",
                             index=message.get("index", 0), total=message.get("total", 0),
                             name=message.get("name", "")))
        elif event == "structure_failed":
            self.failures.append((message.get("name", ""), message.get("reason", "")))
        elif event == "failed":
            self.reason = message.get("reason") or self.reason

    def say(self, text):
        handle = self.handle
        if handle is not None:
            handle.progress(text)

    def failure(self, code):
        """The error of a run whose worker ended with ``code``."""
        if self.failures:
            name, reason = self.failures[0]
            others = len(self.failures) - 1
            error = Message("Folding {name} failed: {reason}", name=name, reason=reason)
            if others:
                error = JoinedMessage([error, Message("%n more structure(s) failed too.", n=others)])
            return error
        if self.reason:
            return Message("The folding worker failed: {reason}", reason=self.reason)
        return Message("The folding worker stopped (exit code {code}); its output is in {log}.",
                       code=code, log=self.log_path)

    # --- Biohub rejected the token ---

    def remaining_records(self):
        """The records of the structures this run has not folded: what a run again is for."""
        return [record for record in self.records if record[0] not in self.folded]

    def ask_replacement(self):
        """Ask, in a dialog, for a token to replace the rejected one, and save it (the GUI thread).

        Returns ("saved", None), or why not: ("unavailable", reason) where no one is at the Viewer
        to ask, ("cancelled", None), or ("not saved", reason). A request from the portal shows
        awaiting_user_input while the dialog is open.
        """
        from Viewer_Command_Portal import CommandInteractionError, bind, user_interaction
        from resources import Biohub_API
        viewer = self.viewer_ref()
        if viewer is None:
            return ("unavailable", "the Viewer is closing")
        parent = getattr(getattr(viewer, "canvas", None), "native", None)
        with bind(getattr(self.job, "command_context", None)):
            try:
                with user_interaction(viewer, "Enter a replacement Biohub API token in the Viewer dialog"):
                    token = token_dialog.ask_token(parent, replacement=True, shown=self._dialog_shown)
            except CommandInteractionError as error:
                return ("unavailable", str(error))
            finally:
                self._dialog = None
        if token is None:
            return ("cancelled", None)
        try:
            token_dialog.save_token(token, replacement=True)
        except Biohub_API.BiohubSettingsError as error:
            return ("not saved", str(error))
        self.prepare_again()
        return ("saved", None)

    def _dialog_shown(self, dialog):
        self._dialog = dialog

    def close_dialog(self):
        """The job was told to stop while its dialog was open: close it (the GUI thread)."""
        dialog = self._dialog
        if dialog is not None:
            dialog.reject()

    def prepare_again(self):
        """Get ready to run the worker again for what is left (the GUI thread).

        The first worker's tracker takes a last look and stops, its status files go, and the
        records become those not folded, in a new private input file (the worker deleted the
        first when it read it); the new worker gets a tracker of its own.
        """
        tracker = self.tracker
        if tracker is not None:
            try:
                tracker.poll()
                tracker.timer.stop()
                tracker.deleteLater()
            except RuntimeError:
                pass  # The tracker's Qt object is deleted already.
        if self.status_path:
            for suffix in ("", ".partial", ".stdout", ".stderr"):
                _discard_worker_input(self.status_path + suffix)
        with self._lock:
            self.tracker = self.process = self.worker = None
        self.records = self.remaining_records()
        self.write_input()
        if self.job is not None:
            self.started(self.job)

    # --- the job's ends ---

    def started(self, job):
        """The scheduler began the job (GUI thread): a portal request follows the worker through its status file."""
        from Viewer_Worker_Tracking import WorkerTracker
        self.job = job
        if job.command_context is not None and self.status_path:
            self.set_tracker(WorkerTracker(job.command_context, job_id=str(job.job_id),
                                           path=self.status_path, add_job=False))

    def finished(self, job, outcome, error):
        """The job ended (GUI thread)."""
        tracker = self.tracker
        if tracker is not None:
            try:
                if outcome == "succeeded":
                    # One last look, for what the worker saved last: it may have ended between two of the tracker's.
                    tracker.poll()
                tracker.timer.stop()
            except RuntimeError:
                pass  # The tracker's Qt object is deleted already.
        self.handle = None


def _remember(run):
    RUNS[run.run_id] = run
    while len(RUNS) > MAX_RUNS_KEPT:
        RUNS.popitem(last=False)


def log_of(run_id):
    """The log of the run this Viewer started with this id, or None: a path is never taken from anyone else.

    The path is the one the run made for itself in the workers' folder; it is checked to lie there all the same.
    """
    from web_ui.Web_Server import contained_path
    run = RUNS.get(run_id)
    if run is None:
        return None
    directory = str(worker_directory())
    path = contained_path(directory, os.path.basename(run.log_path))
    return path if path is not None and os.path.normcase(path) == os.path.normcase(os.path.abspath(run.log_path)) else None


def note_job_event(viewer, message):
    """An esmfold_job event reached the Viewer: pass it to the run it is of, if this Viewer started that run."""
    run = RUNS.get(message.get("run"))
    if run is not None:
        run.note_event(message)


def note_structure_folded(viewer, run_id, node_id):
    """A structure was saved by run ``run_id`` (None for a worker that gave no id)."""
    run = RUNS.get(run_id)
    if run is not None:
        run.folded.add(node_id)


def fold_job(run, job):
    """The scheduler's worker for a fold: launch the folding worker with no window, and wait for it to end.

    Told to stop (by `esmfold stop`, the Session card or the Viewer closing), the job
    ends the worker and everything it started through the function it registered with
    its handle; the structures the worker had saved stay and a half-written one is deleted.
    """
    job.raise_if_cancelled()
    run.handle = job
    job.set_log_path(run.log_path)
    while True:
        code = _run_worker(run, job)
        job.raise_if_cancelled()
        if code == 0:
            return {"message": Message("Folded %n structure(s).", n=len(run.nodes))}
        if code != worker_exit_authentication():
            raise RuntimeError(run.failure(code))
        # Biohub rejected the token: ask for another, and fold the rest with it.
        _replace_rejected_token(run, job)


def worker_exit_authentication():
    """The exit code of a worker that Biohub's rejection of its token ended."""
    from resources.esmfold import esmfold_worker
    return esmfold_worker.EXIT_AUTHENTICATION


def _run_worker(run, job):
    """Launch the worker once and wait for it to end; return its exit code."""
    from Viewer_Worker_Tracking import track_hidden_worker, track_worker
    job.progress(Message("Starting the folding worker…"))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        process = launch_without_window(run.command(), cwd=project_root, log_path=run.log_path)
    except OSError as error:
        raise RuntimeError(Message("Could not launch the folding worker without a window:\n{error}", error=error)) from error
    worker = track_worker(process, input_path=run.input_path, output_dir=run.structures_dir, tracker=None)
    track_hidden_worker(process)
    run.attach(process, worker)
    job.on_cancel(lambda: worker.stop(Message("ESMFold worker stopped")))
    while True:
        try:
            return process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue


def _replace_rejected_token(run, job):
    """Ask for a token to replace the one Biohub rejected and get the run ready to go again; raise to end the job.

    Only once for a run: a replacement that is rejected too ends it. Where no one can be asked (a Viewer
    that shows no windows) or no token is given, the job fails and the structures folded so far stay.
    """
    if run.replaced:
        raise RuntimeError(Message(
            "Biohub rejected the replacement token too; folding stopped. The structures folded before then are kept."))
    job.progress(Message("Biohub rejected the API token; waiting for a replacement…"))
    state, reason = job.call_in_gui(run.ask_replacement, close=run.close_dialog)
    if state == "saved":
        run.replaced = True
        return
    if state == "unavailable":
        raise RuntimeError(Message(
            "Biohub rejected the saved API token, and a replacement cannot be asked for: {reason}", reason=reason))
    if state == "not saved":
        raise RuntimeError(Message(
            "Biohub rejected the API token, and the replacement could not be saved: {reason}", reason=reason))
    raise RuntimeError(Message(
        "Biohub rejected the API token and no replacement was entered. The structures folded before then are kept."))


def _folding_jobs(scheduler):
    """The entries of the scheduler's folding lane that esmfold queued, running one first."""
    return [entry for entry in scheduler.snapshot()
            if entry.lane == FOLDING_LANE and entry.job.command_name == "esmfold"]


def _stop_workers(viewer):
    """`esmfold stop`: end the running fold and everything it started, and take the queued folds out of the line.

    The running fold stops as the Session card's Stop does: the worker and the
    processes it started end, the structures already saved are kept, a half-written
    one is deleted, and the job is reported cancelled. A fold that waits is removed
    and never starts. With none running or waiting, it is a valid no-op that says so.
    """
    from Viewer_Worker_Tracking import stop_workers
    scheduler = getattr(viewer, "background_job_scheduler", None)
    entries = _folding_jobs(scheduler) if scheduler is not None else []
    # One step for all of them, so that the next fold in the line cannot start as the running one ends.
    cancelled = set(scheduler.cancel_many([entry.job.job_id for entry in entries])) if entries else set()
    running = sum(1 for entry in entries if entry.running and entry.job.job_id in cancelled)
    waiting = sum(1 for entry in entries if not entry.running and entry.job.job_id in cancelled)
    # A worker started in some other way, such as by a version that ran before the scheduler did.
    running += stop_workers("ESMFold worker stopped")
    parts = []
    if running:
        parts.append(Message("Stopped %n ESMFold worker(s); structures already saved are kept.", n=running))
    if waiting:
        parts.append(Message("Removed %n queued fold(s).", n=waiting))
    msg = JoinedMessage(parts) if len(parts) > 1 else parts[0] if parts else Message("No ESMFold worker is running.")
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, msg)
    print(msg)
    Command_Engine.command_succeeded(viewer, msg)


def _parse_options(viewer, args):
    normalized = [str(argument).lower() for argument in args]
    allowed = {"large", "multi", "refold"}
    unknown = [argument for argument in normalized if argument not in allowed]
    if unknown:
        _report_usage_error(viewer, Message("Unknown esmfold keyword: {keyword}", keyword=unknown[0]))
        return None
    duplicates = sorted({argument for argument in normalized if normalized.count(argument) > 1})
    if duplicates:
        _report_usage_error(viewer, Message("Duplicate esmfold keyword: {keyword}", keyword=duplicates[0]))
        return None
    return {
        "large": "large" in normalized,
        "multi": "multi" in normalized,
        "refold": "refold" in normalized,
    }


def _model_suffix(is_large):
    """The suffix a large run's PDB names carry: the ESM3 model the Biohub settings choose (None for a local run)."""
    if not is_large:
        return None
    from resources import Biohub_API
    try:
        settings = Biohub_API.read_api_settings()
    except Biohub_API.BiohubSettingsError:
        settings = None
    return (settings or {}).get("ESM3_MODEL", Biohub_API.DEFAULT_ESM3_MODEL)


def _credentials_state():
    """What a run through Biohub has for credentials: ("saved", None), ("missing", None), or ("invalid", error).

    Saved means a settings file that holds a token, or ESM_API_KEY in the environment. A settings file that
    is there but unusable is not "missing": asking for a token would overwrite what the user wrote in it.
    """
    from resources import Biohub_API
    try:
        if Biohub_API.read_api_settings(Biohub_API.API_SETTINGS_FILE) is not None:
            return ("saved", None)
    except Biohub_API.BiohubSettingsError as error:
        return ("invalid", error)
    token = os.environ.get("ESM_API_KEY")
    return ("saved", None) if isinstance(token, str) and token.strip() else ("missing", None)


def _ask_for_token(viewer):
    """Ask for the Biohub API token in a dialog and save it; return whether it is saved. Says why not.

    A command from the portal shows awaiting_user_input while the dialog is open, and a Viewer that
    shows no windows fails it at once, as every command that needs a dialog does. The token is only
    ever typed in the dialog, never taken from a command, and is in no message.
    """
    from Viewer_Command_Portal import CommandInteractionError, user_interaction
    from resources import Biohub_API
    parent = getattr(getattr(viewer, "canvas", None), "native", None)
    try:
        with user_interaction(viewer, "Enter the Biohub API token in the Viewer dialog"):
            token = token_dialog.ask_token(parent)
    except CommandInteractionError as error:
        _report_failure(viewer, Message("Could not ask for the Biohub API token: {error}", error=error))
        return False
    if token is None:
        message = Message("Biohub API token entry was cancelled; nothing was queued.")
        print(message)
        _set_console_text(viewer, message)
        Command_Engine.command_cancelled(viewer, message)
        return False
    try:
        token_dialog.save_token(token)
    except Biohub_API.BiohubSettingsError as error:
        _report_failure(viewer, Message("Could not save the Biohub API token: {error}", error=error))
        return False
    return True


def _report_failure(viewer, message):
    print(f"Error: {message}")
    Command_Engine.command_failed(viewer, f"Error: {message}")
    _set_console_text(viewer, Message("Error: {error}", error=message))


def _structure_is_saved(path):
    """Whether the structure file at ``path`` is saved: a file with something in it.

    A fold writes its file whole or not at all (the worker renames it into place), so a file
    that is there is a finished structure; an empty one is not counted.
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _show_saved(viewer, saved, structures_dir):
    """Show the structures saved for selected nodes in the Fold View page, as a fold's are; return how many got residue data.

    ``saved`` holds (node index, node name, sequence, PDB file name) for each node. A node the loaded
    alignment holds gets its residue data file written now, as a fold writes it beside the structure it
    saves (the data comes from the alignment, not from the model), before the page is told: the page
    finds the file when it loads the structure. Without an alignment, or for a node it lacks, an older
    file is left as it is. A request from the portal lists each file written among its artifacts.
    """
    from Viewer_Command_Portal import report
    from resources.esmfold import esmfold_worker as worker_files
    records, _notes = _worker_records(viewer, [item[0] for item in saved], [item[1:3] for item in saved])
    written = 0
    for (_index, name, sequence, pdb_filename), record in zip(saved, records):
        sidecar = len(record) == 3 and worker_files._write_residue_data(structures_dir, pdb_filename, name, sequence, record[2])
        if sidecar:
            written += 1
            report(artifact=os.path.join(structures_dir, sidecar), viewer=viewer)
    esmfold_backend.show_structures(viewer, [(item[1], item[3]) for item in saved])
    return written


def _saved_notes(saved, written, command):
    """The sentences for the selected nodes that are shown from saved structures, as (Message, names) pairs.

    ``written`` is how many of them got residue data, and ``command`` the command that folds them again.
    """
    names = [item[1] for item in saved]
    notes = [(Message("Showed %n structure(s) from saved files instead of folding them again.", n=len(names)), names)]
    if written:
        notes.append((Message("Per-residue alignment data was saved beside %n shown structure(s).", n=written), []))
    notes.append((Message("Run {command} to fold them again.", command=command), []))
    return notes


def _left_out_notes(skipped_ids, taken):
    """The sentences for the selected nodes that are not folded: without a sequence, and held by another fold."""
    notes = []
    if skipped_ids:
        notes.append((Message("Skipped %n selected node(s) without a sequence.", n=len(skipped_ids)), skipped_ids))
    if taken:
        notes.append((Message("Left out %n selected node(s) another fold holds: {nodes}.", n=len(taken), nodes=", ".join(taken)), []))
    return notes


def _joined_notes(notes):
    """(the sentences of ``notes``, their English for the outcome with a space before each).

    A note is a (Message, names) pair; when it names a few nodes, they follow its sentence in the outcome.
    """
    text = ""
    for note, names in notes:
        text += f" {note}"
        if names and len(names) <= MAX_LISTED_SKIPPED:
            text += f" ({', '.join(names)})"
    return [note for note, _names in notes], text


def _report_outcome(viewer, headless, page_opened, message, outcome):
    """End the command with success: ``message`` is its console line and ``outcome`` its result.

    A Viewer that shows no windows opens no page, and the outcome gives the structure viewer's address;
    a page that did not open and is not connected is mentioned.
    """
    if headless:
        address = None if _structure_viewer_connected(viewer) else _structure_viewer_address(viewer)
        if address is not None:
            outcome += f" A Viewer with no window opens no browser page; the structure viewer is served at {address}."
        Command_Engine.command_succeeded(viewer, outcome)
        return
    if not page_opened and not _structure_viewer_connected(viewer):
        outcome += " The structure viewer page was not opened; use the Fold View button if it does not appear."
    _set_console_text(viewer, message)
    Command_Engine.command_succeeded(viewer, outcome)


def run(viewer, args):
    import warnings
    # Suppress library-level user warnings from esm library
    warnings.filterwarnings("ignore", category=UserWarning, module="esm")

    # 1. Registration callback support
    if args and args[0] == '--register-only':
        esmfold_backend.register(viewer)
        Command_Engine.command_succeeded(viewer, 'Registered the structure viewer interface.')
        return

    # 2. Help & Usage Check
    if args and args[0].lower() in ['help', '-h', '--help']:
        if len(args) != 1:
            _report_usage_error(viewer, Message("Help cannot be combined with other keywords."))
            return
        print_help()
        _set_console_text(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return

    if args and args[0].lower() == 'stop':
        if len(args) != 1:
            _report_usage_error(viewer, Message("Stop cannot be combined with other keywords."))
            return
        _stop_workers(viewer)
        return

    options = _parse_options(viewer, args)
    if options is None:
        return
    headless = is_headless(viewer)
    is_large = options["large"]
    is_multi = options["multi"]

    # 3. Determine selected nodes: the selection this command was given, as a queued command uses it
    # (Command_Engine.command_selection). With nothing selected nothing is folded, whether or not a node
    # was clicked (the Viewer keeps a click as a transient highlight, not as part of the selection): a plain
    # esmfold opens the structure viewer, and the forms that fold fail.
    selected_indices = Command_Engine.command_selection(viewer)

    if not selected_indices:
        if not args:
            esmfold_backend.register(viewer)
            if _open_structure_viewer(viewer):
                Command_Engine.command_succeeded(viewer, "Opened the structure viewer; no folding job was requested.")
            elif _structure_viewer_connected(viewer):
                Command_Engine.command_succeeded(viewer, "The structure viewer is already open; no folding job was requested.")
            elif headless and (address := _structure_viewer_address(viewer)) is not None:
                Command_Engine.command_succeeded(
                    viewer,
                    f"A Viewer with no window opens no browser page; the structure viewer is served at {address}. "
                    "No folding job was requested.",
                )
            else:
                Command_Engine.command_failed(viewer, "The structure viewer was not opened; the Viewer's console line says why.")
            return

        print("Error: No nodes selected. Please select a node in the visualizer first.")
        Command_Engine.command_failed(viewer, 'Error: No nodes selected. Please select a node in the visualizer first.')
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: No nodes selected.")
            Command_Engine.show_status(viewer, failure)
        return

    # 4. Check for multiple selections vs "multi" command flag
    if len(selected_indices) > 1 and not is_multi:
        print("Error: Multiple nodes selected. Run 'esmfold multi' to fold them, or select a single node.")
        Command_Engine.command_failed(viewer, "Error: Multiple nodes selected. Run 'esmfold multi' to fold them, or select a single node.")
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: Multiple nodes selected. Use 'esmfold multi'.")
            Command_Engine.show_status(viewer, failure)
        return

    # 4b. The fold is a job of the background scheduler's folding lane: it waits its turn behind
    # the folds already queued, where one worker at a time keeps two models out of GPU memory.
    scheduler = getattr(viewer, "background_job_scheduler", None)
    if scheduler is None:
        failure = Message("The background job scheduler is unavailable.")
        print(f"Error: {failure}")
        Command_Engine.command_failed(viewer, f"Error: {failure}")
        _set_console_text(viewer, Message("Error: {error}", error=failure))
        return

    # 5. Parse sequences from FASTA subset/main database
    if not hasattr(viewer, 'sequences_map'):
        viewer.sequences_map = {}
        fasta_path = getattr(cfg, 'NODE_FASTA_FILE', None) or getattr(cfg, 'SEQUENCES_FILE', '')
        if fasta_path and os.path.exists(fasta_path):
            try:
                from Bio import SeqIO
                for rec in SeqIO.parse(fasta_path, "fasta"):
                    viewer.sequences_map[rec.id] = str(rec.seq)
                    viewer.sequences_map[rec.description] = str(rec.seq)
            except Exception as e:
                print(f"Warning: Failed to parse FASTA for sequences: {e}")

    # Resolve target sequences to fold
    nodes_to_fold = []
    fold_indices = []
    skipped_ids = []
    for idx in selected_indices:
        full_header = viewer.full_headers[idx]
        rec_id = full_header.split()[0]

        sequence = None
        if full_header in viewer.sequences_map:
            sequence = viewer.sequences_map[full_header]
        elif rec_id in viewer.sequences_map:
            sequence = viewer.sequences_map[rec_id]

        if sequence:
            nodes_to_fold.append((rec_id, sequence))
            fold_indices.append(idx)
        else:
            print(f"Warning: Sequence not found in FASTA for node: {rec_id}")
            skipped_ids.append(rec_id)

    if not nodes_to_fold:
        print("Error: Could not retrieve sequences for selected nodes.")
        Command_Engine.command_failed(viewer, 'Error: Could not retrieve sequences for selected nodes.')
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: Sequence retrieval failed.")
            Command_Engine.show_status(viewer, failure)
        return

    # 6. Set up Directory & Web Registration
    structures_dir = esmfold_backend.get_structures_directory()
    os.makedirs(structures_dir, exist_ok=True)

    # Register web button and route mapping
    esmfold_backend.register(viewer)

    try:
        action_url = viewer.get_web_url("/api/action")
    except Exception as error:
        message = Message("ESMFold cannot start because the Viewer web server is unavailable:\n{error}", error=error)
        print(f"Error: {message}")
        Command_Engine.command_failed(viewer, f'Error: {message}')
        parent = getattr(viewer, 'main_window', None)
        from Viewer_Command_Portal import CURRENT
        if CURRENT.get() is None and not headless:
            QMessageBox.critical(parent, translate("Viewer", "ESMFold Web Server Error"), message.display())
        _set_console_text(viewer, Message("Error: Viewer web server unavailable."))
        return

    # 7. A node another fold waits to write, or is writing, is left out of this one: two folds never
    # write the same file. The node says which job holds it. A node whose structure is saved already (the
    # file this fold would write) is shown from it instead of being folded again, unless the command says refold.
    from resources.esmfold import esmfold_worker as worker_files
    abs_structures_dir = os.path.abspath(structures_dir)
    suffix = _model_suffix(is_large)
    taken = []
    saved = []
    kept_nodes, kept_indices = [], []
    for index, (rec_id, sequence) in zip(fold_indices, nodes_to_fold):
        pdb_filename = worker_files.prediction_filename(rec_id, suffix)
        pdb_path = os.path.join(abs_structures_dir, pdb_filename)
        holder = scheduler.reservation_holder(pdb_path)
        if holder is not None:
            taken.append(f"{rec_id} (#{holder})")
        elif not options["refold"] and _structure_is_saved(pdb_path):
            saved.append((index, rec_id, sequence, pdb_filename))
        else:
            kept_nodes.append((rec_id, sequence))
            kept_indices.append(index)
    if not kept_nodes and not saved:
        refusal = Message("Every selected node is already queued or being folded: {nodes}.", nodes=", ".join(taken))
        print(f"Error: {refusal}")
        Command_Engine.command_failed(viewer, f'Error: {refusal}')
        _set_console_text(viewer, Message("Error: {error}", error=refusal))
        return
    words = " ".join(["esmfold"] + [word for word in ("large", "multi", "refold") if options[word]])
    if not kept_nodes:
        # Nothing is left to fold: no job, no worker, no model, and none of what a fold needs.
        written = _show_saved(viewer, saved, abs_structures_dir)
        page_opened = False if headless else esmfold_backend.open_esmfold_ui(viewer, show_existing_dialog=False)
        notes = _saved_notes(saved, written, f"{words} refold") + _left_out_notes(skipped_ids, taken)
        parts, outcome = _joined_notes(notes)
        print(outcome.strip())
        _report_outcome(viewer, headless, page_opened, JoinedMessage(parts), outcome.strip())
        return
    nodes_to_fold, fold_indices = kept_nodes, kept_indices

    # 8. Select hardware only for local inference. Biohub runs remotely.
    device_str = None
    if not is_large:
        try:
            from utilities import Hardware_Acceleration as Hardware_Utils
            import torch
        except ImportError:
            print("Error: PyTorch or Hardware_Utils could not be imported.")
            Command_Engine.command_failed(viewer, 'Error: PyTorch or Hardware_Utils could not be imported.')
            _set_console_text(viewer, Message("Error: PyTorch/Hardware_Utils missing"))
            return

        device = Hardware_Utils.get_optimal_device()
        device_str = str(device)
        print(f"Optimal device selected: {device_str}")
        if device.type == 'cpu':
            print("Warning: Running local ESM3 on CPU will be extremely slow.")

    # 8b. A run through Biohub needs the credentials. A worker with no window cannot ask for them, so the
    # Viewer does, below, once it is sure there is something to fold; a settings file it cannot use is not
    # written over.
    credentials = _credentials_state() if is_large else ("saved", None)
    if credentials[0] == "invalid":
        _report_failure(viewer, Message(
            "The Biohub API settings cannot be used: {error} Fix or delete the file, then run esmfold large again.",
            error=credentials[1]))
        return

    # 8c. Nothing is left to ask in vain about: ask for the token a run through Biohub still lacks.
    if credentials[0] == "missing" and not _ask_for_token(viewer):
        return

    # 9. The records are fixed now, with the per-residue data of the alignment as it is now: they go to a
    # private file the worker reads when its turn comes.
    from Viewer_Command_Portal import CURRENT
    context = CURRENT.get()
    records, residue_notes = _worker_records(viewer, fold_indices, nodes_to_fold)
    run = FoldRun(
        records, nodes_to_fold, mode="large" if is_large else "local", device=device_str,
        structures_dir=abs_structures_dir, action_url=action_url, portal=context is not None, viewer=viewer,
    )
    run.write_input()
    paths = []
    for rec_id, _sequence in nodes_to_fold:
        pdb_filename = worker_files.prediction_filename(rec_id, suffix)
        paths += [os.path.join(abs_structures_dir, pdb_filename),
                  os.path.join(abs_structures_dir, worker_files.residue_data_filename(pdb_filename))]
    count = len(nodes_to_fold)
    description = f"{words} -> {nodes_to_fold[0][0]}" if count == 1 else f"{words} -> {count} structures"
    _remember(run)
    try:
        job_id = scheduler.enqueue(
            command_name="esmfold",
            description=description,
            payload=run,
            worker=fold_job,
            output_path=paths[0],
            allow_overwrite=True,
            lane=FOLDING_LANE,
            lane_title=Message("Folding"),
            extra_paths=paths[1:],
            cleanup=run.discard_input,
            on_started=run.started,
            on_finished=run.finished,
            log_path=run.log_path,
        )
    except (FileExistsError, RuntimeError) as error:
        RUNS.pop(run.run_id, None)
        run.discard_input()
        message = Message("Could not queue the fold: {error}", error=error)
        print(f"Error: {message}")
        Command_Engine.command_failed(viewer, f'Error: {message}')
        _set_console_text(viewer, Message("Error: {error}", error=message))
        return
    run.job_id = job_id

    # 10. The structures that were saved already are shown now that the fold is queued, and the Mol* web
    # browser tab is opened immediately. A headless Viewer has no browser to open it in.
    saved_notes = []
    if saved:
        written = _show_saved(viewer, saved, abs_structures_dir)
        saved_notes = _saved_notes(saved, written, f"{words} refold")
        # The status line has the counts only, as a run that shows alone does: the Console names the nodes.
        print(_joined_notes(saved_notes)[1].strip())
    if headless:
        page_opened = False
    else:
        page_opened = esmfold_backend.open_esmfold_ui(viewer, show_existing_dialog=False)
    position = next((entry.position for entry in scheduler.snapshot() if entry.job.job_id == job_id), 1)
    mode_label = "Biohub ESM3" if is_large else "local ESM3"
    if is_large:
        queued = Message("Queued %n structure(s) for folding with Biohub ESM3 as job #{job}.", n=len(nodes_to_fold), job=job_id)
    else:
        queued = Message("Queued %n structure(s) for folding with local ESM3 as job #{job}.", n=len(nodes_to_fold), job=job_id)
    outcome = f"Queued {mode_label} folding for {len(nodes_to_fold)} structure(s) as job #{job_id}."
    notes = []
    if position > 1:
        notes.append((Message("It is number {position} in the folding line.", position=position), []))
    notes += _left_out_notes(skipped_ids, taken) + saved_notes
    for note, _names in residue_notes:
        print(note)
    parts, notes_text = _joined_notes(notes + residue_notes)
    outcome += notes_text + f" The worker runs without a window; its output is in {run.log_path}."
    _report_outcome(viewer, headless, page_opened, JoinedMessage([queued, *parts]), outcome)
