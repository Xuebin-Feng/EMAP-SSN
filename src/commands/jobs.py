# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""jobs: list the background jobs, the files being written and the folds, and cancel them.

A background job is work the Viewer's background job scheduler runs in a lane of its
own, numbered in the order the jobs are given. A command that waits or runs in the
Viewer's command queue is not one: it has no number, this command does not list it,
and its cancel does not reach it (undo typed by hand does).

It runs at once, even while commands wait: a command that computes holds the ones given
after it, and the jobs listed here must not wait behind it.
"""
import re
import sys

import Command_Engine
from utilities.Localization import JoinedMessage, Message
from desktop.Command_Metadata import argument, choice, entry

# --- What this command declares about itself ---
# Listing the jobs is harmless. Cancelling throws away the user's own commands.
AGENT_POLICY = {'default': 'run', 'first_argument': {'cancel': 'approve'}}

CATALOG_ENTRY = entry('List the background jobs that wait or run (the files label, logo and print write, and the structures esmfold folds), or cancel them. Each is numbered in the order given. A command that computes, such as cluster or subcluster, is not a background job: jobs neither lists nor cancels it. jobs itself runs at once.',
    argument('action', 'Omit to list the background jobs, numbered, with the folding workers and Python scripts that run. cancel alone cancels the newest background job.', choice('cancel')),
    argument('jobs', 'After cancel: job numbers (14 or #14), every one of which must name a waiting or running background job, or nothing is cancelled. A running job is only told to stop: it is listed as stopping and ends, reported cancelled with the file it had begun deleted (a file of that name from before stays as it was), at its next step; one with no step left to stop at completes and is reported completed. A job not started is dropped. all cancels every background job, running or not.', choice('all')))

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 250
AGENT_PROMPT = """
`jobs | jobs cancel [N ...] | jobs cancel all`
- Commands run in the order given; a slow one (`cluster`, `subcluster`) computes in the background, and the commands after it wait; it is not a background job, has no number, and `jobs` does not list it. `label`, `logo` and `print` write their files, and `esmfold` folds, as numbered background jobs that run beside the commands: files share one line (`files`) and folds have one of their own (`folding`), each job waiting behind those queued before it. A fold requested while another runs is queued, not refused.
- `jobs` lists the background jobs, numbered, with each one's state (`waiting`, `running` or `stopping`) and line. It is read-only and runs at once.
- `cancel` cancels the newest background job; `cancel N [N ...]` (`14` or `#14`) cancels those jobs, every number naming a job that waits or runs or nothing is cancelled; `cancel all` cancels every background job. A waiting job is removed at once. A running background job is only told to stop: `jobs` lists it as `stopping`, and it is reported cancelled only once it has stopped (a fold keeps the structures it had saved); one with no step left to stop at completes instead.
- `jobs cancel` is reviewed (see "Review by the user"). When the user asks whether something has finished, emit `jobs` and report from its output: that `label`, `logo`, `print` or `esmfold` was accepted is not that it has completed.
"""

# How the Viewer dispatches this command (Command_Declarations): it never waits behind the commands
# in the queue, and the command portal runs it ahead of a request that waits for jobs.
QUEUE = 'instant'
ACTS_ON_JOBS = True

_JOB_NUMBER = re.compile(r"#?([0-9]+)")


def print_help():
    print("""
    Jobs: Background Jobs
    =====================
    Usage: jobs
           jobs cancel [N ...]
           jobs cancel all
           jobs help

    Description:
      Slow work that writes a file (logo, label, print) or folds structures
      (esmfold) runs in the background, in lines of its own, while the Viewer
      keeps responding. Each such job is numbered, #1, #2 and so on, in the
      order the jobs were given. jobs lists them and cancels them.
      A slow command that changes the network (cluster, subcluster) is not a
      background job. It computes while the Viewer keeps responding, the commands
      given after it wait for it, and jobs neither lists nor cancels it.

    Arguments:
      (none)            - Lists the background jobs that wait or run, and the
                          folding workers and Python scripts that run.
      cancel            - Cancels the newest background job.
      cancel N [N ...]  - Cancels those jobs, by number (14 or #14). A job that
                          is running is told to stop, and a job not started is
                          dropped. A number that names no background job cancels
                          nothing.
      cancel all        - Cancels every background job, running or not.

    Notes:
      A running job stops at its next step, which takes a moment, longer where
      one step is a large picture or report. Until it has, it is not cancelled:
      jobs lists it as stopping, and it is reported cancelled when it ends,
      with what it had written deleted (a file of that name from before stays as
      it was). A job with no step left to stop at completes instead, and is
      reported as completed. The Viewer's Session card lists the jobs too,
      where a waiting one can also be moved.
      To take back a command that waits or computes, type undo (or press
      Ctrl+Z) while it does: that cancels the newest of them. A cancelled
      command that is computing stops at its next step (MCL) or runs to its end
      in the background (Leiden, Jaccard); its result is thrown away, and the
      commands after it go on at once. esmfold stop and run stop end folding
      workers and Python scripts.

    Examples:
      jobs
      jobs cancel
      jobs cancel 14 15
      jobs cancel all
    """)


def _background_jobs(viewer):
    """The background jobs not finished, in the order of the line, as (job, state) pairs.

    The state is 'waiting', 'running', or 'stopping' for a job already told to stop.
    """
    scheduler = getattr(viewer, 'background_job_scheduler', None)
    snapshot = getattr(scheduler, 'snapshot', None)
    if snapshot is not None:
        return [(entry.job, 'stopping' if entry.stopping else 'running' if entry.running else 'waiting')
                for entry in snapshot()]
    listing = getattr(scheduler, 'listing', None)
    return [(job, 'running' if running else 'waiting') for job, running in (listing() if listing is not None else [])]


def _cancel_background_jobs(viewer, entries):
    """Cancel these background jobs, (job, state) pairs, in one step; return (removed, told to stop, already stopping).

    A waiting job is gone at once. A running one is only told to stop: it ends
    at its next check point, so it is not reported cancelled here, and the job
    that has been told already is named as such.
    """
    cancel_many = getattr(getattr(viewer, 'background_job_scheduler', None), 'cancel_many', None)
    if cancel_many is None or not entries:
        return [], [], []
    cancelled = set(cancel_many([job.job_id for job, _state in entries]))
    removed = [job for job, state in entries if job.job_id in cancelled and state == 'waiting']
    stopping = [job for job, state in entries if job.job_id in cancelled and state != 'waiting']
    already = [job for job, state in entries if job.job_id not in cancelled and state == 'stopping']
    return removed, stopping, already


def _named(job):
    return f"#{job.job_id} {job.description}"


def _external_counts():
    """(folding workers, Python scripts) that run now.

    Viewer_Worker_Tracking counts them. It is loaded by the command that starts a worker or a script, so
    until then there are none; it is not imported here, since it needs Qt, which the VR viewer must not load.
    """
    tracking = sys.modules.get('Viewer_Worker_Tracking')
    if tracking is None:
        return 0, 0
    return tracking.running_worker_count(), tracking.running_script_count()


def _lines(viewer):
    """The listing's lines, and how many background jobs wait or run."""
    lines = []
    jobs = _background_jobs(viewer)
    for job, state in jobs:
        lines.append(f"#{job.job_id:<5} {state:<9} {' ' * 8}  {job.description}  ({job.lane})")
    workers, scripts = _external_counts()
    if workers:
        lines.append(f"Folding workers running: {workers} (esmfold stop ends them)")
    if scripts:
        lines.append(f"Python scripts running: {scripts} (run stop ends them)")
    return lines, len(jobs)


def list_jobs(viewer):
    lines, count = _lines(viewer)
    print("\n" + "\n".join(lines) + "\n" if lines else "")
    if count:
        msg = Message("%n job(s) waiting or running; the list is in the Console.", n=count)
    else:
        msg = Message("No background job waits or runs.")
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def _refuse(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg, report_message=False)


def cancel_jobs(viewer, words):
    entries = _background_jobs(viewer)
    if not words:
        # The newest job is the one given last: the highest number, whatever its place in its line.
        targets = [max(entries, key=lambda pair: pair[0].job_id)] if entries else []
    elif len(words) == 1 and words[0].lower() == 'all':
        targets = entries
    else:
        # Every number must name a background job, or nothing is cancelled.
        targets = []
        for word in words:
            match = _JOB_NUMBER.fullmatch(word)
            if not match:
                _refuse(viewer, Message("Error: '{word}' is not a job number. Use jobs cancel N, or jobs cancel all.",
                                        word=word))
                return
            number = int(match.group(1))
            entry = next((pair for pair in entries if pair[0].job_id == number), None)
            if entry is None:
                _refuse(viewer, Message("Error: No background job #{job} waits or runs; nothing was cancelled.",
                                        job=number))
                return
            if entry not in targets:
                targets.append(entry)
    removed, stopping, already = _cancel_background_jobs(viewer, targets)
    parts = []
    if removed:
        parts.append(Message("Cancelled %n job(s): {jobs}.", n=len(removed), jobs=", ".join(_named(job) for job in removed)))
    if stopping:
        # Not "cancelled": the job ends at its next step, and jobs lists it as stopping until it has.
        parts.append(Message("Told %n running job(s) to stop: {jobs}. Each ends at its next step; until then jobs lists it as stopping.",
                             n=len(stopping), jobs=", ".join(_named(job) for job in stopping)))
    if already:
        parts.append(Message("Already stopping: {jobs}.", jobs=", ".join(_named(job) for job in already)))
    if not parts:
        msg = Message("Nothing to cancel: no background job waits or runs.")
    else:
        msg = JoinedMessage(parts) if len(parts) > 1 else parts[0]
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return
    if not args:
        list_jobs(viewer)
        return
    if args[0].lower() == 'cancel':
        cancel_jobs(viewer, args[1:])
        return
    msg = JoinedMessage([
        Message("Error: Unrecognized jobs argument '{argument}'.", argument=args[0]),
        Message("Usage: {syntax}", syntax="jobs [cancel [N ... | all]]"),
    ], separator="\n")
    _refuse(viewer, msg)
