# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""jobs: list the commands that wait or run and the files being written, and cancel them.

It runs at once, even while other commands wait (Command_Engine.queue_kind),
since it acts on them.
"""
import re

import Command_Engine
from utilities.Localization import JoinedMessage, Message

_JOB_NUMBER = re.compile(r"#?([0-9]+)")


def print_help():
    print("""
    Jobs: Waiting and Running Commands
    ==================================
    Usage: jobs
           jobs cancel [N ...]
           jobs cancel all
           jobs help

    Description:
      Commands run one at a time, in the order they were given; each runs on
      the network as the commands before it left it. A slow one (cluster,
      subcluster) computes in the background while the Viewer keeps responding,
      and the commands given after it wait. Files (logo, label, print) are
      written in the background too. jobs lists them all, numbered.

    Arguments:
      (none)            - Lists the commands that wait or run, the files being
                          written, and the folding workers and Python scripts
                          that run, then the last few commands that finished.
      cancel            - Cancels the newest command that waits or runs.
      cancel N [N ...]  - Cancels those jobs, by number (14 or #14). A job that
                          is running is told to stop, and a job not started is
                          dropped. A number that names no job cancels nothing.
      cancel all        - Cancels every command that waits or runs, and every
                          file job, running or not.

    Notes:
      A cancelled command that is computing stops at its next step (MCL) or
      runs to its end in the background (Leiden, Jaccard); its result is
      thrown away, and the commands after it go on at once.
      A running job stops at its next step, which takes a moment, longer where
      one step is a large picture or report. Until it has, it is not cancelled:
      jobs lists it as stopping, and it is reported cancelled when it ends,
      with what it had written deleted (a file of that name from before stays as
      it was). A job with no step left to stop at completes instead, and is
      reported as completed. The Viewer's Session card lists the jobs too,
      where a waiting one can also be moved.
      Undo typed by hand (or Ctrl+Z) while commands wait cancels the newest
      of them. esmfold stop and run stop end folding workers and Python scripts.

    Examples:
      jobs
      jobs cancel
      jobs cancel 14 15
      jobs cancel all
    """)


def _queue(viewer):
    return getattr(viewer, 'command_queue', None)


def _file_jobs(viewer):
    """The file jobs not finished, in the order of the line, as (job, state) pairs.

    The state is 'waiting', 'running', or 'stopping' for a job already told to stop.
    """
    scheduler = getattr(viewer, 'background_job_scheduler', None)
    snapshot = getattr(scheduler, 'snapshot', None)
    if snapshot is not None:
        return [(entry.job, 'stopping' if entry.stopping else 'running' if entry.running else 'waiting')
                for entry in snapshot()]
    listing = getattr(scheduler, 'listing', None)
    return [(job, 'running' if running else 'waiting') for job, running in (listing() if listing is not None else [])]


def _cancel_file_jobs(viewer, entries):
    """Cancel these file jobs, (job, state) pairs, in one step; return (removed, told to stop, already stopping).

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
    """(folding workers, Python scripts) that run now, or (0, 0) where they are not tracked."""
    try:
        import Viewer_Worker_Tracking as tracking
    except ImportError:
        return 0, 0
    with tracking._started_lock:
        workers = sum(1 for worker in tracking._started_workers
                      if worker.tracker is not None and tracking._timer_active(worker.tracker))
    with tracking._running_lock:
        scripts = len(tracking._running_scripts)
    return workers, scripts


def _lines(viewer):
    """The listing's lines, and how many jobs wait or run."""
    lines = []
    count = 0
    queue = _queue(viewer)
    for job in (queue.pending() if queue is not None else []):
        count += 1
        state = {'queued': 'waiting', 'running': 'running', 'applying': 'applying'}.get(job.status, job.status)
        timing = f"{job.seconds():6.1f} s" if job.status != 'queued' else " " * 8
        progress = f"  [{job.progress_text}]" if job.status == 'running' and job.progress_text else ""
        lines.append(f"#{job.job_id:<5} {state:<9} {timing}  {job.line}{progress}  ({job.source})")
    for job, state in _file_jobs(viewer):
        count += 1
        lines.append(f"#{job.job_id:<5} {state:<9} {' ' * 8}  {job.description}  ({job.lane})")
    # The desktop Viewer tracks its folding workers and scripts; the VR viewer, which has no queue, does not.
    workers, scripts = _external_counts() if queue is not None else (0, 0)
    if workers:
        lines.append(f"Folding workers running: {workers} (esmfold stop ends them)")
    if scripts:
        lines.append(f"Python scripts running: {scripts} (run stop ends them)")
    finished = list(queue.finished)[-5:] if queue is not None else []
    if finished:
        lines.append("Finished lately:")
        for job in finished:
            lines.append(f"#{job.job_id:<5} {job.status:<9} {job.seconds():6.1f} s  {job.line}")
    return lines, count


def list_jobs(viewer):
    lines, count = _lines(viewer)
    print("\n" + "\n".join(lines) + "\n" if lines else "")
    if count:
        msg = Message("%n job(s) waiting or running; the list is in the terminal.", n=count)
    else:
        msg = Message("No commands wait or run, and no file is being written.")
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def _refuse(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg, report_message=False)


def cancel_jobs(viewer, words):
    queue = _queue(viewer)
    reason = Message("Cancelled with jobs cancel.")
    files = [], [], []
    if not words:
        cancelled = queue.cancel_newest(1, reason) if queue is not None else []
        names = [job.label() for job in cancelled]
    elif len(words) == 1 and words[0].lower() == 'all':
        cancelled = queue.cancel_all(reason) if queue is not None else []
        names = [job.label() for job in cancelled]
        files = _cancel_file_jobs(viewer, _file_jobs(viewer))
    else:
        # Every number must name a job that can be cancelled, or nothing is cancelled.
        targets = []
        for word in words:
            match = _JOB_NUMBER.fullmatch(word)
            if not match:
                _refuse(viewer, Message("Error: '{word}' is not a job number. Use jobs cancel N, or jobs cancel all.",
                                        word=word))
                return
            number = int(match.group(1))
            job = queue.find(number) if queue is not None else None
            if job is not None:
                targets.append(('command', job))
                continue
            entry = next((pair for pair in _file_jobs(viewer) if pair[0].job_id == number), None)
            if entry is None:
                _refuse(viewer, Message("Error: No job #{job} waits or runs; nothing was cancelled.", job=number))
                return
            targets.append(('file', entry))
        names = [job.label() for kind, job in targets if kind == 'command' and queue.cancel(job, reason)]
        files = _cancel_file_jobs(viewer, [pair for kind, pair in targets if kind == 'file'])
    removed, stopping, already = files
    names.extend(_named(job) for job in removed)
    parts = []
    if names:
        parts.append(Message("Cancelled %n job(s): {jobs}.", n=len(names), jobs=", ".join(names)))
    if stopping:
        # Not "cancelled": the job ends at its next step, and jobs lists it as stopping until it has.
        parts.append(Message("Told %n running job(s) to stop: {jobs}. Each ends at its next step; until then jobs lists it as stopping.",
                             n=len(stopping), jobs=", ".join(_named(job) for job in stopping)))
    if already:
        parts.append(Message("Already stopping: {jobs}.", jobs=", ".join(_named(job) for job in already)))
    if not parts:
        msg = Message("Nothing to cancel: no command waits or runs.")
    else:
        msg = JoinedMessage(parts) if len(parts) > 1 else parts[0]
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
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
