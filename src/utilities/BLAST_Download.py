# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Download and install the pinned NCBI BLAST+ programs on request.

src/resources/blast/blast_pin.json names one BLAST+ release and, for each
supported host, the NCBI archive (URL, size, SHA-256) and the files taken from
it (blastp and makeblastdb, plus nghttp2.dll on Windows), each with its own
size, SHA-256 and mode. ``install_blast`` is the whole procedure, meant to run
in a worker thread:

1. download the archive over HTTPS into ``blast-<version>+.download.partial``,
   hashing as it streams, and refuse a wrong size or SHA-256;
2. unpack only the listed files, flattened into ``blast-<version>+.partial``
   (tarfile's "data" filter, which keeps the owner's execute bit), and check
   each one's SHA-256 as it lies on the disk;
3. run ``blastp -version`` from there, so a computer that cannot start BLAST+
   learns it before anything is replaced;
4. swap the folder in as ``blast-<version>+``: an existing one is renamed to
   ``.previous`` first, removed after the swap and put back if the swap fails.

A cancel or any failure removes every partial file and leaves the previous
install as it was. Every failure is a BlastDownloadError whose message is for
the user.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.request
import zlib
from pathlib import Path
from typing import NamedTuple

from utilities import BLAST_Executables

PIN_PATH = Path(__file__).resolve().parents[1] / "resources" / "blast" / "blast_pin.json"

# The hosts the program runs on, and so the keys of the pin's platforms.
WINDOWS = "windows-x86_64"
LINUX = "linux-x86_64"
MACOS = "macos-arm64"
PLATFORMS = (WINDOWS, LINUX, MACOS)

CHUNK_BYTES = 256 * 1024
CONNECTION_TIMEOUT = 30
VERSION_CHECK_TIMEOUT = 60

VISUAL_CPP_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
# What a BLAST+ that will not start most likely lacks, by platform.
START_HINTS = {
    WINDOWS: (
        "On Windows this usually means the Microsoft Visual C++ 2015-2022 Redistributable (x64) is "
        "missing (msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll). Install it from "
        f"{VISUAL_CPP_URL} and try again."
    ),
    LINUX: (
        "BLAST+ needs glibc 2.28 or newer and the libgomp1 package (libgomp on Fedora and RHEL); "
        "it does not run on musl-based systems such as Alpine."
    ),
    MACOS: "BLAST+ needs macOS 14 or newer on Apple Silicon.",
}

_HEX_64 = re.compile(r"[0-9a-f]{64}")
_MODE = re.compile(r"0?[0-7]{3,4}")
_VERSION = re.compile(r"\d+(\.\d+)+")


class BlastDownloadError(Exception):
    """The install did not happen; ``str(error)`` says why, for the user."""


class BlastDownloadCancelled(BlastDownloadError):
    """The caller's cancel flag was set."""


class BlastInstall(NamedTuple):
    folder: str   # holds blastp and makeblastdb: the value for the BLASTP Directory
    version: str  # the first line of `blastp -version`


# -- the host and the pin ------------------------------------------------------

def _detect_platform():
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine in ("amd64", "x86_64"):
            return WINDOWS, None
        return None, f"The BLAST+ download is built for Windows on x64, and this computer is {machine or 'unknown'}."
    if sys.platform == "linux":
        if machine in ("x86_64", "amd64"):
            return LINUX, None
        return None, f"The BLAST+ download is built for Linux on x86-64, and this computer is {machine or 'unknown'}."
    if sys.platform == "darwin":
        if machine in ("arm64", "aarch64"):
            return MACOS, None
        return None, f"The BLAST+ download is built for macOS on Apple Silicon, and this computer is {machine or 'unknown'}."
    return None, f"The BLAST+ download has no build for {sys.platform}."


def platform_key():
    """The pin's platform key for this host, or None on a host the download does not cover."""
    return _detect_platform()[0]


def unsupported_reason():
    """Why ``platform_key()`` is None, or None on a supported host. Install BLAST+ yourself there."""
    return _detect_platform()[1]


def _plain_name(name):
    return (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and not any(character in name for character in '/\\:*?"<>|')
    )


def check_pin(pin):
    """Return ``pin`` when it is a well-formed pin document; BlastDownloadError otherwise."""

    def refuse(reason):
        raise BlastDownloadError(f"The BLAST+ pin file is not valid: {reason}.")

    def number(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    if not isinstance(pin, dict):
        refuse("it is not a JSON object")
    if not isinstance(pin.get("version"), str) or not _VERSION.fullmatch(pin["version"]):
        refuse("'version' is missing or is not a version number")
    platforms = pin.get("platforms")
    if not isinstance(platforms, dict) or not platforms:
        refuse("'platforms' is missing")
    for key, entry in platforms.items():
        where = f"platform {key}"
        if not isinstance(entry, dict):
            refuse(f"{where} is not an object")
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            refuse(f"{where} has no https URL")
        if not number(entry.get("size")) or not number(entry.get("installed_size")):
            refuse(f"{where} has no positive 'size' and 'installed_size'")
        if not isinstance(entry.get("sha256"), str) or not _HEX_64.fullmatch(entry["sha256"]):
            refuse(f"{where} has no lowercase hexadecimal SHA-256")
        prefix = entry.get("strip_prefix")
        if not isinstance(prefix, str):
            refuse(f"{where} has no 'strip_prefix'")
        members = entry.get("members")
        if not isinstance(members, list) or not members:
            refuse(f"{where} lists no members")
        names = []
        for member in members:
            if not isinstance(member, dict):
                refuse(f"a member of {where} is not an object")
            name = member.get("install_name")
            if not _plain_name(name):
                refuse(f"a member of {where} has an install_name that is not a plain file name")
            if member.get("archive_path") != prefix + name:
                refuse(f"{name} of {where} is not at strip_prefix plus its name in the archive")
            if not number(member.get("size")):
                refuse(f"{name} of {where} has no positive size")
            if not isinstance(member.get("sha256"), str) or not _HEX_64.fullmatch(member["sha256"]):
                refuse(f"{name} of {where} has no lowercase hexadecimal SHA-256")
            if not isinstance(member.get("mode"), str) or not _MODE.fullmatch(member["mode"]):
                refuse(f"{name} of {where} has no octal mode such as 0755")
            names.append(name)
        if len(set(names)) != len(names):
            refuse(f"{where} lists a file twice")
        stems = {name[:-4] if name.endswith(".exe") else name for name in names}
        if not {"blastp", "makeblastdb"} <= stems:
            refuse(f"{where} does not include blastp and makeblastdb")
    return pin


def read_pin(path=None):
    """The validated pin document (src/resources/blast/blast_pin.json unless ``path`` names another)."""
    path = PIN_PATH if path is None else Path(path)
    try:
        with open(path, encoding="utf-8") as handle:
            pin = json.load(handle)
    except (OSError, ValueError) as error:
        raise BlastDownloadError(f"The BLAST+ pin file {path} cannot be read: {error}") from error
    return check_pin(pin)


def install_folder_name(pin=None):
    """The name of the folder an install creates inside the destination, such as ``blast-2.17.0+``."""
    return f"blast-{(read_pin() if pin is None else pin)['version']}+"


# -- the steps -----------------------------------------------------------------

def _raise_if_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise BlastDownloadCancelled("The BLAST+ download was cancelled.")


def _remove(path, strict=False):
    """Delete a file or folder if it exists; with ``strict`` a failure is an error, else it is ignored."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as error:
        if strict:
            raise BlastDownloadError(
                f"{path} is left over from an earlier install and cannot be removed: {error}"
            ) from error


def _check_free_space(destination, entry):
    needed = entry["size"] + entry["installed_size"]  # the archive and the unpacked files together
    try:
        free = shutil.disk_usage(destination).free
    except OSError:
        return
    if free < needed:
        raise BlastDownloadError(
            f"There is not enough free disk space in {destination}: the install needs about "
            f"{needed / 1e6:.0f} MB while it runs and {free / 1e6:.0f} MB are free."
        )


def _recover(final, previous):
    """Undo what an install that was killed between its two renames left behind."""
    if not previous.exists():
        return
    if final.exists():
        _remove(previous, strict=True)
    else:
        os.rename(previous, final)  # the old install was moved aside and never replaced


def _open_source(source):
    if os.path.isfile(source):
        return open(source, "rb")
    if source.lower().startswith("https://"):
        request = urllib.request.Request(source, headers={"User-Agent": "EMAP-SSN BLAST+ installer"})
        return urllib.request.urlopen(request, timeout=CONNECTION_TIMEOUT)
    raise BlastDownloadError(f"{source} is neither an https URL nor a file on this computer.")


def _download(entry, source, archive, report, cancel):
    total = entry["size"]
    hasher = hashlib.sha256()
    done = 0
    report(0, total, "download")
    try:
        with _open_source(source) as stream, open(archive, "wb") as target:
            while True:
                _raise_if_cancelled(cancel)
                chunk = stream.read(CHUNK_BYTES)
                if not chunk:
                    break
                done += len(chunk)
                if done > total:
                    raise BlastDownloadError(
                        f"The download from {source} is larger than the {total} bytes the pin file expects."
                    )
                hasher.update(chunk)
                target.write(chunk)
                report(done, total, "download")
    except (OSError, http.client.HTTPException) as error:
        raise BlastDownloadError(
            f"The download from {source} failed: {error}. Check the connection and try again, or install "
            f"NCBI BLAST+ yourself and set the {BLAST_Executables.SOURCE_SETTING}."
        ) from error
    if done != total:
        raise BlastDownloadError(
            f"The download from {source} ended after {done} of {total} bytes. Nothing was installed."
        )
    if hasher.hexdigest() != entry["sha256"]:
        raise BlastDownloadError(
            f"The download from {source} failed its SHA-256 check, so nothing was installed "
            f"(expected {entry['sha256']}, got {hasher.hexdigest()})."
        )


def _has_modes():
    """Whether files carry an execute bit here, which Windows files do not."""
    return os.name != "nt"


def _file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract(entry, archive, staging, report, cancel):
    wanted = {member["archive_path"]: member for member in entry["members"]}
    total = entry["installed_size"]
    done = 0
    found = set()
    report(0, total, "extract")
    try:
        staging.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tar:
            for info in tar:
                _raise_if_cancelled(cancel)
                member = wanted.get(info.name[2:] if info.name.startswith("./") else info.name)
                if member is None:
                    continue
                name = member["install_name"]
                if not info.isreg() or info.size != member["size"]:
                    raise BlastDownloadError(
                        f"{info.name} in the archive is not the file the pin file describes, so nothing was installed."
                    )
                # Flattened: the file is written under its install name, not its path in the archive.
                tar.extract(info.replace(name=name), path=staging, filter="data")
                target = staging / name
                if _file_sha256(target) != member["sha256"]:
                    raise BlastDownloadError(f"{name} failed its SHA-256 check, so nothing was installed.")
                mode = int(member["mode"], 8)
                if _has_modes() and mode & 0o100 and not target.stat().st_mode & 0o100:
                    target.chmod(mode)
                found.add(name)
                done += info.size
                report(done, total, "extract")
                if len(found) == len(wanted):
                    break
    except (tarfile.TarError, EOFError, zlib.error, OSError) as error:
        raise BlastDownloadError(f"The BLAST+ archive could not be unpacked: {error}") from error
    missing = sorted(member["install_name"] for member in wanted.values() if member["install_name"] not in found)
    if missing:
        raise BlastDownloadError(f"The BLAST+ archive does not hold {', '.join(missing)}.")


def _check_runs(folder, key, entry, report):
    """Run blastp from the unpacked folder; its first -version line, or an error that says what to install."""
    report(0, 0, "check")
    blastp = next(
        member["install_name"] for member in entry["members"] if member["install_name"] in ("blastp", "blastp.exe")
    )
    try:
        return BLAST_Executables.query_version(str(folder / blastp), timeout=VERSION_CHECK_TIMEOUT)
    except BLAST_Executables.BlastError as error:
        raise BlastDownloadError(
            f"BLAST+ was downloaded and checked, but blastp cannot be started on this computer: {error} "
            f"{START_HINTS.get(key, '')} Nothing was installed."
        ) from error


def _swap(final, staging, previous):
    """Put ``staging`` in place as ``final``, keeping the old ``final`` until the new one is there."""
    moved_aside = False
    try:
        if final.exists() or final.is_symlink():
            os.rename(final, previous)
            moved_aside = True
        try:
            os.rename(staging, final)
        except OSError:
            if moved_aside:
                os.rename(previous, final)
            raise
    except OSError as error:
        raise BlastDownloadError(
            f"BLAST+ could not be put in place as {final}: {error}. If BLAST+ is running (a job "
            f"that uses it, for instance), close it and try again."
        ) from error


# -- the whole procedure -------------------------------------------------------

def install_blast(destination_folder, progress=None, cancel=None, *, source=None, pin=None, host=None):
    """Install the pinned BLAST+ as ``<destination_folder>/blast-<version>+`` and return a BlastInstall.

    ``progress(done_bytes, total_bytes, stage)`` is called from the calling
    thread for the stages "download" and "extract" (bytes of the archive and
    of the unpacked files) and "check" (0, 0: no size). ``cancel`` is a
    threading.Event or anything with ``is_set()``, looked at for every chunk
    and every archive entry; setting it raises BlastDownloadCancelled.

    ``source`` replaces the pinned URL with a local file or another https URL
    (for tests and for a computer that is offline: the archive must still
    match the pin's size and SHA-256). ``pin`` and ``host`` (a platform key)
    replace the pin file and the detected host, for tests.
    """
    report = progress or (lambda done, total, stage: None)
    pin = read_pin() if pin is None else check_pin(pin)
    key = platform_key() if host is None else host
    if key is None:
        raise BlastDownloadError(f"{unsupported_reason()} Install NCBI BLAST+ yourself and set the BLASTP Directory.")
    entry = pin["platforms"].get(key)
    if entry is None:
        raise BlastDownloadError(f"The BLAST+ pin file has no download for {key}.")
    if destination_folder is None or not str(destination_folder).strip():
        raise BlastDownloadError("Choose a folder to install BLAST+ in.")

    destination = Path(os.path.abspath(os.fspath(destination_folder)))
    name = install_folder_name(pin)
    final = destination / name
    staging = destination / f"{name}.partial"
    previous = destination / f"{name}.previous"
    archive = destination / f"{name}.download.partial"

    try:
        destination.mkdir(parents=True, exist_ok=True)
        _check_free_space(destination, entry)
        _recover(final, previous)
    except OSError as error:
        raise BlastDownloadError(f"BLAST+ cannot be installed in {destination}: {error}") from error
    for leftover in (archive, staging):
        _remove(leftover, strict=True)

    try:
        _download(entry, entry["url"] if source is None else str(source), archive, report, cancel)
        _extract(entry, archive, staging, report, cancel)
        _remove(archive)
        version = _check_runs(staging, key, entry, report)
        _raise_if_cancelled(cancel)
        _swap(final, staging, previous)
    finally:
        # After a swap there is no staging folder; after anything else there is nothing worth keeping.
        _remove(archive)
        _remove(staging)
    _remove(previous)  # housekeeping: a leftover is removed by the next install
    return BlastInstall(str(final), version)
