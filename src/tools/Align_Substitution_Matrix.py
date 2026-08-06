# Copyright 2026 Xuebin Feng
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

"""
File: Align_Substitution_Matrix.py
===================================
Description:
This script computes an all-vs-all sequence similarity network utilizing traditional NCBI BLASTP logic.
It formats the input FASTA file into a local BLAST database and aligns it against itself to compute 
substitution matrix similarity metrics.

Input:
- A FASTA file containing all sequences to be aligned (`{SEQUENCE_SET}.fasta`).

Output:
- An HDF5 file containing the list of pairwise edges, their source/target indices, and their negative Log10(E-Value) score (`{SEQUENCE_SET}_BLAST_EValue.h5`).

Settings:
- SEQUENCE_SET: The base name of the sequence group.
- MATRIX: The amino acid substitution matrix to use for scoring (e.g. "BLOSUM62", "PAM30").
- NUM_THREADS: Number of parallel multiprocessing workers/cores to spawn for BLAST.
- E_VALUE_CUTOFF: The maximum statistical E-value threshold to save an alignment. Alignments worse than this are ignored.
- MAX_TARGET_SEQS: Limits the total number of hits returned per query sequence. Note that BLAST may cut off valid alignments if this is too low.
- BATCH_SIZE: Maximum number of parsed edges buffered or copied at once.

Algorithm:
1. Validates the FASTA inputs and initializes the workspace by clearing out any old run fragments if settings have changed.
2. Uses sequence splitting algorithms to partition the FASTA query into parallel chunks.
3. Prepares the target dataset by spawning an `NCBI makeblastdb` background task.
4. Concurrently maps the `blastp` binary across all query chunks. Output format is enforced as a tabular 12 column format strings.
5. Recursively parses the output chunk text files to match sequence IDs back to an integer map.
6. Converts resulting statistical E-Values into mathematical -Log10(E) variables to provide a linearly comparable edge score.
7. Deduplicates results to only store unique non-directional combinations (i < j).
8. Streams validated HDF5 batches into an atomically written final network.
"""
# %% Import Necessary Libraries
# Limit threads to prevent CPU thrashing
import os
try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import subprocess
import shutil
import tempfile
import glob
import hashlib
import numpy as np
import math
import multiprocessing
import json
import h5py
from Bio import SeqIO
from tqdm import tqdm

from utilities.FASTA_Sanitization import load_sanitized_fasta
from utilities.Embedding_HDF5 import validate_manifest_records

# %% =======================================
# CONFIGURATION
# ==========================================

INPUT_FASTA = None

# SETTINGS
MATRIX = "BLOSUM62"
NUM_THREADS = 12
BATCH_SIZE = 500000
BLASTP_DIR = ""

# EXECUTABLE PATHS (Will be resolved dynamically after settings load)
NCBI_BIN_DIR    = r"C:\Program Files\NCBI"
MAKEBLASTDB_CMD = "makeblastdb"
BLASTP_CMD      = "blastp"

FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
NETWORK_DIR = os.path.join("..", "Input_Files", "Networks_EValues")

# --- JSON Settings Override ---
import json
import ast
import os

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            all_settings = json.load(f)
            
            # 1. Load GLOBAL directories and convert relative paths to absolute paths
            if "DIRECTORIES" in all_settings:
                for k, v in all_settings["DIRECTORIES"].items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        # Expand relative paths dynamically based on the current PC
                        if not os.path.isabs(str(v)):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, str(v)))
                        globals()[k] = v
                        
            # 2. Load script-specific settings
            script_name = os.path.basename(__file__)
            if script_name in all_settings:
                user_settings = all_settings[script_name]
                for k, v in user_settings.items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        orig = globals()[k]
                        
                        # Type casting to match the original Python variable type
                        if isinstance(orig, int) and not isinstance(orig, bool):
                            try: v = int(v)
                            except: pass
                        elif isinstance(orig, float):
                            try: v = float(v)
                            except: pass
                        elif isinstance(orig, list):
                            try: v = ast.literal_eval(v) if isinstance(v, str) else v
                            except: pass
                        elif orig is None:
                            if v == "None": v = None
                            elif str(v).replace('.', '', 1).isdigit():
                                v = float(v) if '.' in str(v) else int(v)
                                
                        # Convert any script-specific directory paths to absolute paths
                        if isinstance(v, str) and k.endswith("_DIR") and not os.path.isabs(v):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, v))
                            
                        globals()[k] = v
    except Exception as e:
        print(f"Failed to load user settings: {e}")

# Resolve BLAST commands after config overrides

# Resolve BLASTP and MAKEBLASTDB paths
if BLASTP_DIR and os.path.exists(BLASTP_DIR):
    MAKEBLASTDB_CMD = os.path.join(BLASTP_DIR, "makeblastdb.exe" if os.name == "nt" else "makeblastdb")
    BLASTP_CMD = os.path.join(BLASTP_DIR, "blastp.exe" if os.name == "nt" else "blastp")
else:
    # Fallback search if not specified or empty
    if shutil.which("blastp") or shutil.which("blastp.exe"):
        MAKEBLASTDB_CMD = "makeblastdb"
        BLASTP_CMD = "blastp"
    else:
        # Check standard default installation folders depending on OS
        if os.name == "nt":
            # Search C:\Program Files\NCBI dynamically for any version of BLAST
            ncbi_dir = r"C:\Program Files\NCBI"
            found_dir = None
            if os.path.exists(ncbi_dir):
                try:
                    valid_dirs = []
                    for d in os.listdir(ncbi_dir):
                        bin_path = os.path.join(ncbi_dir, d, "bin")
                        if os.path.exists(os.path.join(bin_path, "blastp.exe")):
                            valid_dirs.append(bin_path)
                    if valid_dirs:
                        valid_dirs.sort(reverse=True)
                        found_dir = valid_dirs[0]
                except:
                    pass
            
            if found_dir:
                MAKEBLASTDB_CMD = os.path.join(found_dir, "makeblastdb.exe")
                BLASTP_CMD = os.path.join(found_dir, "blastp.exe")
            else:
                MAKEBLASTDB_CMD = "makeblastdb"
                BLASTP_CMD = "blastp"
        else:
            # Unix fallback search
            unix_fallbacks = [
                "/usr/local/ncbi/blast/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/opt/homebrew/bin"
            ]
            found_dir = None
            for path in unix_fallbacks:
                if os.path.exists(os.path.join(path, "blastp")):
                    found_dir = path
                    break
            
            if found_dir:
                MAKEBLASTDB_CMD = os.path.join(found_dir, "makeblastdb")
                BLASTP_CMD = os.path.join(found_dir, "blastp")
            else:
                MAKEBLASTDB_CMD = "makeblastdb"
                BLASTP_CMD = "blastp"

# ADVANCED
E_VALUE_CUTOFF = 1e300 # Maximum E-value threshold; sequence hit pairs evaluated above this cutoff are entirely discarded.
MAX_TARGET_SEQS = 1000000 # The maximum threshold of mathematically aligned sequence hit traces retained per query.

SEQUENCE_SET = INPUT_FASTA.replace(".fasta", "") if INPUT_FASTA else "Unknown_Set"

FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA) if INPUT_FASTA else ""
OUTPUT_HDF5 = os.path.join(NETWORK_DIR, f"{SEQUENCE_SET}_[BLAST]_EValue.h5")

# Keep resumable BLAST work beside the network it will produce. This path is
# derived rather than user-configurable so it always follows NETWORK_DIR.
SAFE_TEMP_DIR = os.path.join(
    NETWORK_DIR,
    f"{SEQUENCE_SET}_[BLAST]_EValue_temp",
)

# WORKSPACE SETUP
CHUNKS_DIR    = os.path.join(SAFE_TEMP_DIR, "chunks")
RESULTS_DIR   = os.path.join(SAFE_TEMP_DIR, "results")
BATCH_DIR     = os.path.join(SAFE_TEMP_DIR, "batches")
CONFIG_FILE   = os.path.join(SAFE_TEMP_DIR, "job_config.json")

# %% =======================================
# HELPER FUNCTIONS
# ==========================================

def calculate_file_sha256(file_path):
    """Calculate a SHA-256 digest without loading the complete file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_manifest_sha256(headers, sequences):
    """Hash the ordered sanitized header/sequence manifest unambiguously."""
    hasher = hashlib.sha256()
    for header, sequence in zip(headers, sequences):
        for value in (header, sequence):
            encoded = value.encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
    return hasher.hexdigest()


def get_blastp_version(executable=BLASTP_CMD):
    """Return the first non-empty line reported by ``blastp -version``."""
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Unable to query BLASTP version using '{executable}': {error}"
        ) from error

    output = completed.stdout or completed.stderr or ""
    version = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not version:
        raise RuntimeError(f"BLASTP at '{executable}' returned no version information.")
    return version


def load_sanitized_records(input_fasta):
    """Load exactly the sanitized record manifest used by embedding generation."""
    headers, sequences, _ = load_sanitized_fasta(input_fasta)
    validate_manifest_records(headers, sequences)
    return headers, sequences


def write_numeric_fasta(sequences, output_fasta):
    """Write numeric BLAST identifiers atomically for sanitized sequences."""
    output_directory = os.path.dirname(output_fasta)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    partial_path = output_fasta + ".partial"
    with open(partial_path, "w", encoding="utf-8", newline="\n") as out_fasta:
        for index, sequence in enumerate(sequences):
            out_fasta.write(f">{index}\n{sequence}\n")
    os.replace(partial_path, output_fasta)


def prepare_blast_fasta(input_fasta, output_fasta):
    """
    Apply the canonical embedding FASTA sanitization and assign numeric IDs.

    The returned headers are the sanitized, deduplicated headers written to
    the final network. BLAST sees stable integer IDs while evaluating exactly
    the same sanitized sequences used by Generate_Embeddings.py.
    """
    headers, sequences = load_sanitized_records(input_fasta)
    write_numeric_fasta(sequences, output_fasta)
    return headers, sequences


# --- BLAST space-free path aliasing -----------------------------------------
#
# NCBI BLAST parses database paths as a whitespace-delimited list of names, so
# an absolute path containing a space is truncated at the first space: a project
# stored under "E:\OneDrive - University of Toronto\..." is looked up as the
# database "E:\OneDrive". Quoting cannot help because the splitting happens
# inside BLAST, after the argument has been received intact.
#
# Passing relative paths is not sufficient either. makeblastdb re-resolves its
# own -out value against the working directory when it verifies the finished
# database, so a space in the working directory reintroduces the failure even
# though every database file was written correctly.
#
# The only reliable fix is to hand the BLAST binaries a path to the workspace
# that contains no spaces at all.


def _windows_short_path(path):
    """Return the 8.3 short path for ``path``, or None when unavailable."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    try:
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short_path.restype = wintypes.DWORD
        required = get_short_path(path, None, 0)
        if not required:
            return None
        buffer = ctypes.create_unicode_buffer(required)
        if not get_short_path(path, buffer, required):
            return None
        return buffer.value
    except (AttributeError, OSError, ValueError):
        return None


def _space_free_link_bases(workspace):
    """Yield writable base directories whose own paths contain no spaces."""
    candidates = [tempfile.gettempdir(), os.environ.get("LOCALAPPDATA")]
    workspace_drive = os.path.splitdrive(workspace)[0]
    if workspace_drive:
        candidates.append(workspace_drive + os.sep)
    if os.name == "nt":
        system_drive = os.environ.get("SystemDrive")
        if system_drive:
            candidates.append(system_drive + os.sep)
    else:
        candidates.append("/tmp")

    seen = set()
    for base in candidates:
        if not base:
            continue
        base = os.path.abspath(base)
        if " " in base or base in seen or not os.path.isdir(base):
            continue
        seen.add(base)
        yield base


def _create_directory_link(link_path, target_path):
    """Link ``link_path`` to ``target_path`` without copying any data."""
    if os.name == "nt":
        # Junctions, unlike symlinks, need no elevation or developer mode.
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", link_path, target_path],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise OSError(detail or "mklink failed")
    else:
        os.symlink(target_path, link_path, target_is_directory=True)


def _remove_directory_link(link_path):
    """Remove only the link itself, never the workspace it points at."""
    try:
        if os.name == "nt":
            os.rmdir(link_path)
        else:
            os.unlink(link_path)
    except OSError:
        pass


def _links_to(link_path, target_path):
    try:
        return os.path.realpath(link_path) == os.path.realpath(target_path)
    except OSError:
        return False


def blast_path(path, workspace_root, blast_workspace):
    """
    Rewrite a workspace file as an absolute path under the space-free alias.

    Every BLAST argument is passed this way rather than as a relative name,
    because the working directory cannot be relied upon: POSIX resolves a
    symlinked cwd back to its physical (spaced) path before a child process
    sees it, whereas Windows junctions keep the logical path. Absolute alias
    paths behave identically on both.
    """
    relative = os.path.relpath(os.path.abspath(path), workspace_root)
    return os.path.normpath(os.path.join(blast_workspace, relative))


def resolve_blast_workspace(workspace):
    """
    Expose ``workspace`` to BLAST through a path that contains no spaces.

    Returns ``(blast_path, release)``. ``blast_path`` is the root that every
    BLAST argument is rebased onto; ``release`` removes any temporary link
    that was created and never touches the workspace contents.
    """
    absolute = os.path.abspath(workspace)
    if " " not in absolute:
        return absolute, lambda: None

    short_path = _windows_short_path(absolute)
    if short_path and " " not in short_path:
        print(f"-> Workspace path contains spaces; using short path: {short_path}")
        return short_path, lambda: None

    link_name = "ssn_blast_" + hashlib.sha256(absolute.encode("utf-8")).hexdigest()[:16]
    failures = []
    for base in _space_free_link_bases(absolute):
        link_path = os.path.join(base, link_name)
        try:
            # lexists, so a link left behind by an interrupted run is still
            # seen even when its former target no longer exists.
            if os.path.lexists(link_path):
                if _links_to(link_path, absolute):
                    print(f"-> Reusing space-free BLAST link: {link_path}")
                    return link_path, lambda: _remove_directory_link(link_path)
                _remove_directory_link(link_path)
            _create_directory_link(link_path, absolute)
        except OSError as error:
            failures.append(f"{base}: {error}")
            continue
        print(f"-> Workspace path contains spaces; linking BLAST to: {link_path}")
        return link_path, lambda: _remove_directory_link(link_path)

    if os.name == "nt":
        drive = os.path.splitdrive(absolute)[0] or "E:"
        suggestion = f'    subst S: "{drive}\\"     (then set NETWORK_DIR under S:)'
    else:
        suggestion = "    ln -s '" + absolute + "' /tmp/ssn_blast   (then set NETWORK_DIR there)"

    raise RuntimeError(
        "BLAST cannot open a database whose path contains a space, and no "
        "space-free link to the workspace could be created:\n"
        f"  {absolute}\n"
        "  Attempts: " + ("; ".join(failures) if failures else "no candidate location") + "\n"
        "  Fix this by setting NETWORK_DIR to a path without spaces, or by "
        "creating the link yourself, for example:\n"
        + suggestion
    )


def _validate_workspace_path():
    network_dir = os.path.abspath(NETWORK_DIR)
    workspace = os.path.abspath(SAFE_TEMP_DIR)
    if workspace == network_dir or os.path.commonpath([network_dir, workspace]) != network_dir:
        raise RuntimeError(
            f"Refusing workspace operation outside NETWORK_DIR: {workspace}"
        )


def _unique_backup_path(path):
    backup_path = f"{path}_BackUp"
    counter = 1
    while os.path.exists(backup_path):
        backup_path = f"{path}_BackUp_{counter}"
        counter += 1
    return backup_path


def quarantine_workspace(reason):
    """Move an incompatible unfinished workspace aside with a reason report."""
    _validate_workspace_path()
    backup_path = _unique_backup_path(SAFE_TEMP_DIR)
    shutil.move(SAFE_TEMP_DIR, backup_path)
    report_path = os.path.join(backup_path, "resume_mismatch.txt")
    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        report.write("BLAST alignment workspace was not resumed.\n")
        report.write(f"Reason: {reason}\n")
    print(f"\nWARNING: Existing workspace is incompatible: {reason}")
    print(f"  > Moved to: {backup_path}")
    return backup_path


def _write_json_atomic(file_path, data):
    partial_path = file_path + ".partial"
    with open(partial_path, "w", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, indent=4, sort_keys=True)
        output.write("\n")
    os.replace(partial_path, file_path)


def check_and_initialize_workspace(current_config):
    """Resume only a workspace with an exact matching exposed configuration."""
    _validate_workspace_path()
    resume = False
    if os.path.isdir(SAFE_TEMP_DIR):
        existing_entries = os.listdir(SAFE_TEMP_DIR)
        if existing_entries:
            if not os.path.exists(CONFIG_FILE):
                quarantine_workspace("missing job_config.json")
            else:
                try:
                    with open(CONFIG_FILE, "r", encoding="utf-8") as source:
                        saved_config = json.load(source)
                except (OSError, ValueError) as error:
                    quarantine_workspace(f"unreadable job_config.json: {error}")
                else:
                    if saved_config != current_config:
                        differing_keys = sorted(
                            key
                            for key in set(saved_config) | set(current_config)
                            if saved_config.get(key) != current_config.get(key)
                        )
                        quarantine_workspace(
                            "configuration mismatch for " + ", ".join(differing_keys)
                        )
                    else:
                        resume = True
                        print("\nResuming compatible previous run...")

    for directory in [SAFE_TEMP_DIR, CHUNKS_DIR, RESULTS_DIR, BATCH_DIR]:
        os.makedirs(directory, exist_ok=True)
    _write_json_atomic(CONFIG_FILE, current_config)
    return resume


def split_fasta_into_chunks(fasta_path, num_chunks, output_dir):
    """Regenerate deterministic query chunks and return paths with SHA-256 IDs."""
    print("-> Splitting query file...")
    for existing in glob.glob(os.path.join(glob.escape(output_dir), "chunk_*.fasta*")):
        try:
            os.remove(existing)
        except OSError:
            pass

    records = list(SeqIO.parse(fasta_path, "fasta"))
    chunk_size = math.ceil(len(records) / num_chunks)
    chunk_infos = []

    for i in range(num_chunks):
        chunk_records = records[i*chunk_size : (i+1)*chunk_size]
        if not chunk_records: continue
        chunk_name = f"chunk_{i:05d}.fasta"
        chunk_path = os.path.join(output_dir, chunk_name)
        partial_path = chunk_path + ".partial"
        SeqIO.write(chunk_records, partial_path, "fasta")
        os.replace(partial_path, chunk_path)
        chunk_infos.append((chunk_path, calculate_file_sha256(chunk_path)))

    return chunk_infos

def run_alignment_worker(args):
    """
    Runs the BLAST command.
    """
    (
        query_file,
        target_ref,
        exe_path,
        matrix,
        evalue,
        out_file,
        workspace_root,
        blast_workspace,
    ) = args
    if os.path.exists(out_file):
        return "Skipped"

    limit_count = str(MAX_TARGET_SEQS) 
    partial_out = out_file + ".partial"
    try:
        if os.path.exists(partial_out):
            os.remove(partial_out)
    except OSError:
        pass

    # Address every file through the space-free alias, so no path BLAST
    # parses can contain a space regardless of platform.
    query_argument = blast_path(query_file, workspace_root, blast_workspace)
    database_argument = blast_path(target_ref, workspace_root, blast_workspace)
    output_argument = blast_path(partial_out, workspace_root, blast_workspace)

    cmd = [
        exe_path, "-query", query_argument, "-db", database_argument, "-out", output_argument,
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
        "-matrix", matrix, "-evalue", str(evalue),
        "-max_target_seqs", limit_count, "-max_hsps", "1", "-comp_based_stats", "0" 
    ]
    
    try:
        # Capture stderr for debugging
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=blast_workspace,
        )
        os.replace(partial_out, out_file)
        return "Done"
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except Exception as e:
        return f"SysError: {str(e)}"


BATCH_ATTRIBUTE_NAMES = (
    "complete",
    "model_name",
    "matrix",
    "query_chunk_sha256",
    "source_result_filename",
    "source_result_sha256",
)


def _plain_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bounded_slices(length, max_items):
    """Yield half-open slices containing no more than ``max_items`` entries."""
    max_items = int(max_items)
    if max_items <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    for start in range(0, int(length), max_items):
        yield start, min(start + max_items, int(length))


def _batch_attributes(run_metadata, query_checksum, result_path, result_checksum):
    return {
        "complete": True,
        "model_name": "BLAST",
        "matrix": run_metadata["matrix"],
        "query_chunk_sha256": query_checksum,
        "source_result_filename": os.path.basename(result_path),
        "source_result_sha256": result_checksum,
    }


def _validate_edge_datasets(hf, sequence_count, read_size, index_dtypes):
    required = ("i", "j", "score")
    if not all(name in hf for name in required):
        missing = [name for name in required if name not in hf]
        return False, "missing dataset(s): " + ", ".join(missing), 0

    datasets = [hf[name] for name in required]
    if any(dataset.ndim != 1 for dataset in datasets):
        return False, "edge datasets must be one-dimensional", 0

    edge_count = len(datasets[0])
    if any(len(dataset) != edge_count for dataset in datasets[1:]):
        return False, "edge dataset lengths do not match", 0

    if np.dtype(hf["i"].dtype) not in index_dtypes:
        return False, f"invalid i dtype: {hf['i'].dtype}", 0
    if np.dtype(hf["j"].dtype) not in index_dtypes:
        return False, f"invalid j dtype: {hf['j'].dtype}", 0
    if np.dtype(hf["score"].dtype) != np.dtype(np.float32):
        return False, f"invalid score dtype: {hf['score'].dtype}", 0

    for start, end in _bounded_slices(edge_count, read_size):
        arr_i = hf["i"][start:end]
        arr_j = hf["j"][start:end]
        arr_score = hf["score"][start:end]
        if np.any(arr_i >= sequence_count) or np.any(arr_j >= sequence_count):
            return False, "edge index is outside the header array", 0
        if np.any(arr_i >= arr_j):
            return False, "edge datasets contain a pair that does not satisfy i < j", 0
        if not np.isfinite(arr_score).all():
            return False, "score dataset contains a non-finite value", 0

    return True, "", edge_count


def validate_batch_file(batch_path, expected_attrs, sequence_count, read_size=None):
    """Validate cache identity, schema, and edge values without unbounded reads."""
    read_size = BATCH_SIZE if read_size is None else read_size
    try:
        with h5py.File(batch_path, "r") as hf:
            for name in BATCH_ATTRIBUTE_NAMES:
                if name not in hf.attrs:
                    return False, f"missing attribute '{name}'", 0
                actual = _plain_attr_value(hf.attrs[name])
                expected = expected_attrs[name]
                if actual != expected:
                    return (
                        False,
                        f"attribute '{name}' differs ({actual!r} != {expected!r})",
                        0,
                    )
            return _validate_edge_datasets(
                hf,
                sequence_count,
                read_size,
                {np.dtype(np.uint32)},
            )
    except (OSError, ValueError, KeyError) as error:
        return False, f"unable to read batch: {error}", 0


def quarantine_batch(batch_path, reason):
    """Move one recoverable invalid batch aside and record why."""
    invalid_dir = os.path.join(BATCH_DIR, "invalid_batches")
    os.makedirs(invalid_dir, exist_ok=True)
    base_name = os.path.basename(batch_path)
    destination = os.path.join(invalid_dir, base_name)
    counter = 1
    while os.path.exists(destination):
        stem, extension = os.path.splitext(base_name)
        destination = os.path.join(invalid_dir, f"{stem}_{counter}{extension}")
        counter += 1
    shutil.move(batch_path, destination)
    with open(destination + ".reason.txt", "w", encoding="utf-8", newline="\n") as report:
        report.write(reason + "\n")
    print(f"WARNING: Rebuilding invalid batch '{base_name}': {reason}")
    return destination


def _append_edge_buffer(hf, sources, targets, scores):
    if not sources:
        return
    old_size = len(hf["i"])
    new_size = old_size + len(sources)
    for name in ("i", "j", "score"):
        hf[name].resize((new_size,))
    hf["i"][old_size:new_size] = np.asarray(sources, dtype=np.uint32)
    hf["j"][old_size:new_size] = np.asarray(targets, dtype=np.uint32)
    hf["score"][old_size:new_size] = np.asarray(scores, dtype=np.float32)


def parse_result_to_batch(result_path, query_checksum, run_metadata, batch_size=None):
    """Reuse or atomically create one bounded-memory HDF5 parsing batch."""
    batch_size = BATCH_SIZE if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"BLAST result is missing: {result_path}")

    result_checksum = calculate_file_sha256(result_path)
    expected_attrs = _batch_attributes(
        run_metadata,
        query_checksum,
        result_path,
        result_checksum,
    )
    batch_path = os.path.join(BATCH_DIR, f"batch_{query_checksum}.h5")

    if os.path.exists(batch_path):
        valid, reason, edge_count = validate_batch_file(
            batch_path,
            expected_attrs,
            int(run_metadata["sequence_count"]),
            batch_size,
        )
        if valid:
            return {
                "path": batch_path,
                "attrs": expected_attrs,
                "edges": edge_count,
                "reused": True,
                "valid_lines": None,
                "mismatches": None,
            }
        quarantine_batch(batch_path, reason)

    partial_path = batch_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    source_buffer = []
    target_buffer = []
    score_buffer = []
    valid_lines = 0
    mismatches = 0
    sequence_count = int(run_metadata["sequence_count"])
    hdf5_chunk = max(1, min(batch_size, 65536))

    with h5py.File(partial_path, "w") as hf:
        for name, value in expected_attrs.items():
            hf.attrs[name] = False if name == "complete" else value
        hf.create_dataset(
            "i", shape=(0,), maxshape=(None,), chunks=(hdf5_chunk,), dtype=np.uint32
        )
        hf.create_dataset(
            "j", shape=(0,), maxshape=(None,), chunks=(hdf5_chunk,), dtype=np.uint32
        )
        hf.create_dataset(
            "score", shape=(0,), maxshape=(None,), chunks=(hdf5_chunk,), dtype=np.float32
        )

        with open(result_path, "r", encoding="utf-8", errors="replace") as result_file:
            for line in result_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                columns = stripped.split()
                if len(columns) < 12:
                    mismatches += 1
                    continue

                try:
                    source = int(columns[0])
                    target = int(columns[1])
                    raw_evalue = float(columns[10])
                except ValueError:
                    mismatches += 1
                    continue
                if (
                    source < 0
                    or target < 0
                    or source >= sequence_count
                    or target >= sequence_count
                    or raw_evalue < 0
                    or not math.isfinite(raw_evalue)
                ):
                    mismatches += 1
                    continue

                valid_lines += 1
                if source < target:
                    source_buffer.append(source)
                    target_buffer.append(target)
                    score_buffer.append(-math.log10(raw_evalue + 1e-300))
                    if len(source_buffer) >= batch_size:
                        _append_edge_buffer(
                            hf,
                            source_buffer,
                            target_buffer,
                            score_buffer,
                        )
                        source_buffer.clear()
                        target_buffer.clear()
                        score_buffer.clear()

        _append_edge_buffer(hf, source_buffer, target_buffer, score_buffer)
        hf.attrs["complete"] = True
        hf.flush()

    os.replace(partial_path, batch_path)
    valid, reason, edge_count = validate_batch_file(
        batch_path,
        expected_attrs,
        sequence_count,
        batch_size,
    )
    if not valid:
        quarantine_batch(batch_path, reason)
        raise RuntimeError(f"Newly written batch failed validation: {reason}")

    return {
        "path": batch_path,
        "attrs": expected_attrs,
        "edges": edge_count,
        "reused": False,
        "valid_lines": valid_lines,
        "mismatches": mismatches,
    }


def validate_final_output(output_path, expected_edges, sequence_count, read_size=None):
    """Validate a completed final network before publishing it."""
    read_size = BATCH_SIZE if read_size is None else read_size
    try:
        with h5py.File(output_path, "r") as hf:
            if "headers" not in hf:
                return False, "missing headers dataset"
            if len(hf["headers"]) != sequence_count:
                return False, "header count does not match sequence_count"
            valid, reason, edge_count = _validate_edge_datasets(
                hf,
                sequence_count,
                read_size,
                {np.dtype(np.uint16), np.dtype(np.uint32)},
            )
            if not valid:
                return False, reason
            if edge_count != expected_edges:
                return False, "final edge count does not match batch total"
    except (OSError, ValueError, KeyError) as error:
        return False, f"unable to read final output: {error}"
    return True, ""


def compile_final_output(headers, batch_records, run_metadata, output_path=None, batch_size=None):
    """Compile validated batches using reads bounded by ``batch_size``."""
    output_path = OUTPUT_HDF5 if output_path is None else output_path
    batch_size = BATCH_SIZE if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")

    total_edges = 0
    for record in batch_records:
        valid, reason, edge_count = validate_batch_file(
            record["path"],
            record["attrs"],
            len(headers),
            batch_size,
        )
        if not valid:
            raise RuntimeError(
                f"Cannot compile invalid batch '{record['path']}': {reason}"
            )
        total_edges += edge_count

    output_path = os.path.normpath(output_path)
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    partial_path = output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    index_dtype = np.uint16 if len(headers) <= 65535 else np.uint32
    print(f"  > Selected {index_dtype.__name__} for {len(headers)} sequences.")
    print(f"Saving Combined Scores to {output_path} ({total_edges} edges)...")

    with h5py.File(partial_path, "w") as output:
        output.attrs["model_name"] = "BLAST"
        output.attrs["matrix"] = run_metadata["matrix"]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        output.create_dataset(
            "headers",
            data=np.asarray(headers, dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset("i", shape=(total_edges,), dtype=index_dtype)
        output.create_dataset("j", shape=(total_edges,), dtype=index_dtype)
        output.create_dataset("score", shape=(total_edges,), dtype=np.float32)

        write_offset = 0
        for record in tqdm(batch_records, desc="Merging"):
            with h5py.File(record["path"], "r") as batch:
                edge_count = len(batch["i"])
                for start, end in _bounded_slices(edge_count, batch_size):
                    write_end = write_offset + (end - start)
                    output["i"][write_offset:write_end] = batch["i"][start:end]
                    output["j"][write_offset:write_end] = batch["j"][start:end]
                    output["score"][write_offset:write_end] = batch["score"][start:end]
                    write_offset = write_end
        output.flush()

    valid, reason = validate_final_output(
        partial_path,
        total_edges,
        len(headers),
        batch_size,
    )
    if not valid:
        raise RuntimeError(f"Final output validation failed: {reason}")
    os.replace(partial_path, output_path)
    return total_edges


def cleanup_workspace():
    """Remove only the derived workspace after a validated final publication."""
    _validate_workspace_path()
    shutil.rmtree(SAFE_TEMP_DIR)

# %% =======================================
# MAIN WORKFLOW
# ==========================================

def run_workflow():
    print(f"--- BLAST All-vs-All (-Log10 E-Value Mode) ---")

    # Linux defaults to fork, which is unsafe here: the pool below is created
    # after HDF5 handles are open and each worker spawns a BLAST subprocess.
    # Windows and macOS already default to spawn, so this only changes Linux.
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass

    if not isinstance(NUM_THREADS, int) or NUM_THREADS <= 0:
        raise ValueError("NUM_THREADS must be a positive integer.")
    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")

    # 0. RESOLVE AND SANITIZE INPUT BEFORE TOUCHING WORKSPACE STATE
    clean_fasta_path = os.path.normpath(FULL_INPUT_FASTA)
    if not FULL_INPUT_FASTA or not os.path.exists(clean_fasta_path):
        print(f"❌ Error: Input FASTA not found at:\n{clean_fasta_path}")
        sys.exit(1)

    print(f"Sanitizing {clean_fasta_path}...")
    headers, sequences = load_sanitized_records(clean_fasta_path)
    input_checksum = calculate_file_sha256(clean_fasta_path)
    manifest_checksum = calculate_manifest_sha256(headers, sequences)
    blastp_version = get_blastp_version()
    run_metadata = {
        "input_fasta": str(INPUT_FASTA),
        "input_fasta_sha256": input_checksum,
        "sanitized_manifest_sha256": manifest_checksum,
        "sequence_count": len(headers),
        "matrix": str(MATRIX),
        "num_threads": int(NUM_THREADS),
        "blastp_version": blastp_version,
    }
    check_and_initialize_workspace(run_metadata)

    # 1. GENERATE NUMERIC-ID FASTA
    # Keep the BLAST-facing filename free of spaces as well as making it
    # relative below. The user-facing sequence-set name remains in the final
    # output and provenance metadata.
    safe_fasta_path = os.path.join(SAFE_TEMP_DIR, "sanitized_input.fasta")
    write_numeric_fasta(sequences, safe_fasta_path)
    print(f"Loaded {len(headers)} sanitized sequences from FASTA.")

    # 2. REBUILD TARGET DATABASE
    # Every BLAST invocation runs from a space-free view of the workspace,
    # because BLAST splits database paths on whitespace internally.
    workspace_root = os.path.abspath(SAFE_TEMP_DIR)
    blast_workspace, release_blast_workspace = resolve_blast_workspace(SAFE_TEMP_DIR)
    try:
        db_name = os.path.join(SAFE_TEMP_DIR, "temp_db")
        for database_file in glob.glob(glob.escape(db_name) + ".*"):
            try:
                os.remove(database_file)
            except OSError:
                pass
        print("Building BLAST Database...")
        # makeblastdb re-opens the finished database by absolute path to
        # verify it, so -in and -out are addressed through the space-free
        # alias rather than as relative names.
        safe_fasta_argument = blast_path(
            safe_fasta_path, workspace_root, blast_workspace
        )
        database_argument = blast_path(db_name, workspace_root, blast_workspace)
        cmd_db = [
            MAKEBLASTDB_CMD,
            "-in", safe_fasta_argument,
            "-dbtype", "prot",
            "-out", database_argument,
        ]
        try:
            subprocess.run(
                cmd_db,
                check=True,
                capture_output=True,
                text=True,
                cwd=blast_workspace,
            )
        except subprocess.CalledProcessError as error:
            print(
                "❌ BLAST DATABASE ERROR\nSTDOUT:",
                error.stdout,
                "\nSTDERR:",
                error.stderr,
            )
            sys.exit(1)

        target_reference = db_name

        # 3. SPLIT & RUN
        chunk_infos = split_fasta_into_chunks(safe_fasta_path, NUM_THREADS, CHUNKS_DIR)
        print(f"Running {len(chunk_infos)} alignment jobs...")

        tasks = []
        result_infos = []
        for chunk_path, query_checksum in chunk_infos:
            result_path = os.path.join(RESULTS_DIR, f"result_{query_checksum}.txt")
            result_infos.append((query_checksum, result_path))
            tasks.append(
                (
                    chunk_path,
                    target_reference,
                    BLASTP_CMD,
                    MATRIX,
                    E_VALUE_CUTOFF,
                    result_path,
                    workspace_root,
                    blast_workspace,
                )
            )

        # Run and check for errors
        error_log = []
        if tasks:
            with multiprocessing.Pool(processes=NUM_THREADS) as pool:
                for result in tqdm(
                    pool.imap(run_alignment_worker, tasks),
                    total=len(tasks),
                    desc="Aligning",
                ):
                    if result.startswith("Error") or result.startswith("SysError"):
                        error_log.append(result)

        if error_log:
            print("\n❌ CRITICAL: BLAST Worker Failed!")
            print("Last Error Message:")
            print(error_log[-1])
            sys.exit(1)
    finally:
        release_blast_workspace()

    # 4. PARSE OR RESUME VALIDATED HDF5 BATCHES
    batch_records = []
    parsed_files = 0
    reused_files = 0
    parsed_valid_lines = 0
    parsed_mismatches = 0
    total_edges = 0
    for query_checksum, result_path in result_infos:
        record = parse_result_to_batch(
            result_path,
            query_checksum,
            run_metadata,
            BATCH_SIZE,
        )
        batch_records.append(record)
        total_edges += record["edges"]
        if record["reused"]:
            reused_files += 1
        else:
            parsed_files += 1
            parsed_valid_lines += record["valid_lines"]
            parsed_mismatches += record["mismatches"]

    print("\n" + "-"*40)
    print("PARSING DIAGNOSTICS")
    print("-"*40)
    print(f"Result Files Parsed: {parsed_files}")
    print(f"Batches Reused:      {reused_files}")
    print(f"Valid Parsed Hits:   {parsed_valid_lines}")
    print(f"Rejected Lines:      {parsed_mismatches}")
    possible_edges = len(headers) * (len(headers) - 1) // 2
    edge_coverage = (100.0 * total_edges / possible_edges) if possible_edges else 0.0
    print(
        f"Edges Saved:         {total_edges}/{possible_edges} "
        f"({edge_coverage:.1f}%)"
    )
    if total_edges == 0:
        print(">> No non-self u < v alignments were retained.")
    print("-"*40 + "\n")

    # 5. COMPILE WITHOUT LOADING THE COMPLETE NETWORK
    print("Consolidating...")
    output_full_path = os.path.normpath(OUTPUT_HDF5)
    compile_final_output(
        headers,
        batch_records,
        run_metadata,
        output_full_path,
        BATCH_SIZE,
    )

    print(f"Done! Saved to {output_full_path}")

    print("\n🧹 Cleaning up temporary workspace...")
    try:
        cleanup_workspace()
        print(f"✅ Successfully deleted temporary folder: {SAFE_TEMP_DIR}")
    except Exception as e:
        print(f"⚠️ Could not automatically delete temp folder (files might be in use): {e}")

if __name__ == "__main__":
    run_workflow()
