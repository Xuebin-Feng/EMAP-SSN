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

"""
File: Embedding_MSA.py
===================================
Description:
This script performs a Multiple Sequence Alignment (MSA) heavily optimized around protein language model (pLM) embeddings.
Traditional tools like MAFFT and MUSCLE rely solely on amino acid substitution matrices. This tool instead calculates mathematical 
consensus averages of structural embeddings to align sequences, often performing better on sequences with extremely low literal identity.

It implements a robust "auto-intersection" algorithm. It always intersects the Network File (Topology/Scores) with the
Embeddings File. When sequence filtering is enabled, it also intersects an explicit sequence FASTA file and restricts the
alignment to sequences common to all three inputs.

Input:
- Network File HDF5: Used to construct the evolutionary guide tree utilizing existing alignment scores (`INPUT_NETWORK`).
- Embeddings HDF5: Supplies the dense tensor representations of each sequence used during the active alignment phase (`INPUT_EMBED`).
- Sequence FASTA: Optional filter and sequence source when `USE_SEQUENCE_FILTER` is enabled (`INPUT_FASTA`). When filtering is
  disabled, sequences stored in the embedding manifest are aligned and padded with gaps.

Output:
- A completed Multiple Sequence Alignment FASTA file padded with '-' gap characters (`OUTPUT_FASTA`).

Settings:
- TARGET_SET: A prefix for the output MSA file name.
- PARENT_SET: The prefix shared by the broader input files to pull from.
- MODEL_NAME: The model used for the embeddings.
- ALIGNMENT_SCORE: Whether to weight the guide tree based on "global" or "local" connectivity scores from the network.
- NUM_WORKERS: CPU threads for parallel bootstrap generation of the consensus tree.
- NUM_TREES: How many bootstrap replicate iterations to average for the consensus guide tree (higher = more stable topology).
- NOISE_SCALE: Standard deviation of structural noise applied during bootstrap resampling.
- INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS: Whether missing pairs receive replicate-averaged cophenetic distances in the final matrix. Imputed pairs always participate in replicate trees.
- GAP_OPEN: Penalty scoring for opening gaps in the sequence.
- DEVICE_SELECTION: Auto-benchmarked or manually selected score-matrix device.

Algorithm:
1. Loads IDs from all three inputs and computes the mathematical intersection set.
2. Constructs a dense square distance matrix utilizing ONLY the pairwise connectivity scores explicitly found in the input network.
3. Builds an ensemble of randomized bootstrap neighbor-joining trees from the distance matrix (simulated via structural noise addition).
4. Computes the geometric average (consensus) graph of all random bootstraps to form the master guide tree.
5. In ascending order of linkage closeness, extracts sequence pairs/groups.
6. Forms profile columns as cluster-size-weighted averages of unit residue embeddings, treating gaps as zero.
7. Aligns profiles using Needleman-Wunsch dynamic programming, reciprocal cosine similarity, and column-norm confidence so sparse or internally inconsistent columns contribute less evidence.
8. Distributes the calculated optimal gap padding into all underlying string literal FASTA sequences.
9. Saves the final alignment block.
"""
# %% --- Imports ---
import os
import time

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import shutil  
import gc      
import h5py
import numpy as np
import torch
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import multiprocessing as mp
from functools import partial
from numba import jit, prange
from tqdm import tqdm
import sys
from utilities import Hardware_Acceleration as Hardware_Utils
from utilities import Numba_Threads
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from Cache_Manifest import validate_network_schema
from utilities.Sequence_Utils import load_sanitized_fasta
from utilities.HDF5_Storage import read_embedding_manifest


# ==========================================
# CONFIGURATION
# ==========================================

# Inputs - Now using .h5
INPUT_FASTA   = ""
INPUT_EMBED   = None
INPUT_NETWORK = None
USE_SEQUENCE_FILTER = False

# Metric for Guide Tree: "local" or "global"
ALIGNMENT_SCORE = "global"
NORMALIZATION_MODE = "alignment_length" # (alignment_length, shorter_sequence, longer_sequence, average_sequence)
TREE_METHOD = "UPGMA (Fast)" # (UPGMA (Fast), Neighbor-joining (Slow))

# Consensus Parameters
BOOTSTRAP_TREE = True
NUM_TREES = 100             
NOISE_SCALE = 0.02          
RANDOM_SEED = 42
INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS = False

# --- DIRECTORY DEFAULTS ---
from tools.tool_helpers.Tool_Pipeline import (
    inherited_settings_path,
    load_tool_settings,
    project_directory_defaults,
    select_settings_path,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]
NETWORK_DIR = _DEFAULT_DIRECTORIES["NETWORK_DIR"]
MSA_DIR = _DEFAULT_DIRECTORIES["MSA_DIR"]
# None keeps the replicate-tree memory-map cache in the run's MSA_DIR, which
# is only known once settings are applied; a settings value overrides it.
SAFE_TEMP_DIR = None

# Alignment Settings
GAP_OPEN = -0.5
GAP_EXTEND = 0.0           
WORKERS = 1   
DEVICE_SELECTION = "auto"
SHOW_REGRESSION_PLOT = False
POOLING_METHOD = "max"    # ("mean", "max") - method to pool residue embeddings into sequence vectors
LENGTH_RATIO_POWER = 2.0  # (float) - exponent to scale the sequence length ratio penalty

# --- JSON Settings Override ---
import json
import ast
import os

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "tools_settings.json")

if __name__ != "__main__" and os.path.exists(SETTINGS_FILE):
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

# Resolve directories after config overrides

# --- INFERRED PATHS ---
# Resolved at runtime so an inactive FASTA remains optional and the validated
# embedding manifest can supply the authoritative model name.
import re


class MSAConfigurationError(ValueError):
    """Raised when the MSA utility cannot resolve its configured inputs or settings."""


def _embedding_sequence_set(input_embed):
    """Infer the sequence-set label from a configured embedding filename."""
    filename = os.path.basename(input_embed)
    match = re.match(r"^(.*)_\[[^\]]+\]_embeddings\.h5$", filename, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    stem = os.path.splitext(filename)[0]
    if stem.lower().endswith("_embeddings"):
        stem = stem[:-len("_embeddings")]
    return stem


ALIGNMENT_MODES = ("global", "local")
NORMALIZATION_MODES = ("alignment_length", "shorter_sequence", "longer_sequence", "average_sequence")


def validate_score_normalization(alignment_score, normalization_mode):
    """Reject unknown modes, and normalizing local scores by their own
    alignment length.

    The Tools window, the Viewer and layout generation forbid this pairing too:
    dividing a local score by its own path length rewards short local matches
    however little of either sequence they cover.
    """
    if alignment_score not in ALIGNMENT_MODES:
        raise MSAConfigurationError(
            f"Unknown ALIGNMENT_SCORE {alignment_score!r}; choose global or local."
        )
    if normalization_mode not in NORMALIZATION_MODES:
        raise MSAConfigurationError(
            f"Unknown NORMALIZATION_MODE {normalization_mode!r}; choose "
            "alignment_length, shorter_sequence, longer_sequence, or average_sequence."
        )
    if alignment_score == "local" and normalization_mode == "alignment_length":
        raise MSAConfigurationError(
            "NORMALIZATION_MODE alignment_length is unavailable for local alignment "
            "scores; choose shorter_sequence, longer_sequence, or average_sequence."
        )


def resolve_msa_configuration(
    fasta_dir,
    embed_dir,
    network_dir,
    input_fasta,
    input_embed,
    input_network,
    use_sequence_filter,
):
    """Resolve configured input paths without joining an inactive FASTA value."""
    if not isinstance(input_embed, str) or input_embed == "":
        raise MSAConfigurationError("INPUT_EMBED must select an embedding HDF5 file.")
    if not isinstance(input_network, str) or input_network == "":
        raise MSAConfigurationError("INPUT_NETWORK must select a network HDF5 file.")
    if not isinstance(input_fasta, str):
        raise MSAConfigurationError("INPUT_FASTA must be a filename or an empty string.")
    if use_sequence_filter and input_fasta == "":
        raise MSAConfigurationError(
            "INPUT_FASTA must select a FASTA file when USE_SEQUENCE_FILTER is enabled."
        )

    full_input_fasta = (
        os.path.join(fasta_dir, input_fasta) if input_fasta != "" else ""
    )
    sequence_set = (
        os.path.splitext(os.path.basename(input_fasta))[0]
        if use_sequence_filter
        else _embedding_sequence_set(input_embed)
    )

    return {
        "full_input_fasta": full_input_fasta,
        "full_input_embed": os.path.join(embed_dir, input_embed),
        "full_input_network": os.path.join(network_dir, input_network),
        "sequence_set": sequence_set,
    }


def build_msa_output_path(msa_dir, sequence_set, model_name):
    """Build the final MSA filename from resolved, validated metadata."""
    return os.path.join(msa_dir, f"{sequence_set}_[{model_name}]_alignment.fasta")


def build_memmap_cache_dir(temp_dir, msa_dir, sequence_set, model_name):
    """Build the replicate-tree memory-map folder for one run.

    The cache sits beside the alignment in ``msa_dir`` unless a temporary
    directory was set explicitly.
    """
    root = temp_dir if temp_dir is not None and str(temp_dir).strip() else msa_dir
    return os.path.join(root, f"{sequence_set}_[{model_name}]_Memmap_Cache")


FULL_INPUT_FASTA = ""
FULL_INPUT_EMBED = ""
FULL_INPUT_NETWORK = ""
OUTPUT_FASTA = ""
_seq_set = ""
_model_name = ""

# ==========================================
# CORE CLASSES
# ==========================================
def _normalize_residue_embeddings(embedding):
    """Return unit residue vectors while leaving zero vectors unchanged."""
    embedding = np.asarray(embedding, dtype=np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = np.zeros_like(embedding, dtype=np.float32)
    np.divide(embedding, norms, out=normalized, where=norms > 0.0)
    # Leaves are transient, so retain float32 until their first merge. An
    # immediate float16 cast can make a unit vector's norm slightly less than
    # one and incorrectly attenuate an otherwise full-confidence leaf score.
    return normalized


# Alignments hold one ASCII code per residue; this is the gap.
GAP_CODE = ord("-")


class NonAsciiSequenceError(ValueError):
    """Raised when a sequence cannot be stored as one byte per residue."""


def encode_leaf_alignments(seq_dict, valid_headers):
    """Return each sequence as a one-sequence alignment of shape (length, 1).

    Sanitized FASTA files and embedding manifests hold only ASCII residue
    codes. Checking before the guide tree is built rejects anything else in
    seconds instead of after hours of tree building.
    """
    rejected = [header for header in valid_headers if not seq_dict[header].isascii()]
    if rejected:
        shown = ", ".join(rejected[:5])
        more = f" and {len(rejected) - 5} more" if len(rejected) > 5 else ""
        raise NonAsciiSequenceError(
            f"{len(rejected)} sequence(s) contain non-ASCII characters, which an "
            f"alignment cannot store: {shown}{more}."
        )
    return [
        np.frombuffer(seq_dict[header].encode("ascii"), dtype=np.uint8).reshape(-1, 1)
        for header in valid_headers
    ]


def fit_leaf_to_embedding(aligned, length):
    """Trim or gap-pad a leaf alignment to its embedding's row count."""
    width = aligned.shape[0]
    if width > length:
        return aligned[:length]
    if width < length:
        padding = np.full((length - width, aligned.shape[1]), GAP_CODE, dtype=np.uint8)
        return np.concatenate((aligned, padding))
    return aligned


class MSACluster:
    def __init__(self, idx, aligned, ids, embedding=None):
        self.idx = idx
        # One alignment column per row: shape (width, len(ids)), uint8 ASCII
        # codes. A merge then places each column with one contiguous copy.
        # Building strings one character at a time took minutes per merge
        # once clusters held tens of thousands of sequences.
        self.aligned = aligned
        self.ids = ids
        # For merged nodes, each row is the average of unit residue vectors
        # across every sequence in the cluster. Gaps contribute zero, so the
        # row norm retains both column occupancy and residue agreement.
        self.embedding = embedding
        self.is_leaf = embedding is None

    def get_embedding(self, embeddings_group, valid_headers):
        """Return profile vectors, lazily initializing leaf residue vectors.

        Leaves are read through the builder's open embeddings group instead of
        reopening the file once per sequence.
        """
        if self.embedding is not None:
            return self.embedding

        header = valid_headers[self.ids[0]]
        safe_h = header.replace("/", "_").replace("\\", "_")
        # Leaves enter the profile as unit residue vectors. Subsequent
        # weighted averages then have norms in [0, 1], where a smaller
        # norm represents gaps and/or disagreement within the column.
        return _normalize_residue_embeddings(embeddings_group[safe_h][:])

# ==========================================
# HELPER: FASTA LOADER & WORKER
# ==========================================
def compute_single_tree_worker(seed, num_seqs, baseline_dist_path, max_dist, noise_scale, tree_method, kernel_threads=None):
    """Build one replicate tree from the complete shared distance baseline.

    ``kernel_threads`` is this worker's share of the Numba thread budget, so
    concurrent neighbor-joining workers do not oversubscribe the CPUs.
    """
    condensed_size = int(num_seqs * (num_seqs - 1) / 2)
    baseline_dist = np.memmap(
        baseline_dist_path,
        dtype=np.float32,
        mode="r",
        shape=(condensed_size,),
    )

    # Allocate only the replicate matrix in worker RAM. Generate additive
    # noise in chunks so a dense imputed baseline does not require another
    # full-size temporary noise array.
    # SciPy's linkage converts its input to float64 and then works on its own
    # copy, so a float32 replicate would only sit beside that conversion. For
    # UPGMA the replicate is therefore stored as float64 from the start. Noise
    # and clipping stay in float32 chunks, and widening float32 to float64 is
    # exact, so the tree is unchanged. Neighbor-joining keeps float32, which
    # halves the square matrix it builds.
    neighbor_joining = tree_method == "Neighbor-joining (Slow)"
    D_perturbed_cond = np.empty(
        condensed_size,
        dtype=np.float32 if neighbor_joining else np.float64,
    )
    rng = np.random.default_rng(seed)
    sigma = float(noise_scale) * float(max_dist)
    noise_chunk_size = 1_000_000

    for start in range(0, condensed_size, noise_chunk_size):
        end = min(start + noise_chunk_size, condensed_size)
        if sigma == 0.0:
            chunk = np.array(baseline_dist[start:end])
        else:
            noise = rng.normal(0.0, sigma, size=end - start).astype(np.float32)
            chunk = baseline_dist[start:end] + noise
        np.clip(chunk, 0.0, max_dist, out=chunk)
        D_perturbed_cond[start:end] = chunk

    # Run Linkage or Neighbor-joining
    if neighbor_joining:
        Z = neighbor_joining_condensed(
            D_perturbed_cond, num_seqs, threads=kernel_threads
        )
    else:
        Z = sch.linkage(D_perturbed_cond, method='average')
    
    # Windows requires explicitly closing the memmap before cleanup.
    del baseline_dist
    
    return Z


def generate_bootstrap_seeds(num_trees):
    """Generate the reproducible per-replicate seeds used by bootstrap workers."""
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.integers(0, int(1e9), size=num_trees)


class SequenceEmbeddingMismatchError(ValueError):
    """Raised when FASTA sequence strings and embedding database sequences do not match."""


def validate_sequence_embedding_match(seq_dict, valid_headers, emb_seq_dict, embeddings_group):
    """
    Validate that for every header in the intersection:
    1. The sanitized sequence in the input FASTA matches the sequence stored in the embedding file.
    2. The FASTA sequence length matches the row count of the residue embedding dataset.
    """
    mismatches = []

    for header in valid_headers:
        safe_header = header.replace("/", "_").replace("\\", "_")
        fasta_seq = seq_dict[header]
        emb_seq = emb_seq_dict.get(header)
        embedding_length = embeddings_group[safe_header].shape[0]

        if emb_seq is None:
            mismatches.append(f"  - {header}: Missing sequence in embedding manifest")
        elif fasta_seq != emb_seq:
            mismatches.append(
                f"  - {header}: Sequence mismatch!\n"
                f"      Input FASTA ({len(fasta_seq)} aa): {fasta_seq[:30]}...\n"
                f"      Embedding   ({len(emb_seq)} aa): {emb_seq[:30]}..."
            )
        elif len(fasta_seq) != embedding_length:
            mismatches.append(
                f"  - {header}: Length mismatch between FASTA ({len(fasta_seq)}) "
                f"and embedding tensor ({embedding_length})"
            )

    if mismatches:
        details = "\n".join(mismatches)
        raise SequenceEmbeddingMismatchError(
            "Intersected FASTA sequences do not match embedding database records for "
            f"{len(mismatches)} sequence(s):\n"
            f"{details}\n"
            "Ensure the input FASTA sequences match the records used to generate the embedding file."
        )


# ==========================================
# ALIGNMENT KERNELS
# ==========================================
def compute_score_matrix_torch(emb_i, emb_j, device):
    t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float32)
    t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float32)

    # Profile-vector magnitude is meaningful: it decreases when a column has
    # gaps or contains disagreeing residue directions. Preserve that signal
    # before normalizing directions for the reciprocal similarity score.
    confidence_i = torch.linalg.vector_norm(t_i, dim=-1, keepdim=True).clamp(max=1.0)
    confidence_j = torch.linalg.vector_norm(t_j, dim=-1, keepdim=True).clamp(max=1.0)
    t_i_norm = torch.nn.functional.normalize(t_i, p=2, dim=-1)
    t_j_norm = torch.nn.functional.normalize(t_j, p=2, dim=-1)
    cos_sim = torch.mm(t_i_norm, t_j_norm.T).clamp(-1.0, 1.0)
    dist_mat = 1.0 - cos_sim
    sim_mat = torch.exp(-dist_mat)
    
    epsilon = 1e-8
    row_mean = sim_mat.mean(dim=1, keepdim=True)
    row_std = sim_mat.std(dim=1, keepdim=True, correction=0)
    col_mean = sim_mat.mean(dim=0, keepdim=True)
    col_std = sim_mat.std(dim=0, keepdim=True, correction=0)
    
    z_r = (sim_mat - row_mean) / (row_std + epsilon)
    z_c = (sim_mat - col_mean) / (col_std + epsilon)
    profile_confidence = confidence_i * confidence_j.T
    final_score = ((z_r + z_c) / 2.0) * profile_confidence
    
    return final_score.to(dtype=torch.float32, device="cpu").numpy()


def representative_leaf_merge_pairs(
    linkage_matrix,
    valid_headers,
    embeddings_group,
):
    """Select real leaf-to-leaf merges near fixed score-matrix cost quantiles."""
    num_sequences = len(valid_headers)
    candidates = []
    for row_index, link in enumerate(linkage_matrix):
        left = int(link[0])
        right = int(link[1])
        if left >= num_sequences or right >= num_sequences:
            continue

        left_header = valid_headers[left]
        right_header = valid_headers[right]
        left_key = left_header.replace("/", "_").replace("\\", "_")
        right_key = right_header.replace("/", "_").replace("\\", "_")
        left_length = int(embeddings_group[left_key].shape[0])
        right_length = int(embeddings_group[right_key].shape[0])
        candidates.append(
            (
                left_length * right_length,
                row_index,
                left,
                right,
                left_length,
                right_length,
            )
        )

    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
    selected = []
    selected_pairs = set()
    for fraction in (0.25, 0.50, 0.90):
        index = round((len(ordered) - 1) * fraction)
        _, _, left, right, left_length, right_length = ordered[index]
        pair = (left, right)
        if pair in selected_pairs:
            continue
        selected_pairs.add(pair)
        selected.append((left, right, left_length, right_length))
    return selected


def _load_benchmark_embedding_pairs(samples, valid_headers, embeddings_group):
    """Load and normalize benchmark inputs before any device timing begins."""
    loaded = []
    for left, right, left_length, right_length in samples:
        left_key = valid_headers[left].replace("/", "_").replace("\\", "_")
        right_key = valid_headers[right].replace("/", "_").replace("\\", "_")
        left_embedding = _normalize_residue_embeddings(
            embeddings_group[left_key][:]
        )
        right_embedding = _normalize_residue_embeddings(
            embeddings_group[right_key][:]
        )
        loaded.append(
            (
                left,
                right,
                left_length,
                right_length,
                left_embedding,
                right_embedding,
            )
        )
    return loaded


def benchmark_msa_devices(
    linkage_matrix,
    valid_headers,
    embeddings_group,
    selection,
):
    """Benchmark one sequential score-matrix lane and return ranked devices."""
    candidates = Hardware_Utils.get_available_devices()
    manual = Hardware_Utils.resolve_device_selection(selection, candidates)
    if manual is not None:
        print(f"[Hardware] Using manually selected {manual.display_name}.")
        return [Hardware_Utils.BenchmarkResult(manual, 0.0)]
    if len(candidates) == 1:
        only_candidate = candidates[0]
        print(f"[Hardware] Only {only_candidate.display_name} is available.")
        return [Hardware_Utils.BenchmarkResult(only_candidate, 0.0)]

    sample_metadata = representative_leaf_merge_pairs(
        linkage_matrix,
        valid_headers,
        embeddings_group,
    )
    if not sample_metadata:
        cpu_candidate = next(
            (candidate for candidate in candidates if candidate.is_cpu),
            candidates[0],
        )
        print(
            "[Hardware] No progressive profile merge requires benchmarking; "
            f"using {cpu_candidate.display_name}."
        )
        return [Hardware_Utils.BenchmarkResult(cpu_candidate, 0.0)]
    samples = _load_benchmark_embedding_pairs(
        sample_metadata,
        valid_headers,
        embeddings_group,
    )
    sample_shapes = [
        f"{left_length}x{right_length}"
        for _, _, left_length, right_length, _, _ in samples
    ]
    print(
        "[Hardware] Benchmarking MSA profile score matrices on "
        f"{len(candidates)} available device(s) using {sample_shapes}."
    )
    print("Device/backend                 Samples   Median time (s)   Status")

    results = []
    for candidate in candidates:
        elapsed_runs = []
        error = None
        try:
            first_left = samples[0][4]
            first_right = samples[0][5]
            compute_score_matrix_torch(
                first_left[:64],
                first_right[:64],
                candidate.device,
            )
            Hardware_Utils.synchronize_device(candidate)

            for _ in range(3):
                elapsed = 0.0
                for (
                    _left,
                    _right,
                    left_length,
                    right_length,
                    left_embedding,
                    right_embedding,
                ) in samples:
                    Hardware_Utils.synchronize_device(candidate)
                    started = time.perf_counter()
                    score_matrix = compute_score_matrix_torch(
                        left_embedding,
                        right_embedding,
                        candidate.device,
                    )
                    Hardware_Utils.synchronize_device(candidate)
                    elapsed += time.perf_counter() - started
                    if score_matrix.shape != (left_length, right_length):
                        raise ValueError(
                            "score matrix shape mismatch: expected "
                            f"{(left_length, right_length)}, got "
                            f"{score_matrix.shape}"
                        )
                    if not np.isfinite(score_matrix).all():
                        raise ValueError("score matrix contains non-finite values")
                elapsed_runs.append(elapsed)
            median_elapsed = float(np.median(elapsed_runs))
            result = Hardware_Utils.BenchmarkResult(candidate, median_elapsed)
        except Exception as failure:
            median_elapsed = None
            error = f"{type(failure).__name__}: {failure}"
            result = Hardware_Utils.BenchmarkResult(
                candidate,
                None,
                error=error,
            )
        finally:
            Hardware_Utils.release_device_cache(candidate)

        elapsed_text = (
            f"{median_elapsed:.4f}" if median_elapsed is not None else "--"
        )
        print(
            f"{candidate.display_name[:30]:30}  {len(samples):>7}   "
            f"{elapsed_text:>15}   {error or 'ok'}"
        )
        results.append(result)

    ranked = Hardware_Utils.rank_benchmark_results(
        results,
        higher_is_better=False,
    )
    if not ranked:
        failures = "; ".join(result.error or "unknown" for result in results)
        raise RuntimeError(f"No device completed the MSA benchmark: {failures}")

    selected = ranked[0]
    fastest = min(
        (result for result in results if result.succeeded),
        key=lambda result: float(result.value),
    )
    tie_applied = selected.candidate.spec != fastest.candidate.spec
    print(
        f"[Hardware] Selected {selected.candidate.display_name} for MSA score "
        "matrices; 3% tie preference "
        f"{'applied' if tie_applied else 'not applied'}."
    )
    return ranked


def compute_score_matrix_with_fallback(
    emb_i,
    emb_j,
    ranked_devices,
    selection,
):
    """Compute one score matrix, permanently dropping failed Auto devices."""
    automatic = (
        Hardware_Utils.normalize_device_selection(selection)
        == Hardware_Utils.AUTO_DEVICE
    )
    failures = []
    while ranked_devices:
        candidate = ranked_devices[0].candidate
        try:
            return compute_score_matrix_torch(emb_i, emb_j, candidate.device)
        except (RuntimeError, MemoryError, NotImplementedError) as error:
            Hardware_Utils.release_device_cache(candidate)
            if not automatic:
                raise RuntimeError(
                    f"MSA score-matrix calculation failed on manually selected "
                    f"device '{candidate.spec}': {error}"
                ) from error
            failures.append(f"{candidate.spec}: {error}")
            ranked_devices.pop(0)
            if ranked_devices:
                print(
                    f"[Hardware] MSA score calculation failed on "
                    f"{candidate.display_name}: {error}. Retrying on "
                    f"{ranked_devices[0].candidate.display_name}."
                )

    raise RuntimeError(
        "MSA score-matrix calculation failed on every ranked device: "
        + "; ".join(failures)
    )

def map_network_indices(indices, net_old_to_new):
    """Map network sequence indices to intersection indices as int32.

    ``net_old_to_new`` holds -1 for network sequences outside the
    intersection, and -1 in the result marks an endpoint whose edge is
    dropped. Indices outside the header table drop too, rather than raising
    (too large) or wrapping around (negative).
    """
    if indices.size == 0 or (
        indices.min() >= 0 and indices.max() < len(net_old_to_new)
    ):
        return net_old_to_new[indices]
    in_range = (indices >= 0) & (indices < len(net_old_to_new))
    mapped = np.full(indices.shape, -1, dtype=np.int32)
    mapped[in_range] = net_old_to_new[indices[in_range]]
    return mapped

@jit(nopython=True, fastmath=True)
def populate_condensed_matrix(D_condensed, num_seqs, edge_i, edge_j, edge_dists):
    """Blazing fast population of the 1D condensed array from sparse data."""
    for k in range(len(edge_i)):
        i = edge_i[k]
        j = edge_j[k]
        if i > j: 
            temp = i
            i = j
            j = temp
        idx = int(num_seqs*i - i*(i+1)/2 + j - i - 1)
        D_condensed[idx] = edge_dists[k]

@jit(nopython=True, fastmath=True)
def compute_sparse_cophenetic(Z, num_seqs, edge_i, edge_j):
    """Traces the linkage tree to find cophenetic distances ONLY for specific edges."""
    num_edges = len(edge_i)
    coph_dists = np.zeros(num_edges, dtype=np.float32)
    
    total_nodes = 2 * num_seqs - 1
    parent = np.arange(total_nodes, dtype=np.int32)
    height = np.zeros(total_nodes, dtype=np.float32)
    
    # Build tree hierarchy from Z matrix
    for i in range(num_seqs - 1):
        idx = num_seqs + i
        child1 = int(Z[i, 0])
        child2 = int(Z[i, 1])
        parent[child1] = idx
        parent[child2] = idx
        height[idx] = Z[i, 2]
        
    visited_marker = np.zeros(total_nodes, dtype=np.int32)
    
    # Trace Lowest Common Ancestor (LCA) for each edge
    for k in range(num_edges):
        u = edge_i[k]
        v = edge_j[k]
        marker = k + 1
        
        curr = u
        visited_marker[curr] = marker
        while parent[curr] != curr:
            curr = parent[curr]
            visited_marker[curr] = marker
            
        curr = v
        while visited_marker[curr] != marker:
            curr = parent[curr]
            
        coph_dists[k] = height[curr]
        
    return coph_dists


@jit(nopython=True, fastmath=True)
def accumulate_full_cophenetic(Z, num_seqs, coph_accumulator):
    """Add float32 cophenetic distances for every condensed pair in place.

    Adding into the caller's condensed accumulator avoids allocating a
    condensed-size array for every bootstrap tree.
    """
    if num_seqs <= 1:
        return

    total_nodes = 2 * num_seqs - 1
    head = np.full(total_nodes, -1, dtype=np.int32)
    tail = np.full(total_nodes, -1, dtype=np.int32)
    next_leaf = np.full(num_seqs, -1, dtype=np.int32)

    for leaf in range(num_seqs):
        head[leaf] = leaf
        tail[leaf] = leaf

    # At each linkage merge, every cross-child leaf pair meets for the first
    # time at this node. Across the complete tree, each pair is updated once.
    for step in range(num_seqs - 1):
        child_a = int(Z[step, 0])
        child_b = int(Z[step, 1])
        parent = num_seqs + step
        merge_height = np.float32(Z[step, 2])

        leaf_a = head[child_a]
        while leaf_a != -1:
            leaf_b = head[child_b]
            while leaf_b != -1:
                i = leaf_a
                j = leaf_b
                if i > j:
                    temp = i
                    i = j
                    j = temp
                condensed_idx = (
                    num_seqs * i - i * (i + 1) // 2 + j - i - 1
                )
                coph_accumulator[condensed_idx] += merge_height
                leaf_b = next_leaf[leaf_b]
            leaf_a = next_leaf[leaf_a]

        head[parent] = head[child_a]
        tail[parent] = tail[child_b]
        next_leaf[tail[child_a]] = head[child_b]


def use_full_cophenetic_consensus(is_sparse, include_imputed_pairs):
    """Complete networks are always full; sparse networks follow the setting."""
    return (not bool(is_sparse)) or bool(include_imputed_pairs)


def finalize_cophenetic_consensus(
    distance_matrix,
    num_seqs,
    edge_i,
    edge_j,
    cophenetic_accumulator,
    num_trees,
    full_consensus,
):
    """Finalize full or observed-edge-only cophenetic accumulation in place."""
    if num_trees <= 0:
        raise ValueError("NUM_TREES must be greater than zero.")
    if full_consensus:
        distance_matrix /= num_trees
    else:
        average_sparse = cophenetic_accumulator / num_trees
        populate_condensed_matrix(
            distance_matrix,
            num_seqs,
            edge_i,
            edge_j,
            average_sparse,
        )
    return distance_matrix

@jit(nopython=True, fastmath=True, parallel=True)
def neighbor_joining_row_minima(D, active_list, r_list, k, row_min, row_arg):
    """Store each active row's first minimum Q over the pairs j > i.

    Rows are independent, so threads never share a write. Row i has
    k - 1 - i pairs; pairing row t with row k - 1 - t gives every parallel
    iteration about k cells.
    """
    for t in prange((k + 1) // 2):
        # The prange index is unsigned; mixing it with signed k would
        # promote the row index to float64.
        first = np.int64(t)
        for side in range(2):
            i = first if side == 0 else k - 1 - first
            if side == 1 and i == first:
                break
            u = active_list[i]
            r_u = r_list[i]
            best_q = 1e15
            best_j = -1
            for j in range(i + 1, k):
                v = active_list[j]
                q = D[u, v] - (r_u + r_list[j])
                if q < best_q:
                    best_q = q
                    best_j = j
            row_min[i] = best_q
            row_arg[i] = best_j


@jit(nopython=True, fastmath=True)
def neighbor_joining_kernel(D, N):
    Z = np.zeros((N - 1, 4), dtype=np.float64)

    # Track active node indices contiguous in memory (sorted)
    active_list = np.arange(N, dtype=np.int32)
    k = N
    
    # Pre-calculate initial row sums for the first N nodes
    R = np.zeros(2 * N - 1, dtype=np.float64)
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += D[i, j]
        R[i] = s
        
    node_height = np.zeros(2 * N - 1, dtype=np.float64)
    num_leaves = np.ones(2 * N - 1, dtype=np.float64)
    
    # Pre-allocate r_list array for reuse
    r_list = np.zeros(2 * N - 1, dtype=np.float64)
    row_min = np.empty(N, dtype=np.float64)
    row_arg = np.empty(N, dtype=np.int64)

    for step in range(N - 1):
        if k > 2:
            inv_k_minus_2 = 1.0 / (k - 2)
            min_Q = 1e15
            idx_u = -1
            idx_v = -1

            # Precompute normalized divergence values for active nodes
            for i in range(k):
                r_list[i] = R[active_list[i]] * inv_k_minus_2

            # Taking rows in order with a strict < keeps the serial scan's
            # choice: the first minimum in row-major order.
            neighbor_joining_row_minima(D, active_list, r_list, k, row_min, row_arg)
            for i in range(k):
                if row_min[i] < min_Q:
                    min_Q = row_min[i]
                    idx_u = i
                    idx_v = row_arg[i]
        else:
            idx_u = 0
            idx_v = 1
            
        u = active_list[idx_u]
        v = active_list[idx_v]
        
        # New parent node
        w = N + step
        
        dist_uv = D[u, v]
        child_max = max(node_height[u], node_height[v])
        node_height[w] = max(dist_uv, child_max)
        
        # Update distances from w to other active nodes and calculate R[w]
        sum_d_wm = 0.0
        for p in range(k):
            if p != idx_u and p != idx_v:
                m = active_list[p]
                d_wm = 0.5 * (D[u, m] + D[v, m] - dist_uv)
                D[w, m] = d_wm
                D[m, w] = d_wm
                sum_d_wm += d_wm
                # Update R[m]
                R[m] = R[m] - D[u, m] - D[v, m] + d_wm
                
        R[w] = sum_d_wm
        
        # Record merge in Z
        c1 = min(u, v)
        c2 = max(u, v)
        Z[step, 0] = float(c1)
        Z[step, 1] = float(c2)
        Z[step, 2] = node_height[w]
        Z[step, 3] = num_leaves[u] + num_leaves[v]
        
        num_leaves[w] = num_leaves[u] + num_leaves[v]
        
        # Update active list keeping it sorted:
        write_idx = 0
        for p in range(k):
            if p != idx_u and p != idx_v:
                active_list[write_idx] = active_list[p]
                write_idx += 1
        active_list[write_idx] = w
        k -= 1
        
    return Z

def neighbor_joining_condensed(D_condensed, num_seqs, threads=None):
    D_square = squareform(D_condensed)
    D_allocated = np.zeros((2 * num_seqs - 1, 2 * num_seqs - 1), dtype=np.float64)
    D_allocated[:num_seqs, :num_seqs] = D_square
    if threads is None:
        threads = Numba_Threads.default_thread_count()
    with Numba_Threads.limited_threads(threads):
        return neighbor_joining_kernel(D_allocated, num_seqs)

@jit(nopython=True, fastmath=True)
def calculate_normalized_scores_kernel(edge_i, edge_j, raw_scores, align_lens, seq_lens, is_evalue, mode_int):
    """C-speed kernel for processing hundreds of millions of edge normalizations."""
    num_edges = len(edge_i)
    norm_scores = np.zeros(num_edges, dtype=np.float32)
    max_norm_score = 0.0
    
    for k in range(num_edges):
        if is_evalue:
            norm_score = raw_scores[k]
        else:
            if mode_int == 0:  # alignment_length
                denom = max(align_lens[k], 1.0)
            else:
                len_src = seq_lens[edge_i[k]]
                len_dst = seq_lens[edge_j[k]]
                
                if mode_int == 1:  # shorter_sequence
                    denom = min(len_src, len_dst)
                elif mode_int == 2:  # longer_sequence
                    denom = max(len_src, len_dst)
                elif mode_int == 3:  # average_sequence
                    denom = (len_src + len_dst) / 2.0
                else:
                    denom = 1.0 
                    
            denom = max(denom, 1e-6)
            norm_score = raw_scores[k] / denom
            
        norm_scores[k] = norm_score
        if norm_score > max_norm_score:
            max_norm_score = norm_score
            
    return norm_scores, max_norm_score

@jit(nopython=True, fastmath=True)
def run_global_traceback(score_matrix, gap_open, gap_extend):
    N, M = score_matrix.shape
    NEG_INF = -1e9
    
    # 3-State Affine DP Matrices (Match, Delete, Insert)
    dp_M = np.full((N + 1, M + 1), NEG_INF, dtype=np.float32)
    dp_D = np.full((N + 1, M + 1), NEG_INF, dtype=np.float32)
    dp_I = np.full((N + 1, M + 1), NEG_INF, dtype=np.float32)
    
    # Pointers to track which matrix the current cell came from
    ptr_M = np.zeros((N + 1, M + 1), dtype=np.int8)
    ptr_D = np.zeros((N + 1, M + 1), dtype=np.int8)
    ptr_I = np.zeros((N + 1, M + 1), dtype=np.int8)
    
    dp_M[0, 0] = 0.0
    
    # Initialize edges
    dp_D[1, 0] = gap_open
    ptr_D[1, 0] = 1 # 1 = came from M(0,0)
    for i in range(2, N + 1):
        dp_D[i, 0] = dp_D[i-1, 0] + gap_extend
        ptr_D[i, 0] = 2 # 2 = came from D
        
    dp_I[0, 1] = gap_open
    ptr_I[0, 1] = 1 # 1 = came from M(0,0)
    for j in range(2, M + 1):
        dp_I[0, j] = dp_I[0, j-1] + gap_extend
        ptr_I[0, j] = 3 # 3 = came from I

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # 1. Update dp_D (Delete / Gap in sequence 2 / move down)
            d_from_m = dp_M[i-1, j] + gap_open
            d_from_d = dp_D[i-1, j] + gap_extend
            if d_from_m >= d_from_d:
                dp_D[i, j] = d_from_m
                ptr_D[i, j] = 1 
            else:
                dp_D[i, j] = d_from_d
                ptr_D[i, j] = 2 
                
            # 2. Update dp_I (Insert / Gap in sequence 1 / move right)
            i_from_m = dp_M[i, j-1] + gap_open
            i_from_i = dp_I[i, j-1] + gap_extend
            if i_from_m >= i_from_i:
                dp_I[i, j] = i_from_m
                ptr_I[i, j] = 1 
            else:
                dp_I[i, j] = i_from_i
                ptr_I[i, j] = 3 
                
            # 3. Update dp_M (Match/Mismatch / diagonal)
            score = score_matrix[i-1, j-1]
            m_from_m = dp_M[i-1, j-1] + score
            m_from_d = dp_D[i-1, j-1] + score
            m_from_i = dp_I[i-1, j-1] + score
            
            best_m = m_from_m
            best_ptr = 1
            if m_from_d > best_m:
                best_m = m_from_d
                best_ptr = 2
            if m_from_i > best_m:
                best_m = m_from_i
                best_ptr = 3
                
            dp_M[i, j] = best_m
            ptr_M[i, j] = best_ptr

    # Traceback
    path_buffer = np.zeros(N + M, dtype=np.int8)
    k = 0
    i, j = N, M
    
    # Find the optimal final state to trace backward from
    best_final = dp_M[N, M]
    state = 1
    if dp_D[N, M] > best_final:
        best_final = dp_D[N, M]
        state = 2
    if dp_I[N, M] > best_final:
        best_final = dp_I[N, M]
        state = 3
        
    while i > 0 or j > 0:
        if state == 1:
            path_buffer[k] = 1
            k += 1
            next_state = ptr_M[i, j]
            i -= 1
            j -= 1
            state = next_state
        elif state == 2:
            path_buffer[k] = 2
            k += 1
            next_state = ptr_D[i, j]
            i -= 1
            state = next_state
        elif state == 3:
            path_buffer[k] = 3
            k += 1
            next_state = ptr_I[i, j]
            j -= 1
            state = next_state
        else:
            break
            
    return path_buffer[:k]

def merge_clusters(cluster_a, cluster_b, path, emb_a, emb_b):
    # The traceback lists moves from the last column to the first. Match (1)
    # and delete (2) take the next column of A, match and insert (3) of B.
    moves = path[::-1]
    take_a = (moves == 1) | (moves == 2)
    take_b = (moves == 1) | (moves == 3)
    used_a = int(take_a.sum())
    used_b = int(take_b.sum())
    if not (used_a == cluster_a.aligned.shape[0] == emb_a.shape[0]
            and used_b == cluster_b.aligned.shape[0] == emb_b.shape[0]):
        # Masked assignment would otherwise broadcast a one-column cluster
        # into the wrong number of columns without complaint.
        raise ValueError(
            "Alignment path does not place every column: cluster A has "
            f"{cluster_a.aligned.shape[0]} columns and {emb_a.shape[0]} profile "
            f"rows, the path takes {used_a}; cluster B has "
            f"{cluster_b.aligned.shape[0]} columns and {emb_b.shape[0]} profile "
            f"rows, the path takes {used_b}."
        )

    width = moves.shape[0]
    n_a = cluster_a.aligned.shape[1]
    aligned = np.full(
        (width, n_a + cluster_b.aligned.shape[1]), GAP_CODE, dtype=np.uint8
    )
    aligned[take_a, :n_a] = cluster_a.aligned
    aligned[take_b, n_a:] = cluster_b.aligned

    w_a = float(len(cluster_a.ids))
    w_b = float(len(cluster_b.ids))
    total_w = w_a + w_b
    # These are cluster-wide averages of unit residue vectors. The weighted
    # mean preserves occupancy and directional agreement. Each column is
    # computed as (a * w_a + b * w_b) / total_w in float32, in the same
    # order as the former column-by-column loop. Rows start at -0.0, the one
    # value that leaves every addend unchanged (+0.0 + -0.0 is +0.0), so
    # B-only columns keep their exact bits and the profile is bit-identical.
    merged = np.full((width, emb_a.shape[1]), -0.0, dtype=np.float32)
    merged[take_a] = emb_a.astype(np.float32) * w_a
    merged[take_b] += emb_b.astype(np.float32) * w_b
    merged /= total_w

    new_cluster = MSACluster(
        idx=-1,
        aligned=aligned,
        # Values remain bounded because they are averages of unit vectors.
        # Downcast to float16 to save RAM for the remaining iterations.
        embedding=merged.astype(np.float16),
        ids=cluster_a.ids + cluster_b.ids
    )
    new_cluster.is_leaf = False
    return new_cluster

# ==========================================
# MAIN EXECUTION
# ==========================================
def report_processing_times(
    total_processing_seconds,
    tree_building_seconds,
    cluster_merging_seconds,
):
    """Print a compact, human-readable timing summary for a completed MSA run."""
    def format_duration(seconds):
        seconds = max(0.0, float(seconds))
        hours, remainder = divmod(seconds, 3600.0)
        minutes, seconds = divmod(remainder, 60.0)
        if hours >= 1.0:
            return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        if minutes >= 1.0:
            return f"{int(minutes)}m {seconds:.2f}s"
        return f"{seconds:.2f}s"

    print("\n--- Processing Time Summary ---")
    print(f"Total processing time: {format_duration(total_processing_seconds)}")
    print(f"Tree building time: {format_duration(tree_building_seconds)}")
    print(f"Cluster merging time: {format_duration(cluster_merging_seconds)}")


def run_msa_builder():
    global FULL_INPUT_FASTA, FULL_INPUT_EMBED, FULL_INPUT_NETWORK
    global OUTPUT_FASTA, _seq_set, _model_name
    total_processing_started = time.perf_counter()

    # Linux defaults to fork, which is unsafe here: the bootstrap pool is
    # created after torch and the memmapped edge arrays are live. Windows and
    # macOS already default to spawn, so this only changes Linux.
    try: mp.set_start_method('spawn')
    except RuntimeError: pass

    try:
        validate_score_normalization(ALIGNMENT_SCORE, NORMALIZATION_MODE)
        resolved = resolve_msa_configuration(
            FASTA_DIR,
            EMBED_DIR,
            NETWORK_DIR,
            INPUT_FASTA,
            INPUT_EMBED,
            INPUT_NETWORK,
            USE_SEQUENCE_FILTER,
        )
    except MSAConfigurationError as error:
        sys.exit(f"❌ Configuration Error: {error}")

    FULL_INPUT_FASTA = resolved["full_input_fasta"]
    FULL_INPUT_EMBED = resolved["full_input_embed"]
    FULL_INPUT_NETWORK = resolved["full_input_network"]
    _seq_set = resolved["sequence_set"]

    # 1. LOAD & VALIDATE INPUTS
    print("--- Loading & Validating Inputs ---")
    
    # A. Open HDF5 files and run rigorous embedding file validation
    print("Opening HDF5 data...")
    try:
        f_emb = h5py.File(FULL_INPUT_EMBED, "r")
        f_net = h5py.File(FULL_INPUT_NETWORK, "r")
    except Exception as e:
        sys.exit(f"❌ Error opening HDF5 files: {e}")

    print("Validating embedding database manifest...")
    try:
        manifest = read_embedding_manifest(
            f_emb,
            require_complete=True,
            validate_embeddings=True,
        )
    except Exception as error:
        sys.exit(f"❌ Critical Error: Embedding file '{FULL_INPUT_EMBED}' validation failed:\n{error}")

    emb_headers = manifest.headers
    emb_seq_dict = manifest.sequence_by_header
    _model_name = manifest.model_name
    OUTPUT_FASTA = build_msa_output_path(MSA_DIR, _seq_set, _model_name)

    raw_net_headers = f_net['headers'][:]
    net_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_net_headers]

    arr_i = f_net['i'][:]
    arr_j = f_net['j'][:]
    
    network_metadata = validate_network_schema(f_net)
    if network_metadata.network_type == "blast":
        target_score = f_net['score'][:]
        target_len   = np.ones_like(target_score)
        is_evalue = True
    else:
        if ALIGNMENT_SCORE == "global":
            target_score = f_net['g_score'][:]
            target_len   = f_net['g_len'][:]
        else:
            target_score = f_net['l_score'][:]
            target_len   = f_net['l_len'][:]
        is_evalue = False

    # Validate dataset lengths match to prevent out-of-bounds IndexError
    if not (len(arr_i) == len(arr_j) == len(target_score) == len(target_len)):
        sys.exit(f"❌ Error: Network file {FULL_INPUT_NETWORK} is corrupted or incomplete.\n"
                 f"Dataset lengths: i={len(arr_i)}, j={len(arr_j)}, score={len(target_score)}, len={len(target_len)}.\n"
                 f"Please delete this network file and re-run the pipeline to re-generate it.")

    os.makedirs(os.path.dirname(OUTPUT_FASTA), exist_ok=True)
    
    set_net = set(net_headers)
    set_emb = set(emb_headers)

    use_filter = bool(USE_SEQUENCE_FILTER)
    if use_filter:
        try:
            clean_headers, clean_sequences, _ = load_sanitized_fasta(FULL_INPUT_FASTA)
        except Exception as e:
            sys.exit(f"❌ Error loading/sanitizing FASTA file: {e}")

        seq_dict = dict(zip(clean_headers, clean_sequences))
        fasta_headers = clean_headers
        set_fas = set(fasta_headers)
        
        common_set = set_net.intersection(set_emb).intersection(set_fas)
        if not common_set:
            sys.exit("❌ Error: No common sequences found between Network, Embeddings, and FASTA!")

        print(f"Intersection Found (3-Way Filtered): {len(common_set)} sequences.")
        print(f"  (Network: {len(set_net)}, Embed: {len(set_emb)}, FASTA: {len(set_fas)})")
    else:
        print("Use Sequence Filter is OFF. Using embedded sequences directly from HDF5 database...")
        common_set = set_net.intersection(set_emb)
        if not common_set:
            sys.exit("❌ Error: No common sequences found between Network and Embeddings database!")

        seq_dict = {h: emb_seq_dict[h] for h in common_set if h in emb_seq_dict}
        print(f"Intersection Found (2-Way Network ∩ Embeddings): {len(common_set)} sequences.")
        print(f"  (Network: {len(set_net)}, Embed: {len(set_emb)})")

    # BUILD VALIDATION LIST (Preserve Network Order)
    valid_headers = []
    for h in net_headers:
        if h in common_set:
            valid_headers.append(h)
    
    num_seqs = len(valid_headers)
    header_to_new_idx = {h: i for i, h in enumerate(valid_headers)}

    # Validate sequence string & length match if sequence filter is active
    if use_filter:
        try:
            validate_sequence_embedding_match(
                seq_dict, valid_headers, emb_seq_dict, f_emb["embeddings"]
            )
        except SequenceEmbeddingMismatchError as e:
            sys.exit(f"❌ Error: {e}")

    try:
        leaf_alignments = encode_leaf_alignments(seq_dict, valid_headers)
    except NonAsciiSequenceError as e:
        sys.exit(f"❌ Error: {e}")

    # 5. BUILD MAPPINGS & FILTER NETWORK EDGES
    print("--- Filtering Network Edges ---")
    
    # Map: Network_Index -> New_Index (-1 for sequences outside the intersection)
    net_old_to_new = np.full(len(net_headers), -1, dtype=np.int32)
    for i, h in enumerate(net_headers):
        new_index = header_to_new_idx.get(h)
        if new_index is not None:
            net_old_to_new[i] = new_index

    # Filter with whole-array operations. A Python tuple per edge costs about
    # 130 bytes, so a complete 44k-sequence network (~974M edges) would need
    # ~125 GB. Each network index array is freed as soon as its int32 mapping
    # exists. Edge order is kept, so duplicate pairs resolve as before.
    edge_i = map_network_indices(arr_i, net_old_to_new)
    del arr_i
    edge_j = map_network_indices(arr_j, net_old_to_new)
    del arr_j
    if edge_i.size and (edge_i.min() < 0 or edge_j.min() < 0):
        keep = (edge_i >= 0) & (edge_j >= 0)
        edge_i = edge_i[keep]
        edge_j = edge_j[keep]
        target_score = target_score[keep]
        target_len = target_len[keep]
        del keep
    raw_scores = target_score.astype(np.float32, copy=False)
    align_lens = target_len.astype(np.float32, copy=False)
    del target_score, target_len
    num_edges = len(edge_i)

    print(f"Retained {num_edges} edges valid for the intersection.")

    import scipy.sparse as sp

    # 7. PREPARE SPARSE DATA ARRAYS
    print(f"Preparing sparse distance metrics...")

    # 7a. Pre-compute Sequence Lengths into a C-compatible array
    num_seqs = len(valid_headers)
    seq_lens_array = np.zeros(num_seqs, dtype=np.int32)
    for idx, h in enumerate(valid_headers):
        seq_lens_array[idx] = len(seq_dict[h])

    # 7b. Map the string mode to an integer for the Numba kernel
    mode_map = {
        "alignment_length": 0,
        "shorter_sequence": 1,
        "longer_sequence": 2,
        "average_sequence": 3
    }
    
    if NORMALIZATION_MODE not in mode_map and not is_evalue:
        raise ValueError(f"❌ Critical Error: Unhandled NORMALIZATION_MODE '{NORMALIZATION_MODE}'. Cannot calculate distance.")
    
    mode_int = mode_map.get(NORMALIZATION_MODE, 0)

    # 7c. Execute C-Speed Kernel
    print("Executing Numba Math Kernel...")
    norm_scores, max_norm_score = calculate_normalized_scores_kernel(
        edge_i=edge_i,
        edge_j=edge_j,
        raw_scores=raw_scores,
        align_lens=align_lens,
        seq_lens=seq_lens_array,
        is_evalue=is_evalue,
        mode_int=mode_int
    )
    del raw_scores, align_lens

    MAX_DISTANCE = max_norm_score + 0.1

    is_sparse = num_edges < int(num_seqs * (num_seqs - 1) / 2)
    iso_reg = None
    cos_sim_mat = None

    # Invert scores to distances
    if is_sparse:
        # The regression below still reads the normalized scores.
        edge_dists = np.maximum(0.0, max_norm_score - norm_scores).astype(np.float32)
    else:
        # Nothing reads the normalized scores again, so invert them in place.
        # Deleting edge_dists before bootstrapping then frees them as well.
        # max_norm_score is a Python float, so NumPy still computes in float32.
        edge_dists = norm_scores
        del norm_scores
        np.subtract(max_norm_score, edge_dists, out=edge_dists)
        np.maximum(0.0, edge_dists, out=edge_dists)

    if is_sparse:
        print(f"Network is sparse ({num_edges} / {int(num_seqs * (num_seqs - 1) / 2)} edges). Activating hybrid cosine-alignment transformation...")
        
        # Load embeddings for valid_headers and calculate their pooled vectors
        print(f"Computing pooled embeddings ({POOLING_METHOD} pooling) for all sequences...")
        mean_embs = []
        for h in tqdm(valid_headers, desc="Computing Pooled Embeddings"):
            safe_h = h.replace("/", "_").replace("\\", "_")
            emb = f_emb["embeddings"][safe_h][:] # shape: (length, dim)
            if POOLING_METHOD == "max":
                pooled = np.max(emb, axis=0)
            else:
                pooled = np.mean(emb, axis=0)
            mean_embs.append(pooled)
        mean_embs = np.array(mean_embs, dtype=np.float32)
        
        print("Calculating all-vs-all length-adjusted similarities (length ratio * cosine similarity)...")
        norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        norm_embs = mean_embs / norms
        cos_sim_mat = np.dot(norm_embs, norm_embs.T)
        cos_sim_mat = np.clip(cos_sim_mat, -1.0, 1.0)
        
        # Apply sequence length ratio adjustment
        lens_col = seq_lens_array[:, np.newaxis]
        lens_row = seq_lens_array[np.newaxis, :]
        min_lens = np.minimum(lens_col, lens_row)
        max_lens = np.maximum(lens_col, lens_row)
        max_lens = np.maximum(max_lens, 1)
        length_ratio_mat = min_lens / max_lens
        
        if LENGTH_RATIO_POWER != 1.0:
            length_ratio_mat = length_ratio_mat ** LENGTH_RATIO_POWER
            
        cos_sim_mat = cos_sim_mat * length_ratio_mat
        
        # Extract overlapping pairs
        X_cos = cos_sim_mat[edge_i, edge_j]
        Y_align = norm_scores
        
        print("Fitting Isotonic Regression (Adjusted Similarity -> Alignment Score)...")
        # If we have a large number of edges, sample 100,000 edges to maintain sub-second speed
        if len(X_cos) > 100000:
            np.random.seed(42)
            sample_idx = np.random.choice(len(X_cos), size=100000, replace=False)
            X_fit = X_cos[sample_idx]
            Y_fit = Y_align[sample_idx]
        else:
            X_fit = X_cos
            Y_fit = Y_align
            
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(X_fit, Y_fit)
        
        # Evaluate fitness on all edges
        Y_pred = iso_reg.predict(X_cos)
        rho, _ = spearmanr(X_cos, Y_align)
        r2 = r2_score(Y_align, Y_pred)
        
        print(f"Isotonic Regression Fit Diagnostics:")
        print(f"  > Spearman Rank Correlation (rho): {rho:.4f}")
        print(f"  > Coefficient of Determination (R^2): {r2:.4f}")

        # ==========================================
        # TEMPORARY PLOTTING SECTION (EASY TO REMOVE)
        # ==========================================
        if SHOW_REGRESSION_PLOT:
            try:
                import matplotlib.pyplot as plt
                print("Displaying Isotonic Regression plot. Close the plot window to continue...")
                plt.figure(figsize=(10, 6))
                
                plot_sample_size = min(len(X_cos), 5000)
                np.random.seed(42)
                plot_idx = np.random.choice(len(X_cos), size=plot_sample_size, replace=False)
                
                plt.scatter(X_cos[plot_idx], Y_align[plot_idx], color='blue', alpha=0.3, label='Edges (Sample)', s=5)
                
                x_line = np.linspace(np.min(X_cos), np.max(X_cos), 1000)
                y_line = iso_reg.predict(x_line)
                plt.plot(x_line, y_line, color='red', linewidth=3, label='Isotonic Fit')
                
                plt.title(f"Isotonic Regression Fit (Spearman rho = {rho:.4f}, R^2 = {r2:.4f})")
                plt.xlabel("Length-Adjusted Embedding Cosine Similarity")
                plt.ylabel("Normalized Network Score")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.show()
            except Exception as plot_err:
                print(f"Could not open plot window: {plot_err}")
        # ==========================================
        # END OF TEMPORARY PLOTTING SECTION
        # ==========================================

    # 8. BUILD CONSENSUS TREE
    tree_building_started = time.perf_counter()
    condensed_size = int(num_seqs * (num_seqs - 1) / 2)
    
    if is_sparse and iso_reg is not None and cos_sim_mat is not None:
        print("Applying hybrid adjusted similarity-alignment imputation for final master tree...")
        dense_scores = iso_reg.predict(cos_sim_mat.ravel()).reshape(num_seqs, num_seqs)
        dense_dists = np.maximum(0.0, max_norm_score - dense_scores).astype(np.float32)
        np.fill_diagonal(dense_dists, 0.0)
        dense_dists = 0.5 * (dense_dists + dense_dists.T)
        D_final_cond = squareform(dense_dists)
    else:
        D_final_cond = np.full(condensed_size, MAX_DISTANCE, dtype=np.float32)

    # The regression supplies missing distances, while observed network edges
    # remain authoritative. This complete baseline is used by every replicate
    # so imputed relationships can influence each tree topology.
    populate_condensed_matrix(D_final_cond, num_seqs, edge_i, edge_j, edge_dists)
    np.clip(D_final_cond, 0.0, MAX_DISTANCE, out=D_final_cond)

    if BOOTSTRAP_TREE:
        full_consensus = use_full_cophenetic_consensus(
            is_sparse,
            INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS,
        )
        consensus_label = "full all-pairs" if full_consensus else "observed-edge partial"
        print(f"\nBuilding Consensus Tree from {NUM_TREES} bootstrap replicates using {WORKERS} cores...")
        print(f"Final cophenetic consensus mode: {consensus_label}.")
        
        # --- MEMORY MAPPING FIX FOR WINDOWS IPC LIMITS ---
        print("Writing the complete distance baseline to a temporary Memory-Mapped file for workers...")
        
        # 1. Define a strictly named, predictable folder in this run's MSA_DIR
        #    (or the explicitly set temporary directory)
        temp_dir = build_memmap_cache_dir(SAFE_TEMP_DIR, MSA_DIR, _seq_set, _model_name)
        
        # 2. Auto-Cleanup Failsafe: Wipe the folder if it was left behind by a previous crash
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print("Cleared residual cache from a previous interrupted run.")
            except Exception as e:
                print(f"Warning: Could not clear old temp directory. It might be locked by another process: {e}")
                
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Temporary memmap cache active at: {temp_dir}")
        
        baseline_dist_path = os.path.join(temp_dir, "baseline_dist.dat")
        mm_baseline = np.memmap(
            baseline_dist_path,
            dtype=np.float32,
            mode="w+",
            shape=D_final_cond.shape,
        )
        mm_baseline[:] = D_final_cond[:]
        mm_baseline.flush()

        if full_consensus:
            # Workers now own the immutable baseline through the memory map.
            # Reuse the main condensed array as the all-pairs accumulator.
            del edge_i
            del edge_j
            del edge_dists
            D_final_cond.fill(0.0)
            mm_i_main = None
            mm_j_main = None
            C_accum_sparse = None
        else:
            # Partial consensus updates only observed pairs. Keep their indices
            # memory-mapped so large sparse networks do not remain in RAM.
            edge_i_path = os.path.join(temp_dir, "edge_i.dat")
            edge_j_path = os.path.join(temp_dir, "edge_j.dat")
            mm_i = np.memmap(edge_i_path, dtype=np.int32, mode="w+", shape=edge_i.shape)
            mm_i[:] = edge_i[:]
            mm_i.flush()
            del mm_i

            mm_j = np.memmap(edge_j_path, dtype=np.int32, mode="w+", shape=edge_j.shape)
            mm_j[:] = edge_j[:]
            mm_j.flush()
            del mm_j

            del edge_i
            del edge_j
            del edge_dists
            mm_i_main = np.memmap(edge_i_path, dtype=np.int32, mode="r", shape=(num_edges,))
            mm_j_main = np.memmap(edge_j_path, dtype=np.int32, mode="r", shape=(num_edges,))
            C_accum_sparse = np.zeros(num_edges, dtype=np.float32)

        seeds = generate_bootstrap_seeds(NUM_TREES)
        # Neighbor-joining workers share the CPU budget; UPGMA ignores it.
        worker_threads = max(
            1, Numba_Threads.default_thread_count() // max(1, int(WORKERS))
        )
        if TREE_METHOD == "Neighbor-joining (Slow)":
            print(f"Neighbor-joining uses {worker_threads} threads per worker.")

        worker_func = partial(compute_single_tree_worker,
                              num_seqs=num_seqs,
                              baseline_dist_path=baseline_dist_path,
                              max_dist=MAX_DISTANCE,
                              noise_scale=NOISE_SCALE,
                              tree_method=TREE_METHOD,
                              kernel_threads=worker_threads)
        
        with mp.Pool(processes=WORKERS) as pool:
            iterator = pool.imap_unordered(worker_func, seeds)
            for Z in tqdm(iterator, total=NUM_TREES, desc="Bootstrapping Trees"):
                if full_consensus:
                    accumulate_full_cophenetic(Z, num_seqs, D_final_cond)
                else:
                    sparse_coph = compute_sparse_cophenetic(
                        Z,
                        num_seqs,
                        mm_i_main,
                        mm_j_main,
                    )
                    C_accum_sparse += sparse_coph

        print("Building final master tree...")
        finalize_cophenetic_consensus(
            D_final_cond,
            num_seqs,
            mm_i_main,
            mm_j_main,
            C_accum_sparse,
            NUM_TREES,
            full_consensus,
        )
        
        # --- CLEANUP ---
        if mm_i_main is not None:
            del mm_i_main
        if mm_j_main is not None:
            del mm_j_main
        del mm_baseline
        
        gc.collect() # Force Windows to release file handles
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Note: Could not clean up temporary directory {temp_dir}: {e}")
    else:
        print("\nBuilding Deterministic Tree (Bootstrapping bypassed)...")
        del edge_i
        del edge_j
        del edge_dists
        gc.collect()

    if TREE_METHOD == "Neighbor-joining (Slow)":
        linkage_matrix = neighbor_joining_condensed(D_final_cond, num_seqs)
    else:
        # SciPy converts to float64 and then copies its input. Converting
        # first lets the float32 matrix go before that copy is made; the
        # widening is exact, so the tree is unchanged.
        D_final_cond = D_final_cond.astype(np.float64)
        linkage_matrix = sch.linkage(D_final_cond, method='average')
    
    # --- CLEANUP ---
    del D_final_cond 
    gc.collect()
    tree_building_seconds = time.perf_counter() - tree_building_started

    ranked_devices = benchmark_msa_devices(
        linkage_matrix,
        valid_headers,
        f_emb["embeddings"],
        DEVICE_SELECTION,
    )

    # 9. INITIALIZE CLUSTERS (No embeddings loaded here!)
    print("Initializing clusters...")
    clusters = {}
    for i in range(num_seqs):
        # Initialize leaves purely with alignment/ID metadata
        c = MSACluster(idx=i, aligned=leaf_alignments[i], ids=[i], embedding=None)
        clusters[i] = c

    # 10. PROGRESSIVE ALIGNMENT
    print(f"Aligning {num_seqs} sequences...")
    cluster_merging_started = time.perf_counter()
    embeddings_group = f_emb["embeddings"]
    for iteration, link in enumerate(tqdm(linkage_matrix, desc="Merging Clusters")):
        idx_a = int(link[0])
        idx_b = int(link[1])

        cluster_a = clusters.pop(idx_a)
        cluster_b = clusters.pop(idx_b)

        # --- LAZY LOAD: Fetch embeddings from disk (or RAM if already merged) ---
        emb_a = cluster_a.get_embedding(embeddings_group, valid_headers)
        emb_b = cluster_b.get_embedding(embeddings_group, valid_headers)

        # Handle sequence padding for raw leaf nodes just before alignment
        if cluster_a.is_leaf:
            cluster_a.aligned = fit_leaf_to_embedding(cluster_a.aligned, emb_a.shape[0])
        if cluster_b.is_leaf:
            cluster_b.aligned = fit_leaf_to_embedding(cluster_b.aligned, emb_b.shape[0])

        # --- ALIGNMENT ---
        score_mat = compute_score_matrix_with_fallback(
            emb_a,
            emb_b,
            ranked_devices,
            DEVICE_SELECTION,
        )
        path = run_global_traceback(score_mat, GAP_OPEN, GAP_EXTEND)

        # You will need to pass emb_a and emb_b into your merge_clusters function now,
        # since they are no longer stored inside the cluster objects by default.
        new_cluster = merge_clusters(cluster_a, cluster_b, path, emb_a, emb_b)
        new_idx = num_seqs + iteration
        new_cluster.idx = new_idx
        clusters[new_idx] = new_cluster
        
        # --- GARBAGE COLLECTION: Drop old arrays immediately ---
        del emb_a
        del emb_b
        del cluster_a
        del cluster_b
    cluster_merging_seconds = time.perf_counter() - cluster_merging_started

    # 11. SAVE
    final_cluster = clusters[num_seqs + len(linkage_matrix) - 1]
    print(f"Saving Consensus MSA to {OUTPUT_FASTA}...")
    # One transpose gives each aligned sequence as a contiguous row.
    aligned_rows = np.ascontiguousarray(final_cluster.aligned.T)
    with open(OUTPUT_FASTA, "w", encoding="utf-8", newline="\n") as f:
        for i, row in enumerate(aligned_rows):
            original_idx = final_cluster.ids[i]
            header = valid_headers[original_idx]
            f.write(f">{header}\n{row.tobytes().decode('ascii')}\n")
    print("Done!")
    total_processing_seconds = time.perf_counter() - total_processing_started
    report_processing_times(
        total_processing_seconds,
        tree_building_seconds,
        cluster_merging_seconds,
    )

def main(argv=None):
    global SHOW_REGRESSION_PLOT
    _, headless = select_settings_path(os.path.basename(__file__), PROJECT_ROOT, argv)
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    # JSON/MCP invocations must never wait for an interactive plot window.
    if headless:
        SHOW_REGRESSION_PLOT = False
    run_msa_builder()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
