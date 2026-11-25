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
- The guide tree as NEXUS in a `Guide_Trees` folder inside the alignment's folder
  (`<set>_[<model>]_guide_tree.nex`), saved before the alignment starts. Its TREE
  line is a Newick tree, and an EMAPSSN block records the settings that built it, the alignment's file name among them. Internal nodes carry the percentage
  of noise-perturbed replicates that hold their split when those replicates run.

Settings:
- TARGET_SET: A prefix for the output MSA file name.
- PARENT_SET: The prefix shared by the broader input files to pull from.
- MODEL_NAME: The model used for the embeddings.
- ALIGNMENT_SCORE: Whether to weight the guide tree based on "global" or "local" connectivity scores from the network.
- NUM_WORKERS: CPU threads for parallel bootstrap generation of the consensus tree.
- NUM_TREES: How many bootstrap replicate iterations to average for the consensus guide tree (higher = more stable topology).
- NOISE_SCALE: Standard deviation of structural noise applied during bootstrap resampling.
- CALIBRATED_IMPUTATION: For a sparse network, whether a sample of this run's missing pairs is aligned, with the network's own
  scoring, at the start of the run (ON) and used to fit the imputation and to check it, or nothing is aligned and the fit takes
  only the observed pairs, giving every missing pair below its range the fit's lowest value (OFF). With a sample, every other
  missing pair is estimated from its embedding similarity (the fit the sample calibrates) and from chains of computed pairs: each
  third sequence with a computed score to both gives the lower of its two scores and the highest of these is the pair's chain
  value; four weights fitted on the sample combine the two (see utilities/Chain_Fill.py). When the chain step does not fit in the
  free memory, the similarity fit alone fills the missing pairs. It is not applied when fewer
  than 10% of the run's pairs are missing, nor when the network does not record how its scores were made or a check on some of
  its own edges does not reproduce them. Every sparse network averages replicate tree distances over all pairs.
- INPUT_NETWORK_EMBED: Optional, with CALIBRATED_IMPUTATION on and a sparse network: the embeddings the network's scores were
  made from, when the run's own embeddings (INPUT_EMBED) are of another model. They must record the network's model and hold
  every sequence of the run; they then serve the similarity of the fill and the calibration alignment. Left empty, INPUT_EMBED
  serves both and must be of the network's model. Ignored otherwise.
- GAP_OPEN: Penalty scoring for opening gaps in the sequence.
- DEVICE_SELECTION: Auto-benchmarked or manually selected score-matrix device.
- SHOW_REGRESSION_PLOT: For a sparse network, shows the isotonic regression fit over the observed pairs as a pop-up
  chart in the Tools window's look (its theme colour, language and window style). Runs with a JSON settings file
  (including MCP runs) never show it.

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
from utilities import Tool_Supervisor
Tool_Supervisor.supervise(__name__, __file__)  # a run of this script starts under a light parent; nothing else here

import shutil  
import gc      
import h5py
import numpy as np
import torch
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import sys
from utilities import Hardware_Acceleration as Hardware_Utils
from utilities import Numba_Threads
from utilities.MSA_Kernels import (
    _accumulate_cophenetic_batch_kernel,
    _accumulate_path_lengths_batch_kernel,
    _nj_dense_row_minima,
    _prep_tree_euler,
    _prep_tree_euler_nj,
    calculate_normalized_scores_kernel,
    linkage_split_keys,
    neighbor_joining_compact_kernel,
    populate_condensed_matrix,
    run_global_traceback,
)
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from Cache_Manifest import validate_network_schema
from utilities.Localization import Message, set_translator
from utilities.Sequence_Utils import load_sanitized_fasta
from utilities.HDF5_Storage import open_embeddings_shared, read_embedding_manifest
from utilities.Guide_Tree_File import guide_tree_path_for_alignment, newick_label, nexus_string
from utilities import Prefilter_Calibration
from utilities import Chain_Fill
from utilities import Align_Library
from tools.tool_helpers import Tool_Charts
import warnings
from dataclasses import dataclass, field


# ==========================================
# CONFIGURATION
# ==========================================

# Inputs - Now using .h5
INPUT_FASTA   = ""
INPUT_EMBED   = None
INPUT_NETWORK = None
INPUT_NETWORK_EMBED = ""   # optional: the embeddings the network was made from, for the calibration (see choose_network_embeddings)
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
CALIBRATED_IMPUTATION = True  # align a sample of the run's missing pairs to fit the imputation (see calibrate_sparse_run)

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
    input_network_embed="",
):
    """Resolve configured input paths without joining an inactive FASTA value.

    input_network_embed, the optional second embeddings file, resolves in embed_dir exactly as input_embed does;
    empty (or None) gives "".
    """
    if input_network_embed is not None and not isinstance(input_network_embed, str):
        raise MSAConfigurationError("INPUT_NETWORK_EMBED must be an embedding HDF5 filename or an empty string.")
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
        "full_input_network_embed": (
            os.path.join(embed_dir, input_network_embed) if input_network_embed else ""
        ),
        "sequence_set": sequence_set,
    }


def build_msa_output_path(msa_dir, sequence_set, model_name):
    """Build the final MSA filename from resolved, validated metadata."""
    return os.path.join(msa_dir, f"{sequence_set}_[{model_name}]_alignment.fasta")


def build_guide_tree_path(output_fasta):
    """Build the guide-tree NEXUS path from the alignment's: the Guide_Trees folder inside its folder."""
    return guide_tree_path_for_alignment(output_fasta)


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

    Returns the linkage matrix and, for neighbor-joining, each join's raw
    branch lengths (None for UPGMA). ``kernel_threads`` is this worker's
    share of the Numba thread budget, so concurrent neighbor-joining workers
    do not oversubscribe the CPUs.
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
    branch_lengths = None
    if neighbor_joining:
        Z, branch_lengths = neighbor_joining_condensed(
            D_perturbed_cond,
            num_seqs,
            threads=kernel_threads,
            return_branch_lengths=True,
        )
    else:
        Z = sch.linkage(D_perturbed_cond, method='average')

    # Windows requires explicitly closing the memmap before cleanup.
    del baseline_dist

    return Z, branch_lengths


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
    with torch.inference_mode():
        t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float32)
        t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float32)

        # Profile-vector magnitude is meaningful: it decreases when a column has
        # gaps or contains disagreeing residue directions. Compute L2 norms once
        # and reuse them for both confidence scaling and unit-vector normalization.
        norm_i = torch.linalg.vector_norm(t_i, dim=-1, keepdim=True)
        norm_j = torch.linalg.vector_norm(t_j, dim=-1, keepdim=True)
        confidence_i = norm_i.clamp(max=1.0)
        confidence_j = norm_j.clamp(max=1.0)
        t_i_norm = t_i / norm_i.clamp(min=1e-12)
        t_j_norm = t_j / norm_j.clamp(min=1e-12)
        cos_sim = torch.mm(t_i_norm, t_j_norm.T).clamp_(-1.0, 1.0)
        sim_mat = torch.exp(-(1.0 - cos_sim))

        epsilon = 1e-8
        try:
            row_std, row_mean = torch.std_mean(sim_mat, dim=1, keepdim=True, correction=0)
            col_std, col_mean = torch.std_mean(sim_mat, dim=0, keepdim=True, correction=0)
        except (RuntimeError, NotImplementedError):
            row_mean = sim_mat.mean(dim=1, keepdim=True)
            row_std = sim_mat.std(dim=1, keepdim=True, correction=0)
            col_mean = sim_mat.mean(dim=0, keepdim=True)
            col_std = sim_mat.std(dim=0, keepdim=True, correction=0)

        z_r = (sim_mat - row_mean) / (row_std + epsilon)
        z_c = (sim_mat - col_mean) / (col_std + epsilon)
        profile_confidence = confidence_i * confidence_j.T
        final_score = ((z_r + z_c) * 0.5) * profile_confidence

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
        failure_status, failure_details = None, []
        free_memory = Hardware_Utils.begin_gpu_trial(candidate)
        try:
            # Warm up on the largest representative pair so cold-shape kernel
            # selection/JIT never inflates the timed trials.
            warm_left = samples[-1][4]
            warm_right = samples[-1][5]
            compute_score_matrix_torch(
                warm_left,
                warm_right,
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
            memory_text = Hardware_Utils.note_gpu_failure(
                candidate, failure, free_memory, where="MSA device test"
            )
            error = f"{type(failure).__name__}: {failure}" + memory_text
            failure_status, *failure_details = Hardware_Utils.failure_lines(failure, memory_text)
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
            f"{elapsed_text:>15}   {failure_status or error or 'ok'}"
        )
        for line in failure_details:
            print(line)
        results.append(result)

    ranked = Hardware_Utils.rank_benchmark_results(
        results,
        higher_is_better=False,
        decision={
            "kind": "msa_device",
            "unit": "s",
            "sample_merges": sample_shapes,
            "repeats": 3,
        },
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


def accumulate_full_cophenetic(Z, num_seqs, coph_accumulator, threads=None):
    """Add float32 cophenetic distances for every condensed pair in place."""
    if num_seqs <= 1:
        return
    par, sib, ml, mh, ss, se, leaves = _prep_tree_euler(Z, num_seqs)
    if threads is None:
        threads = Numba_Threads.default_thread_count()
    with Numba_Threads.limited_threads(threads):
        _accumulate_cophenetic_batch_kernel(
            par[None, :], sib[None, :], ml[None, :], mh[None, :],
            ss[None, :], se[None, :], leaves[None, :], num_seqs, coph_accumulator,
        )


def accumulate_full_path_lengths(Z, num_seqs, node_length, path_accumulator, threads=None):
    """Add float32 leaf-to-leaf path lengths for every condensed pair in place."""
    if num_seqs <= 1:
        return
    par, sib, ml, rd, ss, se, leaves = _prep_tree_euler_nj(Z, num_seqs, node_length)
    if threads is None:
        threads = Numba_Threads.default_thread_count()
    with Numba_Threads.limited_threads(threads):
        _accumulate_path_lengths_batch_kernel(
            par[None, :], sib[None, :], ml[None, :], rd[None, :],
            ss[None, :], se[None, :], leaves[None, :], num_seqs, path_accumulator,
        )


def add_replicate_to_consensus(Z, nj_branch_lengths, num_seqs, accumulator, threads=None):
    """Add one replicate tree's distance for every pair to the consensus, in place."""
    if nj_branch_lengths is None:
        accumulate_full_cophenetic(Z, num_seqs, accumulator, threads=threads)
        return

    _, node_length = guide_tree_branches(Z, num_seqs, nj_branch_lengths)
    accumulate_full_path_lengths(Z, num_seqs, node_length, accumulator, threads=threads)


def add_replicates_batch_to_consensus(replicates, num_seqs, accumulator, threads=None):
    """Add a batch of replicate trees in exact order using one sequential DRAM pass.

    Bit-for-bit identical to calling ``add_replicate_to_consensus`` on each tree
    in order, while reducing DRAM traffic by ``len(replicates)``x and exploiting
    all CPU cores with L1-resident row accumulation.
    """
    if num_seqs <= 1 or not replicates:
        return
    if threads is None:
        threads = Numba_Threads.default_thread_count()
    is_nj = replicates[0][1] is not None
    if not is_nj:
        prepped = [_prep_tree_euler(Z, num_seqs) for Z, _ in replicates]
        with Numba_Threads.limited_threads(threads):
            _accumulate_cophenetic_batch_kernel(
                np.stack([p[0] for p in prepped]),
                np.stack([p[1] for p in prepped]),
                np.stack([p[2] for p in prepped]),
                np.stack([p[3] for p in prepped]),
                np.stack([p[4] for p in prepped]),
                np.stack([p[5] for p in prepped]),
                np.stack([p[6] for p in prepped]),
                num_seqs,
                accumulator,
            )
    else:
        prepped = []
        for Z, bl in replicates:
            _, node_length = guide_tree_branches(Z, num_seqs, bl)
            prepped.append(_prep_tree_euler_nj(Z, num_seqs, node_length))
        with Numba_Threads.limited_threads(threads):
            _accumulate_path_lengths_batch_kernel(
                np.stack([p[0] for p in prepped]),
                np.stack([p[1] for p in prepped]),
                np.stack([p[2] for p in prepped]),
                np.stack([p[3] for p in prepped]),
                np.stack([p[4] for p in prepped]),
                np.stack([p[5] for p in prepped]),
                np.stack([p[6] for p in prepped]),
                num_seqs,
                accumulator,
            )


def finalize_cophenetic_consensus(distance_matrix, num_trees):
    """Average the accumulated replicate distances over every pair, in place."""
    if num_trees <= 0:
        raise ValueError("NUM_TREES must be greater than zero.")
    distance_matrix /= num_trees
    return distance_matrix


def neighbor_joining_kernel(D, N):
    """Compatibility entry point delegating to the compact (N, N) NJ kernel."""
    D_sub = np.array(D[:N, :N], dtype=np.float64, copy=True, order="C")
    return neighbor_joining_compact_kernel(D_sub)


def neighbor_joining_condensed(
    D_condensed, num_seqs, threads=None, return_branch_lengths=False
):
    """Return the neighbor-joining linkage matrix, and its branch lengths
    when ``return_branch_lengths`` is set."""
    D_square = np.ascontiguousarray(squareform(D_condensed), dtype=np.float64)
    if threads is None:
        threads = Numba_Threads.default_thread_count()
    with Numba_Threads.limited_threads(threads):
        Z, branch_lengths = neighbor_joining_compact_kernel(D_square)
    if return_branch_lengths:
        return Z, branch_lengths
    return Z

# ==========================================
# GUIDE TREE EXPORT
# ==========================================
def guide_tree_leaf_keys(num_seqs, seed):
    """Return one random 64-bit key per leaf for naming splits."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, np.iinfo(np.uint64).max, size=num_seqs, dtype=np.uint64, endpoint=True)


def replicate_split_keys(Z, num_seqs, leaf_keys):
    """Return the sorted, distinct split keys of one replicate tree."""
    return np.unique(linkage_split_keys(Z, num_seqs, leaf_keys))


def guide_tree_branches(linkage_matrix, num_seqs, nj_branch_lengths=None):
    """Return each node's parent (-1 at the root) and branch length.

    UPGMA lengths halve the height differences, so a path between two
    leaves is as long as their merge height. A negative neighbor-joining
    length becomes zero and its sibling takes the join's whole distance.
    """
    node_count = 2 * num_seqs - 1
    parent = np.full(node_count, -1, dtype=np.int64)
    length = np.zeros(node_count, dtype=np.float64)
    children = linkage_matrix[:, :2].astype(np.int64)
    parents = np.arange(num_seqs, node_count, dtype=np.int64)
    parent[children[:, 0]] = parents
    parent[children[:, 1]] = parents
    if nj_branch_lengths is None:
        height = np.zeros(node_count, dtype=np.float64)
        height[num_seqs:] = linkage_matrix[:, 2]
        for side in range(2):
            child = children[:, side]
            length[child] = np.maximum(0.0, 0.5 * (linkage_matrix[:, 2] - height[child]))
    else:
        total = np.maximum(nj_branch_lengths.sum(axis=1), 0.0)
        first = np.clip(nj_branch_lengths[:, 0], 0.0, total)
        length[children[:, 0]] = first
        length[children[:, 1]] = total - first
    return parent, length


def rooted_children(parent):
    """Return each node's children and the root of a parent array."""
    children = [[] for _ in range(len(parent))]
    root = -1
    for node, up in enumerate(parent):
        if up < 0:
            root = node
        else:
            children[up].append(node)
    return children, root


def midpoint_rooted_children(parent, length, num_seqs):
    """Re-root a tree at the midpoint of its longest leaf-to-leaf path.

    Returns each node's children, branch lengths and the root. The new root
    is appended as the last node; a node left with one child (the old root)
    is folded into its branch.
    """
    node_count = len(parent)
    neighbours = [[] for _ in range(node_count)]
    for node, up in enumerate(parent):
        if up >= 0:
            neighbours[node].append((up, length[node]))
            neighbours[up].append((node, length[node]))

    def distances_from(start):
        distance = np.full(node_count, -1.0)
        previous = np.full(node_count, -1, dtype=np.int64)
        distance[start] = 0.0
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour, weight in neighbours[node]:
                if distance[neighbour] < 0.0:
                    distance[neighbour] = distance[node] + weight
                    previous[neighbour] = node
                    stack.append(neighbour)
        return distance, previous

    start = int(np.argmax(distances_from(0)[0][:num_seqs]))
    distance, previous = distances_from(start)
    end = int(np.argmax(distance[:num_seqs]))
    half = 0.5 * distance[end]
    if half <= 0.0:
        children, root = rooted_children(parent)
        return children, np.append(length, 0.0), root

    # Walk back from the far end to the edge that holds the midpoint.
    lower = end
    while distance[previous[lower]] > half:
        lower = previous[lower]
    upper = int(previous[lower])
    to_upper = half - distance[upper]
    edge = distance[lower] - distance[upper]

    root = node_count
    children = [[] for _ in range(node_count + 1)]
    branch = np.zeros(node_count + 1, dtype=np.float64)
    stack = []

    def attach(up, node, came_from, weight):
        while node >= num_seqs:
            onward = [pair for pair in neighbours[node] if pair[0] != came_from]
            if len(onward) != 1:
                break
            came_from, (node, extra) = node, onward[0]
            weight += extra
        children[up].append(node)
        branch[node] = weight
        stack.append((node, came_from))

    attach(root, upper, lower, to_upper)
    attach(root, lower, upper, edge - to_upper)
    while stack:
        node, came_from = stack.pop()
        for neighbour, weight in neighbours[node]:
            if neighbour != came_from:
                attach(node, neighbour, node, weight)
    return children, branch, root


def split_support(children, root, num_seqs, leaf_keys, replicate_keys, num_trees):
    """Return the percentage of replicate trees holding each node's split.

    replicate_keys is the sorted concatenation of every replicate's distinct
    split keys. Leaves, the root and clades of all leaves but one get -1.
    """
    node_count = len(children)
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])

    keys = np.zeros(node_count, dtype=np.uint64)
    sizes = np.zeros(node_count, dtype=np.int64)
    keys[:num_seqs] = leaf_keys
    sizes[:num_seqs] = 1
    for node in reversed(order):
        for child in children[node]:
            keys[node] ^= keys[child]
            sizes[node] += sizes[child]

    total = np.bitwise_xor.reduce(leaf_keys)
    support = np.full(node_count, -1, dtype=np.int64)
    named = [
        node for node in order
        if node != root and 2 <= sizes[node] <= num_seqs - 2
    ]
    if named:
        named = np.asarray(named, dtype=np.int64)
        split = np.minimum(keys[named], keys[named] ^ total)
        found = (
            np.searchsorted(replicate_keys, split, side="right")
            - np.searchsorted(replicate_keys, split, side="left")
        )
        support[named] = np.rint(100.0 * found / num_trees).astype(np.int64)
    return support


def newick_text(children, branch, root, headers, support=None):
    """Write a rooted tree as one Newick line, without recursion.

    Internal nodes carry their support percentage as a label when one is
    given; every node except the root carries its branch length.
    """
    def length_of(node):
        return "" if node == root else f":{float(branch[node]) + 0.0:.6g}"

    parts = []
    stack = [(False, root)]
    while stack:
        closing, node = stack.pop()
        if node is None:
            parts.append(",")
        elif closing:
            label = "" if support is None or support[node] < 0 else str(int(support[node]))
            parts.append(f"){label}{length_of(node)}")
        elif not children[node]:
            parts.append(newick_label(headers[node]) + length_of(node))
        else:
            parts.append("(")
            stack.append((True, node))
            for position, child in enumerate(reversed(children[node])):
                if position:
                    stack.append((False, None))
                stack.append((False, child))
    parts.append(";")
    return "".join(parts)


def write_guide_tree(
    path,
    linkage_matrix,
    headers,
    nj_branch_lengths=None,
    leaf_keys=None,
    replicate_keys=None,
    num_trees=0,
    notes=(),
):
    """Save the guide tree as NEXUS: UPGMA keeps its root, neighbor-joining
    is midpoint-rooted, and perturbed replicates label each split.

    The TREE line holds a plain Newick tree with the full headers, so it can
    be copied out as it is. An EMAPSSN block, which tree readers skip, records
    the (name, value) ``notes`` of the run and how to read the tree.
    """
    num_seqs = len(headers)
    neighbor_joining = nj_branch_lengths is not None
    support = None
    if num_seqs < 2:
        text = newick_label(headers[0]) + ";" if num_seqs else ";"
    else:
        parent, length = guide_tree_branches(linkage_matrix, num_seqs, nj_branch_lengths)
        if not neighbor_joining:
            children, root = rooted_children(parent)
            branch = length
        else:
            children, branch, root = midpoint_rooted_children(parent, length, num_seqs)
        if replicate_keys is not None and num_trees > 0:
            support = split_support(
                children, root, num_seqs, leaf_keys, replicate_keys, num_trees
            )
        text = newick_text(children, branch, root, headers, support)

    notes = list(notes) + [("SEQUENCES", num_seqs)]
    if num_seqs >= 2 and neighbor_joining:
        notes += [
            ("ROOT", "Midpoint of the longest path. The alignment merged from the last "
                     "neighbor-joining join instead."),
            ("BRANCH_LENGTHS", "Saitou-Nei neighbor-joining lengths. A negative length became "
                               "zero and its sibling took the rest."),
        ]
    elif num_seqs >= 2:
        notes += [
            ("ROOT", "The UPGMA root. Every sequence is equally far from it."),
            ("BRANCH_LENGTHS", "Half of each merge-height step, so the path between two "
                               "sequences equals the height at which they merge."),
        ]
    if support is not None:
        notes.append(("INTERNAL_LABELS", f"Percentage of the {num_trees} noise-perturbed replicate "
                                         "trees that hold the split. This is stability under "
                                         "distance noise, not bootstrap support."))
    lines = [
        "#NEXUS",
        "[Guide tree saved by the EMAP-SSN embedding MSA. After the equals sign, the TREE line is a Newick tree.]",
        "",
        "BEGIN EMAPSSN;",
        "    [How this guide tree was built. Tree readers skip this block.]",
        *(f"    {name} = {nexus_string(value)};" for name, value in notes),
        "END;",
        "",
        "BEGIN TREES;",
        f"    TREE guide_tree = {'[&U]' if neighbor_joining else '[&R]'} {text}",
        "END;",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def program_version():
    """The release Desktop_App.py names, read without importing Qt; None when unreadable."""
    source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "desktop", "Desktop_App.py"
    )
    try:
        with open(source, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "APPLICATION_VERSION" for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def guide_tree_notes(network_type, imputed, imputation=None, calibration=None):
    """The settings a saved guide tree records, as (name, value) pairs.

    imputation, an ImputationSummary, adds to MISSING_PAIRS how well the
    imputed scores were checked and how many fell outside the fit's range.
    calibration, a CalibrationOutcome, adds what the run did about the calibration sample.
    """
    version = program_version()
    notes = [
        ("WRITTEN", time.strftime("%Y-%m-%d %H:%M:%S %z")),
        ("PROGRAM", f"EMAP-SSN {version}" if version else "EMAP-SSN"),
        ("SEQUENCE_SET", _seq_set),
        ("MODEL", _model_name),
        ("ALIGNMENT", os.path.basename(OUTPUT_FASTA)),
        ("INPUT_NETWORK", os.path.basename(FULL_INPUT_NETWORK)),
        ("INPUT_EMBED", os.path.basename(FULL_INPUT_EMBED)),
    ]
    if USE_SEQUENCE_FILTER:
        notes.append(("INPUT_FASTA", os.path.basename(FULL_INPUT_FASTA)))
    if network_type == "blast":
        notes.append(("NETWORK_SCORES", "BLAST E-values"))
    else:
        notes += [("ALIGNMENT_SCORE", ALIGNMENT_SCORE), ("NORMALIZATION_MODE", NORMALIZATION_MODE)]
    notes.append(("MISSING_PAIRS", missing_pairs_note(imputed, imputation)))
    if imputed or calibration is not None:
        notes.append(("CALIBRATED_IMPUTATION", "true" if CALIBRATED_IMPUTATION else "false"))
    if calibration is not None:
        notes.append(("CALIBRATION", calibration_text(calibration)))
    notes += [
        ("TREE_METHOD", TREE_METHOD),
        ("BOOTSTRAP_TREE", "true" if BOOTSTRAP_TREE else "false"),
    ]
    if BOOTSTRAP_TREE:
        notes += [
            ("NUM_TREES", NUM_TREES),
            ("NOISE_SCALE", NOISE_SCALE),
            ("RANDOM_SEED", RANDOM_SEED),
        ]
    return notes


def heavy_first_merge_order(linkage_matrix, num_seqs):
    """Return merge indices in Sethi-Ullman heavy-first post-order DFS.

    Executing the guide tree's merges in this order produces the exact same
    per-node alignments and final MSA as sequential ``linkage_matrix`` order,
    while bounding the number of simultaneously live intermediate profiles in
    ``active_profiles`` to at most ``floor(log2(num_seqs))`` (O(log N) instead
    of O(N) for breadth-first distance-ordered merges). Uses an explicit stack
    so deeply unbalanced trees never hit Python's recursion limit.
    """
    num_merges = len(linkage_matrix)
    if num_merges == 0:
        return []
    total_nodes = int(num_seqs) + num_merges
    # Minimum simultaneously live internal profiles needed to evaluate each subtree.
    cost = np.zeros(total_nodes, dtype=np.int32)
    size = np.ones(total_nodes, dtype=np.int32)
    for k in range(num_merges):
        a = int(linkage_matrix[k, 0])
        b = int(linkage_matrix[k, 1])
        node = int(num_seqs) + k
        size[node] = size[a] + size[b]
        ca = int(cost[a])
        cb = int(cost[b])
        if ca == cb:
            cost[node] = ca + 1
        else:
            cost[node] = ca if ca > cb else cb

    order = []
    visited = np.zeros(num_merges, dtype=bool)
    # Start from the root (last merge), falling back to any unvisited merge if disconnected.
    for root_k in range(num_merges - 1, -1, -1):
        if visited[root_k]:
            continue
        stack = [(int(num_seqs) + root_k, False)]
        while stack:
            node, ready = stack.pop()
            k = node - int(num_seqs)
            if ready:
                if not visited[k]:
                    visited[k] = True
                    order.append(k)
                continue
            if visited[k]:
                continue
            stack.append((node, True))
            a = int(linkage_matrix[k, 0])
            b = int(linkage_matrix[k, 1])
            # Visit higher-cost (tie-break: larger subtree) child first -> push it last.
            a_key = (int(cost[a]), int(size[a]))
            b_key = (int(cost[b]), int(size[b]))
            first, second = (a, b) if a_key >= b_key else (b, a)
            if second >= num_seqs:
                stack.append((second, False))
            if first >= num_seqs:
                stack.append((first, False))
    return order


def merge_profiles_lazy(width_a, count_a, width_b, count_b, path, emb_a, emb_b):
    """Merge only the float16 profiles and return ``(merged_profile, moves_int8)``.

    Avoids O(N^2 W) intermediate dense alignment copies during progressive merging;
    the exact final alignment is materialized once via top-down tree replay.
    """
    moves = np.ascontiguousarray(path[::-1], dtype=np.int8)
    take_a = (moves == 1) | (moves == 2)
    take_b = (moves == 1) | (moves == 3)
    used_a = int(take_a.sum())
    used_b = int(take_b.sum())
    if not (used_a == width_a == emb_a.shape[0] and used_b == width_b == emb_b.shape[0]):
        raise ValueError(
            "Alignment path does not place every column: cluster A has "
            f"{width_a} columns and {emb_a.shape[0]} profile "
            f"rows, the path takes {used_a}; cluster B has "
            f"{width_b} columns and {emb_b.shape[0]} profile "
            f"rows, the path takes {used_b}."
        )
    width = moves.shape[0]
    w_a = float(count_a)
    w_b = float(count_b)
    total_w = w_a + w_b
    merged = np.full((width, emb_a.shape[1]), -0.0, dtype=np.float32)
    merged[take_a] = emb_a.astype(np.float32) * w_a
    merged[take_b] += emb_b.astype(np.float32) * w_b
    merged /= total_w
    return merged.astype(np.float16), moves


def write_lazy_alignment_fasta(
    output_path, root_node, num_seqs, merge_children, merge_moves, leaf_alignments, valid_headers, final_width
):
    """Stream the exact progressive alignment to FASTA in O(N * W_final) via top-down tree replay."""
    row_buf = np.empty(final_width, dtype=np.uint8)
    stack = [(root_node, np.arange(final_width, dtype=np.int32))]
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        while stack:
            node, col_map = stack.pop()
            if node < num_seqs:
                row_buf.fill(GAP_CODE)
                row_buf[col_map] = leaf_alignments[node][:, 0]
                f.write(f">{valid_headers[node]}\n{row_buf.tobytes().decode('ascii')}\n")
            else:
                child_a, child_b = merge_children[node - num_seqs]
                moves = merge_moves[node - num_seqs]
                # Push B first so A is popped and written first (preserves exact ids_a + ids_b order!)
                stack.append((child_b, col_map[(moves == 1) | (moves == 3)]))
                stack.append((child_a, col_map[(moves == 1) | (moves == 2)]))

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
# SPARSE NETWORK IMPUTATION
# ==========================================
# With fewer calibration pairs than this in a run, half of them could not
# test the fit, so it is made as for a network without a sample.
MIN_CALIBRATION_PAIRS = 10
# The most observed pairs the fit takes; a random sample of them stands in above this.
IMPUTATION_FIT_PAIRS = 500_000


@dataclass
class CalibrationCheck:
    """A fit made without every second calibration pair, scored on those pairs."""

    pairs: int
    spearman: float
    mae: float
    r2: float


@dataclass
class ImputationFit:
    """The isotonic fit a sparse network's missing pairs are filled from.

    fit_x and fit_y are the observed pairs the fit took besides the
    calibration sample (all of them, or a random sample); calibration_x and
    calibration_y are the calibration pairs of the run, None without a
    sample, and check tells how well a fit without half of them predicts that half.
    uniform is True when every missing pair below the fit's range gets the
    fit's lowest value (a fit made without a sample).
    """

    iso: IsotonicRegression
    fit_x: np.ndarray
    fit_y: np.ndarray
    calibration_x: np.ndarray = None
    calibration_y: np.ndarray = None
    check: CalibrationCheck = None
    uniform: bool = True
    # With a sample: the weights of its pairs, and each pair's prediction by a fit made without its
    # half (even positions predicted without the odd ones and the reverse), which the chain fill reads.
    calibration_weights: np.ndarray = None
    out_of_fold: np.ndarray = None


@dataclass
class ChainSummary:
    """What a run records about a chain fill: its held-out check, the share of missing pairs that had a chain, its coefficients."""

    check: CalibrationCheck
    covered: float
    coefficients: tuple
    calibration_coverage: float


@dataclass
class ImputationSummary:
    """What a run records about its imputation: the missing pairs and how they were filled."""

    missing: int
    outside: float
    check: CalibrationCheck = None
    calibration_pairs: int = 0
    uniform_value: float = None
    # Why no calibration sample was used ("" when one was): a short phrase for the notes and the log.
    calibration: str = ""
    # Set once the missing pairs are filled with the chain fill; chain_skipped says why a run with a
    # calibration sample was filled from the calibrated isotonic fit alone instead ("" when it was not).
    chain: ChainSummary = None
    chain_skipped: str = ""


def held_out_check(predicted, actual):
    """Spearman's rho, mean absolute error and R² of predicted against directly computed scores."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.ptp(predicted) > 0 and np.ptp(actual) > 0:
            rho = float(spearmanr(predicted, actual).statistic)
        else:
            rho = float("nan")
    error = predicted - actual
    centred = actual - actual.mean()
    total = float(np.dot(centred, centred))
    r2 = 1.0 - float(np.dot(error, error)) / total if total > 0 else float("nan")
    return CalibrationCheck(int(actual.size), rho, float(np.mean(np.abs(error))), r2)


_KEY_CHUNK = 8_000_000


def canonical_pair_keys(edge_i, edge_j, headers):
    """One int64 key per observed pair that depends only on which two sequences it joins: min(rank) * n + max(rank).

    rank is the position of a sequence's header in the headers sorted in plain code-point order (Python's
    string order, never a locale's), so neither the order of the edges in the network file nor the order of the
    run's sequences changes a pair's key. edge_i and edge_j are indices into headers (a pair's two ends in
    either order); headers are distinct (the run keeps one entry per header). Computed in chunks.
    """
    n = len(headers)
    rank = np.empty(n, dtype=np.int64)
    rank[sorted(range(n), key=headers.__getitem__)] = np.arange(n, dtype=np.int64)
    keys = np.empty(len(edge_i), dtype=np.int64)
    for start in range(0, len(edge_i), _KEY_CHUNK):
        stop = min(start + _KEY_CHUNK, len(edge_i))
        a = rank[np.asarray(edge_i[start:stop], dtype=np.int64)]
        b = rank[np.asarray(edge_j[start:stop], dtype=np.int64)]
        keys[start:stop] = np.minimum(a, b) * n + np.maximum(a, b)
    return keys


def fit_imputation(x, y, calibration_positions=None, calibration_weights=None,
                   fit_pairs=IMPUTATION_FIT_PAIRS, seed=42, pair_keys=None):
    """Fit normalized scores y on similarity x over a sparse network's observed pairs.

    pair_keys (canonical_pair_keys of the observed pairs, one per entry of x) puts the pairs in a canonical
    order before the random pick, so the pick, and so the fit, does not depend on the order of the edges in
    the network file; a stable argsort of the keys, then the same seeded pick as ever. The calibration
    sample's pairs keep their positions and their order. Without pair_keys the pick follows the given order.

    Without a calibration sample, or with fewer than MIN_CALIBRATION_PAIRS of
    it, the fit takes every observed pair or a random fit_pairs of them; every
    missing pair below the lowest of them then gets the fit's lowest value.
    With one, calibration_positions are the sample's edges in sample order and
    calibration_weights how many missing pairs each stands for. The fit then
    takes the other observed pairs (or a random fit_pairs of them, weighted up
    to their count) and every calibration pair, weighted, so it spans the
    missing pairs' range as a fit over all pairs would. A first fit without
    every second calibration pair is scored on those pairs (the check); the
    fit returned uses them all.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if calibration_positions is not None:
        calibration_positions = np.asarray(calibration_positions, dtype=np.int64)
    canonical = None if pair_keys is None else np.argsort(pair_keys, kind="stable")
    if calibration_positions is None or calibration_positions.size < MIN_CALIBRATION_PAIRS:
        if len(x) > fit_pairs:
            np.random.seed(seed)
            pick = np.random.choice(len(x), size=fit_pairs, replace=False)
            if canonical is not None:
                pick = canonical[pick]
            fit_x, fit_y = x[pick], y[pick]
        elif canonical is not None:
            fit_x, fit_y = x[canonical], y[canonical]
        else:
            fit_x, fit_y = x, y
        iso = IsotonicRegression(out_of_bounds="clip").fit(fit_x, fit_y)
        return ImputationFit(iso, fit_x, fit_y)

    is_calibration = np.zeros(len(x), dtype=bool)
    is_calibration[calibration_positions] = True
    if canonical is None:
        others = np.flatnonzero(~is_calibration)
    else:
        others = canonical[~is_calibration[canonical]]
    del canonical
    other_count = others.size
    if others.size > fit_pairs:
        np.random.seed(seed)
        others = others[np.random.choice(others.size, size=fit_pairs, replace=False)]
    fit_x, fit_y = x[others], y[others]
    calibration_x = x[calibration_positions]
    calibration_y = y[calibration_positions]

    weights = np.asarray(calibration_weights, dtype=np.float64)
    other_weight = other_count / max(others.size, 1)

    def fit_with_sample(chosen):
        # The chosen calibration pairs stand in for the whole sample's missing pairs.
        scale = len(weights) / len(chosen)
        return IsotonicRegression(out_of_bounds="clip").fit(
            np.concatenate([fit_x, calibration_x[chosen]]),
            np.concatenate([fit_y, calibration_y[chosen]]),
            sample_weight=np.concatenate(
                [np.full(fit_x.size, other_weight), weights[chosen] * scale]
            ),
        )

    held = np.arange(1, calibration_x.size, 2)
    even = np.arange(0, calibration_x.size, 2)
    from_even = fit_with_sample(even).predict(calibration_x[held])
    check = held_out_check(from_even, calibration_y[held])
    # Each calibration pair predicted by a fit made without its half (the chain fill's c_oof).
    out_of_fold = np.empty(calibration_x.size, dtype=np.float64)
    out_of_fold[held] = from_even
    out_of_fold[even] = fit_with_sample(held).predict(calibration_x[even])
    iso = fit_with_sample(np.arange(calibration_x.size))
    return ImputationFit(
        iso, fit_x, fit_y, calibration_x, calibration_y, check, uniform=False,
        calibration_weights=weights, out_of_fold=out_of_fold,
    )


def uniform_fill_value(iso):
    """The value every missing pair below the fit's range gets: the fit's lowest."""
    return float(iso.predict([iso.X_min_])[0])


def outside_fit_fraction(similarity, iso, observed_x, missing):
    """The share of missing pairs whose similarity lies outside the fit's range.

    Each of them gets the fit's value at the nearer end. similarity is the
    (n, n) matrix, observed_x the observed pairs' similarities, and missing
    how many of the n(n - 1)/2 pairs are not observed.
    """
    if missing <= 0:
        return 0.0
    low, high = float(iso.X_min_), float(iso.X_max_)
    outside = 0
    for start in range(0, similarity.shape[0], 1024):
        block = similarity[start:start + 1024]
        # The upper triangle: column > row, where row = start + local row.
        outside += int(np.count_nonzero(np.triu((block < low) | (block > high), k=start + 1)))
    observed_x = np.asarray(observed_x)
    observed_outside = int(np.count_nonzero((observed_x < low) | (observed_x > high)))
    return max(outside - observed_outside, 0) / missing


def missing_pairs_note(imputed, summary=None):
    """The guide tree's MISSING_PAIRS note: how missing pairs were filled, and how well."""
    if not imputed:
        return "None. Every pair is in the network."
    text = "Regression-imputed from embedding distances"
    if summary is None:
        return text
    if summary.chain is not None:
        check = summary.chain.check
        c = summary.chain.coefficients
        covered = "" if summary.chain.covered is None else f"{100.0 * summary.chain.covered:.1f}"
        return (
            "Imputed from embedding similarity and chains of computed pairs (calibrated); "
            f"checked on {check.pairs:,} held-out calibration pairs: Spearman "
            f"{check.spearman:.4f}, MAE {check.mae:.4f}, R^2 {check.r2:.4f}; "
            f"{covered}% of missing pairs had a chain; coefficients (constant, isotonic fit, chain, no chain) "
            f"{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, {c[3]:.4f}; "
            f"{100.0 * summary.outside:.1f}% of missing pairs outside the fit's range"
        )
    if summary.check is not None:
        check = summary.check
        text += (
            f"; checked on {check.pairs:,} held-out calibration pairs: Spearman "
            f"{check.spearman:.4f}, MAE {check.mae:.4f}, R^2 {check.r2:.4f}"
        )
        if summary.chain_skipped:
            text += f"; chain fill not used: {summary.chain_skipped}"
    else:
        reason = summary.calibration or "no calibration sample was used"
        text += f"; {reason}, so the estimates are unchecked"
    return text + f"; {100.0 * summary.outside:.1f}% of missing pairs outside the fit's range"


def report_imputation(fit, summary):
    """Print the calibration check, or the warning a run without a sample gets.

    Whenever the fit's lowest value fills the missing pairs below its range
    (fit.uniform), the value is printed as well.
    """
    outside = (
        f"  > Missing pairs outside the fit's range: {100.0 * summary.outside:.1f}% of "
        f"{summary.missing:,} (they get the fit's value at the nearer end)"
    )
    if fit.check is not None:
        check = fit.check
        print(
            f"Calibration check ({summary.calibration_pairs:,} directly computed pairs "
            "from this run's missing pairs):"
        )
        print(
            f"  > A fit without every second one predicts those {check.pairs:,}: "
            f"Spearman rho {check.spearman:.4f}, MAE {check.mae:.4f}, R^2 {check.r2:.4f}"
        )
        print(outside)
        if summary.chain is not None:
            chain = summary.chain
            c = chain.coefficients
            print(
                "Chain fill: the missing pairs are estimated from the calibrated isotonic fit and from chains of computed pairs."
            )
            print(
                f"  > Chain + calibration fill, checked on the same {chain.check.pairs:,} pairs: "
                f"Spearman rho {chain.check.spearman:.4f}, MAE {chain.check.mae:.4f}, R^2 {chain.check.r2:.4f}"
            )
            print(
                "  > Coefficients (constant, isotonic fit, chain, no chain): "
                f"{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}, {c[3]:.6f}"
            )
            print(f"  > Calibration pairs with a chain: {100.0 * chain.calibration_coverage:.1f}%")
        elif summary.chain_skipped:
            print(f"⚠️ Chain fill not used: {summary.chain_skipped}")
            print("  > The missing pairs are filled from the calibrated isotonic fit alone.")
        return
    print(outside)
    print(
        "Uniform fill: every missing pair below the fit's range gets the fit's lowest value, "
        f"{uniform_fill_value(fit.iso):.4f}."
    )
    reason = summary.calibration or "no calibration sample was used"
    print(f"⚠️ {reason[:1].upper()}{reason[1:]}, so the imputed scores are unchecked.")


def chain_memory_reason(verdict):
    """The reason a chain fill is not used when its arrays do not fit in the free memory (see Chain_Fill.memory_verdict)."""
    def size(count):
        return f"{count / 1024 ** 3:.1f} GB" if count >= 0.1 * 1024 ** 3 else f"{count / 1024 ** 2:.1f} MB"

    return (
        f"it needs about {size(verdict.need)} and {size(verdict.usable)} of memory is free "
        f"beyond a reserve of {size(verdict.reserve)}"
    )


def prepare_chain_fill(num_seqs, edge_i, edge_j, y, calibration_positions, imputation, summary):
    """Prepare the chain fill of a run whose fit used a calibration sample; return a Chain_Fill.ChainPlan or None.

    None (and summary.chain_skipped says why) when the plan's arrays do not fit in the memory free now minus
    the reserve; the missing pairs are then filled from the calibrated isotonic fit alone, as before. Otherwise
    summary.chain gets the held-out check of the chain + calibration estimates, its coefficients and the share
    of calibration pairs that had a chain.

    The check's scheme is parity halves of the calibration pairs, with c_oof taken from the existing
    out-of-fold isotonic fits (ImputationFit.out_of_fold): the four coefficients are fitted on the even half
    and applied to the odd half, the pairs the isotonic check scores, so the two checks compare. It has one
    small indirect leak: the half the coefficients are fitted on uses c_oof values from isotonic fits that
    saw the other half's scores, so the evaluated half's targets reach the four coefficients indirectly
    (small with four parameters and thousands of pairs). No nested scheme removes it.
    """
    verdict = Chain_Fill.memory_verdict(num_seqs, len(edge_i))
    if not verdict.fits:
        summary.chain_skipped = chain_memory_reason(verdict)
        return None
    plan = Chain_Fill.prepare(
        num_seqs, edge_i, edge_j, y, calibration_positions,
        imputation.out_of_fold, imputation.calibration_weights,
    )
    summary.chain = ChainSummary(
        check=held_out_check(plan.held_out, imputation.calibration_y[1::2]),
        covered=None,
        coefficients=tuple(float(c) for c in plan.coefficients),
        calibration_coverage=plan.calibration_coverage,
    )
    return plan


# ==========================================
# CALIBRATION SAMPLE
# ==========================================
# A sparse network's imputation is fitted with, and checked on, a sample of this run's own missing
# pairs, aligned at the start of the run with the scoring the network records (calibrate_sparse_run).
# It is not applied when fewer than this share of the run's pairs are missing (the Tools window greys the switch then).
CALIBRATION_MIN_MISSING_FRACTION = Prefilter_Calibration.MIN_MISSING_FRACTION  # one constant with the Tools window
# Edges of the network itself that go through the same alignment call as the sample. Their four stored score
# fields must come out as stored: that shows the embeddings, gap penalties and precision are the network's.
# They are chosen with the sample's seed from CALIBRATION_CHECK_POOL random edges of the run, spread over the
# range of their stored scores.
CALIBRATION_CHECK_EDGES = 32
CALIBRATION_CHECK_POOL = 4096
# The CPU pool of the calibration alignment gets one child per this many score-matrix cells (length_i x length_j
# summed over the pairs): about 1.5 s of alignment each, which a child's start-up (a few seconds) should not dwarf.
CALIBRATION_CELLS_PER_WORKER = 8_000_000
# A reproduced score may differ from the stored one by this much (absolute) plus this share of it: the stored
# scores may have been made on another device, and a different order of float32 additions differs in the last digits.
CALIBRATION_CHECK_ABSOLUTE = 1e-4
CALIBRATION_CHECK_RELATIVE = 1e-4
# Pairs aligned in a precision other than the one the network records (a TF32 or BF16 network on a machine
# that aligns in IEEE FP32) differ more. Measured on 32 check edges of a 200-sequence kringle network, GPU
# and CPU runs (2026-11-21): TF32 network in IEEE FP32, largest relative difference 6.7e-5 (l_score) and
# 1.9e-5 (g_score), largest absolute 2.3e-3; BF16 network in IEEE FP32, 6.3e-4 (l_score) and 1.7e-3 (g_score),
# largest absolute 0.215; both lengths equal on every edge in both. The relative limits are 15 and 12 times
# those, and wrong embeddings or gap penalties differ by far more.
CALIBRATION_CHECK_MISMATCHED = {"tf32": 1e-3, "bf16": 2e-2}
# In those mismatched cases this many check edges may end in a different length (a near tie can end
# differently in another precision). The 2 is a margin, NOT a measurement: no edge differed in a length.
CALIBRATION_CHECK_LENGTH_MARGIN = 2
CHECK_FIELDS = ("l_score", "l_len", "g_score", "g_len")


@dataclass
class CalibrationOutcome:
    """What the run did about calibration, for its log and its guide tree.

    status is "applied" (a sample of the missing pairs was aligned and used), "all_aligned" (every missing
    pair was no more than the sample size, so all were aligned and nothing is imputed), "off" (the switch is
    off), "not_applied" (fewer pairs missing than CALIBRATION_MIN_MISSING_FRACTION) or "unavailable" (the
    network's scoring is not known here or was not reproduced). reason is a short phrase for the last three.
    """

    status: str
    reason: str = ""
    sample: int = 0
    missing: int = 0
    share: float = 0.0
    device: str = ""
    plan: str = ""
    precision: str = ""
    seconds: float = 0.0
    check_edges: int = 0
    check_difference: float = 0.0
    tolerance: str = ""  # what the self-check allowed, when it was not the same-precision rule
    embeddings: str = ""  # the embeddings that served the network, "name (model m)", when the run knew them
    notes: list = field(default_factory=list)


@dataclass
class CalibrationResult:
    """The outcome and, when pairs were aligned, the sample (its pairs, weights) and their four score fields."""

    outcome: CalibrationOutcome
    sample: object = None
    scores: dict = None


@dataclass
class CheckEdges:
    """Edges of the network that must be reproduced: their run indices and their stored score fields."""

    rows: np.ndarray
    cols: np.ndarray
    stored: dict


def choose_check_edges(network, edge_i, edge_j, net_old_to_new, score_field,
                       count=None, pool=None, seed=Prefilter_Calibration.SEED):
    """About count of the network's own edges that lie within the run, spread over their stored scores; None if there are none.

    network is the open network file, edge_i and edge_j its index arrays (the network's own indices),
    net_old_to_new the map of those to the run's (-1 outside it), score_field the field the guide tree uses
    ("g_score" or "l_score"). A random pool of the edges is drawn with seed, those in the run are sorted by
    the stored score and count of them are taken at even steps of that order.
    """
    count = CALIBRATION_CHECK_EDGES if count is None else count
    pool = CALIBRATION_CHECK_POOL if pool is None else pool
    total = len(edge_i)
    if total == 0:
        return None
    rng = np.random.default_rng(seed)
    candidates = np.sort(rng.choice(total, size=min(total, pool), replace=False))
    rows = map_network_indices(np.asarray(edge_i[candidates]), net_old_to_new)
    cols = map_network_indices(np.asarray(edge_j[candidates]), net_old_to_new)
    inside = (rows >= 0) & (cols >= 0) & (rows != cols)
    candidates, rows, cols = candidates[inside], rows[inside], cols[inside]
    if candidates.size == 0:
        return None
    stored = {name: np.asarray(network[name][candidates]) for name in CHECK_FIELDS}
    order = np.argsort(stored[score_field], kind="stable")
    steps = np.unique(np.round(np.linspace(0, order.size - 1, min(count, order.size))).astype(np.int64))
    chosen = np.sort(order[steps])
    return CheckEdges(
        rows.astype(np.int64)[chosen], cols.astype(np.int64)[chosen],
        {name: values[chosen] for name, values in stored.items()},
    )


def check_tolerance(aligned):
    """(absolute, relative, check edges that may differ in a length, description) the self-check allows for this alignment.

    The same precision as the network's, or one that is not recorded here: CALIBRATION_CHECK_ABSOLUTE and
    CALIBRATION_CHECK_RELATIVE and every length equal (the description is empty). A TF32 or BF16 network aligned in
    IEEE FP32, which PairAlignment says by precision against requested_precision: the wider relative limit of
    CALIBRATION_CHECK_MISMATCHED and CALIBRATION_CHECK_LENGTH_MARGIN edges allowed a different length.
    """
    requested = getattr(aligned, "requested_precision", None)
    used = getattr(aligned, "precision", None)
    if used == "ieee_fp32" and requested in CALIBRATION_CHECK_MISMATCHED:
        relative = CALIBRATION_CHECK_MISMATCHED[requested]
        name = {"tf32": "TF32", "bf16": "BF16"}[requested]
        return (CALIBRATION_CHECK_ABSOLUTE, relative, CALIBRATION_CHECK_LENGTH_MARGIN,
                f"tolerance for a {name} network aligned in IEEE FP32: absolute {CALIBRATION_CHECK_ABSOLUTE:g} plus "
                f"relative {relative:g}, up to {CALIBRATION_CHECK_LENGTH_MARGIN} check edges may differ in a length")
    return CALIBRATION_CHECK_ABSOLUTE, CALIBRATION_CHECK_RELATIVE, 0, ""


def compare_with_stored(aligned, offset, stored):
    """(reproduced, largest score difference, what differed) of aligned[offset:] against the stored fields.

    The two score fields must be within the absolute plus relative tolerance of check_tolerance(aligned)
    of the stored value, and the two lengths equal on every check edge (all but the allowed few in a
    precision mismatch).
    """
    absolute, relative, length_margin, _ = check_tolerance(aligned)
    worst, problems = 0.0, []
    length_wrong = None
    for name in CHECK_FIELDS:
        got = np.asarray(getattr(aligned, name)[offset:])
        want = np.asarray(stored[name])
        if name.endswith("_len"):
            unequal = got.astype(np.int64) != want.astype(np.int64)
            length_wrong = unequal if length_wrong is None else (length_wrong | unequal)
            wrong = int(np.count_nonzero(unequal))
            if wrong and not length_margin:
                problems.append(f"{wrong} of {want.size} {name} values differ")
            continue
        difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
        worst = max(worst, float(difference.max()) if difference.size else 0.0)
        allowed = absolute + relative * np.abs(want.astype(np.float64))
        wrong = int(np.count_nonzero(~(difference <= allowed)))
        if wrong:
            problems.append(f"{wrong} of {want.size} {name} values differ by more than the tolerance")
    if length_margin and length_wrong is not None:
        edges = int(np.count_nonzero(length_wrong))
        if edges > length_margin:
            problems.append(f"{edges} of {length_wrong.size} check edges differ in a length (at most {length_margin} may)")
    return not problems, worst, "; ".join(problems)


@dataclass
class NetworkEmbeddings:
    """The embeddings that describe the network for this run (see choose_network_embeddings).

    path is the file the calibration alignment reads, label its base name, model the model it records,
    group the open /embeddings group the similarity is pooled from (None: the run's own embeddings),
    handle the open file when it is a given one, given whether INPUT_NETWORK_EMBED supplied it, and problem
    why calibration is unavailable with it ("" when nothing stands in the way).
    """

    path: str
    label: str
    model: str
    group: object = None
    handle: object = None
    given: bool = False
    problem: str = ""

    def description(self):
        return f"{self.label} (model {self.model})" if self.model else self.label


def choose_network_embeddings(given_path, own_path, own_model, network_model, safe_names):
    """Decide which embeddings describe the network: the given second file, or the run's own.

    given_path is INPUT_NETWORK_EMBED resolved ("" when empty), own_path and own_model the run's embeddings
    and the model they record, network_model the model the network records, safe_names the run's dataset names.
    A given file is accepted if (1) it records a model and it is the network's, and (2) it holds every sequence of the run.
    Not accepted: it is not used at all (the run's own embeddings make the similarity), with problem set, so that
    calibration is unavailable and nothing is aligned. Without a given file, the run's own embeddings of a model
    other than the network's make calibration unavailable the same way, naming the remedy; an own file that
    records no model, or a network that records none, leaves the self-check alone to decide.
    """
    own_model = (own_model or "").strip()
    network_model = (network_model or "").strip()
    own = NetworkEmbeddings(own_path, os.path.basename(own_path), own_model)
    if not given_path:
        if own_model and network_model and own_model != network_model:
            own.problem = (
                f"the run's embeddings ({own.label}) are of model {own_model}, but the network was made with "
                f"{network_model}; give the network's embeddings in INPUT_NETWORK_EMBED"
            )
        return own
    label = os.path.basename(given_path)
    problem, handle, model = "", None, ""
    try:
        handle = open_embeddings_shared(given_path)
        manifest = read_embedding_manifest(handle, require_complete=True, validate_embeddings=False)
        model = (manifest.model_name or "").strip()
        if not model:
            problem = f"INPUT_NETWORK_EMBED {label} records no embedding model"
        elif not network_model:
            problem = f"the network records no model, so INPUT_NETWORK_EMBED {label} (model {model}) cannot be checked against it"
        elif model != network_model:
            problem = f"INPUT_NETWORK_EMBED {label} is of model {model}, but the network was made with {network_model}"
        else:
            group = handle["embeddings"]
            missing = [name for name in safe_names if name not in group]
            if missing:
                problem = (
                    f"INPUT_NETWORK_EMBED {label} lacks {len(missing):,} of the run's sequences "
                    f"(for example {missing[0]})"
                )
    except Exception as error:
        problem = f"INPUT_NETWORK_EMBED {label} could not be read ({type(error).__name__}: {error})"
    if problem:
        if handle is not None:
            handle.close()
        own.problem = problem
        return own
    return NetworkEmbeddings(given_path, label, model, handle["embeddings"], handle, True)


def calibration_workers(pairs, lengths):
    """How many CPU pool children the calibration alignment gets: one per CALIBRATION_CELLS_PER_WORKER cells of work.

    cells is the sum over the pairs of length_i x length_j; the count is at least 1 and at most the default
    thread count ("all but two CPUs"), so a small sample does not pay for starting a child per CPU.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    lengths = np.asarray(lengths, dtype=np.int64)
    cells = int(np.sum(lengths[pairs[:, 0]] * lengths[pairs[:, 1]]))
    wanted = max(1, -(-cells // CALIBRATION_CELLS_PER_WORKER))
    return min(Numba_Threads.default_thread_count(), wanted)


def align_calibration_pairs(pairs, embeddings_path, names, gap_penalties, precision, lengths):
    """Align pairs of the run's sequences with the network's scoring: Align_Similarity_Matrix's own path and planner.

    pairs is an (k, 2) array of indices into names (the run's embedding dataset names), embeddings_path
    the embeddings file that describes the network, gap_penalties the network's [local, global], precision
    its matmul precision and lengths the run's sequence lengths (they size the CPU pool, see
    calibration_workers). DEVICE_SELECTION decides the device: a named one is used without timing, with
    "auto" the planner is timed on a few pairs, but only when the sample is large enough for every default CPU
    worker: a smaller one goes to the CPU pool without timing, because timing the devices would cost about what
    the pool needs (the line printed says which). Returns Align_Similarity_Matrix.PairAlignment.
    """
    align = Align_Library.load()
    workers = calibration_workers(pairs, lengths)
    device_selection = DEVICE_SELECTION
    if Hardware_Utils.normalize_device_selection(device_selection) == Hardware_Utils.AUTO_DEVICE:
        default = Numba_Threads.default_thread_count()
        count = len(np.asarray(pairs).reshape(-1, 2))
        if workers < default:
            device_selection = "cpu"
            print(
                f"[Hardware] Calibration alignment of {count:,} pairs on the CPU, without timing the devices: the sample "
                f"needs {workers} CPU worker(s), fewer than the {default} the CPU pool may use, so the pool finishes it "
                "sooner than timing the devices would."
            )
        else:
            print(
                f"[Hardware] Calibration alignment of {count:,} pairs: the sample is large enough for all {default} "
                "CPU workers, so the devices are timed on a small share of it and the fastest plan is used."
            )
    return align.align_pairs_in_memory(
        pairs, embeddings_path, names, gap_penalties, precision,
        device_selection=device_selection, workers=workers,
    )


def calibrate_sparse_run(similarity, edge_i, edge_j, names, gap_penalties, precision, check_edges,
                         embeddings_path, lengths, is_blast=False, embeddings=None):
    """Draw, align and check the calibration sample of a sparse run's missing pairs; return a CalibrationResult.

    With CALIBRATED_IMPUTATION off nothing is drawn or aligned. With it on: first, fewer than
    CALIBRATION_MIN_MISSING_FRACTION of the run's pairs missing means it is not applied; then
    a network that does not record its gap penalties (or is a BLAST network) is unavailable; then the sample is the
    run's missing pairs by Prefilter_Calibration.select_among (all of them when they are no more than the sample
    size). The sample and the check edges go through one call of align_calibration_pairs; if the check edges are
    not reproduced, or the alignment fails, calibration is unavailable and nothing is added.
    similarity is the run's (n, n) adjusted similarity, edge_i and edge_j its observed pairs, names the dataset
    names of its sequences, check_edges the CheckEdges of choose_check_edges (or None), lengths the
    sequences' lengths (they size the alignment's CPU pool). embeddings, a NetworkEmbeddings, says which file
    the sample is aligned from (its path replaces embeddings_path) and, through its problem, that calibration
    is unavailable before anything is aligned (a model other than the network's, a given file refused).
    """
    if not CALIBRATED_IMPUTATION:
        return CalibrationResult(CalibrationOutcome("off", "the calibration switch is off"))
    n = len(names)
    total = n * (n - 1) // 2
    present = Prefilter_Calibration.pair_presence(edge_i, edge_j, n)
    missing = Prefilter_Calibration.missing_count(present)
    share = missing / total if total else 0.0
    if share < CALIBRATION_MIN_MISSING_FRACTION:
        del present
        return CalibrationResult(CalibrationOutcome(
            "not_applied",
            f"calibration not applied: {100.0 * share:.1f}% of pairs missing, below "
            f"{100.0 * CALIBRATION_MIN_MISSING_FRACTION:g}%",
            missing=missing, share=share,
        ))
    used = ""
    if embeddings is not None:
        embeddings_path = embeddings.path
        used = embeddings.description()
    unavailable = None
    if embeddings is not None and embeddings.problem:
        unavailable = embeddings.problem
    elif is_blast:
        unavailable = "a BLAST network's scores were not made by this program's alignment"
    elif gap_penalties is None:
        unavailable = "the network does not record the gap penalties its scores were made with"
    elif check_edges is None:
        unavailable = "no edge of the network could be used to check the alignment"
    if unavailable is not None:
        del present
        return CalibrationResult(CalibrationOutcome(
            "unavailable", f"calibration unavailable: {unavailable}", missing=missing, share=share,
        ))
    values, locate = Prefilter_Calibration.missing_pairs(similarity, present)
    del present
    sample = Prefilter_Calibration.select_among(values, locate)
    del values, locate
    pairs = np.concatenate([
        np.column_stack([sample.rows, sample.cols]),
        np.column_stack([check_edges.rows, check_edges.cols]),
    ])
    try:
        aligned = align_calibration_pairs(pairs, embeddings_path, names, gap_penalties, precision, lengths)
    except Exception as error:
        return CalibrationResult(CalibrationOutcome(
            "unavailable", f"calibration unavailable: the sample could not be aligned ({type(error).__name__}: {error})",
            missing=missing, share=share,
        ))
    reproduced, difference, problems = compare_with_stored(aligned, len(sample), check_edges.stored)
    outcome = CalibrationOutcome(
        "applied" if len(sample) < missing else "all_aligned",
        sample=len(sample), missing=missing, share=share, device=aligned.device, plan=aligned.plan,
        precision=aligned.precision, seconds=aligned.seconds, check_edges=len(check_edges.rows),
        check_difference=difference, notes=list(aligned.notes), tolerance=check_tolerance(aligned)[3],
        embeddings=used,
    )
    if not reproduced:
        outcome.status = "unavailable"
        outcome.reason = (
            "calibration unavailable: the network's own edges were not reproduced by "
            + (f"the embeddings {used}" if used else "this run's embeddings") + " and "
            f"the network's scoring ({problems}"
            + (f"; {outcome.tolerance}" if outcome.tolerance else "") + ")"
        )
        outcome.sample = 0
        return CalibrationResult(outcome)
    scores = {name: np.asarray(getattr(aligned, name))[:len(sample)] for name in CHECK_FIELDS}
    return CalibrationResult(outcome, sample, scores)


def calibration_text(outcome):
    """The calibration lines of a run's log, and the value of its CALIBRATION note, as one sentence."""
    if outcome.status == "off":
        return "Calibration switched off: no sample is drawn or aligned, so the fit takes the observed pairs only"
    if outcome.status in ("not_applied", "unavailable"):
        text = outcome.reason[:1].upper() + outcome.reason[1:] + "; the missing pairs get the uniform fill"
        for note in outcome.notes:
            text += f". {note}"
        return text
    bounds = f"{Prefilter_Calibration.DRAW_MIN_PAIRS:,} to {Prefilter_Calibration.DRAW_MAX_PAIRS:,}"
    if outcome.status == "all_aligned":
        text = (
            f"All {outcome.missing:,} missing pairs ({100.0 * outcome.share:.1f}% of the run's pairs) are no more "
            f"than the sample size, so all were aligned; the network is complete now and nothing is imputed"
        )
    else:
        text = (
            f"Calibration sample of {outcome.sample:,} of the {outcome.missing:,} missing pairs "
            f"({100.0 * outcome.share:.1f}% of the run's pairs; "
            f"{100.0 * Prefilter_Calibration.FRACTION:g}% of them within {bounds}, "
            f"{Prefilter_Calibration.BINS} similarity bands, seed {Prefilter_Calibration.SEED})"
        )
    text += (
        f", aligned on {outcome.device} ({outcome.plan}, {outcome.precision}) in {outcome.seconds:.1f} s; "
        f"self-check on {outcome.check_edges} of the network's own edges"
        + (f", aligned from the network's embeddings {outcome.embeddings}" if outcome.embeddings else "") + ": reproduced "
        f"(largest score difference {outcome.check_difference:.2g}"
        + (f"; {outcome.tolerance}" if outcome.tolerance else "") + ")"
    )
    for note in outcome.notes:
        text += f". {note}"
    return text


# ==========================================
# REGRESSION CHART
# ==========================================
# The regression chart's title, which names its window too.
_REGRESSION_TITLE = Message("Isotonic Regression Fit")
# The shades of the pairs' hexagons, from one pair to the most: the Tools
# theme's (Studio_Theme tokens), from the statistics report's histogram tint
# to the theme colour's darkest text shade.
_DENSITY_SHADES = ("histogram", "separator", "accent", "accent_text")
# Hexagons across and up the plot; for a 10 x 6 inch window's plot, about
# twice as wide as it is tall, these keep them about regular.
_DENSITY_GRID = (64, 20)
# The fit's line, in points, and the ring in the chart's surface colour
# around it, which keeps it apart from the hexagons it crosses.
_FIT_WIDTH = 2.25
_FIT_RING = 2.0


def regression_statistics(rho, r2, pairs):
    """One line on the fit: Spearman's rho, R² and how many observed pairs they were measured on.

    A Message: str() is the English line, and display() the chart's, in the
    Tools window's language. The numbers come formatted as the terminal's
    fit diagnostics print them, the count with thousands separators.
    """
    numbers = {"rho": f"{rho:.4f}", "r2": f"{r2:.4f}", "count": f"{pairs:,}"}
    if pairs == 1:
        return Message("Spearman ρ {rho} · R² {r2} · {count} observed pair", **numbers)
    return Message("Spearman ρ {rho} · R² {r2} · {count} observed pairs", **numbers)


def calibration_statistics(check):
    """The chart's line on the held-out check: Spearman's rho, MAE and R² on the calibration pairs left out."""
    numbers = {
        "count": f"{check.pairs:,}", "rho": f"{check.spearman:.4f}", "mae": f"{check.mae:.4f}",
        "r2": f"{check.r2:.4f}",
    }
    return Message(
        "Held-out check on {count} calibration pairs · Spearman ρ {rho} · MAE {mae} · R² {r2}",
        **numbers,
    )


def _count_ticks(top):
    """The colour bar's ticks for 1 to top pairs: the powers of ten, or 1 and top below 10."""
    ticks = [10 ** power for power in range(int(np.log10(top)) + 1)]
    return ticks if len(ticks) > 1 else [1, top]


def plot_regression_fit(iso_reg, x, y, rho, r2, pairs, calibration=None):
    """Show the fit over the observed pairs as a pop-up chart in the Tools window's look.

    x and y are the length-adjusted similarities and normalized scores the
    fit was made on: every observed pair, or a random sample of them; pairs
    counts them all. The pairs show as hexagons shaded by how many fall in
    each, in the Tools window's theme colour. The fit is drawn through its
    breakpoints (X_thresholds_, y_thresholds_), between which it predicts
    linearly, so the line is the very function the imputation uses. rho and r2
    head the chart. calibration, an ImputationFit with a calibration sample,
    adds that sample's pairs as dots (they are among the pairs counted, but
    not in x and y) and a line on its held-out check. Its texts show in the
    Tools window's language (Tool_Charts.chart_translation), and its window
    takes the Tools window's style (Tool_Charts.show_chart).
    """
    language, translator = Tool_Charts.chart_translation("regression plot")
    previous = set_translator(translator)
    try:
        _plot_regression_fit(iso_reg, x, y, rho, r2, pairs, language, calibration)
    finally:
        set_translator(previous)


# The calibration pairs' dots: their area in points², and how opaque they are.
_CALIBRATION_DOT = 9
_CALIBRATION_ALPHA = 0.6
# The space between the lines above the plot, in points.
_HEADER_LINE = 16


def _plot_regression_fit(iso_reg, x, y, rho, r2, pairs, language, calibration=None):
    """plot_regression_fit's chart, its texts in language with the translator installed."""
    Tool_Charts.use_qt_backend()
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter
    from desktop.Studio_Theme import theme_tokens

    color = Tool_Charts.tools_theme_color()
    tokens = theme_tokens(color)
    families = Tool_Charts.plot_font_families(PROJECT_ROOT, language)
    shades = LinearSegmentedColormap.from_list(
        "pairs", [tokens[name] for name in _DENSITY_SHADES]
    )
    with plt.rc_context(Tool_Charts.chart_style(tokens, families)):
        # The figure's label names the pop-up window.
        figure, axes = plt.subplots(
            num=_REGRESSION_TITLE.display(), figsize=(10, 6), facecolor=tokens["surface"]
        )
        axes.set_facecolor(tokens["surface"])
        # Thin edges in each hexagon's own colour close the hairline seams between them.
        cells = axes.hexbin(
            x, y, gridsize=_DENSITY_GRID, mincnt=1, cmap=shades, norm=LogNorm(),
            linewidths=0.2, edgecolors="face",
        )
        # A colour bar needs two counts apart, even when every hexagon holds one pair.
        top = max(2, int(cells.get_array().max()))
        cells.set_clim(1, top)
        ring = [
            patheffects.Stroke(linewidth=_FIT_WIDTH + 2 * _FIT_RING, foreground=tokens["surface"]),
            patheffects.Normal(),
        ]
        sampled = calibration is not None and calibration.calibration_x is not None
        if sampled:
            axes.scatter(
                calibration.calibration_x, calibration.calibration_y, s=_CALIBRATION_DOT,
                color=tokens["text_muted"], alpha=_CALIBRATION_ALPHA, linewidths=0, zorder=2,
            )
        axes.plot(
            iso_reg.X_thresholds_, iso_reg.y_thresholds_, color=tokens["text"],
            linewidth=_FIT_WIDTH, solid_joinstyle="round", solid_capstyle="round",
            path_effects=ring, zorder=3,
        )

        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.grid(True, color=tokens["border"], linewidth=0.8)
        axes.set_axisbelow(True)
        axes.tick_params(axis="y", length=0, pad=6)
        axes.tick_params(axis="x", length=4)
        axes.set_xlabel(Message("Length-adjusted embedding cosine similarity").display(), labelpad=8)
        axes.set_ylabel(Message("Normalized network score").display(), labelpad=8)

        colorbar = figure.colorbar(cells, ax=axes, aspect=40, pad=0.015, fraction=0.03)
        colorbar.outline.set_visible(False)
        colorbar.ax.yaxis.set_major_locator(FixedLocator(_count_ticks(top)))
        colorbar.ax.yaxis.set_minor_locator(NullLocator())
        colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        colorbar.ax.tick_params(length=0, pad=6)
        colorbar.set_label(Message("Pairs per hexagon").display(), labelpad=8)

        # Above the plot, bottom up: the legend, the statistics lines, the title.
        # The hexagons hold the observed pairs besides the calibration sample.
        # The sample's pairs are among the pairs counted, but not in x and y.
        others = pairs - (calibration.calibration_x.size if sampled else 0)
        if len(x) < others:
            observed = Message("Observed pairs (random sample of {count})", count=f"{len(x):,}")
        else:
            observed = Message("Observed pairs")
        keys = [
            Line2D([], [], linestyle="none", marker="h", markersize=10,
                   markerfacecolor=shades(0.6), markeredgecolor="none"),
        ]
        labels = [observed.display()]
        if sampled:
            keys.append(Line2D([], [], linestyle="none", marker="o", markersize=4,
                               markerfacecolor=tokens["text_muted"], markeredgecolor="none"))
            labels.append(Message("Calibration pairs, a sample of the missing pairs").display())
        keys.append(Line2D([], [], color=tokens["text"], linewidth=_FIT_WIDTH))
        labels.append(Message("Isotonic fit").display())
        axes.legend(
            keys, labels,
            loc="lower left", bbox_to_anchor=(0, 1), ncols=len(keys), frameon=False,
            borderaxespad=0.6, borderpad=0, handlelength=1.6, columnspacing=1.8,
            labelcolor=tokens["text_soft"],
        )
        lines = [regression_statistics(rho, r2, pairs)]
        if sampled and calibration.check is not None:
            lines.append(calibration_statistics(calibration.check))
        for row, line in enumerate(reversed(lines)):
            axes.annotate(
                line.display(), (0, 1), xycoords="axes fraction",
                xytext=(0, 30 + _HEADER_LINE * row), textcoords="offset points", va="bottom",
                color=tokens["text_muted"],
            )
        axes.set_title(
            _REGRESSION_TITLE.display(), loc="left", pad=50 + _HEADER_LINE * (len(lines) - 1),
            fontsize=15, fontweight=Tool_Charts.title_weight(families), color=tokens["text"],
        )

        figure.tight_layout()
        Tool_Charts.show_chart(figure, color, PROJECT_ROOT)

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
            INPUT_NETWORK_EMBED,
        )
    except MSAConfigurationError as error:
        sys.exit(f"❌ Configuration Error: {error}")

    given_network_embed = resolved["full_input_network_embed"]  # "" when INPUT_NETWORK_EMBED is empty
    FULL_INPUT_FASTA = resolved["full_input_fasta"]
    FULL_INPUT_EMBED = resolved["full_input_embed"]
    FULL_INPUT_NETWORK = resolved["full_input_network"]
    _seq_set = resolved["sequence_set"]

    # 1. LOAD & VALIDATE INPUTS
    print("--- Loading & Validating Inputs ---")
    
    # A. Open HDF5 files and run rigorous embedding file validation
    print("Opening HDF5 data...")
    try:
        # Opened as the alignment path opens embeddings (SWMR read): the calibration alignment, run in this
        # process while the file is open, opens it again, and HDF5 refuses a second open with another flag.
        f_emb = open_embeddings_shared(FULL_INPUT_EMBED)
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
    # How the network's scores were made, which the calibration sample is aligned with.
    network_gap_penalties = f_net.attrs.get("gap_penalties")
    network_precision = f_net.attrs.get("matmul_precision", "ieee_fp32")
    if isinstance(network_precision, bytes):
        network_precision = network_precision.decode("utf-8")
    # The embeddings that describe the network: today the run's own embeddings.
    network_embeddings = FULL_INPUT_EMBED
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

    if num_seqs < 2:
        sys.exit(
            f"❌ Error: Only {num_seqs} sequence was kept after matching the inputs, and an "
            "alignment needs at least two. Use a network, embeddings and (if filtering) a FASTA "
            "file that share more sequences."
        )

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

    # A few of the network's own edges, to be reproduced by the calibration alignment (chosen before the
    # index arrays are mapped and freed below).
    check_edges = None
    if CALIBRATED_IMPUTATION and network_metadata.network_type != "blast":
        check_edges = choose_check_edges(
            f_net, arr_i, arr_j, net_old_to_new, "g_score" if ALIGNMENT_SCORE == "global" else "l_score",
        )

    # Filter with whole-array operations. A Python tuple per edge costs about
    # 130 bytes, so a complete 44k-sequence network (~974M edges) would need
    # ~125 GB. Each network index array is freed as soon as its int32 mapping
    # exists. Edge order is kept, so duplicate pairs resolve as before.
    edge_i = map_network_indices(arr_i, net_old_to_new)
    del arr_i
    edge_j = map_network_indices(arr_j, net_old_to_new)
    del arr_j
    if edge_i.size and (
        edge_i.min() < 0 or edge_j.min() < 0 or np.any(edge_i == edge_j)
    ):
        keep = (edge_i >= 0) & (edge_j >= 0) & (edge_i != edge_j)
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
    if num_edges == 0 and num_seqs >= 2:
        sys.exit(
            f"❌ Error: The {num_seqs} sequences kept share no edge in the network, so there "
            "are no scores to build a guide tree from. Use sequences that have alignments "
            "against each other in the network, or a network that stores them."
        )

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

    total_pairs = int(num_seqs * (num_seqs - 1) / 2)
    is_sparse = num_edges < total_pairs
    cos_sim_mat = None
    calibration = None
    sample = None
    if is_sparse:
        print(f"Network is sparse ({num_edges} / {total_pairs} edges). Activating hybrid cosine-alignment transformation...")

        # The all-vs-all similarity of the run's sequences, by the rule every tool shares.
        safe_names = [h.replace("/", "_").replace("\\", "_") for h in valid_headers]
        # Which embeddings describe the network: INPUT_NETWORK_EMBED when it is given and accepted, else the run's own.
        network_embed = None
        if given_network_embed:
            if network_metadata.network_type == "blast":
                print(f"INPUT_NETWORK_EMBED ({os.path.basename(given_network_embed)}) is ignored: a BLAST network was not made from embeddings.")
            elif not CALIBRATED_IMPUTATION:
                print(f"INPUT_NETWORK_EMBED ({os.path.basename(given_network_embed)}) is ignored: calibrated imputation is switched off.")
        if CALIBRATED_IMPUTATION and network_metadata.network_type != "blast":
            network_embed = choose_network_embeddings(
                given_network_embed, FULL_INPUT_EMBED, _model_name, network_metadata.model_name, safe_names,
            )
        similarity_source = f_emb["embeddings"] if network_embed is None or network_embed.group is None else network_embed.group
        print(f"Computing pooled embeddings ({Prefilter_Calibration.POOLING} pooling) for all sequences "
              + (f"from {network_embed.label} " if network_embed is not None and network_embed.given else "")
              + "and their all-vs-all length-adjusted similarities (length ratio * cosine similarity)...")
        cos_sim_mat = Prefilter_Calibration.pair_similarity(
            similarity_source,
            safe_names,
            seq_lens_array,
            progress=lambda names: tqdm(names, desc="Computing Pooled Embeddings"),
        )
        del similarity_source

        # The calibration sample: aligned now, so its pairs are observed pairs of the run from here on.
        result = calibrate_sparse_run(
            cos_sim_mat, edge_i, edge_j, safe_names, network_gap_penalties, network_precision,
            check_edges, network_embeddings, is_blast=network_metadata.network_type == "blast",
            lengths=seq_lens_array, embeddings=network_embed,
        )
        if network_embed is not None and network_embed.handle is not None:
            network_embed.handle.close()
        del network_embed
        calibration = result.outcome
        print(calibration_text(calibration))
        if result.sample is not None:
            sample = result.sample
            edge_i = np.concatenate([edge_i, sample.rows.astype(edge_i.dtype)])
            edge_j = np.concatenate([edge_j, sample.cols.astype(edge_j.dtype)])
            if ALIGNMENT_SCORE == "global":
                sample_scores, sample_lens = result.scores["g_score"], result.scores["g_len"]
            else:
                sample_scores, sample_lens = result.scores["l_score"], result.scores["l_len"]
            raw_scores = np.concatenate([raw_scores, sample_scores.astype(np.float32)])
            align_lens = np.concatenate([align_lens, sample_lens.astype(np.float32)])
            sample_start = num_edges
            num_edges = len(edge_i)
            del sample_scores, sample_lens
            if calibration.status == "all_aligned":
                is_sparse = False
                cos_sim_mat = None  # nothing is imputed, so nothing reads it again
        del result, check_edges
    elif given_network_embed:
        print(f"INPUT_NETWORK_EMBED ({os.path.basename(given_network_embed)}) is ignored: the network is complete, so nothing is imputed or calibrated.")

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

    iso_reg = None
    imputation_summary = None
    chain_plan = None

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
        # Extract overlapping pairs
        X_cos = cos_sim_mat[edge_i, edge_j]
        Y_align = norm_scores

        # The calibration sample's pairs are the last of the observed pairs, in the sample's order.
        calibration_positions = calibration_weights = None
        if sample is not None:
            calibration_positions = np.arange(sample_start, num_edges)
            calibration_weights = sample.weights

        print("Fitting Isotonic Regression (Adjusted Similarity -> Alignment Score)...")
        # The pick of observed pairs is made in an order that does not depend on the network file's edge order.
        pair_keys = canonical_pair_keys(edge_i, edge_j, valid_headers)
        imputation = fit_imputation(X_cos, Y_align, calibration_positions, calibration_weights, pair_keys=pair_keys)
        del pair_keys
        iso_reg = imputation.iso

        # Evaluate fitness on the observed pairs, the calibration sample's among them.
        fitted_x, fitted_y = X_cos, Y_align
        Y_pred = iso_reg.predict(fitted_x)
        rho, _ = spearmanr(fitted_x, fitted_y)
        r2 = r2_score(fitted_y, Y_pred)

        print(f"Isotonic Regression Fit Diagnostics:")
        print(f"  > Spearman Rank Correlation (rho): {rho:.4f}")
        print(f"  > Coefficient of Determination (R^2): {r2:.4f}")
        missing_pairs = total_pairs - num_edges
        imputation_summary = ImputationSummary(
            missing=missing_pairs,
            outside=outside_fit_fraction(cos_sim_mat, iso_reg, X_cos, missing_pairs),
            check=imputation.check,
            calibration_pairs=0 if imputation.calibration_x is None else imputation.calibration_x.size,
            uniform_value=uniform_fill_value(iso_reg),
            calibration="" if sample is not None or calibration is None else calibration.reason,
        )
        if not imputation.uniform:
            # A fit made with a calibration sample: the chain fill replaces the plain isotonic fill.
            chain_plan = prepare_chain_fill(
                num_seqs, edge_i, edge_j, Y_align, calibration_positions, imputation, imputation_summary,
            )
        report_imputation(imputation, imputation_summary)

        if SHOW_REGRESSION_PLOT:
            try:
                print("Displaying Isotonic Regression plot. Close the plot window to continue...")
                plot_regression_fit(
                    iso_reg, imputation.fit_x, imputation.fit_y, rho, r2, len(fitted_x),
                    calibration=imputation,
                )
            except Exception as plot_err:
                print(f"Could not open plot window: {plot_err}")
        # The fit and its report are done, so release the per-edge arrays that only
        # they read. The n x n arrays below are released the same way, each right
        # after its last use. Otherwise they all stay alive through the replicate
        # trees and the whole merge.
        del fitted_x, fitted_y, Y_pred, X_cos, Y_align, norm_scores, calibration_positions, calibration_weights
        del imputation, sample

    # 8. BUILD CONSENSUS TREE
    tree_building_started = time.perf_counter()
    condensed_size = int(num_seqs * (num_seqs - 1) / 2)
    
    if chain_plan is not None and cos_sim_mat is not None:
        print("Applying chain + calibration imputation for final master tree...")
        D_final_cond = np.empty(condensed_size, dtype=np.float32)
        covered, missing = Chain_Fill.fill_distances(chain_plan, D_final_cond, cos_sim_mat, iso_reg, max_norm_score)
        # The score matrix, lists and bit rows are not read again: free them before the replicates start.
        chain_plan.release()
        del chain_plan, cos_sim_mat
        share = covered / missing if missing else 0.0
        imputation_summary.chain.covered = share
        print(f"  > Missing pairs with a chain: {100.0 * share:.1f}% of {missing:,}")
    elif is_sparse and iso_reg is not None and cos_sim_mat is not None:
        print("Applying hybrid adjusted similarity-alignment imputation for final master tree...")
        # The isotonic-only mode of the row-block writer (the chain fill's other mode): the same bits as the
        # dense matrix this replaced, in one block's buffers.
        D_final_cond = np.empty(condensed_size, dtype=np.float32)
        Chain_Fill.fill_isotonic(D_final_cond, cos_sim_mat, iso_reg, max_norm_score)
        del cos_sim_mat
    else:
        D_final_cond = np.full(condensed_size, MAX_DISTANCE, dtype=np.float32)

    # The regression supplies missing distances, while observed network edges
    # remain authoritative. This complete baseline is used by every replicate
    # so imputed relationships can influence each tree topology.
    populate_condensed_matrix(D_final_cond, num_seqs, edge_i, edge_j, edge_dists)
    np.clip(D_final_cond, 0.0, MAX_DISTANCE, out=D_final_cond)

    # Replicate splits label the saved guide tree with how often each recurs.
    leaf_keys = guide_tree_leaf_keys(num_seqs, RANDOM_SEED)
    replicate_keys = None

    if BOOTSTRAP_TREE:
        print(f"\nBuilding Consensus Tree from {NUM_TREES} bootstrap replicates using {WORKERS} cores...")
        print("Final tree-distance consensus mode: full all-pairs.")
        
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

        # Workers now own the immutable baseline through the memory map.
        # Reuse the main condensed array as the all-pairs accumulator.
        del edge_i
        del edge_j
        del edge_dists
        D_final_cond.fill(0.0)

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
        
        split_keys = []
        # Batch replicate trees (in exact seed order) so L1-resident row accumulation
        # cuts DRAM traffic by ~25x while keeping bit-identical float32 consensus sums.
        consensus_batch = []
        consensus_batch_size = 25
        with mp.Pool(processes=WORKERS) as pool:
            # In seed order, so the float32 sums of the consensus do not depend on which
            # worker finishes first and the guide tree is the same for any WORKERS.
            iterator = pool.imap(worker_func, seeds)
            for Z, replicate_lengths in tqdm(
                iterator, total=NUM_TREES, desc="Bootstrapping Trees"
            ):
                split_keys.append(replicate_split_keys(Z, num_seqs, leaf_keys))
                consensus_batch.append((Z, replicate_lengths))
                if len(consensus_batch) >= consensus_batch_size:
                    add_replicates_batch_to_consensus(
                        consensus_batch, num_seqs, D_final_cond, threads=worker_threads
                    )
                    consensus_batch.clear()
        if consensus_batch:
            add_replicates_batch_to_consensus(consensus_batch, num_seqs, D_final_cond)
            consensus_batch.clear()
        replicate_keys = np.sort(np.concatenate(split_keys))
        del split_keys

        print("Building final master tree...")
        finalize_cophenetic_consensus(D_final_cond, NUM_TREES)

        # --- CLEANUP ---
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

    nj_branch_lengths = None
    if TREE_METHOD == "Neighbor-joining (Slow)":
        linkage_matrix, nj_branch_lengths = neighbor_joining_condensed(
            D_final_cond, num_seqs, return_branch_lengths=True
        )
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

    # Saved before the hours-long merge so a failed alignment keeps the tree.
    guide_tree_path = build_guide_tree_path(OUTPUT_FASTA)
    os.makedirs(os.path.dirname(guide_tree_path), exist_ok=True)
    print(f"Saving guide tree to {guide_tree_path}...")
    write_guide_tree(
        guide_tree_path,
        linkage_matrix,
        valid_headers,
        nj_branch_lengths,
        leaf_keys,
        replicate_keys,
        NUM_TREES if BOOTSTRAP_TREE else 0,
        guide_tree_notes(network_metadata.network_type, is_sparse, imputation_summary, calibration),
    )
    del leaf_keys, replicate_keys

    ranked_devices = benchmark_msa_devices(
        linkage_matrix,
        valid_headers,
        f_emb["embeddings"],
        DEVICE_SELECTION,
    )

    # 9. INITIALIZE CLUSTERS (Deferred lazy tree-replay: zero intermediate alignment copies!)
    print("Initializing clusters...")
    # Keep leaf_alignments fitted lazily as leaves are first touched; store only
    # live intermediate profiles in active_profiles (bounded to O(log N) by heavy-first DFS).
    active_profiles = {}
    active_widths = {i: leaf_alignments[i].shape[0] for i in range(num_seqs)}
    active_counts = {i: 1 for i in range(num_seqs)}
    num_merges = len(linkage_matrix)
    merge_children = [None] * num_merges
    merge_moves = [None] * num_merges

    # 10. PROGRESSIVE ALIGNMENT
    print(f"Aligning {num_seqs} sequences...")
    cluster_merging_started = time.perf_counter()
    embeddings_group = f_emb["embeddings"]

    def _get_node_profile(node_idx):
        prof = active_profiles.pop(node_idx, None)
        if prof is not None:
            return prof
        header = valid_headers[node_idx]
        safe_h = header.replace("/", "_").replace("\\", "_")
        raw = embeddings_group[safe_h][:]
        prof = _normalize_residue_embeddings(raw)
        leaf_alignments[node_idx] = fit_leaf_to_embedding(leaf_alignments[node_idx], prof.shape[0])
        active_widths[node_idx] = leaf_alignments[node_idx].shape[0]
        return prof

    merge_order = heavy_first_merge_order(linkage_matrix, num_seqs)
    for iteration in tqdm(merge_order, desc="Merging Clusters"):
        link = linkage_matrix[iteration]
        idx_a = int(link[0])
        idx_b = int(link[1])

        emb_a = _get_node_profile(idx_a)
        emb_b = _get_node_profile(idx_b)
        width_a = active_widths.pop(idx_a)
        width_b = active_widths.pop(idx_b)
        count_a = active_counts.pop(idx_a)
        count_b = active_counts.pop(idx_b)

        # --- ALIGNMENT ---
        score_mat = compute_score_matrix_with_fallback(
            emb_a,
            emb_b,
            ranked_devices,
            DEVICE_SELECTION,
        )
        path = run_global_traceback(score_mat, GAP_OPEN, GAP_EXTEND)

        merged_profile, moves = merge_profiles_lazy(
            width_a, count_a, width_b, count_b, path, emb_a, emb_b
        )
        new_idx = num_seqs + iteration
        active_profiles[new_idx] = merged_profile
        active_widths[new_idx] = moves.shape[0]
        active_counts[new_idx] = count_a + count_b
        merge_children[iteration] = (idx_a, idx_b)
        merge_moves[iteration] = moves

        del emb_a
        del emb_b
    cluster_merging_seconds = time.perf_counter() - cluster_merging_started

    # 11. SAVE (O(N * W_final) top-down tree replay directly to FASTA)
    root_idx = num_seqs + len(linkage_matrix) - 1
    final_width = active_widths[root_idx]
    active_profiles.clear()
    print(f"Saving Consensus MSA to {OUTPUT_FASTA}...")
    write_lazy_alignment_fasta(
        OUTPUT_FASTA,
        root_idx,
        num_seqs,
        merge_children,
        merge_moves,
        leaf_alignments,
        valid_headers,
        final_width,
    )
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
