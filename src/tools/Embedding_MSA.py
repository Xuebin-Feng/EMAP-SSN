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
from numba import jit
from tqdm import tqdm
import sys
from utilities import Hardware_Utils
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from Cache_Manifest import validate_network_schema
from utilities.FASTA_Sanitization import load_sanitized_fasta
from utilities.Embedding_HDF5 import read_embedding_manifest


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
from utilities.Tool_Directories import project_directory_defaults

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]
NETWORK_DIR = _DEFAULT_DIRECTORIES["NETWORK_DIR"]
MSA_DIR = _DEFAULT_DIRECTORIES["MSA_DIR"]
SAFE_TEMP_DIR = MSA_DIR

# Alignment Settings
GAP_OPEN = -0.5
GAP_EXTEND = 0.0           
WORKERS = 1   
SHOW_REGRESSION_PLOT = False
POOLING_METHOD = "max"    # ("mean", "max") - method to pool residue embeddings into sequence vectors
LENGTH_RATIO_POWER = 2.0  # (float) - exponent to scale the sequence length ratio penalty

# --- JSON Settings Override ---
import json
import ast
import os

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
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
                SAFE_TEMP_DIR = MSA_DIR
                        
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
    """Raised when the MSA utility cannot resolve its configured inputs."""


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


FULL_INPUT_FASTA = ""
FULL_INPUT_EMBED = ""
FULL_INPUT_NETWORK = ""
OUTPUT_FASTA = ""
_seq_set = ""
_model_name = ""

DEVICE = Hardware_Utils.get_optimal_device()

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


class MSACluster:
    def __init__(self, idx, sequences, ids, embedding=None):
        self.idx = idx
        self.sequences = sequences   
        self.ids = ids               
        # For merged nodes, each row is the average of unit residue vectors
        # across every sequence in the cluster. Gaps contribute zero, so the
        # row norm retains both column occupancy and residue agreement.
        self.embedding = embedding
        self.is_leaf = embedding is None

    def get_embedding(self, h5_path, valid_headers):
        """Return profile vectors, lazily initializing leaf residue vectors."""
        if self.embedding is not None:
            return self.embedding
        
        # If leaf, open the file, fetch the single array, and close the file
        with h5py.File(h5_path, "r") as f:
            header = valid_headers[self.ids[0]]
            safe_h = header.replace("/", "_").replace("\\", "_")
            # Leaves enter the profile as unit residue vectors. Subsequent
            # weighted averages then have norms in [0, 1], where a smaller
            # norm represents gaps and/or disagreement within the column.
            return _normalize_residue_embeddings(f["embeddings"][safe_h][:])

# ==========================================
# HELPER: FASTA LOADER & WORKER
# ==========================================
def compute_single_tree_worker(seed, num_seqs, baseline_dist_path, max_dist, noise_scale, tree_method):
    """Build one replicate tree from the complete shared distance baseline."""
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
    D_perturbed_cond = np.empty(condensed_size, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sigma = float(noise_scale) * float(max_dist)
    noise_chunk_size = 1_000_000

    if sigma == 0.0:
        D_perturbed_cond[:] = baseline_dist[:]
    else:
        for start in range(0, condensed_size, noise_chunk_size):
            end = min(start + noise_chunk_size, condensed_size)
            noise = rng.normal(0.0, sigma, size=end - start).astype(np.float32)
            D_perturbed_cond[start:end] = baseline_dist[start:end] + noise

    np.clip(D_perturbed_cond, 0.0, max_dist, out=D_perturbed_cond)
    
    # Run Linkage or Neighbor-joining
    if tree_method == "Neighbor-joining (Slow)":
        Z = neighbor_joining_condensed(D_perturbed_cond, num_seqs)
    else:
        Z = sch.linkage(D_perturbed_cond, method='average')
    
    # Windows requires explicitly closing the memmap before cleanup.
    del baseline_dist
    
    return Z


def generate_bootstrap_seeds(num_trees):
    """Generate the reproducible per-replicate seeds used by bootstrap workers."""
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.integers(0, int(1e9), size=num_trees)

def load_fasta_map(filepath):
    print(f"Loading sequences from {filepath}...")
    seq_dict = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            header = None
            seq_accum = []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if header: seq_dict[header] = "".join(seq_accum)
                    header = line[1:]
                    seq_accum = []
                else: seq_accum.append(line)
            if header: seq_dict[header] = "".join(seq_accum)
    except FileNotFoundError:
        sys.exit(f"❌ Error: FASTA file not found at {filepath}")
    return seq_dict


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
def compute_score_matrix_torch(emb_i, emb_j):
    t_i = torch.as_tensor(emb_i, device=DEVICE, dtype=torch.float32)
    t_j = torch.as_tensor(emb_j, device=DEVICE, dtype=torch.float32)

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
def compute_full_cophenetic(Z, num_seqs):
    """Return float32 cophenetic distances for every condensed sequence pair."""
    condensed_size = int(num_seqs * (num_seqs - 1) / 2)
    coph_dists = np.zeros(condensed_size, dtype=np.float32)
    if num_seqs <= 1:
        return coph_dists

    total_nodes = 2 * num_seqs - 1
    head = np.full(total_nodes, -1, dtype=np.int32)
    tail = np.full(total_nodes, -1, dtype=np.int32)
    next_leaf = np.full(num_seqs, -1, dtype=np.int32)

    for leaf in range(num_seqs):
        head[leaf] = leaf
        tail[leaf] = leaf

    # At each linkage merge, every cross-child leaf pair meets for the first
    # time at this node. Across the complete tree, each pair is written once.
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
                coph_dists[condensed_idx] = merge_height
                leaf_b = next_leaf[leaf_b]
            leaf_a = next_leaf[leaf_a]

        head[parent] = head[child_a]
        tail[parent] = tail[child_b]
        next_leaf[tail[child_a]] = head[child_b]

    return coph_dists


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
    
    for step in range(N - 1):
        if k > 2:
            inv_k_minus_2 = 1.0 / (k - 2)
            min_Q = 1e15
            idx_u = -1
            idx_v = -1
            
            # Precompute normalized divergence values for active nodes
            for i in range(k):
                r_list[i] = R[active_list[i]] * inv_k_minus_2
                
            for i in range(k):
                u = active_list[i]
                r_u = r_list[i]
                for j in range(i + 1, k):
                    v = active_list[j]
                    q = D[u, v] - (r_u + r_list[j])
                    if q < min_Q:
                        min_Q = q
                        idx_u = i
                        idx_v = j
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

def neighbor_joining_condensed(D_condensed, num_seqs):
    D_square = squareform(D_condensed)
    D_allocated = np.zeros((2 * num_seqs - 1, 2 * num_seqs - 1), dtype=np.float64)
    D_allocated[:num_seqs, :num_seqs] = D_square
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
    path = path[::-1]
    new_seqs_a = ["" for _ in cluster_a.sequences]
    new_seqs_b = ["" for _ in cluster_b.sequences]
    merged_vecs = []
    
    idx_a, idx_b = 0, 0
    
    w_a = float(len(cluster_a.ids))
    w_b = float(len(cluster_b.ids))
    total_w = w_a + w_b

    for move in path:
        if move == 1: 
            for i, s in enumerate(cluster_a.sequences): new_seqs_a[i] += s[idx_a]
            for i, s in enumerate(cluster_b.sequences): new_seqs_b[i] += s[idx_b]
            # These are cluster-wide averages of unit residue vectors. The
            # weighted mean preserves occupancy and directional agreement.
            vec = (emb_a[idx_a].astype(np.float32) * w_a + emb_b[idx_b].astype(np.float32) * w_b) / total_w
            merged_vecs.append(vec)
            idx_a += 1; idx_b += 1
            
        elif move == 2: 
            for i, s in enumerate(cluster_a.sequences): new_seqs_a[i] += s[idx_a]
            for i, s in enumerate(cluster_b.sequences): new_seqs_b[i] += "-"
            vec = (emb_a[idx_a].astype(np.float32) * w_a) / total_w
            merged_vecs.append(vec)
            idx_a += 1
            
        elif move == 3: 
            for i, s in enumerate(cluster_a.sequences): new_seqs_a[i] += "-"
            for i, s in enumerate(cluster_b.sequences): new_seqs_b[i] += s[idx_b]
            vec = (emb_b[idx_b].astype(np.float32) * w_b) / total_w
            merged_vecs.append(vec)
            idx_b += 1

    new_cluster = MSACluster(
        idx=-1,
        sequences=new_seqs_a + new_seqs_b,
        # Values remain bounded because they are averages of unit vectors.
        # Downcast to float16 to save RAM for the remaining iterations.
        embedding=np.stack(merged_vecs, axis=0).astype(np.float16),
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

    # 5. BUILD MAPPINGS & FILTER NETWORK EDGES
    print("--- Filtering Network Edges ---")
    
    # Map: Network_Index -> New_Index
    net_old_to_new = {}
    for i, h in enumerate(net_headers):
        if h in header_to_new_idx:
            net_old_to_new[i] = header_to_new_idx[h]

    processed_edges = []

    for k in range(len(arr_i)):
        u_old, v_old = int(arr_i[k]), int(arr_j[k])
        
        if u_old in net_old_to_new and v_old in net_old_to_new:
            u_new = net_old_to_new[u_old]
            v_new = net_old_to_new[v_old]
            processed_edges.append((u_new, v_new, target_score[k], target_len[k]))

    print(f"Retained {len(processed_edges)} edges valid for the intersection.")

    import scipy.sparse as sp

    # 7. PREPARE SPARSE DATA ARRAYS
    print(f"Preparing sparse distance metrics...")
    
    num_edges = len(processed_edges)
    
    # Pre-allocate exactly sized arrays for Numba
    edge_i = np.zeros(num_edges, dtype=np.int32)
    edge_j = np.zeros(num_edges, dtype=np.int32)
    raw_scores = np.zeros(num_edges, dtype=np.float32)
    align_lens = np.zeros(num_edges, dtype=np.float32)
    
    # 7a. Fast Data Unpacking (Moving memory, no math)
    print("Unpacking edges into memory arrays...")
    for k, e in enumerate(tqdm(processed_edges, desc="Unpacking Edges")):
        edge_i[k] = e[0]
        edge_j[k] = e[1]
        raw_scores[k] = e[2]
        align_lens[k] = e[3]
        
    # 7b. Pre-compute Sequence Lengths into a C-compatible array
    num_seqs = len(valid_headers)
    seq_lens_array = np.zeros(num_seqs, dtype=np.int32)
    for idx, h in enumerate(valid_headers):
        seq_lens_array[idx] = len(seq_dict[h])

    # 7c. Map the string mode to an integer for the Numba kernel
    mode_map = {
        "alignment_length": 0,
        "shorter_sequence": 1,
        "longer_sequence": 2,
        "average_sequence": 3
    }
    
    if NORMALIZATION_MODE not in mode_map and not is_evalue:
        raise ValueError(f"❌ Critical Error: Unhandled NORMALIZATION_MODE '{NORMALIZATION_MODE}'. Cannot calculate distance.")
    
    mode_int = mode_map.get(NORMALIZATION_MODE, 0)

    # 7d. Execute C-Speed Kernel
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

    MAX_DISTANCE = max_norm_score + 0.1 

    # Invert scores to distances
    edge_dists = np.maximum(0.0, max_norm_score - norm_scores).astype(np.float32)

    is_sparse = num_edges < int(num_seqs * (num_seqs - 1) / 2)
    iso_reg = None
    cos_sim_mat = None

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
        
        # 1. Define a strictly named, predictable folder in the user-defined Temp directory
        temp_dir = os.path.join(SAFE_TEMP_DIR, f"{_seq_set}_[{_model_name}]_Memmap_Cache")
        
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
        
        worker_func = partial(compute_single_tree_worker, 
                              num_seqs=num_seqs, 
                              baseline_dist_path=baseline_dist_path,
                              max_dist=MAX_DISTANCE, 
                              noise_scale=NOISE_SCALE,
                              tree_method=TREE_METHOD)
        
        with mp.Pool(processes=WORKERS) as pool:
            iterator = pool.imap_unordered(worker_func, seeds)
            for Z in tqdm(iterator, total=NUM_TREES, desc="Bootstrapping Trees"):
                if full_consensus:
                    full_coph = compute_full_cophenetic(Z, num_seqs)
                    D_final_cond += full_coph
                    del full_coph
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
        linkage_matrix = sch.linkage(D_final_cond, method='average')
    
    # --- CLEANUP ---
    del D_final_cond 
    gc.collect()
    tree_building_seconds = time.perf_counter() - tree_building_started

    # 9. INITIALIZE CLUSTERS (No embeddings loaded here!)
    print("Initializing clusters...")
    clusters = {}
    for i in range(num_seqs):
        header = valid_headers[i]
        seq = seq_dict[header]
        
        # Initialize leaves purely with string/ID metadata
        c = MSACluster(idx=i, sequences=[seq], ids=[i], embedding=None)
        clusters[i] = c

    # 10. PROGRESSIVE ALIGNMENT
    print(f"Aligning {num_seqs} sequences...")
    cluster_merging_started = time.perf_counter()
    for iteration, link in enumerate(tqdm(linkage_matrix, desc="Merging Clusters")):
        idx_a = int(link[0])
        idx_b = int(link[1])
        
        cluster_a = clusters.pop(idx_a)
        cluster_b = clusters.pop(idx_b)
        
        # --- LAZY LOAD: Fetch embeddings from disk (or RAM if already merged) ---
        emb_a = cluster_a.get_embedding(FULL_INPUT_EMBED, valid_headers)
        emb_b = cluster_b.get_embedding(FULL_INPUT_EMBED, valid_headers)

        # Handle sequence padding for raw leaf nodes just before alignment
        if cluster_a.is_leaf and len(cluster_a.sequences[0]) != emb_a.shape[0]:
            seq = cluster_a.sequences[0]
            if len(seq) > emb_a.shape[0]: cluster_a.sequences[0] = seq[:emb_a.shape[0]]
            else: cluster_a.sequences[0] = seq.ljust(emb_a.shape[0], "-")
            
        if cluster_b.is_leaf and len(cluster_b.sequences[0]) != emb_b.shape[0]:
            seq = cluster_b.sequences[0]
            if len(seq) > emb_b.shape[0]: cluster_b.sequences[0] = seq[:emb_b.shape[0]]
            else: cluster_b.sequences[0] = seq.ljust(emb_b.shape[0], "-")

        # --- ALIGNMENT ---
        score_mat = compute_score_matrix_torch(emb_a, emb_b)
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
    with open(OUTPUT_FASTA, "w", encoding="utf-8", newline="\n") as f:
        for i, seq_str in enumerate(final_cluster.sequences):
            original_idx = final_cluster.ids[i]
            header = valid_headers[original_idx]
            f.write(f">{header}\n{seq_str}\n")
    print("Done!")
    total_processing_seconds = time.perf_counter() - total_processing_started
    report_processing_times(
        total_processing_seconds,
        tree_building_seconds,
        cluster_merging_seconds,
    )

if __name__ == "__main__":
    run_msa_builder()
