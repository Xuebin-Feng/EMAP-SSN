import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter
import math
import fnmatch
import SSN_Config as cfg

# --- 1. Library Detection ---
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("\nNumba JIT Detected: Acceleration Enabled.")
except ImportError:
    NUMBA_AVAILABLE = False
    print("\nNumba not found. Using standard Python (Slower).")

try:
    import torch
    from utilities import Hardware_Utils
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# --- 2. String & Label Helpers ---
def get_alignment_offset_display(viewer):
    """Return the current offset with an inactive marker when no reference is resolved."""
    alignment = getattr(viewer, 'alignment', None)
    if alignment is not None and getattr(alignment, 'has_reference', False):
        return str(getattr(alignment, 'offset', 0))

    configured_offset = getattr(
        viewer,
        'alignment_offset',
        getattr(cfg, 'ALIGNMENT_OFFSET', 0),
    )
    return f"{configured_offset} (inactive)"


def get_network_suffix():
    suffix = ""
    
    score_mode = getattr(cfg, 'ALIGNMENT_SCORE', None)
    if score_mode:
        suffix += f"_{score_mode}"
        
    norm_mode = getattr(cfg, 'NORM_MODE', None)
    if norm_mode:
        suffix += f"_{norm_mode}"
        
    if cfg.TOP_EDGE_PERCENT is not None:
        suffix += f"_Top{float(cfg.TOP_EDGE_PERCENT)}Pct"
    else:
        suffix += f"_Score{float(cfg.SIMILARITY_THRESHOLD)}"
    return suffix

def get_base_network_name():
    fasta_path = getattr(cfg, 'NODE_FASTA_FILE', None) or getattr(cfg, 'SEQUENCES_FILE', '')
    if fasta_path and isinstance(fasta_path, str):
        fasta_base = os.path.splitext(os.path.basename(fasta_path))[0]
    else:
        fasta_base = "Network"
        
    hdf5_base = os.path.basename(getattr(cfg, 'INPUT_HDF5', ''))
    hdf5_no_ext = hdf5_base[:-3] if hdf5_base.endswith(".h5") else os.path.splitext(hdf5_base)[0]
    
    cfg.INPUT_IS_EVALUE = "EValue" in hdf5_no_ext or "Evalue" in hdf5_no_ext
    
    stripped = re.sub(r'_(network|evalue)$', '', hdf5_no_ext, flags=re.IGNORECASE)
    match = re.search(r'_(e[0-9]+_.*|blast.*)$', stripped, flags=re.IGNORECASE)
    model_str = f"_{match.group(1)}" if match else ""
    
    return f"{fasta_base}{model_str}"

def get_cache_filename():
    """
    Resolves the reference header from the MSA file for graph lookup,
    but ALWAYS uses the clean config string for the filename.
    """
    filename_ref_str = getattr(cfg, 'ALIGNMENT_REFERENCE', "None")
    resolved_ref_full = None
    
    # 1. Resolve Reference Header from FASTA or H5 (Only for graph lookup, NOT for filename)
    msa_file = getattr(cfg, 'MSA_FILE', None)
    if msa_file and os.path.exists(msa_file):
        try:
            if msa_file.endswith(".h5"):
                import h5py
                with h5py.File(msa_file, "r") as hf:
                    if "headers" in hf:
                        raw_headers = hf["headers"][:]
                        ref_str_lower = filename_ref_str.lower() if filename_ref_str else ""
                        for h_bytes in raw_headers:
                            h = h_bytes.decode('utf-8') if isinstance(h_bytes, bytes) else h_bytes
                            if ref_str_lower and ref_str_lower in h.lower():
                                resolved_ref_full = h
                                break
            else:
                with open(msa_file, "r", encoding="utf-8", errors="ignore") as f:
                    ref_str_lower = filename_ref_str.lower() if filename_ref_str else ""
                    for line in f:
                        if line.startswith(">"):
                            # Case-insensitive substring match
                            if ref_str_lower and ref_str_lower in line.lower():
                                resolved_ref_full = line.strip()[1:] # Remove '>'
                                break
        except Exception as e:
            print(f"Utils Warning: Could not resolve reference header: {e}")

    # ---> NEW: MATCH GUI PREDICTION LOGIC EXACTLY <---
    import re
    hdf5_base = os.path.basename(getattr(cfg, 'INPUT_HDF5', ''))
    
    # Extract sequence set from fasta file or fallback
    fasta_file = getattr(cfg, 'NODE_FASTA_FILE', None)
    if fasta_file:
        fasta_base = os.path.splitext(os.path.basename(fasta_file))[0]
    else:
        fasta_base = getattr(cfg, 'SEQUENCE_SET', 'Network')
        
    # Extract model name
    match = re.search(r'(\[.*?\])', hdf5_base)
    if match:
        model_str = f"_{match.group(1)}"
    else:
        hdf5_no_ext = hdf5_base[:-3] if hdf5_base.endswith(".h5") else os.path.splitext(hdf5_base)[0]
        stripped = re.sub(r'_(network|evalue)$', '', hdf5_no_ext, flags=re.IGNORECASE)
        old_match = re.search(r'_(e[0-9]+_.*|blast.*)$', stripped, flags=re.IGNORECASE)
        model_str = f"_{old_match.group(1)}" if old_match else ""
        
    net_prefix = f"{fasta_base}{model_str}"
    
    # Construct Suffixes
    is_blast = "EValue" in hdf5_base or "Evalue" in hdf5_base or "blast" in hdf5_base.lower()
    
    suffix = ""
    if not is_blast:
        norm_m = getattr(cfg, 'NORM_MODE', None)
        if norm_m: suffix += f"_{norm_m}"
        
        score_m = getattr(cfg, 'ALIGNMENT_SCORE', None)
        if score_m: suffix += f"_{score_m}"
    
    is_umap = getattr(cfg, 'UMAP_MODE', False)
    if is_umap:
        umap_k = getattr(cfg, 'UMAP_NEIGHBORS', 15)
        suffix += f"_UMAP_k{int(umap_k)}"
    else:
        # Threshold or Top %
        top_val = getattr(cfg, 'TOP_EDGE_PERCENT', None)
        if top_val is not None and str(top_val).strip() != "None":
            try: suffix += f"_Top{float(top_val)}Pct"
            except: pass
        else:
            thresh = getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0)
            try: suffix += f"_Score{float(thresh)}"
            except: pass
        
    # ---> NEW: Build the folder name and default _ver.00 filename
    base_name = f"{net_prefix}{suffix}"
    saved_layout_dir = getattr(cfg, 'SAVED_LAYOUT_DIR', os.path.join("Cache_Files", "Saved_Layouts"))
    
    target_folder = os.path.join(saved_layout_dir, base_name)
    default_path = os.path.join(target_folder, f"{base_name}_ver.00.h5")
    
    # Check if GUI passed a specific selected cache file from the dropdown
    selected_cache = getattr(cfg, 'TARGET_CACHE_FILE', None)
    if selected_cache and selected_cache.strip() and selected_cache != "None":
        if not selected_cache.startswith("No .h5") and not selected_cache.startswith("Folder does"):
            selected_path = os.path.join(target_folder, selected_cache)
            return selected_path, resolved_ref_full
        
    return default_path, resolved_ref_full

def get_cluster_alignment_dir(viewer):
    """
    Centralized path resolver for the 3 external commands (load.py, label.py, align.py).
    Constructs the standard nested directory path based on current Viewer and Config state.
    """
    if not hasattr(viewer, 'last_cluster_params') or viewer.last_cluster_params is None:
        c_mode_param, c_min = "UNK", "UNK"
    else:
        c_mode_param, c_min = viewer.last_cluster_params

    import re
    hdf5_base = os.path.basename(getattr(cfg, 'INPUT_HDF5', ''))
    
    # --- 1. LEVEL 1: Cache Name until before ALIGNMENT_REFERENCE ---
    fasta_file = getattr(cfg, 'NODE_FASTA_FILE', None)
    if fasta_file:
        fasta_base = os.path.splitext(os.path.basename(fasta_file))[0]
    else:
        fasta_base = getattr(cfg, 'SEQUENCE_SET', 'Network')
        
    match = re.search(r'(\[.*?\])', hdf5_base)
    if match:
        model_str = f"_{match.group(1)}"
    else:
        hdf5_no_ext = hdf5_base[:-3] if hdf5_base.endswith(".h5") else os.path.splitext(hdf5_base)[0]
        stripped = re.sub(r'_(network|evalue)$', '', hdf5_no_ext, flags=re.IGNORECASE)
        old_match = re.search(r'_(e[0-9]+_.*|blast.*)$', stripped, flags=re.IGNORECASE)
        model_str = f"_{old_match.group(1)}" if old_match else ""
        
    lvl1_name = f"{fasta_base}{model_str}"
    
    is_blast = "EValue" in hdf5_base or "Evalue" in hdf5_base or "blast" in hdf5_base.lower()
    if not is_blast:
        norm_m = getattr(cfg, 'NORM_MODE', None)
        if norm_m: lvl1_name += f"_{norm_m}"
        
        score_m = getattr(cfg, 'ALIGNMENT_SCORE', None)
        if score_m: lvl1_name += f"_{score_m}"
        
    # --- 2. LEVEL 2: Rest of Cache Name + Cluster Parameters ---
    lvl2_name = ""
    
    is_umap = getattr(cfg, 'UMAP_MODE', False)
    if is_umap:
        umap_k = getattr(cfg, 'UMAP_NEIGHBORS', 15)
        lvl2_name += f"_UMAP_k{int(umap_k)}"
    else:
        top_val = getattr(cfg, 'TOP_EDGE_PERCENT', None)
        if top_val is not None and str(top_val).strip() != "None":
            try: lvl2_name += f"Top{float(top_val)}Pct"
            except: pass
        else:
            thresh = getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0)
            try: lvl2_name += f"_Score{float(thresh)}"
            except: pass
        
    # Append the cluster parameters
    lvl2_name += f"_{c_mode_param}_Min{c_min}"

    # Return Full Target Path
    return os.path.join(cfg.CLUSTER_ALIGNMENT_DIR, lvl1_name, lvl2_name)

def simplify_node_label(header):
    # 1. Try to match standard NCBI Accession formats
    # - RefSeq: 2 letters, underscore, numbers, optional version (e.g., WP_012345678.1, NP_123456)
    # - GenBank: 3 letters, 5 to 7 numbers, optional version (e.g., AAA12345.1, EAW123456)
    match = re.search(r'\b([A-Z]{2}_\d+(?:\.\d+)?|[A-Z]{3}\d{5,7}(?:\.\d+)?)\b', header)
    if match:
        return match.group(1)
        
    # 2. Legacy fallback for old |gb| flags
    marker = "|gb|"
    if marker in header:
        try: return header.split(marker)[1].split("|")[0]
        except IndexError: pass
        
    # 3. Ultimate fallback: Return the first word (Standard FASTA ID format)
    return header.split()[0] if header else ""

def sort_labels(labels):
    def key_func(k):
        try:
            # Split the label by the decimal point
            parts = str(k).split('.')
            major = int(parts[0])
            # If there's a decimal, grab the integer after it. Otherwise, it's 0.
            minor = int(parts[1]) if len(parts) > 1 else 0
            
            # Return a tuple for sorting (e.g., (188, 10) vs (188, 6))
            return (major, minor)
        except:
            return (0, 0)
            
    return sorted(labels, key=key_func)

def hex_to_rgba(hex_code):
    return mcolors.to_rgba(hex_code)

# --- 3. Clustering & Topology Functions ---

def calculate_jaccard_sparse(csr_matrix):
    intersection = csr_matrix.dot(csr_matrix.T)
    row_sums = csr_matrix.getnnz(axis=1)
    return intersection, row_sums

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def fast_jaccard_filter(edges, indptr, indices, threshold):
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=np.bool_)
        for e in range(n_edges):
            u, v = edges[e, 0], edges[e, 1]
            start_u, end_u = indptr[u], indptr[u+1]
            start_v, end_v = indptr[v], indptr[v+1]
            size_u, size_v = end_u - start_u, end_v - start_v
            
            intersection = 0
            ptr_u, ptr_v = start_u, start_v
            while ptr_u < end_u and ptr_v < end_v:
                val_u, val_v = indices[ptr_u], indices[ptr_v]
                if val_u == val_v:
                    intersection += 1; ptr_u += 1; ptr_v += 1
                elif val_u < val_v: ptr_u += 1
                else: ptr_v += 1
            
            union = size_u + size_v - intersection
            if union > 0 and (intersection / union) >= threshold:
                keep_mask[e] = True
        return keep_mask
else:
    def fast_jaccard_filter(edges, indptr, indices, threshold):
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=bool)
        for e in range(n_edges):
            u, v = edges[e]
            set_u = set(indices[indptr[u]:indptr[u+1]])
            set_v = set(indices[indptr[v]:indptr[v+1]])
            intersection = len(set_u.intersection(set_v))
            union = len(set_u.union(set_v))
            if union > 0 and (intersection / union) >= threshold:
                keep_mask[e] = True
        return keep_mask


# --- 6. Network Building (Updated for Top-N) ---

def normalize_score(raw_score, align_len, len_i, len_j, mode):
    # Vectorized normalization! Handles both scalars and massive numpy arrays effortlessly.
    if mode == "alignment_length":
        denom = align_len
    elif mode == "shorter_sequence":
        denom = np.minimum(len_i, len_j)
    elif mode == "longer_sequence":
        denom = np.maximum(len_i, len_j)
    elif mode == "average_sequence":
        denom = (len_i + len_j) / 2.0
    else:
        denom = align_len
        
    return np.where(denom > 0, raw_score / denom, 0.0)

def plot_score_histogram(scores, threshold):
    print("Displaying Score Histogram... (Close histogram to continue)")

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=100, color='#4488ff', edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', label=f"Threshold {threshold}")
    
    # FIX: Check mode before accessing NORM_MODE
    mode_label = "E-Value" if cfg.INPUT_IS_EVALUE else cfg.NORM_MODE
    plt.title(f"Score Distribution ({mode_label})")
    
    plt.legend(); plt.show()

def build_network_from_raw(data, forced_ref_header=None):
    # ---> FIX: Ensure INPUT_IS_EVALUE is set before accessing it <---
    hdf5_base = os.path.basename(getattr(cfg, 'INPUT_HDF5', ''))
    cfg.INPUT_IS_EVALUE = "blast" in hdf5_base.lower()

    # Decode HDF5 byte-strings into standard python strings
    raw_headers = data['headers'][:]
    headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
    n_total = len(headers)
    
    # Extract arrays efficiently into memory
    if cfg.INPUT_IS_EVALUE:
        sources = data['i'][:]
        targets = data['j'][:]
        scores  = data['score'][:]
        num_edges = len(sources)
    else:
        sources = data['i'][:]
        targets = data['j'][:]
        seq_lens = data['seq_lens'][:]
        if cfg.ALIGNMENT_SCORE == "global":
            arr_score = data['g_score'][:]
            arr_len = data['g_len'][:]
        else:
            arr_score = data['l_score'][:]
            arr_len = data['l_len'][:]
        num_edges = len(sources)
        
    print(f"Raw Data: {n_total} sequences.")
    
    if not cfg.INPUT_IS_EVALUE:
        print(f"Metric: {cfg.ALIGNMENT_SCORE.upper()} Alignment with {cfg.NORM_MODE} Normalization")

    # --- 2. FASTA Subset Filtering Logic ---
    fasta_path = getattr(cfg, "NODE_FASTA_FILE", "")
    kept_indices = []
    
    if os.path.exists(fasta_path):
        # ---> FIX: Normalize the slash direction for the console output <---
        clean_fasta_path = os.path.normpath(fasta_path)
        print(f"Scanning FASTA file for node filter: {clean_fasta_path}")
        fasta_ids = set()
        fasta_headers = set()
        try:
            from Bio import SeqIO
            for rec in SeqIO.parse(fasta_path, "fasta"):
                fasta_ids.add(rec.id)
                fasta_headers.add(rec.description)
                
            net_headers_set = set(headers)
            net_id_set = {h.split()[0] for h in headers}
            
            missing_nodes = []
            for hid in fasta_ids:
                if hid not in net_id_set and hid not in net_headers_set:
                    missing_nodes.append(hid)
            
            if len(missing_nodes) > 0:
                print(f"CRITICAL WARNING: The passed FASTA file is NOT a strict subset of the network file. {len(missing_nodes)} FASTA sequences are missing from the network.")
                
            # Filter network based on FASTA strictly matching header
            for i, h in enumerate(headers):
                rec_id = h.split()[0]
                if h in fasta_headers or rec_id in fasta_ids:
                    kept_indices.append(i)
                    
            kept_indices = np.array(kept_indices)
            print(f"Filtered {n_total} down to {len(kept_indices)} valid FASTA subsets.")
            
        except Exception as e:
            print(f"Error reading FASTA filter: {e}. Retaining all sequences.")
            kept_indices = np.arange(n_total)
    else:
        print(f"No FASTA file found at {fasta_path}. Retaining all {n_total} sequences.")
        kept_indices = np.arange(n_total)

    # --- 3. Rebuild Data Structures (Fully Vectorized) ---
    kept_mask_bool = np.zeros(n_total, dtype=bool)
    kept_mask_bool[kept_indices] = True
    new_headers = [headers[i] for i in kept_indices]
    
    # Create mapping array to instantly translate old index to new index
    map_array = np.zeros(n_total, dtype=np.int32)
    map_array[kept_indices] = np.arange(len(kept_indices))
    
    # Find edges where BOTH nodes are in the "kept" subset
    valid_edges_mask = kept_mask_bool[sources] & kept_mask_bool[targets]
    valid_u = sources[valid_edges_mask]
    valid_v = targets[valid_edges_mask]
    
    # Fetch/calculate scores for those valid edges
    if cfg.INPUT_IS_EVALUE:
        valid_scores = scores[valid_edges_mask]
    else:
        valid_raw_scores = arr_score[valid_edges_mask]
        valid_align_lens = arr_len[valid_edges_mask]
        valid_scores = normalize_score(valid_raw_scores, valid_align_lens, seq_lens[valid_u], seq_lens[valid_v], cfg.NORM_MODE)
        
    scores_for_hist = valid_scores.tolist()
    
    top_percent = getattr(cfg, 'TOP_EDGE_PERCENT', None)
    if top_percent is not None and not getattr(cfg, 'UMAP_MODE', False):
        # ---> NEW: Calculate absolute theoretical max edges for proper Top % <---
        total_active_nodes = len(kept_indices)
        theoretical_max_edges = (total_active_nodes * (total_active_nodes - 1)) / 2.0
        
        # Calculate how many edges we actually need to grab
        k = int(theoretical_max_edges * (top_percent / 100.0))
        
        if len(valid_scores) == 0:
            calculated_cutoff = 0.0 # Fallback if network is completely empty
        else:
            # Safely cap K so we don't crash if BLAST dropped more edges than K requires
            k = max(1, min(k, len(valid_scores))) 
            
            # Sort descending to find the score at the Kth position
            sorted_all = np.sort(valid_scores)[::-1]
            calculated_cutoff = sorted_all[k - 1]
        
        mode_label = "E-Value" if cfg.INPUT_IS_EVALUE else "Similarity"
        print(f"Top {top_percent}% Edges Requested (based on max possible {int(theoretical_max_edges)} edges).")
        print(f"Calculated {mode_label} Cutoff: {calculated_cutoff:.5f}")
        
        # Override the global threshold for this session so the Viewer knows what to use
        cfg.SIMILARITY_THRESHOLD = calculated_cutoff

    is_umap = getattr(cfg, 'UMAP_MODE', False)
    if is_umap:
        print("UMAP Mode enabled: Bypassing global threshold. Filtering top k edges per node...")
        umap_k = int(getattr(cfg, 'UMAP_NEIGHBORS', 15))
        keep_limit = umap_k
        
        import pandas as pd
        df = pd.DataFrame({'u': valid_u, 'v': valid_v, 'score': valid_scores, 'idx': np.arange(len(valid_scores))})
        df_sorted = df.sort_values('score', ascending=False)
        
        top_u = df_sorted.groupby('u').head(keep_limit)['idx']
        top_v = df_sorted.groupby('v').head(keep_limit)['idx']
        
        keep_idx = pd.concat([top_u, top_v]).unique()
        
        thresh_mask = np.zeros(len(valid_scores), dtype=bool)
        thresh_mask[keep_idx] = True
        print(f"Kept {len(keep_idx)} edges for UMAP topology (max {keep_limit} per node direction).")
    else:
        # Threshold filter (only keep edges above cutoff)
        thresh_mask = valid_scores >= cfg.SIMILARITY_THRESHOLD
    
    # Apply new indices
    final_u = map_array[valid_u[thresh_mask]]
    final_v = map_array[valid_v[thresh_mask]]
    springs_arr = np.column_stack((final_u, final_v)).astype(np.int32)
    
    # ---> NEW: Export the scores so the Viewer can use them for annealing <---
    edge_scores = valid_scores[thresh_mask] 


    
    n_nodes_new = len(new_headers)
    
    # Layout Init
    side = int(np.ceil(np.sqrt(n_nodes_new)))
    base_box = np.sqrt(n_nodes_new) * 2.5 + 5.0
    box_limit = base_box * cfg.BOX_SCALE
    x = np.linspace(-box_limit*0.5, box_limit*0.5, side)
    y = np.linspace(-box_limit*0.5, box_limit*0.5, side)
    xv, yv = np.meshgrid(x, y)
    pos = np.column_stack((xv.flatten(), yv.flatten()))[:n_nodes_new].astype(np.float32)

    # ---> NEW: Return edge_scores <---
    return new_headers, springs_arr, edge_scores, pos, box_limit

# --- 8. Theme Utility ---
def force_light_palette(app):
    """Forces the application to use a standard light theme palette on all platforms."""
    from PyQt6.QtGui import QPalette, QColor
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(48, 140, 198))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)



