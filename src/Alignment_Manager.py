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

import os
import numpy as np
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import SSN_Config as cfg
import SSN_Utils as utils

class Alignment_Manager:
    def __init__(self, msa_file, full_headers=None, active_reference=None, alignment_offset=0):
        self.aln = None
        self.valid_cols = None
        self.seq_map = None
        self.col_to_label = None
        self.label_to_col = None
        self.resolved_ref_full = None
        self.has_reference = False
        self.offset = 0
        self._base_col_to_label = None

        if not msa_file or str(msa_file).strip() == "" or str(msa_file).strip().lower() == "none" or "none_[e1_ra]_alignment.fasta" in str(msa_file).lower():
            print("An MSA is not selected and will not be loaded.")
            return

        self.aln, is_sparse = load_alignment_smart(msa_file, filter_headers=full_headers)
        if self.aln is None:
            print('Warning: Failed to load alignment.')
            return

        has_ref = bool(active_reference and str(active_reference).strip().lower() != 'none')
        if has_ref:
            ref_to_use = active_reference # We use active_reference here, but we will print nicely
            if is_sparse:
                self.valid_cols, ref_length, forced_retained = self.aln.get_valid_columns(cfg.FILTER_MIN_OCCUPANCY, ref_header=ref_to_use)
                ref_idx, self.col_to_label = self.aln.get_ref_anchored_mapping(active_reference, self.valid_cols)
                self.seq_map = self.aln.header_map
            else:
                self.valid_cols, ref_length, forced_retained = utils.get_valid_columns_legacy(self.aln, ref_header=ref_to_use)
                ref_idx, self.col_to_label = utils.get_ref_anchored_mapping_legacy(self.aln, active_reference, self.valid_cols)
                self.seq_map = {}
                for i, r in enumerate(self.aln):
                    self.seq_map[r.id] = i
                    self.seq_map[r.description] = i
                    simple = utils.simplify_node_label(r.id)
                    self.seq_map[simple] = i
            if ref_idx is None or ref_idx == -1:
                print(f"Warning: Active reference '{active_reference}' not found in alignment data. MSA disabled.")
                self.aln = None
                return
            if is_sparse:
                self.resolved_ref_full = self.aln.headers[ref_idx]
            else:
                rec = self.aln[ref_idx]
                self.resolved_ref_full = rec.description if rec.description else rec.id
            self.has_reference = True
            self._base_col_to_label = dict(self.col_to_label)
            self.set_offset(alignment_offset)
            print(f"Matched Reference '{active_reference}' to '{self.resolved_ref_full[:40]}...'")
            print(f"Active Reference: {self.resolved_ref_full}")
            print(f"Alignment Offset: {self.offset}")
            print(f"Alignment Ready. Valid Cols: {len(self.valid_cols)} (Reference: {ref_length}, Forced to Retain: {forced_retained})")
        else:
            self.resolved_ref_full = 'None'
            if is_sparse:
                self.valid_cols, ref_length, forced_retained = self.aln.get_valid_columns(cfg.FILTER_MIN_OCCUPANCY, ref_header=None)
                self.seq_map = self.aln.header_map
            else:
                self.valid_cols, ref_length, forced_retained = utils.get_valid_columns_legacy(self.aln, ref_header=None)
                self.seq_map = {}
                for i, r in enumerate(self.aln):
                    self.seq_map[r.id] = i
                    self.seq_map[r.description] = i
                    simple = utils.simplify_node_label(r.id)
                    self.seq_map[simple] = i
            sorted_cols = sorted(list(self.valid_cols))
            self.col_to_label = {col_idx: str(idx + 1) for idx, col_idx in enumerate(sorted_cols)}
            self._base_col_to_label = dict(self.col_to_label)
            print(f"No reference sequence provided. Operating in Pure Occupancy Mode.")
            print(f"Alignment Ready. Valid Cols (Occupancy >= {cfg.FILTER_MIN_OCCUPANCY}%): {len(self.valid_cols)}")

        self.label_to_col = {v: k for k, v in self.col_to_label.items()}

    @staticmethod
    def _offset_label(label, offset):
        """Shift the integer portion of a reference label while preserving insertions."""
        parts = str(label).split('.', 1)
        shifted = str(int(parts[0]) + offset)
        return f"{shifted}.{parts[1]}" if len(parts) == 2 else shifted

    def set_offset(self, offset):
        """Apply an integer offset to a successfully resolved reference mapping."""
        if not self.has_reference or self.aln is None or self._base_col_to_label is None:
            return False

        if isinstance(offset, bool):
            raise ValueError("Alignment offset must be an integer.")
        if isinstance(offset, float) and not offset.is_integer():
            raise ValueError("Alignment offset must be an integer.")
        try:
            new_offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("Alignment offset must be an integer.") from exc

        self.offset = new_offset
        self.col_to_label = {
            col_idx: self._offset_label(label, new_offset)
            for col_idx, label in self._base_col_to_label.items()
        }
        self.label_to_col = {label: col_idx for col_idx, label in self.col_to_label.items()}
        return True

    def calculate_frequencies(self, mapping, exclude=[], aln=None):
        target_aln = aln if aln is not None else self.aln
        return calculate_frequencies(target_aln, mapping, exclude)

# --- 4. Sparse Alignment Loading (Updated for Filtering) ---

class SparseAlignmentLoader:
    def __init__(self, h5_path, filter_headers=None):
        """
        filter_headers: If provided (list of strings), only these sequences 
                        will be loaded/retained in the matrix.
        """
        import h5py
        import json
        from scipy import sparse
        
        with h5py.File(h5_path, "r") as hf:
            # 1. Reconstruct SciPy Sparse Matrix
            mat_group = hf["matrix"]
            shape = tuple(mat_group.attrs["shape"])
            raw_matrix = sparse.csr_matrix(
                (mat_group["data"][:], mat_group["indices"][:], mat_group["indptr"][:]), 
                shape=shape
            )
            
            # 2. Extract Headers
            raw_headers_bytes = hf["headers"][:]
            raw_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers_bytes]
            
            # 3. Extract JSON mappings (convert int_to_aa keys back to integers)
            raw_int_to_aa = json.loads(hf["int_to_aa"][()].decode('utf-8'))
            self.int_to_aa = {int(k): v for k, v in raw_int_to_aa.items()}
            
        if filter_headers is not None:
            header_to_idx = {h: i for i, h in enumerate(raw_headers)}
            
            keep_indices = []
            final_headers = []
            missing_count = 0
            
            for h in filter_headers:
                if h in header_to_idx:
                    keep_indices.append(header_to_idx[h])
                    final_headers.append(h)
                else:
                    missing_count += 1
            
            if missing_count > 0:
                msg = f"CRITICAL ERROR: MSA missing {missing_count} sequences from your FASTA subset! To correctly load an MSA, the fasta file must be a strict subset of the sequence set used to build the MSA."
                print(msg)
                raise ValueError(msg)
                
            extra_count = len(raw_headers) - len(final_headers)
            if extra_count > 0:
                print(f"Warning: The MSA contains {extra_count} more sequences than the provided FASTA subset.")
                
            self.matrix = raw_matrix[keep_indices, :]
            self.headers = final_headers
        else:
            self.matrix = raw_matrix
            self.headers = raw_headers

        self.n_seqs, self.n_cols = self.matrix.shape
        self.header_map = {}
        
        # Re-index simplified headers
        print("Indexing headers...")
        for i, header in enumerate(self.headers):
            # Map Full Header
            self.header_map[header] = i
            
            # Map ID (first word)
            rec_id = header.split()[0]
            self.header_map[rec_id] = i
            
            # Map Simplified
            simple_id = utils.simplify_node_label(header)
            self.header_map[simple_id] = i

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_seqs:
            raise IndexError("Alignment index out of range")
        
        row = self.matrix[idx].toarray()[0]
        seq_chars = [self.int_to_aa.get(val, '-') if val != 0 else '-' for val in row]
        seq_str = "".join(seq_chars)
        
        desc = self.headers[idx]
        rec_id = desc.split()[0] 
        return SeqRecord(Seq(seq_str), id=rec_id, description=desc)

    def __iter__(self):
        for i in range(self.n_seqs):
            yield self[i]

    def get_alignment_length(self):
        return self.n_cols

    def get_valid_columns(self, min_occupancy_pct, ref_header=None):
        # 1. Occupancy Filter (Standard)
        min_count = self.n_seqs * (min_occupancy_pct / 100.0)
        col_counts = self.matrix.getnnz(axis=0) 
        valid_indices = set(np.where(col_counts >= min_count)[0])
        
        ref_length = 0
        added = 0
        
        # 2. Reference Force-Keep (NEW)
        search_targets = []
        if ref_header: search_targets.append(ref_header)
        if hasattr(cfg, 'ALIGNMENT_REFERENCE') and cfg.ALIGNMENT_REFERENCE: 
            search_targets.append(cfg.ALIGNMENT_REFERENCE)

        ref_idx = -1
        for target in search_targets:
            target_lower = target.lower()
            
            # Exact map (Case-Insensitive)
            for h_key, idx in self.header_map.items():
                if target_lower == str(h_key).lower():
                    ref_idx = idx
                    break
            if ref_idx != -1: break
            
            # Substring search (Case-Insensitive)
            for i, h in enumerate(self.headers):
                 if target_lower in h.lower() or h.lower() in target_lower:
                     ref_idx = i; break
            if ref_idx != -1: break
        
        if ref_idx != -1:
            # Get the reference row
            ref_row = self.matrix[ref_idx].toarray()[0]
            # Find columns where reference is NOT a gap (val > 0)
            ref_cols = np.where(ref_row != 0)[0]
            
            ref_length = len(ref_cols)
            
            # Add these columns to valid_indices
            before_len = len(valid_indices)
            valid_indices.update(ref_cols)
            added = len(valid_indices) - before_len
        else:
            print(f"Warning: Reference not found during column filtering.")

        return valid_indices, ref_length, added

    def get_ref_anchored_mapping(self, ref_id_substring, valid_cols):
        ref_idx = -1
        target_lower = ref_id_substring.lower() if ref_id_substring else ""
        
        for h_key, idx in self.header_map.items():
            if target_lower in str(h_key).lower():
                ref_idx = idx; break
        
        if ref_idx == -1: 
            for i, h in enumerate(self.headers):
                if target_lower in h.lower():
                    ref_idx = i; break
        
        if ref_idx == -1: return None, None

        ref_row = self.matrix[ref_idx].toarray()[0]
        mapping = {}
        last_int = 0; dec_cnt = 0
        
        for col_i in range(len(ref_row)):
            val = ref_row[col_i]
            is_gap = (val == 0)
            if is_gap:
                dec_cnt += 1; label = f"{last_int}.{dec_cnt}"
            else:
                last_int += 1; dec_cnt = 0; label = str(last_int)
                
            if valid_cols is not None and col_i in valid_cols:
                mapping[col_i] = label
        return ref_idx, mapping

    def get_frequencies(self, col_idx):
        col_vec = self.matrix[:, col_idx]
        residues = col_vec.data
        n_valid = len(residues)
        if n_valid == 0: return ('-', 0.0, 0.0)
        occupancy = n_valid / self.n_seqs
        counts = Counter(residues)
        top_aa_int, count = counts.most_common(1)[0]
        top_aa = self.int_to_aa.get(top_aa_int, 'X')

        consensus = count / self.n_seqs 
        return (top_aa, consensus, occupancy)
    
    def bulk_residue_check(self, col_idx, target_aa_char):
        """
        Efficiently checks which sequences have a specific amino acid at a specific column.
        Returns a boolean numpy array of shape (n_seqs,).
        """
        if col_idx < 0 or col_idx >= self.n_cols:
            return np.zeros(self.n_seqs, dtype=bool)

        # 1. Get the column vector (sparse)
        col_vec = self.matrix[:, col_idx]
        
        # Convert to dense for easy comparison
        dense_col = col_vec.toarray().flatten()
        
        # ---> NEW GAP INTERCEPT <---
        # In a sparse matrix, 0 represents ANY gap character
        if target_aa_char in cfg.GAP_CHARS:
            return (dense_col == 0)
        
        # 2. Find the integer code for the target AA
        target_code = None
        for code, aa in self.int_to_aa.items():
            if aa == target_aa_char:
                target_code = code
                break
        
        if target_code is None:
            return np.zeros(self.n_seqs, dtype=bool)

        # 3. Compare data
        return (dense_col == target_code)

# --- Sparse In-Memory Conversion Assets ---
AA_MAP = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 
    'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 
    'W': 18, 'Y': 19, 'V': 20, 
    'X': 21, 'B': 3, 'Z': 6, 'J': 10, 'U': 5, 'O': 12
}
INT_TO_AA = {v: k for k, v in AA_MAP.items() if k not in ['B', 'Z', 'J', 'U', 'O']}

class InMemorySparseLoader(SparseAlignmentLoader):
    """Generates a CSR matrix directly from a FASTA file in RAM without saving to disk."""
    def __init__(self, fasta_path, filter_headers=None):
        from Bio import SeqIO
        from scipy import sparse
        import numpy as np

        print(f"--- Parsing FASTA to Sparse Matrix in RAM: {fasta_path} ---")

        row_ind, col_ind, data_vals = [], [], []
        final_headers = []
        max_col, row_idx = 0, 0

        # Pre-compute filter set for O(1) lookup
        keep_set = set(filter_headers) if filter_headers else None

        for record in SeqIO.parse(fasta_path, "fasta"):
            if keep_set and record.description not in keep_set and record.id not in keep_set:
                continue

            final_headers.append(record.description)
            seq_str = str(record.seq).upper()
            length = len(seq_str)
            if length > max_col: max_col = length

            for col_idx, char in enumerate(seq_str):
                if char in AA_MAP:
                    row_ind.append(row_idx)
                    col_ind.append(col_idx)
                    data_vals.append(AA_MAP[char])
            row_idx += 1

        if filter_headers is not None:
            missing_count = len(filter_headers) - len(final_headers)
            if missing_count > 0:
                msg = f"CRITICAL ERROR: MSA missing {missing_count} sequences from your FASTA subset!"
                print(msg)
                raise ValueError(msg)

        # Build state variables identical to HDF5 loader
        self.matrix = sparse.csr_matrix(
            (data_vals, (row_ind, col_ind)), 
            shape=(row_idx, max_col),
            dtype=np.uint8 
        )
        self.headers = final_headers
        self.int_to_aa = INT_TO_AA
        self.n_seqs, self.n_cols = self.matrix.shape
        self.header_map = {}

        print("Indexing headers...")
        for i, header in enumerate(self.headers):
            self.header_map[header] = i
            rec_id = header.split()[0]
            self.header_map[rec_id] = i
            simple_id = utils.simplify_node_label(header)
            self.header_map[simple_id] = i

def load_alignment_smart(msa_path, filter_headers=None):
    """
    Strict loader: Respects the exact file extension provided.
    - .h5: Loads the pre-computed sparse matrix from disk.
    - .fasta: Converts the FASTA to a sparse matrix directly in RAM.
    """
    if not msa_path or str(msa_path).strip() == "" or str(msa_path).strip().lower() == "none":
        return None, False

    if msa_path.endswith(".h5"):
        if os.path.exists(msa_path):
            print(f"--- Loading Sparse Alignment in HDF5 format: {msa_path} ---")
            try:
                loader = SparseAlignmentLoader(msa_path, filter_headers)
                return loader, True
            except Exception as e:
                print(f"Error loading HDF5: {e}")
                return None, False
        else:
            print(f"Error: Specified HDF5 file does not exist: {msa_path}")
            return None, False

    # If it's not an .h5, we bypass Biopython and use our new high-speed RAM converter
    try:
        # The InMemorySparseLoader already prints its own initialization message
        loader = InMemorySparseLoader(msa_path, filter_headers)
        
        # Returning True here forces SSN_Viewer to use the fast downstream sparse filtering logic
        return loader, True 
    except Exception as e:
        print(f"Error loading FASTA into memory: {e}")
        return None, False

# --- 5. Shared Alignment Utilities ---

def calculate_frequencies(aln, mapping, exclude=[]):
    stats = {}
    if isinstance(aln, SparseAlignmentLoader) and not exclude:
        for col_i, label in mapping.items():
            stats[label] = aln.get_frequencies(col_i)
        return stats

    valid_rows = [r for i, r in enumerate(aln) if i not in exclude]
    if not valid_rows: return {}
    
    try: n_cols = aln.get_alignment_length()
    except: n_cols = len(aln[0])

    total_seqs = len(valid_rows)

    for col_i in range(n_cols):
        if col_i not in mapping: continue
        label = mapping[col_i]
        col_chars = [r.seq[col_i] for r in valid_rows]
        valid_aa = [c for c in col_chars if c not in cfg.GAP_CHARS]
        n_valid = len(valid_aa)
        
        if total_seqs > 0: occupancy = n_valid / total_seqs
        else: occupancy = 0.0
        
        if n_valid == 0:
            stats[label] = ('-', 0.0, 0.0); continue
            
        c = Counter(valid_aa).most_common(1)
        consensus = c[0][1] / total_seqs
        stats[label] = (c[0][0], consensus, occupancy)
    return stats

def get_valid_columns_legacy(aln, ref_header=None):
    valid_indices = set()
    ref_length = 0
    added = 0
    try:
        n_cols = aln.get_alignment_length()
        n_seqs = len(aln)
        min_occ = cfg.FILTER_MIN_OCCUPANCY / 100.0
        
        # 1. Occupancy Filter
        for col_i in range(n_cols):
            non_gaps = sum(1 for c in aln[:, col_i] if c not in cfg.GAP_CHARS)
            if (non_gaps / n_seqs) >= min_occ: valid_indices.add(col_i)
            
        # 2. Reference Force-Keep (NEW)
        if ref_header:
            ref_rec = None
            for r in aln:
                if ref_header in r.description or ref_header in r.id:
                    ref_rec = r
                    break
            
            if ref_rec:
                # Add any column where the reference has a residue
                for col_i, char in enumerate(ref_rec.seq):
                    if char not in cfg.GAP_CHARS:
                        ref_length += 1
                        if col_i not in valid_indices:
                            added += 1
                            valid_indices.add(col_i)
                            
    except Exception as e: print(f"Warning: {e}")
    return valid_indices, ref_length, added

def get_ref_anchored_mapping_legacy(aln, ref_id, valid_cols_global):
    ref_idx = -1
    target_lower = ref_id.lower() if ref_id else ""
    for i, r in enumerate(aln):
        if target_lower in r.id.lower() or target_lower in r.description.lower():
            ref_idx = i; break
    if ref_idx == -1: return None, None
    ref_seq = str(aln[ref_idx].seq)
    mapping = {}
    last_int = 0; dec_cnt = 0
    for col_i, char in enumerate(ref_seq):
        if char in cfg.GAP_CHARS:
            dec_cnt += 1; label = f"{last_int}.{dec_cnt}"
        else:
            last_int += 1; dec_cnt = 0; label = str(last_int)
        if valid_cols_global is not None and col_i in valid_cols_global:
            mapping[col_i] = label
    return ref_idx, mapping
