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
import sys
import json
import numpy as np
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import SSN_Config as cfg
import SSN_Utils as utils
from utilities.MSA_Sanitization import (
    AA_TO_INT,
    INT_TO_AA,
    MSAValidationError,
    MSASanitizationStats,
    canonicalize_sparse_values,
    load_sanitized_msa_fasta,
    parse_int_to_aa_mapping,
    print_msa_sanitization_result,
    sanitize_msa_headers,
)


def _print_red_warning(message):
    """Print a bright-red warning to terminals and plain text to redirected logs."""
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        print(f"\033[91m{message}\033[0m")
    else:
        print(message)

class Alignment_Manager:
    def __init__(self, msa_file, full_headers=None, active_reference=None, alignment_offset=0):
        self.msa_file = msa_file
        self.aln = None
        self.sanitization_stats = None
        self.valid_cols = None
        self.seq_map = {}
        self.col_to_label = {}
        self.label_to_col = {}
        self.resolved_ref_full = None
        self.has_reference = False
        self.offset = 0
        self._base_col_to_label = None
        self.network_headers = list(full_headers) if full_headers is not None else []
        self.matched_headers = []
        self.missing_headers = []
        self.viewer_to_aln = np.full(len(self.network_headers), -1, dtype=int)
        self.aligned_node_mask = np.zeros(len(self.network_headers), dtype=bool)

        if not msa_file or str(msa_file).strip() == "" or str(msa_file).strip().lower() == "none" or "none_[e1_ra]_alignment.fasta" in str(msa_file).lower():
            print("An MSA is not selected and will not be loaded.")
            return

        self.aln, is_sparse = load_alignment_smart(msa_file, filter_headers=full_headers)
        if self.aln is None:
            print('Warning: Failed to load alignment.')
            return
        self.sanitization_stats = getattr(self.aln, "sanitization_stats", None)

        self._initialize_coverage(is_sparse)

        has_ref = bool(active_reference and str(active_reference).strip().lower() != 'none')
        reference_fallback = False
        ref_idx = self._find_reference_index(active_reference, is_sparse) if has_ref else -1

        if has_ref and ref_idx not in (None, -1):
            if is_sparse:
                self.valid_cols, ref_length, forced_retained = self.aln.get_valid_columns(
                    cfg.FILTER_MIN_OCCUPANCY,
                    ref_header=active_reference,
                )
                _, self.col_to_label = self.aln.get_ref_anchored_mapping(
                    active_reference,
                    self.valid_cols,
                )
            else:
                self.valid_cols, ref_length, forced_retained = utils.get_valid_columns_legacy(
                    self.aln,
                    ref_header=active_reference,
                )
                _, self.col_to_label = utils.get_ref_anchored_mapping_legacy(
                    self.aln,
                    active_reference,
                    self.valid_cols,
                )
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
            reference_fallback = has_ref
            self._configure_occupancy_mapping(is_sparse)

        self.label_to_col = {v: k for k, v in self.col_to_label.items()}

        if self.missing_headers:
            self._warn_incomplete_coverage(reference_fallback, active_reference)
        elif reference_fallback:
            _print_red_warning(
                f"WARNING: Configured alignment reference '{active_reference}' was not found. "
                "The MSA remains loaded in pure occupancy mode; reference numbering and "
                "alignment offsets are inactive."
            )

    def _initialize_coverage(self, is_sparse):
        """Build the exact network-node to alignment-row mapping once per load."""
        if is_sparse:
            alignment_headers = list(self.aln.headers)
        else:
            alignment_headers = [
                record.description if record.description else record.id
                for record in self.aln
            ]

        exact_header_to_row = {
            header: row_idx for row_idx, header in enumerate(alignment_headers)
        }
        for node_idx, header in enumerate(self.network_headers):
            row_idx = exact_header_to_row.get(header)
            if row_idx is None:
                self.missing_headers.append(header)
                continue
            self.viewer_to_aln[node_idx] = row_idx
            self.aligned_node_mask[node_idx] = True
            self.matched_headers.append(header)

        if is_sparse:
            self.seq_map = self.aln.header_map
        else:
            for i, record in enumerate(self.aln):
                self.seq_map[record.id] = i
                self.seq_map[record.description] = i
                self.seq_map[utils.simplify_node_label(record.id)] = i

    def _find_reference_index(self, active_reference, is_sparse):
        if not active_reference or len(self.aln) == 0:
            return -1
        if is_sparse:
            return self.aln.find_reference_index(active_reference)

        target_lower = str(active_reference).lower()
        for i, record in enumerate(self.aln):
            if target_lower in record.id.lower() or target_lower in record.description.lower():
                return i
        return -1

    def _configure_occupancy_mapping(self, is_sparse):
        self.resolved_ref_full = 'None'
        self.has_reference = False
        self.offset = 0

        if len(self.aln) == 0:
            self.valid_cols = set()
            ref_length = 0
            forced_retained = 0
        elif is_sparse:
            self.valid_cols, ref_length, forced_retained = self.aln.get_valid_columns(
                cfg.FILTER_MIN_OCCUPANCY,
                ref_header=None,
            )
        else:
            self.valid_cols, ref_length, forced_retained = utils.get_valid_columns_legacy(
                self.aln,
                ref_header=None,
            )

        sorted_cols = sorted(self.valid_cols)
        self.col_to_label = {
            col_idx: str(idx + 1) for idx, col_idx in enumerate(sorted_cols)
        }
        self._base_col_to_label = dict(self.col_to_label)
        print("No active reference sequence. Operating in Pure Occupancy Mode.")
        print(
            f"Alignment Ready. Valid Cols (Occupancy >= "
            f"{cfg.FILTER_MIN_OCCUPANCY}%): {len(self.valid_cols)}"
        )

    def _warn_incomplete_coverage(self, reference_fallback, active_reference):
        total = len(self.network_headers)
        aligned = len(self.matched_headers)
        missing = len(self.missing_headers)
        coverage_pct = (100.0 * aligned / total) if total else 100.0
        shown_headers = self.missing_headers[:10]
        lines = [
            "WARNING: Incomplete MSA coverage",
            f"Aligned network nodes: {aligned}/{total} ({coverage_pct:.1f}%)",
            f"Excluded from alignment-dependent analyses: {missing}",
            f"Missing alignment headers (showing {len(shown_headers)} of {missing}):",
        ]
        lines.extend(f"  - {header}" for header in shown_headers)
        omitted = missing - len(shown_headers)
        if omitted > 0:
            lines.append(f"  ... and {omitted} more.")
        if reference_fallback:
            lines.append(
                f"Configured reference '{active_reference}' is missing; the MSA is loaded "
                "in pure occupancy mode and reference numbering is inactive."
            )
        _print_red_warning("\n".join(lines))

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
        from scipy import sparse

        stats = MSASanitizationStats()

        with h5py.File(h5_path, "r") as hf:
            required_paths = (
                "matrix",
                "matrix/data",
                "matrix/indices",
                "matrix/indptr",
                "headers",
                "int_to_aa",
            )
            for object_path in required_paths:
                link = hf.get(object_path, getlink=True)
                if link is None:
                    raise MSAValidationError(
                        f"Sparse HDF5 is missing required object '{object_path}'."
                    )
                if not isinstance(link, h5py.HardLink):
                    raise MSAValidationError(
                        f"Sparse HDF5 required object '{object_path}' must be a local hard link."
                    )

            mat_group = hf["matrix"]
            if not isinstance(mat_group, h5py.Group):
                raise MSAValidationError("Sparse HDF5 'matrix' must be a group.")
            if "shape" not in mat_group.attrs:
                raise MSAValidationError("Sparse HDF5 matrix is missing its shape attribute.")

            shape_array = np.asarray(mat_group.attrs["shape"])
            if (
                shape_array.ndim != 1
                or len(shape_array) != 2
                or not np.issubdtype(shape_array.dtype, np.integer)
            ):
                raise MSAValidationError(
                    "Sparse HDF5 matrix shape must contain two integers."
                )
            shape = tuple(int(value) for value in shape_array)
            if shape[0] <= 0 or shape[1] <= 0:
                raise MSAValidationError(
                    "Sparse HDF5 matrix must contain at least one row and one column."
                )

            data_ds = mat_group["data"]
            indices_ds = mat_group["indices"]
            indptr_ds = mat_group["indptr"]
            for name, dataset in (
                ("matrix/data", data_ds),
                ("matrix/indices", indices_ds),
                ("matrix/indptr", indptr_ds),
            ):
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
                    raise MSAValidationError(
                        f"Sparse HDF5 '{name}' must be a one-dimensional dataset."
                    )
                if not np.issubdtype(dataset.dtype, np.integer):
                    raise MSAValidationError(
                        f"Sparse HDF5 '{name}' must use an integer dtype."
                    )

            headers_ds = hf["headers"]
            if not isinstance(headers_ds, h5py.Dataset) or headers_ds.ndim != 1:
                raise MSAValidationError(
                    "Sparse HDF5 'headers' must be a one-dimensional dataset."
                )
            if len(headers_ds) != shape[0]:
                raise MSAValidationError(
                    f"Sparse HDF5 header count ({len(headers_ds)}) does not match "
                    f"matrix rows ({shape[0]})."
                )

            if len(data_ds) != len(indices_ds):
                raise MSAValidationError(
                    "Sparse HDF5 data and indices datasets have different lengths."
                )
            if len(indptr_ds) != shape[0] + 1:
                raise MSAValidationError(
                    "Sparse HDF5 indptr length must equal row count plus one."
                )

            indices = indices_ds[:]
            indptr = indptr_ds[:]
            nnz = len(data_ds)

            if int(indptr[0]) != 0 or int(indptr[-1]) != nnz:
                raise MSAValidationError(
                    "Sparse HDF5 indptr endpoints do not match the stored entries."
                )
            if np.any(indptr < 0) or np.any(indptr[1:] < indptr[:-1]):
                raise MSAValidationError(
                    "Sparse HDF5 indptr values must be non-negative and monotonic."
                )
            if np.any(indices < 0) or np.any(indices >= shape[1]):
                raise MSAValidationError(
                    "Sparse HDF5 contains column indices outside the matrix shape."
                )
            for row_idx in range(shape[0]):
                start = int(indptr[row_idx])
                end = int(indptr[row_idx + 1])
                row_indices = indices[start:end]
                if len(row_indices) != len(np.unique(row_indices)):
                    raise MSAValidationError(
                        f"Sparse HDF5 row {row_idx} contains duplicate column indices."
                    )

            raw_headers = []
            for row_idx, raw_header in enumerate(headers_ds[:], start=1):
                if isinstance(raw_header, (bytes, np.bytes_)):
                    try:
                        header = bytes(raw_header).decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        raise MSAValidationError(
                            f"Sparse HDF5 header {row_idx} is not valid UTF-8."
                        ) from exc
                elif isinstance(raw_header, str):
                    header = raw_header
                else:
                    raise MSAValidationError(
                        f"Sparse HDF5 header {row_idx} is not a string."
                    )
                raw_headers.append(header)
            raw_headers = sanitize_msa_headers(raw_headers, stats)

            mapping_dataset = hf["int_to_aa"]
            if not isinstance(mapping_dataset, h5py.Dataset) or mapping_dataset.shape != ():
                raise MSAValidationError(
                    "Sparse HDF5 'int_to_aa' must be a scalar JSON dataset."
                )
            mapping_text = mapping_dataset[()]
            if isinstance(mapping_text, (bytes, np.bytes_)):
                try:
                    mapping_text = bytes(mapping_text).decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise MSAValidationError(
                        "Sparse HDF5 int_to_aa is not valid UTF-8."
                    ) from exc
            if not isinstance(mapping_text, str):
                raise MSAValidationError(
                    "Sparse HDF5 int_to_aa must contain UTF-8 JSON text."
                )
            try:
                decoded_mapping = json.loads(mapping_text)
            except json.JSONDecodeError as exc:
                raise MSAValidationError(
                    f"Sparse HDF5 int_to_aa contains invalid JSON: {exc.msg}."
                ) from exc
            source_int_to_aa = parse_int_to_aa_mapping(decoded_mapping)

            canonical_data = np.empty(nnz, dtype=np.uint8)
            keep_mask = np.ones(nnz, dtype=bool)
            validation_chunk_size = 1_000_000
            for chunk_start in range(0, nnz, validation_chunk_size):
                chunk_end = min(chunk_start + validation_chunk_size, nnz)
                chunk_data, chunk_keep = canonicalize_sparse_values(
                    data_ds[chunk_start:chunk_end],
                    source_int_to_aa,
                    stats,
                )
                canonical_data[chunk_start:chunk_end] = chunk_data
                keep_mask[chunk_start:chunk_end] = chunk_keep
            canonical_data[~keep_mask] = 0
            raw_matrix = sparse.csr_matrix(
                (canonical_data, indices, indptr),
                shape=shape,
                dtype=np.uint8,
            )
            if not np.all(keep_mask):
                raw_matrix.eliminate_zeros()

        if filter_headers is not None:
            header_to_idx = {h: i for i, h in enumerate(raw_headers)}
            
            keep_indices = []
            final_headers = []
            for h in filter_headers:
                if h in header_to_idx:
                    keep_indices.append(header_to_idx[h])
                    final_headers.append(h)

            extra_count = len(raw_headers) - len(final_headers)
            if extra_count > 0:
                print(f"Warning: The MSA contains {extra_count} more sequences than the provided FASTA subset.")
                
            self.matrix = raw_matrix[keep_indices, :]
            self.headers = final_headers
        else:
            self.matrix = raw_matrix
            self.headers = raw_headers

        self.n_seqs, self.n_cols = self.matrix.shape
        self.sanitization_stats = stats
        self.int_to_aa = dict(INT_TO_AA)
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

        print_msa_sanitization_result(stats, h5_path)

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

    def find_reference_index(self, ref_header):
        """Resolve a reference within the rows retained for the active network."""
        if not ref_header or self.n_seqs == 0:
            return -1

        target_lower = str(ref_header).lower()
        for header_key, idx in self.header_map.items():
            if target_lower == str(header_key).lower():
                return idx
        for i, header in enumerate(self.headers):
            if target_lower in header.lower() or header.lower() in target_lower:
                return i
        return -1

    def get_valid_columns(self, min_occupancy_pct, ref_header=None):
        if self.n_seqs == 0:
            return set(), 0, 0

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
            ref_idx = self.find_reference_index(target)
            if ref_idx != -1:
                break
        
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
        return valid_indices, ref_length, added

    def get_ref_anchored_mapping(self, ref_id_substring, valid_cols):
        ref_idx = self.find_reference_index(ref_id_substring)
        
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

class InMemorySparseLoader(SparseAlignmentLoader):
    """Generates a CSR matrix directly from a FASTA file in RAM without saving to disk."""
    def __init__(self, fasta_path, filter_headers=None):
        from scipy import sparse

        print(f"--- Parsing FASTA to Sparse Matrix in RAM: {fasta_path} ---")

        row_ind, col_ind, data_vals = [], [], []
        final_headers = []
        headers, sequences, stats = load_sanitized_msa_fasta(fasta_path)
        alignment_length = len(sequences[0])
        row_idx = 0

        keep_set = set(filter_headers) if filter_headers is not None else None

        for header, sequence in zip(headers, sequences):
            if keep_set is not None and header not in keep_set:
                continue

            final_headers.append(header)

            for col_idx, char in enumerate(sequence):
                if char in AA_TO_INT:
                    row_ind.append(row_idx)
                    col_ind.append(col_idx)
                    data_vals.append(AA_TO_INT[char])
            row_idx += 1

        self.matrix = sparse.csr_matrix(
            (data_vals, (row_ind, col_ind)), 
            shape=(row_idx, alignment_length),
            dtype=np.uint8 
        )
        self.headers = final_headers
        self.int_to_aa = dict(INT_TO_AA)
        self.sanitization_stats = stats
        self.n_seqs, self.n_cols = self.matrix.shape
        self.header_map = {}

        print("Indexing headers...")
        for i, header in enumerate(self.headers):
            self.header_map[header] = i
            rec_id = header.split()[0]
            self.header_map[rec_id] = i
            simple_id = utils.simplify_node_label(header)
            self.header_map[simple_id] = i

        print_msa_sanitization_result(stats, fasta_path)

def load_alignment_smart(msa_path, filter_headers=None):
    """
    Strict loader: Respects the exact file extension provided.
    - .h5: Loads the pre-computed sparse matrix from disk.
    - .fasta: Converts the FASTA to a sparse matrix directly in RAM.
    """
    if not msa_path or str(msa_path).strip() == "" or str(msa_path).strip().lower() == "none":
        return None, False

    msa_path = os.fspath(msa_path)
    extension = os.path.splitext(msa_path)[1].lower()

    if extension == ".h5":
        if os.path.exists(msa_path):
            print(f"--- Loading Sparse Alignment in HDF5 format: {msa_path} ---")
            try:
                loader = SparseAlignmentLoader(msa_path, filter_headers)
                return loader, True
            except MSAValidationError as e:
                _print_red_warning(f"ERROR: MSA rejected: {e}")
                return None, False
            except Exception as e:
                print(f"Error loading HDF5: {e}")
                return None, False
        else:
            print(f"Error: Specified HDF5 file does not exist: {msa_path}")
            return None, False

    if extension != ".fasta":
        _print_red_warning(
            f"ERROR: MSA rejected: Unsupported alignment extension '{extension or '(none)'}'. "
            "Expected .fasta or .h5."
        )
        return None, False

    try:
        loader = InMemorySparseLoader(msa_path, filter_headers)
        return loader, True 
    except MSAValidationError as e:
        _print_red_warning(f"ERROR: MSA rejected: {e}")
        return None, False
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
