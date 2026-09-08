# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Read-only structural inspection; numeric HDF5 payloads are never scanned."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FILE_TYPES = ("auto", "fasta", "alignment_fasta", "embedding", "network", "sparse_msa", "blast_tabular", "settings", "layout_cache")
MAX_FINDINGS = 50
MAX_TEXT_LINE = 1024 * 1024
MAX_METADATA_RECORDS = 100000
MAX_JSON_BYTES = 8 * 1024 * 1024


class InspectionLimit(Exception):
    pass


def base_report(path):
    return dict(path=str(path), size_bytes=None, modified_ns=None, detected_format="unknown",
                inspection_finished=False, unstable=False, structural_validity="unknown",
                generation_completion={"status": "unknown", "evidence": "No trustworthy completion marker established."},
                network_pair_coverage={"status": "unknown", "observed": None, "expected": None,
                                       "unique_pairs_verified": False},
                metadata={}, findings=[], findings_omitted=0, checks_performed=[],
                checks_omitted=["Numerical payload values, cross-file compatibility, and job readiness are not validated."])


class Inspector:
    def __init__(self, path, deadline):
        self.path = path
        self.deadline = deadline
        self.report = base_report(path)

    def tick(self):
        if time.monotonic() >= self.deadline:
            raise InspectionLimit("Inspection time budget reached; coverage is incomplete.")

    def finding(self, severity, message):
        if severity == "error":
            self.report["structural_validity"] = "invalid"
        if len(self.report["findings"]) < MAX_FINDINGS:
            self.report["findings"].append(dict(severity=severity, message=str(message)[:512]))
        else:
            self.report["findings_omitted"] += 1

    def require(self, condition, message):
        if not condition:
            raise ValueError(message)

    def flag(self, hf):
        import numpy as np
        flags = []
        for key in ("generation_complete", "complete"):
            if key in hf.attrs:
                value = hf.attrs[key]
                self.require(isinstance(value, (bool, np.bool_)), f"{key} must be boolean.")
                flags.append((key, bool(value)))
        if flags:
            if len({value for _, value in flags}) > 1:
                self.finding("error", "Completion flags contradict each other.")
            complete = all(value for _, value in flags)
            self.report["generation_completion"] = dict(status="complete" if complete else "incomplete",
                evidence="; ".join(f"{key}={value}" for key, value in flags))

    def dataset(self, hf, name, kind=None):
        import h5py
        self.require(name in hf and isinstance(hf[name], h5py.Dataset), f"Missing dataset or wrong object type: {name}.")
        ds = hf[name]
        self.require(ds.ndim == 1, f"{name} must be one-dimensional.")
        if kind:
            self.require(ds.dtype.kind in kind, f"Invalid dtype for {name}: {ds.dtype}.")
        return ds

    def embedding(self, hf):
        import h5py
        from utilities.HDF5_Storage import (
            REQUIRED_ATTRIBUTES, REQUIRED_OBJECTS, validate_manifest_records,
            validate_embedding_array, dtype_for_saving_mode,
        )
        self.flag(hf)
        for key in REQUIRED_ATTRIBUTES:
            self.require(key in hf.attrs, f"Missing embedding attribute: {key}.")
        for key in REQUIRED_OBJECTS:
            self.require(key in hf, f"Missing embedding object: {key}.")
        headers = self.dataset(hf, "headers")
        sequences = self.dataset(hf, "sequences")
        self.require(h5py.check_string_dtype(headers.dtype) is not None and h5py.check_string_dtype(sequences.dtype) is not None,
                     "Embedding headers and sequences must be string datasets.")
        model = hf.attrs["model_name"]
        if isinstance(model, bytes): model = model.decode("utf-8")
        self.require(isinstance(model, str) and bool(model.strip()), "Invalid embedding model_name.")
        mode = hf.attrs["saving_mode"]
        if isinstance(mode, bytes): mode = mode.decode("utf-8")
        dtype_for_saving_mode(mode)
        count = len(headers)
        import numpy as np
        declared = hf.attrs["num_sequences"]
        self.require(isinstance(declared, (int, np.integer)) and not isinstance(declared, (bool, np.bool_)), "num_sequences must be an integer.")
        self.require(count == len(sequences) == int(declared), "Embedding manifest counts disagree.")
        group = hf["embeddings"]
        self.require(isinstance(group, h5py.Group), "embeddings must be a group.")
        self.report["metadata"].update(sequence_count=count, embedding_count=len(group), model_name=model[:256], saving_mode=mode)
        self.report["checks_performed"].append("Embedding root metadata and manifest dimensions")
        seen = set()
        missing = 0
        dimension = None
        for index in range(count):
            self.tick()
            if index >= MAX_METADATA_RECORDS:
                raise InspectionLimit("Embedding metadata record limit reached.")
            header, sequence = headers.asstr()[index], sequences.asstr()[index]
            validate_manifest_records([header], [sequence])
            self.require(header not in seen, "Duplicate embedding manifest header.")
            seen.add(header)
            if header not in group:
                missing += 1
            else:
                ds = group[header]
                self.require(isinstance(ds, h5py.Dataset), "Embedding entry must be a dataset.")
                dimension = validate_embedding_array(ds, sequence, mode, dimension, require_finite=False)
            self.report["metadata"]["records_checked"] = index + 1
        for index, key in enumerate(group):
            self.tick()
            if index >= MAX_METADATA_RECORDS: raise InspectionLimit("Embedding object limit reached.")
            self.require(key in seen, "Unexpected embedding dataset absent from manifest.")
        self.report["metadata"].update(missing_embeddings=missing, feature_dimension=dimension)
        if missing:
            self.require(not bool(hf.attrs["generation_complete"]), "Complete flag contradicts missing embeddings.")
            self.finding("warning", f"{missing} manifest entries have no embedding yet.")
        self.report["checks_performed"].append("All embedding manifest records, dataset presence, shapes, and dtypes")
        self.report["checks_omitted"].append("Embedding numerical arrays were not read.")

    def network(self, hf):
        import h5py
        from Cache_Manifest import validate_network_schema
        self.flag(hf)
        metadata = validate_network_schema(hf)
        n = len(self.dataset(hf, "headers"))
        self.require(h5py.check_string_dtype(hf["headers"].dtype) is not None, "Network headers must be strings.")
        names = ("i", "j", "score") if metadata.network_type == "blast" else ("i", "j", "l_score", "l_len", "g_score", "g_len")
        sizes = []
        for name in names:
            self.tick()
            sizes.append(len(self.dataset(hf, name, "iu" if name in ("i", "j", "l_len", "g_len") else "f")))
        self.require(len(set(sizes)) == 1, "Network edge dataset lengths disagree.")
        if metadata.network_type == "alignment":
            self.require(len(self.dataset(hf, "seq_lens", "iu")) == n, "seq_lens/header count mismatch.")
        edges, expected = sizes[0], n * (n - 1) // 2
        self.require(edges <= expected, "Edge count exceeds possible unique undirected pairs.")
        self.report["network_pair_coverage"].update(status="all-pairs count reached" if edges == expected else "sparse",
                                                    observed=edges, expected=expected)
        self.report["metadata"].update(sequence_count=n, edge_count=edges, network_type=metadata.network_type,
                                         model_name=metadata.model_name[:256])
        if "sparsity_keep_count" in hf.attrs:
            import numpy as np
            selected = hf.attrs["sparsity_keep_count"]
            self.require(isinstance(selected, (int, np.integer)) and not isinstance(selected, (bool, np.bool_)), "sparsity_keep_count must be an integer.")
            selected = int(selected)
            self.require(selected >= 0, "Negative selected-edge count.")
            self.report["metadata"]["selected_edge_count"] = selected
            self.report["metadata"]["selected_count_matches"] = edges == selected
            if edges != selected:
                self.finding("warning", "Selected-edge count differs from stored edges. Extraction can inherit this attribute from the source network; it does not establish interrupted generation.")
        if "_resume" in hf:
            self.require(self.report["generation_completion"]["status"] != "complete", "Complete flag contradicts retained resume state.")
            self.report["generation_completion"] = dict(status="incomplete", evidence="Alignment writer retains _resume state until finalization.")
        self.report["checks_performed"].append("Network type, required datasets, ranks, dtypes, counts, and available completion metadata")
        self.report["checks_omitted"].append("Edge indices, ordering, uniqueness, scores, and finite values were not read or validated.")

    def sparse_msa(self, hf):
        import h5py
        import numpy as np
        self.flag(hf)
        self.require("matrix" in hf and isinstance(hf["matrix"], h5py.Group), "Missing CSR matrix group.")
        shape = hf.attrs.get("shape")
        self.require(shape is not None and np.asarray(shape).shape == (2,) and np.asarray(shape).dtype.kind in "iu", "Invalid root matrix shape.")
        rows, columns = map(int, shape)
        self.require(rows >= 0 and columns >= 0, "Negative matrix dimensions.")
        self.require(np.array_equal(shape, hf["matrix"].attrs.get("shape")), "Root and matrix shapes disagree.")
        data = self.dataset(hf, "matrix/data", "iu")
        indices = self.dataset(hf, "matrix/indices", "iu")
        indptr = self.dataset(hf, "matrix/indptr", "iu")
        self.require(len(data) == len(indices) and len(indptr) == rows + 1, "CSR dataset lengths disagree with shape.")
        headers = self.dataset(hf, "headers")
        self.require(len(headers) == rows and h5py.check_string_dtype(headers.dtype) is not None, "Invalid sparse MSA headers.")
        mappings = {}
        for key in ("header_map", "aa_map", "int_to_aa"):
            self.tick()
            self.require(key in hf and isinstance(hf[key], h5py.Dataset) and hf[key].shape == (), f"Invalid mapping dataset {key}.")
            raw = hf[key][()]
            if isinstance(raw, bytes): raw = raw.decode("utf-8")
            if len(raw) > MAX_JSON_BYTES: raise InspectionLimit("Mapping metadata exceeds inspection limit.")
            mapping = json.loads(raw)
            self.require(isinstance(mapping, dict), f"{key} must encode a JSON object.")
            mappings[key] = mapping
        for index, value in enumerate(mappings["header_map"].values()):
            self.tick()
            if index >= MAX_METADATA_RECORDS: raise InspectionLimit("Mapping entry limit reached.")
            self.require(type(value) is int and 0 <= value < rows, "Header mapping row is out of range.")
        for residue, code in mappings["aa_map"].items():
            self.tick()
            self.require(type(code) is int and mappings["int_to_aa"].get(str(code)) == residue, "Amino-acid mappings disagree.")
        for index in range(rows):
            self.tick()
            if index >= MAX_METADATA_RECORDS: raise InspectionLimit("Sparse MSA header limit reached.")
            self.require(mappings["header_map"].get(headers.asstr()[index]) == index,
                         "Sparse MSA header mapping disagrees with its row.")
        self.report["metadata"].update(sequence_count=rows, alignment_length=columns, stored_values=len(data))
        self.report["checks_performed"].append("Sparse MSA object types, CSR dimensions/dtypes, headers shape, and mapping metadata")
        self.report["checks_omitted"].append("CSR pointers, indices, values, and reconstructed alignment were not scanned.")

    def layout_cache(self, hf):
        import h5py

        for key in ("cache_manifest_id", "layout_compatibility_id", "layout_compatibility_json"):
            self.require(key in hf.attrs, f"Missing layout cache attribute: {key}.")
        headers = self.dataset(hf, "headers")
        positions = hf.get("positions")
        self.require(positions is not None and isinstance(positions, h5py.Dataset), "Missing positions dataset.")
        self.require(positions.ndim == 2 and positions.shape[1] == 2, "positions must be a two-dimensional Nx2 dataset.")
        self.require(positions.dtype.kind in "f", "positions must have float dtype.")
        count = len(headers)
        self.require(count == positions.shape[0], "Headers count and positions row count disagree.")
        self.require(count > 0, "Layout cache contains 0 nodes.")

        manifest_id = hf.attrs["cache_manifest_id"]
        if isinstance(manifest_id, bytes):
            manifest_id = manifest_id.decode("utf-8")
        compat_id = hf.attrs["layout_compatibility_id"]
        if isinstance(compat_id, bytes):
            compat_id = compat_id.decode("utf-8")

        self.report["metadata"].update(
            node_count=count,
            cache_manifest_id=str(manifest_id),
            layout_compatibility_id=str(compat_id),
        )
        self.report["checks_performed"].append("Layout cache datasets (headers, positions) and compatibility attributes")

        manifest_file = self.path.parent / "cache_manifest.json"
        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as handle:
                    manifest_data = json.load(handle)
                if manifest_data.get("manifest_id") == manifest_id:
                    self.report["checks_performed"].append("Companion cache_manifest.json ID verified")
                    self.report["generation_completion"] = dict(
                        status="complete",
                        evidence=f"Layout cache contains {count} nodes with valid manifest_id {manifest_id[:16]}."
                    )
                else:
                    self.finding("warning", "Companion cache_manifest.json manifest_id does not match layout cache.")
            except Exception as error:
                self.finding("warning", f"Could not read companion cache_manifest.json: {error}")
        else:
            self.report["generation_completion"] = dict(
                status="complete",
                evidence=f"Layout cache contains {count} nodes and valid compatibility metadata."
            )
        self.report["checks_omitted"].append("Positions coordinates were not checked for node overlap or canvas bounds.")

    def lines(self):
        with open(self.path, "rb") as handle:
            number = 0
            while True:
                self.tick()
                raw = handle.readline(MAX_TEXT_LINE + 1)
                if not raw: break
                number += 1
                self.report["metadata"]["bytes_examined"] = handle.tell()
                if len(raw) > MAX_TEXT_LINE: raise InspectionLimit("Text line exceeds 1 MiB inspection limit.")
                self.report["metadata"]["lines_examined"] = number
                yield number, raw.decode("utf-8-sig" if number == 1 else "utf-8").rstrip("\r\n")

    def fasta(self, aligned):
        count, length, first_length, active = 0, 0, None, False
        issues = 0
        def finish():
            nonlocal count, first_length
            if not active: return
            count += 1
            if not length: self.finding("warning", f"FASTA record {count} has an empty sequence; sanitization may remove it.")
            if first_length is None: first_length = length
            if aligned and length != first_length: self.finding("error", f"Aligned FASTA record {count} has a different length.")
            self.report["metadata"]["records_checked"] = count
        for number, line in self.lines():
            stripped = line.strip()
            if not stripped: continue
            if stripped.startswith(">"):
                finish()
                if not stripped[1:].strip(): self.finding("warning", f"Empty header on line {number}.")
                active, length = True, 0
            else:
                self.require(active, f"Sequence text precedes the first header on line {number}.")
                length += len(stripped)
                if set(stripped) - set("ACDEFGHIKLMNPQRSTVWYXBZJUO-."):
                    issues += 1
        finish()
        self.require(count > 0, "No FASTA records found.")
        if issues: self.finding("warning", f"{issues} sequence lines contain lowercase or nonstandard characters; inspect sanitization behavior for the selected tool.")
        self.report["metadata"].update(sequence_count=count)
        if aligned: self.report["metadata"]["alignment_length"] = first_length
        self.report["checks_performed"].append("FASTA record structure" + (" and equal aligned lengths" if aligned else ""))
        self.report["checks_omitted"].append("FASTA has no reliable generation completion marker; raw sequences were not altered.")

    def blast(self, parameters):
        from utilities.BLAST_Tabular import _field_index, SUBJECT_FIELD_NAMES, EVALUE_FIELD_NAMES
        layout = parameters.get("BLAST_LAYOUT", "standard_outfmt6")
        columns = [parameters.get(k, default) - 1 for k, default in (("QUERY_COLUMN", 1), ("SUBJECT_COLUMN", 2), ("EVALUE_COLUMN", 11))]
        width, current_query, fields, previous_fields, rows = None, None, None, None, 0
        subject_index, evalue_index = None, None
        for number, line in self.lines():
            if not line.strip(): continue
            if line.startswith("#"):
                if line.startswith("# Query:"):
                    current_query = line[len("# Query:"):].strip()
                    fields = None
                if line.startswith("# Fields:"):
                    fields = [field.strip().lower() for field in line[len("# Fields:"):].split(",")]
                    if layout == "outfmt7_fields":
                        self.require(previous_fields is None or previous_fields == fields, f"BLAST field schema changed on line {number}.")
                        previous_fields = fields
                        subject_index = _field_index(fields, SUBJECT_FIELD_NAMES, "subject", number)
                        evalue_index = _field_index(fields, EVALUE_FIELD_NAMES, "E-value", number)
                continue
            parts = line.split("\t")
            if layout == "standard_outfmt6":
                self.require(len(parts) == 12, f"BLAST line {number} must contain 12 tab-separated columns.")
                required_columns = (0, 1, 10)
            elif layout == "outfmt7_fields":
                self.require(bool(current_query) and fields is not None, f"BLAST line {number} lacks Query/Fields context.")
                self.require(len(parts) == len(fields), f"BLAST line {number} disagrees with Fields declaration.")
                required_columns = (subject_index, evalue_index)
            else:
                if width is None: width = len(parts)
                self.require(len(parts) == width and len(parts) > max(columns), f"Custom BLAST line {number} has inconsistent/missing columns.")
                required_columns = columns
            self.require(all(parts[index].strip() for index in required_columns), f"BLAST line {number} has an empty required field.")
            rows += 1
            self.report["metadata"]["data_rows_checked"] = rows
        self.report["metadata"].update(layout=layout, data_rows=rows)
        if not rows: self.finding("warning", "No BLAST data rows; this may be a legitimate no-hit result.")
        self.report["checks_performed"].append("BLAST text decoding, comment context, and row widths")
        self.report["checks_omitted"].append("BLAST numeric values and FASTA header matching were not validated.")

    def settings(self, tool_id, root):
        from mcp_server.Pipeline_Settings import normalize_pipeline_settings
        from tools.tool_helpers.Tool_Pipeline import list_tool_specs
        if self.path.stat().st_size > MAX_JSON_BYTES: raise InspectionLimit("Settings JSON exceeds 8 MiB inspection limit.")
        with open(self.path, encoding="utf-8") as handle: document = json.load(handle)
        self.require(isinstance(document, dict), "Settings JSON must contain an object.")
        known = {s.settings_section for s in list_tool_specs()} | {"DIRECTORIES"}
        self.require(all(key in known for key in document), "Unknown settings section.")
        self.require(isinstance(document.get("DIRECTORIES"), dict), "Settings require a DIRECTORIES object.")
        self.require(all(isinstance(value, dict) for value in document.values()), "Settings sections must be objects.")
        self.report["metadata"]["sections"] = sorted(document)
        if tool_id:
            preview = normalize_pipeline_settings(tool_id, root, settings_document=document)
            for error in preview["errors"]:
                self.finding("error", f"{error['field']}: Invalid or missing setting; use validate_pipeline_settings for details.")
            self.report["checks_performed"].append("Strict MCP configuration validation for selected tool")
        else:
            self.report["checks_performed"].append("Settings outer document structure and recognized sections")
            self.report["checks_omitted"].append("No intended tool supplied; individual parameter validity and job readiness are not established.")


def inspect_local(path, file_type="auto", tool_id=None, parameters=None, project_root=None, budget=18):
    import h5py
    path = Path(path)
    inspector = Inspector(path, time.monotonic() + budget)
    report = inspector.report
    before = None
    try:
        before = path.stat()
        report.update(size_bytes=before.st_size, modified_ns=before.st_mtime_ns)
        inspector.require(path.is_file(), "Inspection requires a regular file.")
        with open(path, "rb") as handle: prefix = handle.read(65536)
        kind = file_type
        is_hdf = h5py.is_hdf5(path)
        if is_hdf or prefix.startswith(b"\x89HDF") or kind in ("embedding", "network", "sparse_msa", "layout_cache"):
            with h5py.File(path, "r") as hf:
                if kind == "auto":
                    if "positions" in hf and "cache_manifest_id" in hf.attrs:
                        kind = "layout_cache"
                    else:
                        markers = [name for name, present in (("embedding", "embeddings" in hf or "generation_complete" in hf.attrs),
                            ("sparse_msa", "matrix" in hf), ("network", "i" in hf or "j" in hf)) if present]
                        if len(markers) != 1: raise InspectionLimit("HDF5 format is unsupported or ambiguous; specify file_type.")
                        kind = markers[0]
                elif kind != "layout_cache" and "positions" in hf:
                    raise InspectionLimit("File appears to be a layout cache, but requested format is different.")
                inspector.require(kind in ("embedding", "network", "sparse_msa", "layout_cache"), "Requested text format does not match HDF5 contents.")
                report["detected_format"] = kind
                getattr(inspector, kind)(hf)
        else:
            if kind == "auto":
                text = prefix.decode("utf-8-sig", errors="replace").lstrip()
                if text.startswith(">"):
                    kind = "alignment_fasta" if tool_id == "sparse_msa_converter" else "fasta"
                elif text.startswith(("{", "[")): kind = "settings"
                elif text.startswith("# BLAST"): kind = "blast_tabular"
                else: raise InspectionLimit("Unrecognized or ambiguous text; specify file_type, especially for plain BLAST tables.")
            report["detected_format"] = kind
            if kind in ("fasta", "alignment_fasta"): inspector.fasta(kind == "alignment_fasta")
            elif kind == "blast_tabular": inspector.blast(parameters or {})
            elif kind == "settings": inspector.settings(tool_id, project_root)
            else: raise InspectionLimit("Unsupported file type.")
        report["inspection_finished"] = True
        if report["structural_validity"] != "invalid": report["structural_validity"] = "valid"
    except InspectionLimit as error:
        inspector.finding("warning", str(error))
        report["checks_omitted"].append("Inspection stopped before all structural checks completed.")
    except (OSError, PermissionError) as error:
        inspector.finding("warning", f"Unable to read file: {error}")
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError) as error:
        inspector.finding("error", str(error))
    finally:
        if before is not None:
            try:
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    report.update(unstable=True, structural_validity="unknown", inspection_finished=False)
                    inspector.finding("warning", "File changed during inspection; results are unstable.")
            except OSError:
                report.update(unstable=True, structural_validity="unknown", inspection_finished=False)
        if report["unstable"] or (report["generation_completion"]["status"] == "complete" and
                (not report["inspection_finished"] or report["structural_validity"] != "valid")):
            report["generation_completion"] = dict(status="unknown", evidence="Completion could not be confirmed because inspection failed, was incomplete, or the file changed.")
    return report


def inspect_pipeline_file(path, project_root, file_type="auto", tool_id=None, parameters=None):
    from mcp_server.Pipeline_Settings import inspection_parameters
    if file_type not in FILE_TYPES: raise ValueError(f"file_type must be one of {FILE_TYPES}.")
    if parameters is not None and tool_id is None: raise ValueError("parameters requires tool_id.")
    effective = inspection_parameters(tool_id, {} if parameters is None else parameters, project_root) if tool_id else {}
    resolved = Path(os.path.abspath(os.path.join(project_root, os.fspath(path))))
    report = base_report(resolved)
    request = dict(path=str(resolved), file_type=file_type, tool_id=tool_id, parameters=effective, project_root=str(project_root))
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())], input=json.dumps(request),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if result.returncode: raise ValueError(f"Inspection helper exited with code {result.returncode}.")
        output = json.loads(result.stdout)
        if not isinstance(output, dict) or not report.keys() <= output.keys(): raise ValueError("Invalid helper report.")
        return output
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["findings"] = [dict(severity="warning", message=f"Inspection unavailable: {error}"[:512])]
        return report


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(io.StringIO()):
        result = inspect_local(**request)
    print(json.dumps(result, allow_nan=False))
