# 🧬 Pairwise Embedding Alignment (`Embedding_PWA.py`)

This script aligns two sequences using their residue-level language model embeddings. It calculates dynamic programming alignment strings (Needleman-Wunsch or Smith-Waterman) and maps user-specified residue positions from the reference sequence directly onto target sequence positions to facilitate active site and feature comparison.

### 📥 Input

#### Embedding Database `INPUT_EMBED`
*   **Format**: Metadata-first HDF5 embedding database (`.h5`) containing sanitized headers, sequences, model metadata, and residue-level tensors. It is required whenever either sequence is selected by header.
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Reference Header **`REF_HEADER`** | The reference header to find in the embedding database. Typed text receives the same canonical header sanitization used by Generate Embeddings. |
| Target Header **`TAR_HEADER`** | The target header to find in the embedding database. Typed text receives the same canonical header sanitization used by Generate Embeddings. |
| Manual Reference Toggle **`MANUAL_REF_SEQ`** | Enables the optional manual reference-sequence field. When ON, the manual sequence is used even if its sanitized header matches a stored database header. It is OFF by default. |
| Manual Reference String **`REF_SEQUENCE`** | Used only while `MANUAL_REF_SEQ` is ON and canonically sanitized before embedding generation; otherwise the reference is loaded by header. |
| Manual Target Toggle **`MANUAL_TAR_SEQ`** | Enables the optional manual target-sequence field. When ON, the manual sequence is used even if its sanitized header matches a stored database header. It is OFF by default. |
| Manual Target String **`TAR_SEQUENCE`** | Used only while `MANUAL_TAR_SEQ` is ON and canonically sanitized before embedding generation; otherwise the target is loaded by header. |
| Highlight Mapping Positions **`HIGHLIGHT_POSITIONS`** | A comma-separated list of 1-indexed residue positions in the reference sequence to map and highlight in the target sequence alignment. |
| Embedding Model **`EMBEDDING_MODEL`** | Model used when both sequences are manual. Available choices are discovered from `src/resources/pLM_models`; if either sequence comes from the embedding set, that set's model is used. |
| Alignment Metric **`ALIGNMENT_MODE`** | The alignment mode to run (either 'global' or 'local'). |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap penalty score for local alignments. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap penalty score for global alignments. |
| Generate Report **`GENERATE_REPORT`** | Toggle to compile and save a comprehensive HTML alignment report showing residue highlights and scores. |

### 📤 Output

#### Alignment Result and Optional HTML Report
*   **Console Output**: Always prints the alignment mode and score, input lengths, aligned residue strings, match marks, and any requested reference-to-target position mappings.
*   **Optional File**: When **Generate Report** (`GENERATE_REPORT`) is enabled, writes a timestamped `PWA_Report_*.html` document to the configured report directory with the same alignment and highlighted mappings. No file is written when the toggle is disabled.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Embedding Extraction / Generation**:
     Loads stored sequence text and tensors directly from the HDF5 manifest. Typed headers and manual sequences are canonically sanitized first. If only one sequence is manual, its embedding uses the database model. If both are manual, they use the explicitly selected embedding model.

2. **Z-Score Score Matrix Construction**:
     Calculates the normalized residue-level score matrix:
     $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$

3. **Traceback Alignment**:
     * **global**: Computes Needleman-Wunsch recurrence matrix with `GLOBAL_GAP_P`.
     * **local**: Subtracts 2.0 from the scores and computes Smith-Waterman recurrence matrix with `LOCAL_GAP_P`:
       $$\text{Score}_{\text{local}}(a, b) = \text{Score}(a, b) - 2.0$$
     
     Traceback yields the alignment string mapping.

4. **Residue Position Mapping**:
     For each 1-indexed reference highlight position $p_{\text{ref}}$, it tracks the aligned index:
     $$p_{\text{ref}} \to p_{\text{aligned}} \to p_{\text{tar}}$$
     
     This maps catalytic residues or features from the reference directly onto the target sequence.

</details>

---

# 🔍 Embedding Database Search (`Embedding_SSEARCH.py`)

This script queries a single sequence against an entire database using residue-level language model embeddings. By running parallel pairwise alignments against all database sequences, it ranks matching proteins by normalized local or global similarity scores, operating similarly to FASTA ssearch.

### 📥 Input

#### Embedding Database `INPUT_EMBED`
*   **Format**: A complete metadata-first HDF5 database (`.h5`) containing sanitized headers, sequences, and embeddings.
*   **Created By**: `Generate_Embeddings.py` or another active embedding writer. SSEARCH does not generate a missing database automatically.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Query Header ID **`QUERY_HEADER`** | A header stored in the embedding database. It is sanitized before lookup. |
| Manual Query Toggle **`MANUAL_QUERY_SEQ`** | Enables the optional manual query-sequence field. When ON, the query remains distinct from a same-header database record, which can still appear as a search hit. It is OFF by default. |
| Manual Query String **`QUERY_SEQUENCE`** | Used only while `MANUAL_QUERY_SEQ` is ON, then sanitized in memory and embedded with the database model's pLM plugin. |
| Output Spreadsheet Prefix **`OUTPUT_NAME`** | The prefix for the exported search results spreadsheet and optional FASTA files. |
| Max Database Hits **`TOP_K`** | The maximum number of top-scoring database hits to include in the output report. |
| Normalized Score Cutoff **`NORM_THRESHOLD`** | A filter to exclude hits scoring below a normalized similarity cutoff. Set to 'None' to disable. |
| Alignment Mode **`ALIGNMENT_MODE`** | The search alignment mode (either 'global' or 'local'). |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap penalty score for local alignments. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap penalty score for global alignments. |
| Score Normalization Mode **`NORM_MODE`** | The score normalization method (e.g., alignment_length, shorter_sequence, longer_sequence, average_sequence). |
| CPU Worker Threads **`WORKERS`** | The number of CPU threads allocated for parallel alignment calculations. |
| Device **`DEVICE_SELECTION`** | Selects automatic hardware benchmarking or a concrete CPU/accelerator device. |
| Accelerator Precision **`ACCELERATOR_PRECISION`** | `auto` uses IEEE FP32 for small searches and considers validated TF32 only from 4,096 targets. Forced TF32 requires NVIDIA CUDA. |
| Export Top Hits FASTA **`GENERATE_FASTA`** | Toggle to export a FASTA file containing the sequences of the top *K* database hits. |

### 📤 Output

#### Embedding Search Results
*   **Text Report**: `Report_<name>.txt`, containing parameters and the full ranked hit table; the console shows at most the first 100 hits.
*   **Excel Workbook**: `Report_<name>.xlsx`, with a metadata-viewer-compatible `Search Results` sheet and a `Search Parameters` sheet. The first sheet follows `docs/metadata_template.xlsx`: row 1 contains property names, row 2 contains data types, and column A contains exact sequence headers for strict node matching.
*   **Optional FASTA**: When `GENERATE_FASTA` is enabled, `Hits_<name>.fasta` contains the query followed by ranked hit sequences.
*   **Metadata Columns**: `Node ID`, `Rank`, `Norm_Score`, `Raw_Score`, `Sequence_Length`, and `Alignment_Length`.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Query Setup**:
     Reads stored sequences from `/sequences`. The manual switch explicitly selects the source: OFF reuses the stored header and embedding, while ON requires a sanitized manual sequence and embeds it through the model adapter recorded by `model_name`. A colliding manual header does not replace or suppress the same-header database record.

2. **Database Alignment Queue**:
     Iterates through all database sequences $j$ in the HDF5 file. Below 512 targets it retains scalar execution. For larger CUDA searches, the query is normalized and uploaded once while targets are read once, grouped by length, and scored in VRAM-bounded batches.

3. **Multithreaded dynamic programming**:
     Allocates alignments to multiprocessing workers. Each worker:
     - Computes the residue-level normalized similarity matrix:
       $$\text{Score}_j(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$
     - Solves alignment scores:
       * **global**: $$S_{\text{raw}} = \text{NW}(\text{Score}_j, \text{gap}_g)$$
       * **local**: $$S_{\text{raw}} = \text{SW}(\text{Score}_j - 2.0, \text{gap}_l)$$

4. **Score Normalization**:
     Applies the length normalization factor based on `NORM_MODE`:
     $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\text{Normalization\_Factor}(L_q, L_j)}$$

5. **Sorting & Filtering**:
     Collects results, filters by `NORM_THRESHOLD`, sorts in descending order of $S_{\text{norm}}$, and keeps the top $K$ hits.

</details>
