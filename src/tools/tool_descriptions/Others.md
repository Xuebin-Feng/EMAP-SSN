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
*   **Console Output**: Always prints the alignment mode and score, input lengths, alignment length, percent identity, aligned residue strings, match marks, and any requested reference-to-target position mappings.
*   **Percent Identity**: Printed as `identical/alignment length (percent)` with its definition: identical standard amino acids divided by the alignment length, which includes internal gaps for local alignments and the full length, end gaps included, for global alignments. A `|` mark shows each counted pair, `.` any other aligned pair, and a blank a gap; X, B, Z, J, U, and O never count as identical.
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
     
     Traceback follows the move recorded for each cell, so the displayed path is the optimal one. Ties resolve as match, then deletion, then insertion, the same order the `Embedding_SSEARCH.py` kernels use.

4. **Percent Identity**:
     Counts the identical standard amino acids $N_{\text{id}}$ on the traced path and divides by its length $L_{\text{aln}}$:
     $$\text{Identity} = 100 \times \frac{N_{\text{id}}}{L_{\text{aln}}}$$

5. **Residue Position Mapping**:
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
| Score Normalization Mode **`NORM_MODE`** | The score normalization method (e.g., alignment_length, shorter_sequence, longer_sequence, average_sequence). `alignment_length` is unavailable for local alignments. |
| CPU Worker Threads **`WORKERS`** | The number of CPU threads allocated for parallel alignment calculations. |
| Device **`DEVICE_SELECTION`** | Selects automatic hardware benchmarking or a concrete CPU/accelerator device. |
| Accelerator Precision **`ACCELERATOR_PRECISION`** | `automatic_32bit` uses IEEE FP32 for small searches and considers validated TF32 only from 4,096 targets. Explicit `bf16` uses BF16 matmul operands with FP32 normalization and postprocessing and requires a capable CUDA/ROCm, XPU, or MPS accelerator. It prints a low-precision warning and reports FP32-relative selected-mode length and raw-score statistics on a separate length-stratified sample of up to 2,048 targets for each device/execution variant. Finite numerical differences are informational and never reject BF16. The legacy alias `auto` remains accepted. Forced TF32 requires NVIDIA CUDA. |
| Export Top Hits FASTA **`GENERATE_FASTA`** | Toggle to export a FASTA file containing the sequences of the top *K* database hits. |

### 📤 Output

#### Embedding Search Results
*   **Text Report**: `Report_<name>.txt`, containing parameters, the percent-identity definition, and the full ranked hit table; the console shows at most the first 100 hits.
*   **Excel Workbook**: `Report_<name>.xlsx`, with a metadata-viewer-compatible `Search Results` sheet and a `Search Parameters` sheet. The first sheet follows `docs/metadata_template.xlsx`: row 1 contains property names, row 2 contains data types, and column A contains exact sequence headers for strict node matching.
*   **Optional FASTA**: When `GENERATE_FASTA` is enabled, `Hits_<name>.fasta` contains the query followed by ranked hit sequences.
*   **Metadata Columns**: `Node ID`, `Rank`, `Norm_Score`, `Raw_Score`, `Sequence_Length`, `Alignment_Length`, and `Percent_Identity`.
*   **Percent Identity**: The number of identical standard amino acids on the reported alignment path as a percentage (0–100) of `Alignment_Length`. Local alignments count internal gaps; global alignments count the full length, end gaps included. X, B, Z, J, U, and O never count as identical.

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

5. **Percent Identity**:
     The same dynamic-programming pass counts the identical standard amino acids $N_{\text{id}}$ on the selected path, whose length $L_{\text{aln}}$ includes internal gaps for **local** alignments and every column, end gaps included, for **global** alignments:
     $$\text{Identity} = 100 \times \frac{N_{\text{id}}}{L_{\text{aln}}}$$
     Because the path maximizes embedding similarity rather than residue matches, the identity can differ from that of a substitution-matrix alignment of the same pair.

6. **Sorting & Filtering**:
     Collects results, filters by `NORM_THRESHOLD`, sorts in descending order of $S_{\text{norm}}$, and keeps the top $K$ hits.

</details>

---

# ⏱️ Benchmark

**Run Benchmark** times EMAP-SSN's heavy calculations on a fixed set of public protein sequences, so reports from different computers and versions can be compared. It asks before it starts, then runs `src/resources/benchmark/Run_Benchmark.py` in a console window of its own, which shows the progress and, at the end, the report.

### 📥 Input

#### Bundled Sequence Set
*   **Sequences**: 860 reviewed UniProtKB/Swiss-Prot entries of the metallo-β-lactamase superfamily (InterPro IPR001279, release 2026_03), with 92 more held out for the injection stage. `src/resources/benchmark/README.md` says where they come from and how they were made.
*   **Model**: ESM-2 8M (`esm2_t6_8m`). The first run downloads it into the Hugging Face cache, about 30 MB; the download is not timed.

### ⚙️ Stages

| Stage | What It Times |
| :--- | :--- |
| 1. Sanitize sequences | `Sanitize_Sequences.py` on the main set. |
| 2. Embeddings | `Generate_Embeddings.py` with ESM-2 8M. |
| 3. All-against-all embedding alignment | `Align_Similarity_Matrix.py`: 369,370 pairs. |
| 4. SSN layout | The force-directed layout of the top 5% of edges. |
| 5. UMAP layout | The same network laid out with UMAP. |
| 6. Clustering | Leiden, MCL and the Jaccard filter, as the Viewer's `cluster` command runs them. |
| 7. Embedding database search | `Embedding_SSEARCH.py`: one query against the main set. |
| 8. Injection of new sequences | `Embedding_Injection.py` and `Network_Injection.py` with the 92 held-out sequences. |
| 9. Embedding MSA | `Embedding_MSA.py` on 100 glyoxalase II sequences. |
| 10. BLAST all-against-all alignment | `Align_Substitution_Matrix.py`, with NCBI BLAST+. |

Every device setting stays on **Auto**, so each tool runs its own hardware trials, and the report records what each one chose. A stage this computer can't run, such as BLAST without NCBI BLAST+, is skipped with the reason, and so are the stages that need it.

### 📤 Output

#### Benchmark Report
*   **Text Report**: `Benchmark_Report_<date>_<time>.txt` in the benchmark folder, which **Report Folder** opens. It is written in the window's language and records each stage's time, throughput, CPU time, memory peaks and device; every Auto decision; the hardware and software; and the conditions that affect the numbers.
*   **Data**: An English `.json` with the same results beside the report.
*   **Kept**: Reports are never deleted. A second run started in the same second adds `_2` to the name.

<details markdown="1">
<summary><b>How a Run Works</b></summary>

1. **Its own files**:
     A run keeps every file it makes in the benchmark folder's `temp/`, with a lock that stops a second benchmark while one runs, and clears `temp/` when it starts and when it ends. It never reads or changes the saved tool settings or the directories set in this window.

2. **One process per stage**:
     Each stage runs in a process of its own, which the benchmark measures from outside: wall time, CPU time and the peak memory of every process the stage starts.

3. **Time and space**:
     A run took about 6 minutes on a computer with an RTX 5070 Ti and a Core Ultra 7 265KF, and takes longer without a GPU. It needs about 1 GB of free disk space.

4. **Stopping**:
     Pressing Ctrl+C in the console stops the run and still writes the report of the stages that ran. Closing the console stops it without a report; the next run clears what it left.

5. **Comparing**:
     Compare reports only when their benchmark protocol and sequence set match. Other programs running at the same time, a laptop on battery, or a first run that still compiles its kernels make a run slower, and the report notes each of these.

</details>
