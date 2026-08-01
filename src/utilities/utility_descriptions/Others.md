# 🧬 Pairwise Embedding Alignment (`Embedding_PWA.py`)

This script aligns two sequences using their residue-level language model embeddings. It calculates dynamic programming alignment strings (Needleman-Wunsch or Smith-Waterman) and maps user-specified residue positions from the reference sequence directly onto target sequence positions to facilitate active site and feature comparison.

### 📥 Input

#### Pairwise Sequence Set `INPUT_FASTA`
*   **Format**: FASTA sequence file (`.fasta`) containing reference and target sequences.
*   **Created By**: `Sanitize_Sequences.py` (Sequence Sanitization utility) or user-provided raw FASTA.

#### Embedding Database `INPUT_EMBED`
*   **Format**: HDF5 embedding database (`.h5`). If missing, calculations run on the fly.
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Reference Header **`REF_HEADER`** | The exact FASTA header of the reference sequence. |
| Target Header **`TAR_HEADER`** | The exact FASTA header of the target sequence. |
| Manual Reference String **`REF_SEQUENCE`** | Manually input the reference sequence (ignores headers and FASTA files if provided). |
| Manual Target String **`TAR_SEQUENCE`** | Manually input the target sequence (ignores headers and FASTA files if provided). |
| Highlight Mapping Positions **`HIGHLIGHT_POSITIONS`** | A comma-separated list of 1-indexed residue positions in the reference sequence to map and highlight in the target sequence alignment. |
| Alignment Metric **`ALIGNMENT_MODE`** | The alignment mode to run (either 'global' or 'local'). |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap penalty score for local alignments. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap penalty score for global alignments. |
| Generate Report **`GENERATE_REPORT`** | Toggle to compile and save a comprehensive HTML alignment report showing residue highlights and scores. |

### 📤 Output

#### HTML Alignment Report
*   **Format**: HTML document (`.html`).
*   **Description**: Visual report showing aligned residue matching arrays, gap spaces, and target mapping positions.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Embedding Extraction / Generation**:
     Fetches residue embeddings for the reference sequence $v_{\text{ref}}$ and target sequence $v_{\text{tar}}$ from the HDF5 database. If missing, it uses the default local model (e.g., ESM-C) to compute them.

2. **Z-Score Score Matrix Construction**:
     Calculates the normalized residue-level score matrix:
     $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$

3. **Traceback Alignment**:
     * **global**: Computes Needleman-Wunsch recurrence matrix with `GLOBAL_GAP_P`.
     * **local**: Subtracts 2.0 from the scores and computes Smith-Waterman recurrence matrix with `LOCAL_GAP_P`:
       $$\text{Score}_{\text{local}}(a, b) = \text{Score}(a, b) - 2.0$$
     
     Traceback yields the alignment string mapping.

4. **Residue Position Mapping**:
     For each 1-indexed target highlight position $p_{\text{ref}}$ in the reference sequence, it tracks the aligned index:
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
| Manual Query String **`QUERY_SEQUENCE`** | A sequence sanitized in memory and embedded with the database model's pLM plugin; overrides header lookup. |
| Output Spreadsheet Prefix **`OUTPUT_NAME`** | The prefix for the exported search results spreadsheet and optional FASTA files. |
| Max Database Hits **`TOP_K`** | The maximum number of top-scoring database hits to include in the output report. |
| Normalized Score Cutoff **`NORM_THRESHOLD`** | A filter to exclude hits scoring below a normalized similarity cutoff. Set to 'None' to disable. |
| Alignment Mode **`ALIGNMENT_MODE`** | The search alignment mode (either 'global' or 'local'). |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap penalty score for local alignments. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap penalty score for global alignments. |
| Score Normalization Mode **`NORM_MODE`** | The score normalization method (e.g., alignment_length, shorter_sequence, longer_sequence, average_sequence). |
| CPU Worker Threads **`WORKERS`** | The number of CPU threads allocated for parallel alignment calculations. |
| Export Top Hits FASTA **`GENERATE_FASTA`** | Toggle to export a FASTA file containing the sequences of the top *K* database hits. |

### 📤 Output

#### Embedding Search Results
*   **Format**: spreadsheet table (`.csv` or `.xlsx`).
*   **Description**: Ranked sheet listing sequence IDs, local/global alignment scores, and alignment lengths.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Query Setup**:
     Reads stored sequences from `/sequences`. A header query reuses its stored embedding; a manual query is sanitized and embedded through the same model adapter recorded by `model_name`.

2. **Database Alignment Queue**:
     Iterates through all database sequences $j$ in the HDF5 file. For each sequence, it adds the pair (query, j) to a parallel queue.

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
