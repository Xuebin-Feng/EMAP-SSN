# Embedding Injection (`Embedding_Injection.py`)

This script injects new sequence embeddings into an existing HDF5 embedding database. It scans the incoming FASTA sequence list, extracts pre-computed embeddings directly from the database for existing matches, and computes embeddings only for the newly added sequences to optimize compute time.

### Input

#### Target Embedding Database `INPUT_EMBED`
*   **Format**: Complete metadata-first HDF5 database (`.h5`) with `/headers`, `/sequences`, `/embeddings`, `model_name`, `saving_mode`, `num_sequences`, and `generation_complete`.
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

#### Incoming Sequence Set `INPUT_FASTA`
*   **Format**: FASTA file (`.fasta`) containing original sequences plus newly appended targets.
*   **Created By**: User-compiled updated sequence set.
*   **Sanitization**: Records are sanitized automatically in memory without header-substring or sequence-length filtering. A sanitization result is printed only when records change.

### Parameters

This script does not require additional configuration parameters.

### Output

#### Updated HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`).
*   **Description**: Re-indexed metadata-first database containing sanitized headers and sequences plus embeddings for every record in the new FASTA. Existing embeddings are reused only when both sanitized header and sequence match.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Inventory and Discrepancy Parsing**:
     Reads and sanitizes the new input FASTA file, then collects all target headers:
     $$H_{\text{fasta}} = \{h_1, h_2, \dots, h_M\}$$
     
     Opens the existing HDF5 embedding file and reads the pre-computed headers:
     $$H_{\text{exist}} = \{e_1, e_2, \dots, e_N\}$$
     
     Identifies the subset of new sequences to embed using set subtraction:
     $$H_{\text{new}} = H_{\text{fasta}} \setminus H_{\text{exist}}$$
     Every existing header must remain present with an identical sanitized sequence; otherwise injection stops before creating the output.

2. **Model Identification and Setup**:
     Reads the metadata of the existing HDF5 file to identify the model architecture and precision (`float16` or `float32`) used. It loads the exact same model (e.g. ESM-C) to ensure vector consistency.

3. **Incremental Embedding Computation**:
     Feeds the sequence segments belonging to **H<sub>new</sub>** through the language model, calculating their residue-level embeddings.

4. **Synchronized Merge and HDF5 Serialization**:
     Iterates through $H_{\text{fasta}}$ in order. If a header belongs to $H_{\text{exist}}$, it copies the embedding dataset directly from the old file. If it belongs to $H_{\text{new}}$, it writes the newly computed embedding tensor:
     $$v_{\text{final}}(i) = \begin{cases} v_{\text{exist}}(i) & \text{if } h_i \in H_{\text{exist}} \\ v_{\text{new}}(i) & \text{otherwise} \end{cases}$$
     
     Writes and flushes sanitized `/headers` and `/sequences` first, copies or generates each embedding, then sets `generation_complete=true` only after final validation.

</details>

---

# Embedding Extraction (`Embedding_Extraction.py`)

This script extracts a subset of sequence embeddings from a master HDF5 database. By providing a list of target sequence headers (either as a FASTA or text file), it creates a smaller, filtered HDF5 embedding archive without running any model calculations.

### Input

#### Source Embedding Database `INPUT_EMBED`
*   **Format**: Complete metadata-first HDF5 embedding database (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

#### Target Whitelist Set `INPUT_FASTA`
*   **Format**: Target whitelist FASTA file (`.fasta`) or plain text file containing selected headers.
*   **Created By**: User-defined subset whitelist.
*   **Validation**: FASTA records are sanitized exactly as in generation and their sequences must match the source metadata. Text lists sanitize headers and obtain sequences from the source.

### Parameters

This script does not require additional configuration parameters.

### Output

#### Extracted HDF5 Embedding Archive
*   **Format**: HDF5 (`.h5`).
*   **Description**: Contains whitelisted embeddings plus one-to-one sanitized `/headers` and `/sequences` metadata.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Target List Gathering**:
     Parses the target sequence whitelist (from a FASTA file or text list) to compile the target headers:
     $$H_{\text{target}} = \{t_1, t_2, \dots, t_K\}$$

2. **Index Alignment and Intersection**:
     Iterates through the master HDF5 file's header dataset and filters out any sequence datasets not present in $H_{\text{target}}$:
     $$H_{\text{extract}} = H_{\text{target}} \cap H_{\text{master}}$$

3. **Validated dataset extraction**:
     Reads each selected residue-level embedding array, validates its dtype, shape, sequence length, and shared feature dimension, and writes it unchanged to the new database without model inference.

4. **Metadata Serialization**:
     Writes and flushes the selected headers and stored sequences before copying embeddings. The output is marked complete only after dtype, shape, sequence length, and feature dimension validation.

</details>

---

# Network Injection (`Network_Injection.py`)

This script performs incremental similarity network calculations. When new sequences are added to a project, it copies all pre-existing sequence-to-sequence alignment scores directly from the old network cache, aligning only the pairs the new network needs and the old one lacks, to save time and compute resources.

**Which pairs are aligned.** An old edge whose two sequences are both still in the new set is always kept, with its old scores. With a complete input network, every pair that involves a new sequence is aligned. With a sparse input network (made with Edge Pre-filtering), the tool measures every pair of the new set with the same similarity Edge Pre-filtering uses (max-pooled embeddings, cosine times the shorter-to-longer length ratio squared) and aligns every pair, old-old pairs without an edge included, whose similarity reaches a cut. The cut is the lower of two values:

1. the lowest similarity among the input network's edges that stay in the new set (recorded calibration pairs left out), and
2. the similarity that keeps the same share of the new set's pairs as the input network kept (its edges divided by all pairs of its sequences, recorded calibration pairs left out).

If the lowest old edge lies below the second value, everything above that edge is aligned, even though this keeps more pairs than the input network did. In the opposite case the kept share stays the same as in the input network. A sparse input network that has no edge at all has nothing to compare with, so every pair with a new sequence is aligned (and no pair of two old sequences, as with a complete input network). The tool prints the share, both values, the cut and which of the two set it, and how many pairs it queues (old-old, old-new, new-new).

### Input

#### Target Network File `OLD_NETWORK`
*   **Format**: Pre-existing HDF5 network database file (`.h5`).
*   **Created By**: `Align_Similarity_Matrix.py` (Embedding Alignment utility).

#### Updated Embedding Database `NEW_EMBEDDINGS`
*   **Format**: Target HDF5 embedding database (`.h5`) containing all embeddings.
*   **Created By**: `Embedding_Injection.py` (Embedding Injection utility).

### Parameters

| Parameter | Description |
| :--- | :--- |
| Gap Penalties | Automatically inherited directly from the input network (`OLD_NETWORK`). |
| CPU Workers **`WORKERS`** | CPU worker count for the CPU processing plan and a concurrency input for accelerator-plan tuning. The tool benchmarks available CPU/accelerator plans on representative pending pairs and falls back through successful plans if needed. |
| Processing Batch Size **`BATCH_SIZE`** | The number of sequence alignments calculated per write block, minimizing memory consumption and optimizing file write performance. |
| Device **`DEVICE_SELECTION`** | Selects automatic hardware benchmarking or one concrete device for calculating new residue alignments. Tiled CUDA/ROCm runs fuse Z-score normalization and dynamic programming on GPU (`SSN_ALIGN_GPU_KERNELS=0` disables); XPU, MPS, scalar, and CPU runs fall back to CPU dynamic programming. |
| Execution Mode **`EXECUTION_MODE`** | `Auto` compares scalar and tiled plans where supported. `Scalar` restricts tuning and production to one-matrix-at-a-time plans. `Tiled` uses memory-bounded embedding tiles and, on CUDA/ROCm, fused GPU alignment kernels (or padded PyTorch microbatches on XPU or when `SSN_ALIGN_GPU_KERNELS=0`), failing early if no compatible accelerator is available. |
| Host Cache **`HOST_CACHE_GB`** | Maximum GiB used to retain packed embeddings across batches; `auto` applies a safe RAM budget capped at 128 GiB and `0` disables it. |
| Matmul Precision | Inherited from `OLD_NETWORK`. Legacy networks are IEEE FP32; TF32 networks require NVIDIA CUDA so copied and new edges are never mixed. BF16 networks require a capable accelerator. Before calculating new edges, Network Injection prints a low-precision warning and reports FP32-relative alignment-length and raw-score statistics on up to 2,048 representative pairs; finite numerical differences never reject BF16. |
| Memory | A complete input network needs no similarity. A sparse one computes the similarity of every pair of the new set, which needs 12 bytes per squared sequence count at its peak; afterwards it holds the matrix (8 bytes per squared sequence count) and, for the cut, the values of the upper triangle (4 bytes per squared sequence count): about 23 GB at 44,000 sequences, the same order as Edge Pre-filtering in Align Similarity Matrix. The similarity is released once the pairs are chosen. |

### Output

#### Updated HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Description**: Re-indexed alignment network containing `/headers`, `/seq_lens`, `/i`, `/j`, `/g_score`, `/g_len`, `/l_score`, and `/l_len`, plus `model_name`, `saving_mode`, `gap_penalties`, `embedding_checksum`, and `matmul_precision` attributes. Existing gap penalties and arithmetic precision are inherited from `OLD_NETWORK`.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Mapping Setup**:
     Let the old network headers be $H_{\text{old}} = \{h_1, \dots, h_N\}$ and the new embedding headers be $H_{\text{new}} = \{h'_1, \dots, h'_M\}$. Cached edges are reusable only when both endpoint headers still occur in $H_{\text{new}}$.
     The script creates an index mapping dictionary to resolve old indices to new indices:
     $$\text{Map}_{\text{old} \to \text{new}}(i) = j \quad \text{such that} \quad h_i = h'_j$$

2. **Edge Classification**:
     For all pairwise combinations in the new network (u, v) (where 0 ≤ u < v < M):
     - **Case 1 (Cached Pair)**: If the old network contains that exact header pair, its global/local scores and lengths are copied.
     - **Case 2 (New Pair, Complete Old Network)**: If either endpoint is new, the pair is scheduled for active dynamic-programming alignment.
     - **Case 3 (Sparse Old Network)**: Let $s(u, v)$ be the max-pooled, length-adjusted cosine similarity of Edge Pre-filtering, $K$ the share of all old pairs the old network kept (recorded calibration pairs left out), $s_{\min}$ the lowest $s$ among the old edges that stay in the new set (calibration pairs left out) and $t'$ the $100\,(1-K)$-th percentile of $s$ over the pairs of the new set. The cut is $\min(s_{\min}, t')$. Every pair with $s \ge$ cut that is not an old edge is scheduled for alignment, old-old pairs included; every old edge is kept. When $s_{\min} < t'$ the kept share rises above $K$; otherwise it stays at $K$. With no old edge in the new set the cut is $t'$; with no old edge at all every pair with a new sequence is aligned and no old-old pair.

3. **Incremental Alignment**:
     Benchmarks available CPU/accelerator processing plans and sends scheduled new pairs through the best successful plan. On CUDA/ROCm (unless `SSN_ALIGN_GPU_KERNELS=0`), tiled runs use runtime-compiled fused GPU kernels (`Align_GPU_Kernels.py`) to run similarity scoring, exact two-pass Z-score normalization, and global+local dynamic programming end-to-end on the GPU without transferring score matrices to host memory. On XPU, MPS, scalar GPU, CPU, or when fused kernels are disabled, tiled/scalar plans compute score matrices (with persistent normalized-embedding tiles, length-bucketed targets with bounded padding, and device-memory preflight on tiled accelerators) and solve dynamic programming on CPU workers. Each pair:
     - Retrieves residue embeddings from the HDF5 database.
     - Calculates the normalized score matrix:
       $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$
     - Solves global and local dynamic programming alignments:
       $$\text{Global Pass} \to \text{NW}(\text{Score}, \text{gap}_g)$$
       $$\text{Local Pass} \to \text{SW}(\text{Score} - 2.0, \text{gap}_l)$$

4. **Consolidation**:
     Merges the copied scores with the newly calculated scores and writes the updated re-indexed network datasets (`i`, `j`, `g_score`, `g_len`, `l_score`, `l_len`) to the new output file.

</details>

---

# Network Extraction (`Network_Extraction.py`)

This script extracts sub-networks from a master HDF5 network based on a whitelist FASTA file. It retains only the alignment connections where both sequence nodes are in the whitelist, and re-indexes all remaining edge indices to produce a clean, self-contained filtered sub-network.

### Input

#### Source Network File `INPUT_NET`
*   **Format**: Master HDF5 network database file (`.h5`).
*   **Created By**: `Align_Similarity_Matrix.py` (Embedding Alignment utility) or `Align_Substitution_Matrix.py` / `Parse_BLAST_Output.py`.

#### Target Whitelist Set `INPUT_FASTA`
*   **Format**: Whitelist sequence FASTA file (`.fasta`) containing nodes to retain.
*   **Created By**: User-defined subset whitelist.

### Parameters

This script does not require additional configuration parameters.

### Output

#### Extracted HDF5 Sub-Network Archive
*   **Format**: HDF5 (`.h5`).
*   **Embedding-network schema**: Copies source attributes and writes re-indexed `/headers`, `/seq_lens`, `/i`, `/j`, `/g_score`, `/g_len`, `/l_score`, and `/l_len`.
*   **BLAST schema** (E-value or EFI alignment score): Copies source attributes, including `score_kind` of an EFI alignment score network, and writes re-indexed `/headers`, `/i`, `/j`, and `/score`. The file is named `<whitelist stem>_[BLAST]_EValue.h5`, with `[BLAST-EFI]` for an EFI alignment score network (and `[DIAMOND]` or `[DIAMOND-EFI]` for an imported DIAMOND network).

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Whitelist Indexing**:
     Loads the target headers whitelist $H_{\text{whitelist}}$ from the FASTA file. Maps each whitelist header to its corresponding index in the master network file:
     $$\text{Map}_{\text{header} \to \text{master\_idx}}(h) = x$$

     It then establishes a new index mapping for the subset:
     $$\text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(x) = y$$

2. **Edge Filtering**:
     Scans the master network edges $(i_k, j_k)$. An edge is retained if and only if both indices are in the whitelist:
     $$i_k \in \text{Map}_{\text{master\_idx} \to \text{subset\_idx}} \quad \text{and} \quad j_k \in \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}$$

3. **Re-Indexing**:
     For all retained edges, the script re-indexes the source and target node values to fit the smaller subset matrix coordinate space:
     $$i'_k = \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(i_k)$$
     $$j'_k = \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(j_k)$$

4. **Output Assembly**:
     Preserves the detected source schema: embedding score/length datasets for alignment networks or the single `score` dataset for BLAST networks (E-value or EFI alignment score). Output names are derived from the whitelist FASTA basename and the source network's `model_name` metadata.

</details>
