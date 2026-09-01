# 🧬 Embedding Multiple Sequence Alignment (`Embedding_MSA.py`)

This script generates progressive Multiple Sequence Alignments (MSAs) using protein language model embeddings. By evaluating the high-dimensional similarity of sequence residue representations, it maps out evolutionary pathways and constructs robust guide trees. It then progressively aligns sequence clusters using gap-aware dynamic programming.

### 📥 Input

#### Sequence Set `INPUT_FASTA`
*   **Format**: Standard FASTA sequence database file (`.fasta`).
*   **Created By**: `Sanitize_Sequences.py` (Sequence Sanitization utility) or user-provided raw FASTA.
*   **Optional Behavior**: Used only when **Use Sequence Filter** (`USE_SEQUENCE_FILTER`) is enabled. When filtering is disabled, the GUI saves an empty string and the alignment uses sequences stored in the embedding database.

#### Embedding Database `INPUT_EMBED`
*   **Format**: Pre-computed HDF5 embedding database (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

#### Network Database `INPUT_NETWORK`
*   **Format**: Pairwise HDF5 alignment network (`.h5`).
*   **Created By**: `Align_Similarity_Matrix.py` (Embedding Alignment utility) or `Align_Substitution_Matrix.py` / `Parse_BLAST_Output.py`.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Use Sequence Filter **`USE_SEQUENCE_FILTER`** | If enabled, restricts the alignment to sequences shared by the network, embedding database, and selected FASTA file. If disabled, no FASTA is required and the network/embedding intersection is aligned using sequences stored in the embedding manifest. |
| Show Regression Plot **`SHOW_REGRESSION_PLOT`** | Toggle to show the diagnostic Isotonic Regression plot when using a sparse network. This visualizes the fit between mean embedding cosine distances and actual alignment scores. |
| Alignment Scoring Type **`ALIGNMENT_SCORE`** | Select whether to construct the guide tree using 'global' or 'local' connectivity scores from the network. |
| Score Normalization Mode **`NORMALIZATION_MODE`** | The normalization method applied to alignment scores (e.g., alignment_length, shorter_sequence, longer_sequence, average_sequence). Disabled for BLAST networks. |
| Tree Building Method **`TREE_METHOD`** | `UPGMA (Fast)` uses average-linkage hierarchical clustering; `Neighbor-joining (Slow)` constructs a neighbor-joining tree from the same complete distance matrix. The selected method is used for deterministic and noise-perturbed trees. |
| Noise-Perturbed Consensus Guide Tree **`BOOTSTRAP_TREE`** | Builds and averages randomly perturbed guide-tree replicates when enabled. This is a sensitivity ensemble, not classical bootstrap support. The existing setting key is retained for backward compatibility. |
| Perturbed Tree Replicates **`NUM_TREES`** | The number of noise-perturbed replicate trees used to construct the consensus guide tree. |
| Include Imputed Pairs in Final Consensus **`INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS`** | For an incomplete network, OFF averages cophenetic distances only for originally observed pairs and retains baseline regression-imputed values for missing pairs; ON replaces every pair with its replicate-averaged cophenetic distance. Imputed pairs participate in every replicate tree in both modes. Complete networks automatically use full consensus. |
| Normalized Additive Noise Scale **`NOISE_SCALE`** | Gaussian standard deviation expressed as a fraction of the valid distance range. For example, `0.02` applies an additive standard deviation equal to 2% of the maximum guide-tree distance. Every observed and regression-imputed distance is perturbed. |
| Gap Open Penalty **`GAP_OPEN`** | The gap open penalty score. More negative values penalize opening new gaps more severely. |
| Gap Extend Penalty **`GAP_EXTEND`** | The gap extend penalty score. More negative values penalize extending existing gaps. |
| CPU Worker Processes **`WORKERS`** | The number of CPU processes allocated for parallel noise-perturbed guide-tree calculations. |
| Compute Device **`DEVICE_SELECTION`** | Selects `auto` or a specific available CPU, CUDA, XPU, or MPS device for sequential profile score-matrix construction. Auto benchmarks three real leaf-to-leaf guide-tree merges near the 25th, 50th, and 90th score-matrix cost percentiles and selects one device for the full progressive merge. Guide-tree calculations and dynamic-programming traceback remain on CPU. |
| Temporary Working Directory **`SAFE_TEMP_DIR`** | The temporary directory used to cache intermediate files and memory-mapped matrices. |

### 📤 Output

#### Multiple Sequence Alignment FASTA File
*   **Format**: Aligned FASTA (`.fasta`).
*   **Structure**: Standard FASTA, containing aligned sequence strings padded with gaps (`-`) to match the final alignment length.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Intersection Filtering**:
     Always intersects sequence headers from the embedding HDF5 file and network HDF5 file. If `USE_SEQUENCE_FILTER` is enabled, the selected FASTA file is added as a third intersection source; otherwise, sequence strings come from the embedding manifest.

2. **Network Score Normalization**:
     Normalizes the network alignment scores based on `NORMALIZATION_MODE` to account for sequence length discrepancies.
     * **alignment_length**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{L_{\text{align}}}$$
     * **shorter_sequence**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\min(L_i, L_j)}$$
     * **longer_sequence**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\max(L_i, L_j)}$$
     * **average_sequence**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\text{mean}(L_i, L_j)}$$

3. **Distance Matrix Formulation**:
     Inverts scores to represent distances:
     $$D(i, j) = \max(S_{\text{norm}}) - S_{\text{norm}}(i, j)$$

4. **Isotonic Regression for Sparse Networks**:
     If the network is sparse, the distance matrix is incomplete. The script:
     - Max- or Mean-pools residue embeddings to generate a global representation vector **u<sub>i</sub>** for all sequences.
     - Computes all-vs-all length-adjusted cosine similarities:
       $$\text{Adj}(i, j) = \text{Sim}_{\text{cos}}(i, j) \times \left(\frac{\min(L_i, L_j)}{\max(L_i, L_j)}\right)^P$$
     - Fits an **Isotonic Regression** function $f$ mapping known $\text{Adj}(i, j)$ values to normalized network scores $S_{\text{norm}}(i, j)$.
     - Predicts missing network scores:
       $$S_{\text{predicted}}(i, j) = f(\text{Adj}(i, j))$$
       
       This guarantees a fully populated distance matrix.

5. **Noise-Perturbed Consensus Guide Tree Construction**:
     Constructs guide trees using the selected `TREE_METHOD` (average-linkage UPGMA or Neighbor-Joining). When `BOOTSTRAP_TREE` is enabled, the program generates `NUM_TREES` replicate trees using normalized additive Gaussian distance perturbations:

     $$D_{\max} = \max(S_{\text{norm}}) + 0.1$$

     $$\sigma_{\text{absolute}} = \text{NOISE\_SCALE} \times D_{\max}$$

     $$D_{\text{perturbed}}(i,j) =
     \operatorname{clip}\left(
     D(i,j) + \mathcal{N}(0,\sigma_{\text{absolute}}^2),
     0,D_{\max}
     \right)$$

     The valid range is $[0,D_{\max}]$. Zero represents the closest possible relationship. $D_{\max}$ is the maximum distance used for the weakest or unconnected sequence pairs. Clamping prevents invalid negative distances and prevents an observed edge from exceeding the program's maximum-distance sentinel.

     For a sparse network, the complete baseline matrix combines observed network distances with regression-imputed distances for missing pairs. This complete baseline is shared with every replicate worker, and the same additive perturbation is applied to every distance. Imputed relationships therefore participate in every replicate tree instead of being replaced by $D_{\max}$.

     The final averaging mode is selected after the network/embedding/FASTA intersection. A complete induced network automatically uses full all-pairs cophenetic consensus. For an incomplete induced network, **Include Imputed Pairs in Final Consensus** controls only the final matrix: OFF replaces observed-pair entries with their replicate-averaged cophenetic distances while missing pairs retain their baseline regression-imputed values; ON replaces every pair with its replicate-averaged cophenetic distance. Turning this switch OFF does not disable imputation or remove imputed pairs from replicate tree construction. When noise-perturbed trees are disabled, the switch is ignored and the complete observed-plus-imputed baseline directly builds the deterministic tree.

     These replicates measure sensitivity to randomized distance perturbations; they are not classical bootstrap resamples and do not provide classical bootstrap support values.

6. **Progressive Profile Alignment**:
     Traverses the guide tree from leaves to the root. At each internal node, it merges the two child clusters (which could be single sequences or alignment profiles) by:
     - Resolving a manual compute device or, in Auto mode, benchmarking representative real leaf merges after guide-tree construction. The benchmark includes score-matrix host/device transfers but excludes HDF5 loading and CPU traceback.
     - Unit-normalizing each leaf residue embedding before it enters a profile.
     - Averaging those unit vectors across every sequence in the cluster while treating gaps as zero. The resulting vector norm therefore records both non-gap occupancy and directional agreement within the column.
     - Calculating reciprocal cosine similarity from the profile-vector directions, then weighting each cell by the two column norms so weakly supported or inconsistent columns contribute less alignment evidence.
     - Running dynamic programming with `GAP_OPEN` and `GAP_EXTEND` penalties to find the optimal path.
     - Merging alignments and outputting the final FASTA multiple sequence alignment (MSA).

</details>

---

# 📉 Sparse MSA Converter (`Sparse_MSA_Converter.py`)

This script compresses multiple sequence alignments (MSAs) into compact HDF5 files. It validates and sanitizes the aligned FASTA, converts residue strings to a SciPy Compressed Sparse Row (CSR) matrix, writes lookup metadata, and then moves the successfully converted source FASTA into a `Full_Alignments` subdirectory.

### 📥 Input

#### MSA Alignment File `INPUT_FASTA`
*   **Format**: Aligned FASTA (`.fasta`).
*   **Created By**: `Embedding_MSA.py` (Embedding Multiple Alignment utility) or other external MSA tools (e.g. Clustal, MUSCLE).
*   **Description**: Pre-calculated standard multiple sequence alignment file.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Convert All Alignments **`CONVERT_ALL`** | Toggle to convert all FASTA multiple alignments inside the input directory. If disabled, only the selected alignment file is converted. |

### 📤 Output

#### Compressed Sparse MSA HDF5 File
*   **Format**: HDF5 (`.h5`) named `<input_basename>_sparse.h5` beside the selected alignment.
*   **Structure**:
    - `/matrix/data`: 1D array of integer residue codes for non-gap entries.
    - `/matrix/indices`: CSR column index array.
    - `/matrix/indptr`: CSR row pointers mapping sequences.
    - `/matrix` attribute `shape`: Matrix row/column dimensions.
    - `/headers`: Array of sequence headers.
    - `/header_map`: JSON mapping of full and first-token headers to row indices.
    - `/aa_map` and `/int_to_aa`: JSON residue-code lookup tables.
    - Root attribute `shape`: Overall dimensions of the alignment.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Amino Acid Mapping**:
     Maps amino acid characters to integer indices (e.g., A → 1, R → 2, ...). The gap character `"-"` is not saved (implicitly zero).

2. **Compressed Sparse Row (CSR) Compilation**:
     Translates the aligned sequence matrix into SciPy Compressed Sparse Row (CSR) format:
     $$\text{Matrix} \to (\text{data}, \text{indices}, \text{indptr})$$
     
     Using `uint8` data types compresses the file size by up to 95 percent.

3. **Atomic HDF5 Serialization and Source Archival**:
     Writes the CSR arrays, headers, mappings, and shape metadata to a temporary HDF5 file and atomically publishes it as `<input>_sparse.h5`. Only after publication succeeds is the original FASTA moved to `<MSA_DIR>/Full_Alignments/`. With `CONVERT_ALL` enabled, every top-level `*.fasta` file in `MSA_DIR` is processed this way.

</details>
