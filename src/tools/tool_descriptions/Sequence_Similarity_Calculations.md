# 🧬 Dynamic Programming Embedding Alignment (`Align_Similarity_Matrix.py`)

This script computes an all-vs-all sequence similarity network (SSN) using residue-level protein embeddings. Instead of using traditional amino acid substitution matrices, it calculates similarity matrices by comparing the dense high-dimensional embedding vectors of each residue.

It performs sequence alignment scoring using dynamic programming (Smith-Waterman for local alignment and Needleman-Wunsch for global alignment) implemented in optimized Numba JIT functions for maximum performance. To accelerate alignment of massive datasets, it includes an optional pre-filtering step that computes the global cosine similarity of sequence mean embeddings to skip full alignments for highly dissimilar sequences.

### 📥 Input

#### HDF5 Embedding Database `INPUT_HDF5`
*   **Format**: HDF5 (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).
*   **Description**: Contains pre-calculated residue embeddings as $L \times D$ arrays.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Enable Edge Pre-filtering **`EDGE_PREFILTERING`** | Toggle to enable pre-filtering sequence pairs using global cosine similarity. If enabled, full residue-level alignments are skipped for pairs below the similarity cutoff. |
| Pre-filtering Strength **`PREFILTER_STRENGTH`** | The percentage of sequence pairs with the lowest global cosine similarity scores to exclude from alignment calculations. Higher values increase speed but risk missing alignments between distantly related proteins. |
| CPU Worker Threads **`WORKERS`** | The number of CPU processes/threads allocated for parallel alignment calculations. |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap penalty score applied for local alignments. More negative values penalize gaps more heavily, resulting in fewer gaps. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap penalty score applied for global alignments. |
| Processing Batch Size **`BATCH_SIZE`** | The number of sequence pairs processed in a single chunk before writing to disk. Larger batches maximize CPU core utilization but increase memory consumption. Can be set to a number or 'auto'. |

### 📤 Output

#### HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/i`: Source sequence node indices.
    - `/j`: Target sequence node indices.
    - `/g_score`: Global alignment scores.
    - `/g_len`: Global alignment lengths.
    - `/l_score`: Local alignment scores.
    - `/l_len`: Local alignment lengths.
    - `/headers`: Array of sequence headers.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

The alignment pipeline is executed in the following steps:

1. **Global Embedding Pooling**:
     For each protein sequence, its residue embeddings are pooled (either via mean or max pooling) to generate a single global sequence representation vector:
     $$u_i = \text{mean}_{\text{residues}}(\text{emb}_i(\text{residue}))$$

2. **Edge Pre-filtering (Optional)**:
     If pre-filtering is enabled, all-vs-all cosine similarities are calculated on the CPU:
     $$\text{Sim}_{\text{cos}}(i, j) = \frac{u_i \cdot u_j}{\|u_i\|_2 \times \|u_j\|_2}$$
     
     This similarity is adjusted by the sequence length ratio:
     $$\text{Adj}(i, j) = \text{Sim}_{\text{cos}}(i, j) \times \left(\frac{\min(L_i, L_j)}{\max(L_i, L_j)}\right)^P$$
     
     Pairs scoring in the bottom percentile corresponding to `PREFILTER_STRENGTH` are skipped.

3. **Residue-Level Similarity Matrix Computation**:
     For each sequence pair $(i, j)$ passing the pre-filter, a pairwise Cosine distance matrix is computed. First, the residue embeddings are L2-normalized along the hidden dimension:
     $$\hat{v}_i(a) = \frac{v_i(a)}{\|v_i(a)\|_2}$$
     
     Then, the cosine distance is calculated using matrix multiplication:
     $$D(a, b) = 1.0 - \hat{v}_i(a) \cdot \hat{v}_j(b)$$
     
     where $v_i(a)$ is the embedding vector for residue $a$ in sequence $i$. This distance is converted into a similarity matrix:
     $$S(a, b) = \exp(-D(a, b))$$

4. **Dual Z-Score Normalization**:
     The similarity matrix is normalized row-wise and column-wise to adjust for residue-specific background similarities:
     $$Z_{\text{row}}(a, b) = \frac{S(a, b) - \mu_{\text{row}}(a)}{\sigma_{\text{row}}(a) + \varepsilon}$$
     $$Z_{\text{col}}(a, b) = \frac{S(a, b) - \mu_{\text{col}}(b)}{\sigma_{\text{col}}(b) + \varepsilon}$$
     
     The final alignment scoring matrix is the average of these Z-scores:
     $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$

5. **Dynamic Programming Alignment**:
     * **Needleman-Wunsch (Global Pass)**: Solves the standard global recurrence relation using the scoring matrix $\text{Score}(a, b)$ and `GLOBAL_GAP_P` to output a global score.
     * **Smith-Waterman (Local Pass)**: Subtracts a shift value of 2.0 from the scoring matrix (to ensure dissimilar matches have negative scores):
       $$\text{Score}_{\text{local}}(a, b) = \text{Score}(a, b) - 2.0$$
       
       It then runs the standard local dynamic programming recurrence with `LOCAL_GAP_P` to identify the optimal local alignment score.

</details>

---

# 🧬 Substitution Matrix Alignment (`Align_Substitution_Matrix.py`)

This script runs all-vs-all local sequence alignments using traditional amino acid substitution matrices. It constructs a local NCBI BLAST database from the input sequence set and executes parallelized BLASTP queries. The resulting E-values are converted into linearly comparable negative Log10(E) edge weights for network mapping.

### 📥 Input

#### Sequence FASTA File `INPUT_FASTA`
*   **Format**: Standard FASTA (`.fasta`).
*   **Created By**: `Sanitize_Sequences.py` (Sequence Sanitization utility) or user-provided raw FASTA.
*   **Description**: Raw sequence database to run BLAST against. Records first undergo the same canonical header, residue, empty-record, and duplicate-sequence sanitization used by `Generate_Embeddings.py`.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Substitution Matrix **`MATRIX`** | The amino acid substitution matrix used for traditional scoring during BLAST alignments (e.g., BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90, PAM30, PAM70, PAM250). |
| BLAST Threads **`NUM_THREADS`** | The number of CPU threads allocated for BLASTP execution and parsing. |
| Processing Batch Size **`BATCH_SIZE`** | The maximum number of parsed edges buffered or copied at once during both intermediate HDF5 writing and final network compilation. |

Intermediate query segments, BLAST databases, result files, and parser batches are stored automatically in a sequence-specific temporary folder inside the configured Network Directory.
Interrupted runs reuse only complete HDF5 batches whose input, sanitized manifest, substitution matrix, thread count, BLASTP version, query chunk, and source-result checksums still match. The temporary workspace is removed after the final network is validated and published successfully.

### 📤 Output

#### HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/i`: Source sequence node indices.
    - `/j`: Target sequence node indices.
    - `/l_score`: Calculated $-\log_{10}(E_{\text{value}})$ scores.
    - `/headers`: Array of sequence headers.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Canonical FASTA Sanitization and Numeric IDs**:
     The input first passes through the same shared sanitization used by `Generate_Embeddings.py`. This sanitizes headers, uppercases and masks residue strings, removes empty records, merges identical sequences, and makes retained headers unique. To avoid parser failures in NCBI BLAST, the script then generates a temporary mapping:
     $$\text{Header} \to \text{Index} \quad (0, 1, 2, \dots)$$
     
     It creates a temporary `safe_fasta` containing sequence headers renamed as their respective integer indices.

2. **Local BLAST Database Construction**:
     Executes `makeblastdb` to generate a local database using the sanitized FASTA.

3. **Multithreaded Query Chunking**:
     Splits the query FASTA into parallelized temporary chunks to distribute search workloads across the selected number of CPU workers (`NUM_THREADS`).

4. **Pairwise BLASTP Execution**:
     Executes `blastp` for each query chunk against the database:
     * **blastp** → tabular output format 6

5. **E-Value Conversion**:
     Extracts E-value scores for each hit and linearizes them into edge connectivity weights using a negative base-10 logarithm:
     $$\text{Score} = -\log_{10}(E_{\text{value}} + 10^{-300})$$
     
     The lower bound offset of $10^{-300}$ is added to avoid mathematical division by zero when the E-value is 0.0.

6. **Consolidation**:
     Merges chunked output files, resolves temporary indices back to original headers, and exports the final network file.

</details>

---

# 🔍 Parse BLAST Output (`Parse_BLAST_Output.py`)

This script parses pre-computed, tab-separated tabular BLAST outfmt 6 output files and converts them into standard HDF5 network files. It automatically detects which columns contain sequence headers and E-value variables, performs E-value conversion, and filters out redundant alignment edges to build clean networks.

### 📥 Input

#### Tabular BLAST Output File `INPUT_BLAST_TABULAR`
*   **Format**: Tab-separated tabular values (`.txt`, `.tab`, `.tsv`).
*   **Created By**: Externally run NCBI BLASTP command (`blastp -outfmt 6`).
*   **Structure**: Output columns from `blastp -outfmt 6`. Must contain query ID, subject ID, and E-value columns.

### ⚙️ Parameters

This script does not require additional configuration parameters.

### 📤 Output

#### HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/i`: Source sequence node indices.
    - `/j`: Target sequence node indices.
    - `/l_score`: Parsed $-\log_{10}(E_{\text{value}})$ scores.
    - `/headers`: Array of sequence headers.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Automatic Column Detection**:
     Scans the first 1000 lines of the tabular document using regular expressions to detect the column index $c_{\text{evalue}}$ that contains E-values (identifying scientific notations or exact 0.0 matches).

2. **Header Index Resolution**:
     Constructs a header-to-index mapping dictionary dynamically as it reads lines.

3. **Edge Parsing and Score Conversion**:
     Extracts query ID, subject ID, and E-value, and converts the E-value:
     $$\text{Score} = -\log_{10}(E_{\text{value}} + 10^{-300})$$

4. **Deduplication**:
     Retains only the highest-scoring alignment edge between any undirected sequence pair (u, v) to filter out redundant alignments.

5. **HDF5 Serialization**:
     Writes headers, sequence indices, and edge score datasets to the output network.

</details>
