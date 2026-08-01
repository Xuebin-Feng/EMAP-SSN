# 🧼 Sanitize Sequences (`Sanitize_Sequences.py`)

This script cleans raw FASTA sequence databases to prepare them for language model embedding. It filters out sequences outside target size boundaries, excludes sequences containing specific header keywords (such as fragments or partials), replaces unsafe non-standard characters, and reports sequence length distributions.

### 📥 Input

#### Raw Sequence FASTA File `INPUT_FASTA`
*   **Format**: Standard FASTA format (`.fasta`, `.fa`, `.txt`).
*   **Created By**: User-provided raw protein sequence database.
*   **Structure**:
    ```text
    >Sequence_Header_1 [Optional Description]
    MNSGVSRRQ...
    >Sequence_Header_2
    MKVLLVSDA...
    ```

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Overwrite **`OVER_WRITE`** | Toggle to overwrite the input FASTA file with the cleaned sequences. If disabled, saves to a new file with a `_clean.fasta` suffix. |
| Enable Length Filter **`ENABLE_LENGTH_FILTER`** | Toggle to filter out sequences that do not meet the minimum or maximum length constraints. |
| Minimum Sequence Length **`MIN_SEQ_LENGTH`** | The minimum sequence length (in amino acids) required to keep a sequence. |
| Maximum Sequence Length **`MAX_SEQ_LENGTH`** | The maximum sequence length allowed. |
| Remove by Header String **`REMOVE_BY_HEADER_STRING`** | Excludes sequences whose headers contain this exact case-sensitive substring (e.g., `partial`, `fragment`, or `low quality`). Leave empty to disable; `None` is treated as a literal search word. |

### 📤 Output

#### Sanitized FASTA File
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: Contains cleaned sequences where headers are sanitized (special characters replaced with underscores) and invalid characters are removed.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Sanitization**:
     Cleans header strings to make them safe for downstream pipeline tools. It replaces slashes, backslashes, colons, semi-colons, brackets, parentheses, spaces, and commas with underscores:
     $$h_{\text{clean}} = \text{replace\_unsafe}(h_{\text{raw}})$$

2. **Sequence Cleaning**:
     Parses each sequence, converting it to uppercase and stripping out whitespace, asterisks (`*`), numbers, periods, and any characters outside the standard IUPAC amino acid set {A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y}:
     $$s_{\text{clean}} = \{c \in s_{\text{raw}} \mid c \text{ is a standard amino acid}\}$$

3. **Length and Substring Filtering**:
     If filtering is enabled, a sequence is discarded if:
     $$\text{Length}(s_{\text{clean}}) < \text{MIN\_SEQ\_LENGTH} \quad \text{or} \quad \text{Length}(s_{\text{clean}}) > \text{MAX\_SEQ\_LENGTH}$$
     
     Or if the header contains the excluded substring:
     $$\text{Substring} \subseteq h_{\text{clean}}$$

4. **Serialization and Diagnostics**:
     Writes the sanitized sequences to the target FASTA file. It then analyzes the sequence length distribution and displays statistics (mean, median, standard deviation).

</details>

---

# 🧬 Generate Embeddings (`Generate_Embeddings.py`)

This script extracts sequence embeddings from pre-trained protein language models (like ESM-2, ESM-C, ProtBERT, or ProstT5). It maps residues to high-dimensional representation vectors and bundles them into HDF5 database files using compressed arrays to optimize speed and disk efficiency.

### 📥 Input

#### FASTA File `INPUT_FASTA`
*   **Format**: Standard FASTA (`.fasta`), either raw or previously sanitized.
*   **Created By**: A user-provided sequence set or `Sanitize_Sequences.py`.
*   **Description**: Records are sanitized automatically in memory before embedding. This simplified pass does not filter by header text or sequence length and prints a result only when records change.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Model Name **`MODEL_NAME`** | The protein language model architecture to use and the label written into the output filename (e.g. `esmc_600m`, `esmc_6b`, `esm2_t33_650m`, `esm2_t30_150m`, `protbert`, `prostt5`). |
| Saving Precision **`SAVING_MODE`** | The numeric precision format used to store vectors in HDF5 (`float16` or `float32`). `float16` is recommended to reduce disk space by 50% with negligible loss of accuracy. |

> **ESMC 6B API access:** Select `esmc_6b`; the plugin maps that filename-friendly label to Biohub's `esmc-6b-2024-12` API identifier. Store the token beside the plugin in the Git-ignored `src/resources/pLM_models/esmc_6b_api_key.json` file as `{"ESM_API_TOKEN": "your-token"}`. Future API plugins can follow the same `*_api_key.json` naming convention.

### 📤 Output

#### HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/headers`: UTF-8 sanitized sequence headers.
    - `/sequences`: UTF-8 sanitized sequences in one-to-one order with `/headers`.
    - `/embeddings/{sanitized_header}`: Dataset of shape $L \times D$.
    - Attributes `model_name`, `saving_mode`, `num_sequences`, and `generation_complete`.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **In-Memory FASTA Sanitization**:
     Cleans headers and sequences, removes empty or duplicate records, retains the longest header for identical sequences, and assigns collision-safe unique headers. Header-substring and sequence-length filters are not applied.

2. **Model Loader and Weight Cache**:
     Downloads and caches model weights from Hugging Face. Loads the transformer model and tokenizes the input.

3. **Hardware Target Selection**:
     Checks system capabilities and assigns tensor operations to the optimal accelerator:
     $$\text{Device} = \begin{cases} \text{cuda} & \text{if Nvidia GPU available} \\ \text{mps} & \text{if Apple Silicon available} \\ \text{cpu} & \text{otherwise} \end{cases}$$

4. **Residue Embedding Generation**:
     For each sanitized sequence $s$:
     - Tokenizes and formats the sequence with start/stop tokens:
       $$s_{\text{token}} = \langle\text{cls}\rangle \, s_1 \, s_2 \, \dots \, s_L \, \langle\text{eos}\rangle$$
     - Executes a forward pass without gradient calculations:
       $$H = \text{TransformerEncoder}(s_{\text{token}})$$
     - Extracts the final hidden states tensor $E \in \mathbb{R}^{(L+2) \times D}$ from the last hidden layer.
     - Slices off the start/stop boundary tokens, yielding the residue embedding matrix:
       $$E_{\text{residue}} = E_{1 \dots L} \in \mathbb{R}^{L \times D}$$

5. **HDF5 Database Compilation**:
     Writes and flushes sanitized `/headers` and `/sequences` before generating any embeddings. Each validated residue matrix is then stored under its sanitized header and flushed individually. `generation_complete` becomes true only after the complete database passes validation.

</details>

---

# ✂️ Embedding Cropping (`Embedding_Cropping.py`)

This script produces embeddings for cropped/partial sequences by slicing them directly out of an existing full-sequence embedding database, instead of embedding the cropped fragment in isolation. Protein language models compute every residue's representation using full self-attention context, so directly embedding a short fragment yields a different (context-impoverished) vector than the same residues would get inside their native full-length sequence. This script never re-runs the language model — it only reads a full-sequence HDF5 database (produced by `Generate_Embeddings.py`) and slices out the requested residue range for each cropped sequence, preserving the full-context representation.

### 📥 Input

#### Full Embedding Database `INPUT_EMBED`
*   **Format**: HDF5 embedding database (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility). The required full sanitized sequences are stored in `/sequences`; a separate full-length FASTA is not needed.

#### Cropped Sequence Set `CROPPED_FASTA`
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: Partial sequences to produce contextual embeddings for. Records are sanitized identically to embedding generation. Each sanitized header must appear in `INPUT_EMBED`, and each sanitized sequence must be an exact contiguous substring of the stored full sequence.

### ⚙️ Parameters

This script does not require additional configuration parameters — behavior is fully determined by the source embedding database and cropped FASTA.

### 📤 Output

#### HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`), named `{CROPPED_FASTA}_[{model_name}]_embeddings.h5` — identical in structure to what `Generate_Embeddings.py` would produce if run directly on `CROPPED_FASTA`, so it is a drop-in input for downstream tools (`Embedding_PWA.py`, `Embedding_SSEARCH.py`, `Embedding_MSA.py`, etc.).
*   **Structure**:
    - `/embeddings/{sanitized_header}`: Dataset of shape $L_{\text{crop}} \times D$, sliced from the full-length embedding.
    - `/headers`: Array of resolved cropped-sequence headers.
    - `/sequences`: One-to-one array of resolved sanitized cropped sequences.
    - Attributes `model_name`, `saving_mode`, `num_sequences`, and `generation_complete`.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Correspondence**:
     For each sanitized cropped sequence $s_{\text{crop}}$ with header $h$, locates the stored full sequence $s_{\text{full}}$ and embedding matrix $E_{\text{full}} \in \mathbb{R}^{L_{\text{full}} \times D}$ sharing that header in `INPUT_EMBED`. Missing headers are reported and skipped.

2. **Consistency Check**:
     Verifies that the full embedding's row count matches the full sequence's length:
     $$L_{\text{full}} \overset{?}{=} \text{Length}(s_{\text{full}})$$
     A mismatch makes the source embedding database invalid and cropping stops rather than slicing stale data.

3. **Offset Resolution**:
     Finds the position of the cropped sequence within its full parent via exact substring search:
     $$\text{offset} = \arg\min \{ i : s_{\text{full}}[i : i+L_{\text{crop}}] = s_{\text{crop}} \}$$
     If the crop occurs more than once, the first occurrence is used and a warning is logged. If it is not found at all, the header is skipped and reported.

4. **Context-Preserving Slice**:
     Because the residue embeddings in $E_{\text{full}}$ already account for full sequence context, slicing is a simple index range with no recomputation:
     $$E_{\text{crop}} = E_{\text{full}}[\text{offset} : \text{offset} + L_{\text{crop}}]$$

5. **HDF5 Database Compilation**:
     Writes and flushes the resolved sanitized headers and sequences first, streams each crop under its header key, and marks the output complete only after validation.

</details>
