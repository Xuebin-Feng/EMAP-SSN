# Sanitize Sequences (`Sanitize_Sequences.py`)

This script cleans raw FASTA sequence databases to prepare them for language model embedding. It filters out sequences outside target size boundaries, excludes sequences containing specific header keywords (such as fragments or partials), replaces unsafe non-standard characters, and reports sequence length distributions.

### Input

#### Raw Sequence FASTA File `INPUT_FASTA`
*   **Format**: Standard protein FASTA (`.fasta`) selected from the configured Sequence Sets directory.
*   **Created By**: User-provided raw protein sequence database.
*   **Structure**:
    ```text
    >Sequence_Header_1 [Optional Description]
    MNSGVSRRQ...
    >Sequence_Header_2
    MKVLLVSDA...
    ```

### Parameters

| Parameter | Description |
| :--- | :--- |
| Overwrite **`OVER_WRITE`** | Toggle to atomically replace the input FASTA with the cleaned sequences. If disabled, saves a sibling file with a `_sanitized.fasta` suffix. An empty result is refused in overwrite mode. |
| Enable Length Filter **`ENABLE_LENGTH_FILTER`** | Toggle to filter out sequences that do not meet the minimum or maximum length constraints. |
| Minimum Sequence Length **`MIN_SEQ_LENGTH`** | The minimum sequence length (in amino acids) required to keep a sequence. |
| Maximum Sequence Length **`MAX_SEQ_LENGTH`** | The maximum sequence length allowed. |
| Remove by Header String **`REMOVE_BY_HEADER_STRING`** | Excludes sequences whose headers contain this exact case-sensitive substring (e.g., `partial`, `fragment`, or `low quality`). Leave empty to disable; `None` is treated as a literal search word. |
| Show Length Histogram **`SHOW_LENGTH_HISTOGRAM`** | Displays the length histogram of the sanitized sequences when the run ends. A run from a settings file (CLI or MCP) prints the same bins as a table instead. Turn off to skip both. |

### Output

#### Sanitized FASTA File
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: Contains canonically sanitized, deduplicated sequences. Headers are made safe and globally unique; terminal sequence artifacts are trimmed and non-residue characters inside the retained sequence span are replaced with `X`.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Sanitization**:
     Converts square/curly brackets to parentheses, replaces `? * " # % @ $ / \` with underscores, collapses repeated underscores, and normalizes whitespace while preserving readable spaces:
     $$h_{\text{clean}} = \text{replace\_unsafe}(h_{\text{raw}})$$

2. **Sequence Cleaning**:
     Converts sequences to uppercase. The accepted residue alphabet is `ACDEFGHIKLMNPQRSTVWYBZJXUO`. Artifacts before the first and after the last accepted residue are trimmed; invalid characters inside that span are replaced one-for-one with `X`, preserving residue coordinates.

     Records are then deduplicated by sanitized sequence. Identical sequences keep the longest deterministic header, and different sequences that collide on the same sanitized header receive collision-safe numeric suffixes.

3. **Length and Substring Filtering**:
     If filtering is enabled, a sequence is discarded if:
     $$\text{Length}(s_{\text{clean}}) < \text{MIN\_SEQ\_LENGTH} \quad \text{or} \quad \text{Length}(s_{\text{clean}}) > \text{MAX\_SEQ\_LENGTH}$$
     
     The case-sensitive header substring filter is applied to each raw header before canonical header sanitization:
     $$\text{Substring} \subseteq h_{\text{raw}}$$

4. **Serialization and Diagnostics**:
     Writes the sanitized sequences to the target FASTA file. It then prints the sequence count and the minimum, median, mean, standard deviation and maximum of the lengths, and shows the length histogram unless **Show Length Histogram** is off.

</details>

---

# Generate Embeddings (`Generate_Embeddings.py`)

This script extracts sequence embeddings from pre-trained protein language models such as ESM-2, ESM-C, Ankh, ProtBERT, ProtT5, and ProstT5. It maps residues to high-dimensional representation vectors and stores them in metadata-first HDF5 databases using the selected `float16` or `float32` precision.

### Input

#### FASTA File `INPUT_FASTA`
*   **Format**: Standard FASTA (`.fasta`), either raw or previously sanitized.
*   **Created By**: A user-provided sequence set or `Sanitize_Sequences.py`.
*   **Description**: Records are sanitized automatically in memory before embedding. This simplified pass does not filter by header text or sequence length and prints a result only when records change.

### Parameters

| Parameter | Description |
| :--- | :--- |
| Model Name **`MODEL_NAME`** | The protein language model architecture to use. The dropdown names each model, such as ESM-2 650M or ESM C 6B; the setting stores its identifier, which is also the label written into the output filename. Identifiers are always lower case (e.g. `esmc_600m`, `esmc_6b`, `esm2_t33_650m`, `esm2_t30_150m`, `ankh_base`, `prot_bert`, `prot_t5`, `prost_t5`). |
| Saving Precision **`SAVING_MODE`** | The numeric precision format used to store vectors in HDF5 (`Float16 (Half Size)`, the default, or `Float32 (Full Precision)`). `Float16 (Half Size)` uses roughly half the embedding storage of `Float32 (Full Precision)`. Models always compute in float32; rounding the stored values changes alignment scores only slightly and did not change MSA accuracy on BAliBASE and HomFam benchmark sets. Use `Float32 (Full Precision)` to store the values exactly as computed. |
| Compute Device **`DEVICE_SELECTION`** | `auto` benchmarks the available installer-validated CPU/accelerator candidates on representative sequence lengths and uses the fastest successful device, with ranked fallback if a runtime failure occurs. Selecting a specific CPU, CUDA, XPU, or MPS device pins generation to that device and reports an error instead of silently switching. |

> **Shared Biohub API access:** Select ESM C 6B (`esmc_6b`); the plugin maps that filename-friendly identifier to Biohub's `esmc-6b-2024-12` API identifier. On first use, the `Generate_Embeddings.py` terminal prompts for a hidden token and stores it in the Git-ignored `src/resources/Biohub_API.json`. The same token and optional `ESM_API_URL` are shared with `esmfold large`, whose worker terminal uses the same hidden prompt when the shared file and `ESM_API_KEY` environment variable are absent.

> **Model terms:** Model weights are downloaded separately and retain their publishers' licenses. In particular, Ankh Base and Ankh Large weights are under CC-BY-NC-SA-4.0; the program requires an acknowledgement before access. See `THIRD_PARTY_LICENSES.md` for the model inventory. These model terms do not relicense the Apache-2.0 program source.

### Output

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
     Discovers installer-validated CPU, CUDA, Intel XPU, and Apple MPS devices. In `auto` mode, it benchmarks representative sanitized sequences on every available candidate, ranks successful devices, and can fall back to the next ranked device if embedding fails. A manually selected device is used exclusively.

4. **Residue Embedding Generation**:
     For each sanitized sequence $s$, the selected model adapter applies that model's required residue formatting and tokenization, executes inference without gradients, and removes its model-specific special tokens. ESM and Rostlab-family adapters remove their boundary tokens; Ankh uses its own tokenizer layout and removes its trailing special token. Every adapter validates that the final residue matrix has exactly one row per sanitized residue:
     $$E_{\text{residue}} \in \mathbb{R}^{L \times D}$$

5. **HDF5 Database Compilation**:
     Writes and flushes sanitized `/headers` and `/sequences` before generating any embeddings. Each validated residue matrix is then stored under its sanitized header and flushed individually. `generation_complete` becomes true only after the complete database passes validation.

</details>

---

# Embedding Cropping (`Embedding_Cropping.py`)

This script produces embeddings for cropped/partial sequences by slicing them directly out of an existing full-sequence embedding database, instead of embedding the cropped fragment in isolation. Protein language models compute every residue's representation using full self-attention context, so directly embedding a short fragment yields a different (context-impoverished) vector than the same residues would get inside their native full-length sequence. This script never re-runs the language model — it only reads a full-sequence HDF5 database (produced by `Generate_Embeddings.py`) and slices out the requested residue range for each cropped sequence, preserving the full-context representation.

### Input

#### Full Embedding Database `INPUT_EMBED`
*   **Format**: HDF5 embedding database (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility). The required full sanitized sequences are stored in `/sequences`; a separate full-length FASTA is not needed.

#### Cropped Sequence Set `CROPPED_FASTA`
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: Partial sequences to produce contextual embeddings for. Records are sanitized identically to embedding generation. Each sanitized header must appear in `INPUT_EMBED`, and each sanitized sequence must be an exact contiguous substring of the stored full sequence.

### Parameters

This script does not require additional configuration parameters — behavior is fully determined by the source embedding database and cropped FASTA.

### Output

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

---

# Embedding Import (`Embedding_Import.py`)

This tool converts per-residue embeddings computed outside EMAP-SSN, such as UniProt's ProtT5 per-residue files, the output of the ProtTrans scripts or a model of your own, into an EMAP-SSN embedding database, so the alignment, MSA and network tools can use them. No model is run. The FASTA decides what the database holds: it is sanitized as `Sanitize_Sequences.py` does, and every sanitized record must find one residues-by-features array in the source.

### Input

#### Sequence Set `INPUT_FASTA`
*   **Format**: Standard FASTA (`.fasta`), raw or sanitized: the sequences the embeddings were computed for.
*   **Description**: Records are sanitized in memory (uppercased, invalid residues masked, identical sequences merged). Each array must have one row per residue of the sanitized sequence.

#### Embedding Source `SOURCE_PATH`
*   **Format**: One of these:
    - an HDF5 file (`.h5`) with one dataset per protein at its top level, named by its ID, such as UniProt's per-residue files, named by accession. A dataset's `original_id` attribute also counts as its ID, which covers bio_embeddings files;
    - a `.npz` file holding one array per ID;
    - a folder of `.npy` files, one per protein, each named by its ID (`P12345.npy`).
*   **Description**: Chosen with **File...** or **Folder...** from anywhere on this computer. It is not saved: the field is empty the next time the window opens. A relative path is read inside the FASTA directory.

#### ID Mapping `ID_MAP_FILE` (optional)
*   **Format**: A `.tsv`, `.csv` or `.txt` file of two columns. The first holds a FASTA header of the sequence set (as written there, its first word or the sanitized one); the second holds the ID that protein has in the source. A `.csv` file is comma-separated, with a value that contains a comma in double quotes; any other file is tab-separated. A column after the second is ignored.
*   **Lines**: Blank lines and lines whose first column starts with `#` are skipped. A line of column names is not recognised and would be read as a pair, so start it with `#`. A line with fewer than two columns, or an empty second column, stops the run with an error naming the file and the line; so does a header given two different IDs. A byte-order mark at the start of the file (Excel adds one to CSV files) is ignored.
*   **Description**: Tried before the automatic rules. A record the file does not list is matched by the automatic rules; a listed ID that the source lacks is reported as `unmapped` and not retried. Not saved, like the source.

### Parameters

| Parameter | Description |
| :--- | :--- |
| Model Label **`MODEL_LABEL`** | Names the model in the database and in the names of the files made from it. Lower-case letters, digits, `_` and `-`, at most 64 characters. It may not be the identifier of a model EMAP-SSN computes itself, such as `prot_t5`: the file would pass for one computed here, and Embedding Injection would extend it with embeddings computed locally. Use a label of your own, such as `uniprot_prot_t5`. |
| Saving Mode **`SAVING_MODE`** | As in Generate Embeddings: `Float32 (Full Precision)` or `Float16 (Half Size)`. Every array is converted to it, and an array with values beyond the range of float16 is refused. |

### Output

#### HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`), named `{INPUT_FASTA}_[{MODEL_LABEL}]_embeddings.h5`, in the format described under Embedding File Format below.
*   **Completion**: Marked complete only when every sanitized record has its array. Otherwise it stays incomplete, every reader refuses it, and running the tool again resumes it: records already imported are kept and the rest are matched again, with a mapping file, say.
*   **Existing File**: One with the same label, saving mode and sequences is kept or resumed; one with another label or saving mode, or with changed sequences, is refused.
*   **Provenance**: Attributes `import_source` (the source's file or folder name), `import_reader` (`hdf5`, `npz` or `npy_folder`), `import_source_dtype` and `import_matched_by` (records per matching rule).

#### Import Report
*   **Format**: Tab-separated (`.tsv`), named `{INPUT_FASTA}_[{MODEL_LABEL}]_import_report.tsv`, beside the database, written only when records were not imported: `header`, `status` (`missing`, `ambiguous`, `unmapped` or `invalid`), `source_id` and `reason`. A later run that imports everything deletes it.
*   **Terminal**: The counts of records imported, already in the file, missing, ambiguous, mapped to an absent ID and refused, and of source IDs not used.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **ID Matching**:
     For each sanitized record, the first rule that finds an ID in the source is used: the mapping file; the original header; its first word; the sanitized header; a UniProt accession (`sp|P12345|NAME ...` and `tr|...` headers give `P12345`). Records that sanitization merged, because their sequences were identical, try every original header. An ID that more than one record could take is not used: those records are reported as ambiguous, and a mapping file settles them.

2. **Array Checks**:
     An array must be two-dimensional: a single vector per protein, a per-protein embedding, is refused, because alignment needs one vector per residue. It must have one row per residue of the sanitized sequence: rows = length + 2 means the model's start and end tokens were left in, and rows = length + 1 one of them. Its values must be floating-point, with one feature width for the whole database, and finite at the saving mode.

3. **HDF5 Database Compilation**:
     Writes and flushes the sanitized headers and sequences first, then one array per record, and marks the database complete only after validation, as `Generate_Embeddings.py` does.

</details>

---

# Embedding File Format

Every EMAP-SSN tool and the Viewer read embeddings in one HDF5 layout. A file in another layout is refused with a message that points to Embedding Import, which converts it.

| Object | Content |
| :--- | :--- |
| Attribute `model_name` | The model's label, in lower case; also in the file name `{set}_[{model_name}]_embeddings.h5`. |
| Attribute `saving_mode` | `float16` or `float32`: the type of every embedding array. |
| Attribute `num_sequences` | The number of records. |
| Attribute `generation_complete` | Boolean. Readers refuse a file in which it is false. |
| Dataset `/headers` | UTF-8 strings: the sanitized FASTA headers, each unique. |
| Dataset `/sequences` | UTF-8 strings: the sanitized sequences, in the order of `/headers`. |
| Group `/embeddings` | One dataset per header, `/embeddings/{sanitized header}`: an $L \times D$ array of the saving mode's type, where $L$ is the length of its sequence and $D$ is the same for every array. |

Sanitized means as `Sanitize_Sequences.py` writes it: readers match nodes by sanitized header and identical sequence. Other attributes, such as Embedding Import's `import_*`, are allowed and ignored.

A script of your own can write such a database with `EmbeddingWriter` from `src/utilities/HDF5_Storage.py` (with `src` on the Python path). It sanitizes the FASTA, checks each array as Embedding Import does, resumes an incomplete database, and marks it complete when every sequence has its array:

```python
from utilities.HDF5_Storage import EmbeddingWriter

with EmbeddingWriter("Embeddings/my_set_[my_model]_embeddings.h5", "my_set.fasta", "my_model", "float16") as writer:
    for header, sequence in writer.pending:      # sanitized headers and sequences still missing
        writer.add(header, my_model(sequence))   # residues x features
```
