# Embedding Multiple Sequence Alignment (`Embedding_MSA.py`)

This script generates progressive Multiple Sequence Alignments (MSAs) using protein language model embeddings. It builds a complete guide-tree distance matrix from network scores, with regression-imputed values when the network is sparse, and then progressively aligns sequence clusters using gap-aware dynamic programming.

### Input

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

#### Network Embeddings `INPUT_NETWORK_EMBED`
*   **Format**: Optional HDF5 embedding database (`.h5`), chosen like the Embedding Database. Left empty, it is not used.
*   **Use**: Only with Calibrated Imputation on and a sparse network. It holds the embeddings the network's scores were made from, and you need it only when the Embedding Database is of another model. It must record the network's model and hold every sequence of the run. It then serves the similarity of the fill and the calibration alignment. Left empty, the Embedding Database serves both and must be of the network's model.

### Parameters

| Parameter | Description |
| :--- | :--- |
| Use Sequence Filter **`USE_SEQUENCE_FILTER`** | If enabled, restricts the alignment to sequences shared by the network, embedding database, and selected FASTA file. If disabled, no FASTA is required and the network/embedding intersection is aligned using sequences stored in the embedding manifest. |
| Show Regression Plot **`SHOW_REGRESSION_PLOT`** | Toggle to show the diagnostic Isotonic Regression chart when using a sparse network. It shows the fit between length-adjusted pooled-embedding similarity and normalized network scores and, when Calibrated Imputation is on, the calibration pairs and the fit's held-out check. |
| Alignment Scoring Type **`ALIGNMENT_SCORE`** | Select whether to construct the guide tree using 'global (Needleman-Wunsch)' or 'local (Smith-Waterman)' connectivity scores from the network. |
| Score Normalization Mode **`NORMALIZATION_MODE`** | The normalization method applied to alignment scores (e.g., Alignment Length, Shorter Sequence, Longer Sequence, Average Sequence). `Alignment Length` is unavailable for local scores. Disabled for BLAST networks. |
| Tree Building Method **`TREE_METHOD`** | `UPGMA (Fast)` uses average-linkage hierarchical clustering; `Neighbor-joining (Slow)` constructs a neighbor-joining tree from the same complete distance matrix. The selected method is used for deterministic and noise-perturbed trees. |
| Noise-Perturbed Consensus Guide Tree **`BOOTSTRAP_TREE`** | Builds and averages randomly perturbed guide-tree replicates when enabled. This is a sensitivity ensemble, not classical bootstrap support. The existing setting key is retained for backward compatibility. The Tools window and MCP jobs turn it off for `Neighbor-joining (Slow)`, because each replicate is another full neighbor-joining tree; a settings file run from the command line keeps it. |
| Perturbed Tree Replicates **`NUM_TREES`** | The number of noise-perturbed replicate trees used to construct the consensus guide tree. |
| Calibrated Imputation **`CALIBRATED_IMPUTATION`** | Fills a sparse network's missing pairs from their embedding similarity and from chains of computed pairs ("computed pairs" are the pairs with an alignment score), calibrated on a sample of the missing pairs. ON (the default) draws a sample of this run's own missing pairs: 1% of them, at least 5,000 and at most 100,000 (all of them when there are no more), the same number from each of 20 similarity bands, drawn with seed 42. At the start of the run these pairs are aligned with the network's own scoring (its gap penalties and precision, through the Align Similarity Matrix tool's own alignment code). The similarity fit is made on them, weighted, together with the pairs the network has, so it covers the missing pairs. Every other missing pair is then estimated from two things: its embedding similarity, through that fit, and chains of computed pairs. For two sequences A and B, each third sequence that has a computed score with both gives the lower of its two scores, and the highest of these is the pair's chain value. The estimate combines the two with four weights fitted on the sample; a pair without any such third sequence uses the similarity estimate with a correction term. Half of the sample, held back, checks this estimate beside the check of the similarity fit alone. The log and the guide tree give both checks, the share of missing pairs that had a chain, and the four weights. **Memory**: the chain step holds a table of n × n scores (about 0.7 GB for 13,000 sequences, 7.7 GB for 44,000) and lists of the computed pairs while it runs. When that does not fit in the free memory, the run says so and fills from the similarity fit alone. **Results**: a run with Calibrated Imputation on gives another guide tree and alignment than version 0.5.0 gave. OFF aligns nothing: the fit takes the pairs the network has, and every missing pair below its range gets the fit's lowest value. The sample is used only when at least 10% of the run's pairs are missing; a run that counts less says so and uses the lowest value. **Cost**: the alignment runs before the tree is built. On a CPU, 5,000 pairs of short sequences (about 80 residues) took about 5 to 8 seconds, most of it starting the worker processes. Measured speeds of 1,300 to 3,700 pairs per second after start-up put 100,000 pairs at roughly half a minute to a minute and a half (an estimate). **Device**: with Compute Device `auto`, a sample that needs fewer CPU workers than the machine's default (one worker per 8 million score-matrix cells, at most all but two logical CPUs) goes to the CPU and no device is timed; a larger sample lets the program time the CPU and the accelerators on about 5% of its pairs and use the fastest. A named device is used as named. **Self-check**: 32 of the network's own edges are aligned in the same call. Their stored scores must come out again (within 0.0001 plus 0.0001 of the score, with the same lengths; a TF32 or BF16 network aligned in IEEE FP32 may differ by a relative 0.001 or 0.02, and two lengths). If they do not, or the network records no gap penalties, or it is a BLAST network, or the embeddings are of another model than the network's, calibration is unavailable: nothing is used, the missing pairs get the lowest value, and the log and the guide tree say why. A network from an earlier build that still stores a calibration sample is read as before: its sample pairs are ordinary edges, and a fresh sample is drawn. Every sparse network uses the full all-pairs consensus. |
| Network Embeddings **`INPUT_NETWORK_EMBED`** | Optional. The embeddings the network's scores were made from, for when the Embedding Database is of another model. Used only for a sparse network with Calibrated Imputation on. The file must record the network's model and hold every sequence of the run; otherwise it is not used and calibration is unavailable. Left empty, the Embedding Database serves, and calibration is unavailable if it is of another model than the network's. The log and the guide tree name the embeddings that served, and their model. |
| Normalized Additive Noise Scale **`NOISE_SCALE`** | Gaussian standard deviation expressed as a fraction of the valid distance range. For example, `0.02` applies an additive standard deviation equal to 2% of the maximum guide-tree distance. Every observed and regression-imputed distance is perturbed. |
| Gap Open Penalty **`GAP_OPEN`** | The gap open penalty score. More negative values penalize opening new gaps more severely. |
| Gap Extend Penalty **`GAP_EXTEND`** | The gap extend penalty score. More negative values penalize extending existing gaps. |
| CPU Worker Processes **`WORKERS`** | The number of CPU processes allocated for parallel noise-perturbed guide-tree calculations. With UPGMA, each worker holds about 16 bytes per sequence pair (about 16 GB for 44,000 sequences), so peak memory grows with this setting. Neighbor-joining workers also share all but two logical CPUs as threads, so a single worker already uses several cores. |
| Compute Device **`DEVICE_SELECTION`** | Selects `auto` or a specific available CPU, CUDA, XPU, or MPS device for sequential profile score-matrix construction. Auto benchmarks three real leaf-to-leaf guide-tree merges near the 25th, 50th, and 90th score-matrix cost percentiles and selects one device for the full progressive merge. Guide-tree calculations and dynamic-programming traceback remain on CPU. |
| Temporary Working Directory **`SAFE_TEMP_DIR`** | The temporary directory used to cache intermediate files and memory-mapped matrices. When unset, the cache is created in the run's alignment folder (`MSA_DIR`) and removed after the guide tree is built. |

### Output

#### Multiple Sequence Alignment FASTA File
*   **Format**: Aligned FASTA (`.fasta`).
*   **Structure**: Standard FASTA, containing aligned sequence strings padded with gaps (`-`) to match the final alignment length.

#### Guide Tree NEXUS File
*   **Format**: NEXUS tree (`.nex`) named `<set>_[<model>]_guide_tree.nex`, saved in a `Guide_Trees` folder inside the alignment's folder (`MSA_DIR`) as soon as the guide tree is built, before the progressive alignment starts. A tree follows its alignment: `<name>_alignment.fasta` (or `<name>_alignment_sparse.h5`) has the tree `Guide_Trees/<name>_guide_tree.nex`. An existing file is replaced.
*   **Origin**: An `EMAPSSN` block records how the tree was built: the program version and date, the alignment file name, the input files, the score and tree settings, how missing pairs were imputed and how well a calibration sample confirmed it (`MISSING_PAIRS`) and what calibration did (`CALIBRATION`), the perturbed-tree settings when they run, how the tree is rooted, and what its branch lengths and labels mean. Tree readers skip this block.
*   **Structure**: The `TREES` block holds one tree, `guide_tree`. Its text after the equals sign is plain Newick, so it can be copied out for tools that read only Newick. Its leaves are the aligned sequence headers, quoted unless they hold only letters, digits, dots and bars, because NEXUS reads an unquoted underscore as a space. Branch lengths are guide-tree distances. A `UPGMA (Fast)` tree keeps its root and is marked rooted (`[&R]`). A `Neighbor-joining (Slow)` tree is rooted at the midpoint of its longest path and marked unrooted (`[&U]`), because that root is a choice, not something the tree shows.
*   **Split Labels**: When `BOOTSTRAP_TREE` is enabled, each internal node is labelled with the percentage of noise-perturbed replicate trees that contain its split. These measure stability under distance noise, not classical bootstrap support.
*   **Use**: Tree viewers such as iTOL or FigTree open it, and so does the Viewer's Tree View (the `tree` command), which draws it beside the network as its Guide tree (the tree saved for the loaded alignment), next to the network threshold tree and any tree of your own. Biopython's NEXUS reader splits quoted headers that contain commas, so read the copied-out tree line with its Newick reader instead. The tree records the order in which the alignment was built, not a phylogeny: its distances are inverted alignment scores, and on a sparse network many of them are regression-imputed.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Intersection Filtering**:
     Always intersects sequence headers from the embedding HDF5 file and network HDF5 file. If `USE_SEQUENCE_FILTER` is enabled, the selected FASTA file is added as a third intersection source; otherwise, sequence strings come from the embedding manifest.

2. **Network Score Normalization**:
     Normalizes the network alignment scores based on `NORMALIZATION_MODE` to account for sequence length discrepancies.
     * **Alignment Length**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{L_{\text{align}}}$$
     * **Shorter Sequence**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\min(L_i, L_j)}$$
     * **Longer Sequence**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\max(L_i, L_j)}$$
     * **Average Sequence**: $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\text{mean}(L_i, L_j)}$$

3. **Distance Matrix Formulation**:
     Inverts scores to represent distances:
     $$D(i, j) = \max(S_{\text{norm}}) - S_{\text{norm}}(i, j)$$

4. **Isotonic Regression for Sparse Networks**:
     If the network is sparse, the distance matrix is incomplete. The script:
     - Max- or Mean-pools residue embeddings to generate a global representation vector **u<sub>i</sub>** for all sequences.
     - Computes all-vs-all length-adjusted cosine similarities:
       $$\text{Adj}(i, j) = \text{Sim}_{\text{cos}}(i, j) \times \left(\frac{\min(L_i, L_j)}{\max(L_i, L_j)}\right)^P$$
       This is the one similarity rule every tool uses (max pooling, $P = 2$, the matrix product on one BLAS thread), so Edge Pre-filtering, this tool and Network Injection get the same numbers.
     - Fits an **Isotonic Regression** function $f$ mapping known $\text{Adj}(i, j)$ values to normalized network scores $S_{\text{norm}}(i, j)$. Above 500,000 observed pairs, a random 500,000 stand in for them.
     - **Calibration sample**: A fit on the observed pairs alone gives every missing pair below the lowest of them the same value. With `CALIBRATED_IMPUTATION` on, the run draws a sample of its own missing pairs (see Calibrated Imputation), aligns it at the start and adds it to the observed pairs before the scores are normalized. Each sample pair is weighted by the number of missing pairs it stands for, so $f$ spans the missing pairs' range. A first fit without every second sample pair is scored on those pairs: Spearman's $\rho$, the mean absolute error and $R^2$ of its estimates against their computed scores are printed, shown in the regression chart and recorded in the guide tree (`MISSING_PAIRS`, and `CALIBRATION` for what the sample was). The similarity estimate then uses the fit with every sample pair. When the missing pairs are no more than the sample size, all of them are aligned and nothing is imputed.
     - **Chain fill**: The sample pairs are computed pairs from here on. For a pair $(i, j)$, every third sequence $k$ with computed scores to both gives $\min\big(S_{\text{norm}}(i,k),\, S_{\text{norm}}(k,j)\big)$, and the highest of these is the chain value $b(i, j)$. A pair's own score is never used for itself. With $\hat{s}$ the similarity estimate $f(\text{Adj}(i, j))$, a missing pair is estimated as
       $$S(i, j) = c_0 + c_1\,\hat{s} + c_2\,\tilde{b} + c_3\,u$$
       where $\tilde{b}$ is the chain value $b$ when the pair has a chain and $\hat{s}$ otherwise, and $u$ is 1 for a pair without a chain and 0 otherwise. The four weights $c_0$ to $c_3$ are fitted on the sample pairs by weighted least squares, each sample pair's $\hat{s}$ coming from a similarity fit made without its half of the sample, so no pair is predicted by a fit that saw it. The half of the sample held back by the first fit scores the result (the weights fitted on the other half): its Spearman's $\rho$, mean absolute error and $R^2$ are printed beside those of the similarity fit alone and recorded in `MISSING_PAIRS`, with the share of missing pairs that had a chain and the four weights. The step holds an $n \times n$ table of scores, the lists of each sequence's computed neighbours and a bit table, for the length of the step only; when they do not fit in the free memory (the free memory less 2 GB or a tenth of the total, whichever is more), the log says so and the missing pairs are estimated from the similarity fit alone (see "Predicts missing network scores" below).
     - **Uniform fill**: With the switch off, with fewer than 10% of the run's pairs missing, or when calibration is unavailable, nothing is aligned. The fit takes only the observed pairs, unweighted, so every missing pair below the lowest of them gets the fit's lowest value $f(x_{\min})$, and the log says why there was no sample. A BLAST network, or one that records no gap penalties, is always filled this way, with a warning that its estimates are unchecked. Every run that fills missing pairs this way prints the value, `Uniform fill: ...`.
     - Predicts missing network scores. Without a calibration sample, or when the chain fill does not fit in memory:
       $$S_{\text{predicted}}(i, j) = f(\text{Adj}(i, j))$$
       With a calibration sample, this is the similarity estimate the chain fill combines with the chains.
       
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

     Each replicate tree contributes one tree distance per sequence pair. A UPGMA replicate contributes the cophenetic distance, the height at which the two sequences merge. A Neighbor-Joining merge height only keeps the linkage format monotone, so a Neighbor-Joining replicate contributes the path length along its branches instead:

     $$d_{\text{tree}}(i,j) = \delta(i) + \delta(j) - 2\,\delta(\operatorname{LCA}(i,j))$$

     where $\delta$ is a node's distance from the replicate's root and $\operatorname{LCA}(i,j)$ is the two sequences' lowest common ancestor. Branches are clamped as in the guide tree export, so no path is negative. Path lengths do not depend on where a tree is rooted, and averaging them keeps the differences between sister sequences that merge heights would erase.

     The final matrix is the replicate-averaged tree distance of every pair, observed or imputed, for a sparse network as for a complete one. Imputed pairs take part in every replicate tree and are replaced by their replicate averages in the final matrix. When noise-perturbed trees are disabled, the complete observed-plus-imputed baseline directly builds the deterministic tree. A settings file that still holds the retired setting `INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS` runs; the setting is ignored, and a note says so.

     These replicates measure sensitivity to randomized distance perturbations; they are not classical bootstrap resamples and do not provide classical bootstrap support values.

6. **Guide Tree Export**:
     Saves the final guide tree as NEXUS before the progressive alignment, so a failed alignment keeps it.
     * **UPGMA**: each branch is half the height difference between a node and its parent, so the path between two leaves equals the height at which they merge.
     * **Neighbor-Joining**: each join gives its two branches the Saitou–Nei lengths
       $$b_u = \frac{1}{2}\left(D(u,v) + \frac{R_u - R_v}{k - 2}\right), \qquad b_v = D(u,v) - b_u$$
       where $R_u$ is node $u$'s row sum over the $k$ active nodes; the last join splits its distance evenly. A negative length becomes zero and its sibling takes the join's whole distance. The tree is then rooted at the midpoint of its longest leaf-to-leaf path.
     * **Split Labels**: a split is counted in every replicate tree that contains it, wherever that replicate is rooted. Splits that separate a single sequence are not labelled.

7. **Progressive Profile Alignment**:
     Traverses the guide tree from leaves to the root. At each internal node, it merges the two child clusters (which could be single sequences or alignment profiles) by:
     - Resolving a manual compute device or, in Auto mode, benchmarking representative real leaf merges after guide-tree construction. The benchmark includes score-matrix host/device transfers but excludes HDF5 loading and CPU traceback.
     - Unit-normalizing each leaf residue embedding before it enters a profile.
     - Averaging those unit vectors across every sequence in the cluster while treating gaps as zero. The resulting vector norm therefore records both non-gap occupancy and directional agreement within the column.
     - Calculating reciprocal cosine similarity from the profile-vector directions, then weighting each cell by the two column norms so weakly supported or inconsistent columns contribute less alignment evidence.
     - Running dynamic programming with `GAP_OPEN` and `GAP_EXTEND` penalties to find the optimal path.
     - Merging alignments and outputting the final FASTA multiple sequence alignment (MSA).

</details>

---

# Sparse MSA Converter (`Sparse_MSA_Converter.py`)

This script compresses multiple sequence alignments (MSAs) into compact HDF5 files. It validates and sanitizes the aligned FASTA, converts residue strings to a SciPy Compressed Sparse Row (CSR) matrix, writes lookup metadata, and then moves the successfully converted source FASTA into a `Full_Alignments` subdirectory.

### Input

#### MSA Alignment File `INPUT_FASTA`
*   **Format**: Aligned FASTA (`.fasta`).
*   **Created By**: `Embedding_MSA.py` (Embedding Multiple Alignment utility) or other external MSA tools (e.g. Clustal, MUSCLE).
*   **Description**: Pre-calculated standard multiple sequence alignment file.

### Parameters

| Parameter | Description |
| :--- | :--- |
| Convert All Alignments **`CONVERT_ALL`** | Toggle to convert all FASTA multiple alignments inside the input directory. If disabled, only the selected alignment file is converted. |

### Output

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
