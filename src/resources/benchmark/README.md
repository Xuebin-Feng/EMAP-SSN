# EMAP-SSN benchmark

The benchmark runs the program's heavy calculations on a fixed set of public
protein sequences, so reports from different machines can be compared. This
folder holds the benchmark and that sequence set.

| File | What it holds |
|---|---|
| `Run_Benchmark.py` | The benchmark. |
| `benchmark_sequences.fasta` | The main set: 860 sequences. |
| `injection_sequences.fasta` | 92 more sequences, held out for the injection stage. |
| `README.md` | This file: how to run the benchmark, where the sequences come from, how the sets were made, and their licence. |

## Running the benchmark

From the EMAP-SSN folder, with the program's Python:

```
.venv\Scripts\python.exe -u src\resources\benchmark\Run_Benchmark.py
```

On Linux and macOS the Python is `.venv/bin/python`. `--stages 3,7` runs only
those stages and the stages they need, by number or name. `--result <file>` also
writes an English summary as JSON, which MCP jobs read.

It runs ten stages, each in a process of its own, with every device setting on
Auto:

| # | Stage | What it times |
|---|---|---|
| 1 | Sanitize sequences | `Sanitize_Sequences.py` on the main set |
| 2 | Embeddings | `Generate_Embeddings.py` with ESM-2 8M (`esm2_t6_8m`) |
| 3 | All-against-all embedding alignment | `Align_Similarity_Matrix.py`: 369,370 pairs |
| 4 | SSN layout | the force-directed layout of the top 5% of edges |
| 5 | UMAP layout | the same network laid out with UMAP |
| 6 | Clustering | Leiden, MCL and the Jaccard filter, as the `cluster` command runs them |
| 7 | Embedding database search | `Embedding_SSEARCH.py`: E. coli's glyoxalase II (B1XD76) against the main set |
| 8 | Injection of new sequences | `Embedding_Injection.py` and `Network_Injection.py` with the 92 held-out sequences |
| 9 | Embedding MSA | `Embedding_MSA.py` on the first 100 glyoxalase II entries |
| 10 | BLAST all-against-all alignment | `Align_Substitution_Matrix.py`, with NCBI BLAST+ |

A stage this machine can't run, such as BLAST without NCBI BLAST+ on the PATH or
in `C:\Program Files\NCBI`, is skipped with the reason, and so are the stages
that need it. The first run downloads ESM-2 8M (about 30 MB) into the Hugging
Face cache; that download is not timed.

The report is named by the time the run started, as
`Benchmark_Report_<date>_<time>.txt`, with an English `.json` of its data beside
it, in this folder. It records each stage's time, throughput, CPU time, memory
peaks and device; the choices the tools' own Auto benchmarks made; the
hardware and software; and the conditions that affect the numbers. Reports are
never deleted, and a second run in the same second adds `_2`. The terminal shows
the report in English and ends with the report's path.

While it runs, the benchmark keeps every file it makes in `temp/` here, with a
lock that stops a second benchmark from starting. `temp/` is cleared when a run
starts and when it ends, so files a killed run left behind go at the next start.
The benchmark never reads or changes `tools_settings.json` or the folders
configured in the Tools window. A run needs about 1 GB of free disk space and
took about 6 minutes on an RTX 5070 Ti with a Core Ultra 7 265KF.

It exits with 0 when every stage ran or was skipped for a stated reason, 1 when
a stage failed or the run was interrupted (Ctrl+C still writes the report), and
2 when it could not start: another benchmark is running, the folder is not
writable, or the disk is too full.

## The sequences

The sequences are the reviewed UniProtKB/Swiss-Prot entries of the
metallo-β-lactamase superfamily: every reviewed entry that InterPro entry
IPR001279 (Metallo-beta-lactamase) matches. The family splits into distinct
functions, which gives the layout and clustering stages realistic structure.

- **Source:** UniProt release 2026_03, released on 2 September 2026.
- **Query:** made on 2026-10-09 through UniProt's REST service:
  `https://rest.uniprot.org/uniprotkb/stream?query=(xref:interpro-IPR001279)%20AND%20(reviewed:true)&format=fasta`
- **Download:** 1,094 entries in 519,679 bytes, SHA-256
  `33cc5027b566120a0f7dee3c32ee5c5eeabaf69a330bed910bd88dda3f73b332`.
- The REST service serves only the current release. Earlier releases are kept
  as archives on UniProt's FTP site, under
  `https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/`.

| | Main set | Injection set |
|---|---|---|
| Sequences | 860 | 92 |
| Residues | 284,647 | 31,825 |
| Median length | 280 aa | 257 aa |
| Shortest and longest | 132 and 911 aa | 219 and 880 aa |
| Genera | 279 | 66 |
| Pairs to align | 369,370, all against all | 83,306: 92 × 860, plus 4,186 among the new sequences |

The main set's largest groups:

| Protein | Sequences |
|---|---|
| Hydroxyacylglutathione hydrolase (glyoxalase II) | 196, plus 8 mitochondrial |
| Ribonuclease Z | 124 |
| Coenzyme PQQ synthesis protein B (PqqB) | 50 |
| Anaerobic nitric oxide reductase flavorubredoxin | 21 |
| Ribonuclease J | 13 |
| Metallo-beta-lactamase type 2 | 13 |
| Atrochrysone carboxyl ACP thioesterase | 12 |
| Ribonuclease BN | 9 |

The embedding MSA stage aligns the first 100 hydroxyacylglutathione hydrolase
entries of the main set, by accession: A0KIK2 to Q04RQ6, from 233 to 269 aa.
They are the entries whose protein name is exactly "Hydroxyacylglutathione
hydrolase".

## How the sets were made

1. Drop the one entry longer than 1,022 residues, the longest sequence ESM-2
   embeds: Q8MM62, 1,096 aa.
2. Keep one entry per distinct sequence. 141 entries are exact copies of
   another entry's sequence, mostly the same protein from several strains. Of
   each group of copies, the entry whose accession sorts first is kept, which
   leaves 952 entries.
3. Sort the entries by accession and shuffle them with Python's
   `random.Random(42)`.
4. The first 860 form the main set and the other 92 the injection set.
5. The records are otherwise unchanged: UniProt's headers, and its sequence
   lines of 60 residues, with LF line ends. The benchmark's first stage
   sanitizes the headers, as it would for a user's file.

In a folder that holds the download as `IPR001279_reviewed.fasta`, this
Python 3 code makes both files byte for byte:

```python
import random
import re

with open("IPR001279_reviewed.fasta", "rb") as download:
    records = re.findall(r">[^>]*", download.read().decode("ascii"))


def accession(record):
    return record.split("|")[1]


def sequence(record):
    return "".join(record.split("\n")[1:])


distinct = {}
for record in sorted(records, key=accession):
    if len(sequence(record)) <= 1022:
        distinct.setdefault(sequence(record), record)
chosen = sorted(distinct.values(), key=accession)
random.Random(42).shuffle(chosen)
with open("benchmark_sequences.fasta", "wb") as main_set:
    main_set.write("".join(chosen[:860]).encode("ascii"))
with open("injection_sequences.fasta", "wb") as injection_set:
    injection_set.write("".join(chosen[860:]).encode("ascii"))
```

| File | Bytes | SHA-256 |
|---|---|---|
| `benchmark_sequences.fasta` | 409,376 | `a3bc03cfe07a1f2c110e79560e48d644a61313c5b00024f1b1fbf500b4ada23a` |
| `injection_sequences.fasta` | 45,355 | `59494479a99aa4cc47b8266afd3f5486bb14eab55ed5afc8ac0bd4c3f617b9e8` |

`.gitattributes` keeps both files' bytes exactly as they are on every platform,
and `tests/test_benchmark.py` checks them against this table.

## Licence and attribution

UniProt applies the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>) to all
copyrightable parts of its databases (<https://www.uniprot.org/help/license>).

The two FASTA files are adapted from UniProtKB/Swiss-Prot release 2026_03 by
the UniProt Consortium. They were selected with the query above on 2026-10-09,
then filtered, deduplicated and split as described; no record was edited.
They are shared under the same licence.

UniProt asks to be cited as:

> The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in
> 2025. *Nucleic Acids Research* 53:D609–D617 (2025).
> <https://doi.org/10.1093/nar/gkae1010>

UniProt makes no warranties regarding the correctness of the data.
