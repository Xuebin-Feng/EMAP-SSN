<!-- Translation of Sequence_Similarity_Calculations.md, sha256 308e94142c1acd26bbe8f56dc88032df5e6e4b233f6efa1cba4bc525f7b42e57 -->
# 动态规划 embedding 比对 (`Align_Similarity_Matrix.py`)

该脚本使用残基级蛋白质 embedding 计算全对全的序列相似性网络（SSN）。它不使用传统的氨基酸替换矩阵（substitution matrix），而是通过比较每个残基（residue）稠密的高维 embedding 向量来计算相似性矩阵。

它使用以优化的 Numba JIT 函数实现的动态规划（局部比对用 Smith-Waterman，全局比对用 Needleman-Wunsch）为序列比对（alignment）打分。为加快大型数据集的计算，它提供一个可选的预过滤步骤：根据池化后的序列 embedding 计算全局余弦相似度（默认最大池化；内部也支持平均池化），并跳过得分最低的序列对的完整比对。

### 输入

#### HDF5 embedding 数据库 `INPUT_HDF5`
*   **格式**：HDF5（`.h5`）。
*   **创建者**：`Generate_Embeddings.py`（embedding 生成工具）。
*   **说明**：以 $L \times D$ 数组的形式包含预先计算的残基 embedding。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 启用边预过滤 **`EDGE_PREFILTERING`** | 切换是否用全局余弦相似度（cosine similarity）对序列对进行预过滤。启用时，相似度低于阈值的序列对会跳过完整的残基级比对。 |
| 预过滤强度 **`PREFILTER_STRENGTH`** | 全局余弦相似度得分最低、要从比对计算中排除的序列对所占的百分比。数值越高速度越快，但可能漏掉亲缘关系较远的蛋白质之间的比对。 |
| CPU 工作线程数 **`WORKERS`** | 分配给并行比对计算的 CPU 进程/线程数。 |
| 局部空位罚分 **`LOCAL_GAP_P`** | 局部比对（local alignment）使用的空位罚分（gap penalty）值。数值越负，对空位（gap）的惩罚越重，产生的空位越少。 |
| 全局空位罚分 **`GLOBAL_GAP_P`** | 全局比对（global alignment）使用的空位罚分值。 |
| 处理批大小 **`BATCH_SIZE`** | 每个分块（tile）中处理、随后写入磁盘的序列对数量（整数）。批次（batch）越大，资源利用率可能越高，但内存占用也越大。 |
| 计算设备 **`DEVICE_SELECTION`** | 为构建残基得分矩阵选择 `auto` 或某个可用的 CPU、CUDA、XPU 或 MPS 设备。自动选择设备时会估算工作内存，在最多 256 个按成本分层的序列对上调优安全的并行通道数，并在最多 2,048 个按正式计算顺序排列的序列对上确认候选方案。动态规划（dynamic programming）打分始终在 CPU 上进行。 |
| 执行模式 **`EXECUTION_MODE`** | `Auto`（自动）在支持时比较标量方案与分块方案。`Scalar`（标量）在每个加速器通道（lane）上一次处理一个双序列得分矩阵，在 CPU 和受支持的加速器上始终可用。`Tiled`（分块）强制在 CUDA/ROCm 或 XPU 上使用内存受限的 embedding 分块和填充后的微批次（microbatch）；若没有兼容的加速器，会在处理序列对之前失败。 |
| 主机缓存 **`HOST_CACHE_GB`** | 打包 embedding 缓存可使用的主机内存上限（GiB）。`auto` 至少保留 8 GiB 或物理内存的 25%，并将缓存上限设为 128 GiB；`0` 则只使用有界的 HDF5 分块。 |
| 加速器精度 **`ACCELERATOR_PRECISION`** | `automatic_32bit` 会在最多 2,048 个按正式计算顺序排列的序列对上，用 `EXECUTION_MODE` 允许的每种执行模式测试 IEEE FP32 和 TF32。只有当这些方案保持比对长度不变、通过逐残基的得分容差检查、结果均为有限值，且最快的 TF32 方案比最快的 FP32 方案至少快 10% 时，才会选用 TF32。`bf16` 表示明确的低精度执行：归一化（normalization）和得分后处理仍用 FP32，而归一化后的矩阵乘法操作数使用 BF16。它需要具备相应能力的加速器，会打印低精度警告，并针对每种设备/执行变体，在最多 2,048 个具有代表性的序列对上报告相对于 FP32 的比对长度和原始得分统计。有限的数值差异仅供参考，不会导致拒绝 BF16。旧的输入别名 `auto` 会被当作 `automatic_32bit`。 |

### 输出

#### HDF5 比对网络
*   **格式**：HDF5（`.h5`）。
*   **结构**：
    - `/i`：源序列节点（node）索引。
    - `/j`：目标序列（target sequence）节点索引。
    - `/g_score`：全局比对得分。
    - `/g_len`：全局比对长度。
    - `/l_score`：局部比对得分。
    - `/l_len`：局部比对长度。
    - `/seq_lens`：按序列标题（FASTA header）顺序排列的已清理序列长度。
    - `/headers`：序列标题数组。
    - 属性 `model_name`、`embedding_checksum` 和 `gap_penalties`，用于校验以及兼容的增量复用。

<details markdown="1">
<summary><b>算法详情</b></summary>

比对流程按以下步骤执行：

1. **全局 embedding 池化**：
     对每条蛋白质序列，将其残基 embedding 池化为一个全局表示向量。当前默认使用最大池化（`POOLING_METHOD="max"`；内部仍支持平均池化）：
     $$u_i = \max_{\text{residues}}(\text{emb}_i(\text{residue}))$$

2. **边预过滤（可选）**：
     若启用预过滤，会在 CPU 上计算所有序列两两之间的余弦相似度：
     $$\text{Sim}_{\text{cos}}(i, j) = \frac{u_i \cdot u_j}{\|u_i\|_2 \times \|u_j\|_2}$$
     
     该相似度再按序列长度之比进行校正：
     $$\text{Adj}(i, j) = \text{Sim}_{\text{cos}}(i, j) \times \left(\frac{\min(L_i, L_j)}{\max(L_i, L_j)}\right)^P$$
     
     得分处于 `PREFILTER_STRENGTH` 所对应最低百分位的序列对会被跳过。

3. **残基级相似性矩阵计算**：
     对每个通过预过滤的序列对 $(i, j)$，计算一个两两余弦距离矩阵。首先，沿隐藏维度对残基 embedding 做 L2 归一化：
     $$\hat{v}_i(a) = \frac{v_i(a)}{\|v_i(a)\|_2}$$
     
     然后通过矩阵乘法计算余弦距离：
     $$D(a, b) = 1.0 - \hat{v}_i(a) \cdot \hat{v}_j(b)$$
     
     其中 $v_i(a)$ 是序列 $i$ 中残基 $a$ 的 embedding 向量。该距离再被转换为相似性矩阵：
     $$S(a, b) = \exp(-D(a, b))$$

     在 CUDA/ROCm 或 XPU 上，待计算的序列对会重新分组为内存受限的源/目标分块。归一化 embedding、流和即时张量工作区在各输出批次之间保持复用。长度相近的目标序列最多填充 15%，多个矩阵通过一次批量乘法完成计算。填充部分不计入任何统计，也不进入 CPU 上的比对矩阵。原有的逐序列对路径仍作为经过基准测试的后备方案保留。

4. **双重 z-score 归一化**：
     对相似性矩阵按行和按列分别归一化，以校正残基特有的背景相似度：
     $$Z_{\text{row}}(a, b) = \frac{S(a, b) - \mu_{\text{row}}(a)}{\sigma_{\text{row}}(a) + \varepsilon}$$
     $$Z_{\text{col}}(a, b) = \frac{S(a, b) - \mu_{\text{col}}(b)}{\sigma_{\text{col}}(b) + \varepsilon}$$
     
     最终的比对打分矩阵是这两个 z-score 的平均值：
     $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$

5. **动态规划比对**：
     * **Needleman-Wunsch（全局计算）**：使用打分矩阵 $\text{Score}(a, b)$ 和 `GLOBAL_GAP_P` 求解标准的全局递推关系，输出全局得分。
     * **Smith-Waterman（局部计算）**：从打分矩阵中减去偏移值 2.0（以确保不相似的匹配得分为负）：
       $$\text{Score}_{\text{local}}(a, b) = \text{Score}(a, b) - 2.0$$
       
       然后使用 `LOCAL_GAP_P` 运行标准的局部动态规划递推，求出最优的局部比对得分。

</details>

---

# 替换矩阵比对 (`Align_Substitution_Matrix.py`)

该脚本使用传统的氨基酸替换矩阵进行全对全的局部序列比对。它根据输入序列集构建一个本地 NCBI BLAST 数据库，并并行执行 BLASTP 查询。得到的期望值（E-value）会被转换为可线性比较的负 Log10(E) 边（edge）权重，用于构建网络；当 `SCORE_METRIC` 要求时，则改由 BLAST 的比特得分（bit score）和序列长度计算 EFI-EST 比对得分。

### 输入

#### 序列 FASTA 文件 `INPUT_FASTA`
*   **格式**：标准 FASTA（`.fasta`）。
*   **创建者**：`Sanitize_Sequences.py`（序列清理工具）或用户提供的原始 FASTA。
*   **说明**：用于运行 BLAST 的原始序列数据库。记录会先经过与 `Generate_Embeddings.py` 相同的规范清理（sanitize）：序列标题、残基、空记录和重复序列。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 替换矩阵 **`MATRIX`** | BLAST 比对中用于传统打分的氨基酸替换矩阵（例如 BLOSUM45、BLOSUM50、BLOSUM62、BLOSUM80、BLOSUM90、PAM30、PAM70、PAM250）。 |
| 得分 **`SCORE_METRIC`** | 网络为每个序列对存储的边得分。`evalue`（默认）是期望值的负 Log10；`efi_alignment_score` 是 EFI-EST 比对得分，由每个命中的比特得分和两条序列的长度计算。无论哪种，都保留一个序列对（任一方向）中最强的命中。 |
| BLAST 线程数 **`NUM_THREADS`** | 分配给 BLASTP 执行和结果解析的 CPU 线程数。 |
| 处理批大小 **`BATCH_SIZE`** | 在写入中间 HDF5 文件和汇总最终网络时，一次缓冲或复制的已解析边的最大数量。 |
| BLASTP 目录 **`BLASTP_DIR`** | 可选的目录，其中包含 `blastp` 和 `makeblastdb`。留空（默认）表示由工具自行查找 NCBI BLAST+，空字段会显示查找的结果。字段旁的**下载**（Download）按钮会从 NCBI 安装 BLAST+。见下文《查找或安装 BLAST+》。 |

#### 查找或安装 BLAST+

已设置的 `BLASTP_DIR` 会按所写内容原样使用，不会再尝试其他位置：它不是文件夹，或缺少 `blastp` 或 `makeblastdb` 时，运行会停止，并给出指明所缺内容的错误。相对路径按项目文件夹解析。字段留空时，工具每次运行都会依次查找：`PATH`；然后在 Windows 上查找 `Program Files` 下 NCBI 文件夹中版本最新的文件夹（按版本号比较，所以 2.17 比 2.9 新），在 Linux 和 macOS 上查找 `/usr/local/ncbi/blast/bin`、`/usr/local/bin`、`/usr/bin` 和 `/opt/homebrew/bin`。空字段中的灰色文字显示此刻的查找结果，或说明未找到 BLAST+。工具从不会把找到的文件夹写入该字段或保存的设置。

**下载**按钮（字段后两个图标按钮中的第一个，第二个是**浏览**）从 NCBI 安装 BLAST+ 2.17.0：

*   它先请你确认，并写明版本、本机的下载大小（Windows 约 143 MB，Linux 约 296 MB，macOS 约 206 MB）和安装后的大小（分别约 34、63 和 47 MB），以及 BLAST+ 是美国政府作品、属于公有领域（public domain）。然后让你选择文件夹（从项目文件夹开始），并安装到 `<所选文件夹>/blast-2.17.0+`。
*   压缩包会按 EMAP-SSN 中记录的大小和 SHA-256 校验和检查，只安装 `blastp` 和 `makeblastdb`（Windows 上还有 `blastp.exe` 所需的 `nghttp2.dll`），每个文件再按各自的 SHA-256 检查。`blastp -version` 必须能够启动，已有的安装才会被替换。下载失败或取消后不会留下临时文件。
*   下载在后台进行：窗口仍可使用，进度对话框显示当前阶段，其取消按钮可停止下载。关闭窗口也会取消下载。取消或失败时字段保持原样，失败时会说明原因。
*   成功后，字段显示新文件夹（位于项目文件夹内时为相对路径）。与其他修改一样，它会在你 Save & Run 时随工具的设置一起保存。
*   在下载不支持的计算机上，下载按钮被禁用，提示文字说明原因；请在那里自行安装 BLAST+ 并设置 `BLASTP_DIR`。

BLAST+ 需要以下组件，下载无法提供：

*   **Windows 10 或更高版本（x64）**：Microsoft Visual C++ 2015–2022 Redistributable (x64)。
*   **Linux（x86-64）**：glibc 2.28 或更高版本以及 `libgomp1` 软件包（Fedora 和 RHEL 上为 `libgomp`）；它不能在 Alpine 等基于 musl 的系统上运行。
*   **macOS**：14 或更高版本，Apple silicon。

BLAST+ 默认会向 NCBI 报告使用统计。本工具运行的每个 BLAST+ 程序启动时都带有 `BLAST_USAGE_REPORT=false`，从而关闭报告。（若你设置了 BLAST+ 自己的 `DO_NOT_TRACK` 和 `NCBI_USAGE_REPORT_ENABLED`，则它们优先。）

BLASTP 目前使用宽松的固定期望值阈值 `1e300`，每个查询最多 `1,000,000` 条目标序列，每个目标只保留一个 HSP，并使用条件性组成统计（`-comp_based_stats 2`）。这些是实现中的常量，而不是界面中的字段。

中间的查询片段、BLAST 数据库、结果文件和解析批次会自动保存在所设网络目录中一个按序列集区分的临时文件夹里。
中断后重新运行时，只复用输入、已清理清单、替换矩阵、线程数、BLASTP 版本、查询分块和源结果校验和仍然一致的完整 HDF5 批次。最终网络通过校验并成功发布后，临时工作区会被删除。

### 输出

#### HDF5 比对网络
*   **格式**：HDF5（`.h5`），命名为 `<set>_[BLAST]_EValue.h5`；使用 EFI 比对得分时为 `<set>_[BLAST]_EFIScore.h5`。
*   **结构**：
    - `/i`：源序列节点索引。
    - `/j`：目标序列节点索引。
    - `/score`：每个无向序列对经双向去重后的最佳 $-\log_{10}(E_{\text{value}})$ 得分，或最佳 EFI 比对得分。
    - `/headers`：序列标题数组。
    - 属性 `model_name="BLAST"` 和 `matrix`；EFI 比对得分网络还有 `score_kind="efi_alignment_score"` 和 `score_transform`。没有 `score_kind` 的网络存储的是期望值得分。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **FASTA 规范清理与数字 ID**：
     输入首先经过与 `Generate_Embeddings.py` 相同的共享清理：清理序列标题，将残基字符串转为大写并屏蔽无效字符，删除空记录，合并相同的序列，并使保留的序列标题唯一。为避免 NCBI BLAST 解析失败，脚本随后生成一个临时映射：
     $$\text{Header} \to \text{Index} \quad (0, 1, 2, \dots)$$
     
     它会创建一个临时的 `safe_fasta`，其中的序列标题被重命名为各自的整数索引。

2. **构建本地 BLAST 数据库**：
     执行 `makeblastdb`，用清理后的 FASTA 生成本地数据库。

3. **多线程查询分块**：
     将查询 FASTA 拆分为多个并行的临时分块，把搜索工作量分配给所选数量的 CPU 工作线程（`NUM_THREADS`）。

4. **执行双序列 BLASTP**：
     对每个查询分块，针对数据库执行 `blastp`：
     * **blastp** → 表格输出格式 6

5. **期望值转换**：
     提取（extraction）每个命中（hit）的期望值，并用以 10 为底的负对数将其线性化为边的连接权重：
     $$\text{Score} = -\log_{10}(E_{\text{value}} + 10^{-300})$$
     
     加上下限偏移 $10^{-300}$ 是为了在期望值为 0.0 时避免数学上的除零问题。

     当 `SCORE_METRIC` 设为 `efi_alignment_score` 时，改由比特得分和两条已清理序列的长度给出 EFI-EST 比对得分，且不做取整（EFI-EST 会将其向下取整为整数）：
     $$\text{Score} = \text{bit score} \cdot \log_{10}(2) - \log_{10}(L_{\text{query}} \cdot L_{\text{subject}})$$

     与 $-\log_{10}(E)$ 一样，它随相似度升高而增大，但没有 300 的上限，不依赖序列集的大小，并且长序列之间的弱命中可能得 0 分或负分。EFI-EST 没有给出通用的阈值：请从较低的值开始，逐步提高相似性阈值，直到功能已知的序列彼此分开。

6. **整合**：
     合并分块输出文件，把临时索引还原为原始序列标题，并导出最终的网络文件。

</details>

---

# 解析 BLAST 输出 (`Parse_BLAST_Output.py`)

该脚本解析严格的、以制表符分隔的 BLAST+ 或 DIAMOND 输出（需配合必需的配套 FASTA），并将其转换为期望值得分的标准 HDF5 网络；当 `SCORE_METRIC` 要求时，则转换为 EFI 比对得分的网络。完整的 FASTA 序列标题和所选的 BLAST 序列标题都按查看器共享的序列标题规则清理，然后精确匹配。双向命中（reciprocal hit）和重复命中会合并为最强的无向边，没有任何命中的 FASTA 记录仍会作为孤立节点保留。

### 输入

#### 表格格式 BLAST 输出文件 `INPUT_BLAST_TABULAR`
*   **格式**：UTF-8 表格格式 BLAST 输出（`.tabular`、`.txt`、`.tab`、`.tsv`）。
*   **创建者**：在外部运行的 NCBI BLASTP 或 DIAMOND `blastp`。
*   **结构**：通过 `BLAST_LAYOUT` 选择 `standard_outfmt6`、`outfmt7_fields` 或 `custom_columns`。DIAMOND 默认的 `--outfmt 6` 包含 12 个标准列。

#### 配套 FASTA 文件 `INPUT_FASTA`
*   **格式**：UTF-8 FASTA（`.fasta`）。
*   **用途**：定义查看器中的每个节点（包括没有 BLAST 命中的序列）以及节点的规范顺序。
*   **序列标题规则**：`>` 之后的完整序列标题会被清理。重复的原始序列标题、清理后冲突、空序列标题、空序列以及无效的 FASTA 结构都会被拒绝。

### 参数

*   `SCORE_METRIC`：边得分。`evalue`（默认）存储期望值的负 Log10；`efi_alignment_score` 存储 EFI-EST 比对得分，由每个命中的比特得分和配套 FASTA 中两条序列的长度计算。
*   `BLAST_LAYOUT`：输入的解析方式。
    - `standard_outfmt6`：恰好 12 个字段；查询、目标和期望值分别位于第 1、2 和 11 列，比特得分位于第 12 列，EFI 比对得分读取它而不是期望值。
    - `outfmt7_fields`：使用完整的 `# Query:` 值，并根据一致的 `# Fields:` 声明确定目标列和期望值列（EFI 比对得分则为比特得分列）。可识别目标的标题、ID 和 accession 等变体。
    - `custom_columns`：使用明确给出的、从 1 开始编号的 `QUERY_COLUMN`、`SUBJECT_COLUMN` 和 `EVALUE_COLUMN` 值；EFI 比对得分用 `BITSCORE_COLUMN` 取代 `EVALUE_COLUMN`。
*   `QUERY_COLUMN`、`SUBJECT_COLUMN`、`EVALUE_COLUMN`、`BITSCORE_COLUMN`：从 1 开始编号的列号，仅供 `custom_columns` 使用。`SCORE_METRIC` 决定读取最后两者中的哪一个，另一个被忽略。

导入的网络记录固定的矩阵来源标签 `Imported`。解析使用固定的批大小，即 `1,000,000` 行；这两个值都不是界面中的设置。

对于含有描述的 FASTA 序列标题，推荐使用带目标标题的 BLAST outfmt 7 格式，例如 `-outfmt "7 qseqid stitle evalue bitscore"`。查询序列（query sequence）标题取自 `# Query:`，目标序列标题取自 `stitle`。DIAMOND 的对应写法为 `--outfmt 6 qtitle stitle evalue bitscore`，导入时使用 `custom_columns` 的第 1、2、3 列（EFI 比对得分还要将 `BITSCORE_COLUMN` 设为 4）。同时报告比特得分，就可以用任一种 `SCORE_METRIC` 导入同一个搜索文件。只有当标准的 `qseqid`/`sseqid` 输出值与已清理的完整 FASTA 序列标题完全相同时才会被接受：BLAST+ 和 DIAMOND 在这两列中都只报告每个 FASTA 序列标题的第一个词，若某个无法匹配的序列标题恰好等于这样的第一个词，会被报告为截断。

#### 推荐的 DIAMOND 全对全搜索

```
diamond makedb --in sequences.fasta -d sequences
diamond blastp -d sequences -q sequences.fasta -o sequences_diamond.tsv --very-sensitive -k 0 --max-hsps 1 --evalue 1e-5 --header verbose
```

使用 `standard_outfmt6` 和同一个 `sequences.fasta` 导入 `sequences_diamond.tsv`。DIAMOND 默认的 `--outfmt 6` 已在第 12 列包含 EFI 比对得分所需的比特得分。

*   **必须使用 `-k 0`。** DIAMOND 默认每个查询最多报告 25 条目标序列（`--max-target-seqs 25`），因此不加 `-k 0` 导入的全对全网络会悄悄丢失每个查询前 25 个命中之外的所有边。`--top` 也会以同样的方式截断命中列表，不应使用。
*   **`--header verbose`** 会写入 `# DIAMOND v…` 和 `# Invocation: …` 注释行。导入程序会把它们记录为网络的搜索程序、版本和命令，将输出命名为 `<name>_[DIAMOND]_EValue.h5`（EFI 比对得分则为 `_EFIScore.h5`），并检查记录的 `--max-target-seqs`。没有这一头部时，DIAMOND 输出仍可导入，但来源未知，并使用 `[BLAST]` 名称。`--header simple` 会写入一行列名，这会被拒绝。
*   **序列标题**：请选择序列标题中不含空格的 FASTA 进行搜索（例如由 `Sanitize_Sequences.py` 写出的文件），使 `qseqid`/`sseqid` 包含完整的序列标题；或者使用上面的标题列。
*   **灵敏度与阈值**：`--very-sensitive` 或 `--ultra-sensitive` 能找到 SSN 所依赖的远缘同源序列；更快的模式会漏掉更多远缘序列对。`--max-hsps 1`（DIAMOND 的默认值）对每个序列对只保留一个比对。期望值阈值决定了之后的得分阈值所能用到的最弱的边。

两种程序生成的网络都保留 `model_name="BLAST"`，因此查看器会把它们作为 BLAST 网络加载，其得分是期望值还是 EFI 比对得分由 `score_kind` 决定。

### 输出

#### HDF5 比对网络
*   **格式**：HDF5（`.h5`），命名为 `<blast file name>_[BLAST]_EValue.h5`；若文件的注释头部声明了 DIAMOND，则为 `_[DIAMOND]_EValue.h5`。使用 EFI 比对得分时，名称改以 `_EFIScore.h5` 结尾。
*   **结构**：
    - `/i`：源序列节点索引。
    - `/j`：目标序列节点索引。
    - `/score`：每个无向序列对解析出的最佳 $-\log_{10}(E_{\text{value}})$ 得分，或最佳 EFI 比对得分。
    - `/headers`：按源顺序排列的已清理完整 FASTA 序列标题。
    - 属性包括 `model_name="BLAST"`、矩阵和布局元数据、解析出的列、源文件和清单的哈希值、解析计数、清理计数，以及可用的 outfmt-7 来源信息。EFI 比对得分网络还有 `score_kind="efi_alignment_score"` 和 `score_transform`；没有 `score_kind` 的网络存储的是期望值得分。
    - `search_program`、`search_version` 和 `search_invocation` 保存文件开头注释中的程序行（`# BLASTP 2.17.0+` 或 `# DIAMOND v2.1.23.`）和 DIAMOND 的 `# Invocation:` 命令，没有时为 `Unknown`。
    - `queries_observed`、`max_targets_per_query`、`queries_at_max_targets` 和 `import_warnings`（一个 JSON 列表）记录下文所述的搜索完整性检查。

解析器总是分别报告 FASTA 和 BLAST 序列标题的清理情况。任何无法匹配的序列标题或冲突都会导致失败；若解析或输出校验失败，已有的最终 HDF5 文件会被保留。

#### 搜索完整性检查

解析完成后，诊断信息会报告有多少条 FASTA 记录作为查询出现，以及任一查询报告的不同目标的最大数量。下面每项检查都会打印一条警告并存入 `import_warnings`，但不会中止导入：

*   `--header verbose` 记录的 DIAMOND 命令使用了 `--top`，或者使用的 `--max-target-seqs` 限制小于至少一个查询达到的序列数。
*   未记录命令时：在达到最大目标数的查询中，至少有三个、且至少占其 10% 的查询，作为命中被多于该数量的查询报告。期望值阈值附近的弱命中偶尔会缺少对应的双向命中，但许多查询在同一个共同数量上出现这种情况，正是每查询数量限制的特征：DIAMOND 的默认值 25、BLAST+ 的默认值 500，或者自行选定的 `-k`。成员恰好都达到最大数量的完整家族不会被标记。若同一查询的行不连续，则跳过此项检查。
*   一些 FASTA 记录从未作为查询出现，而其他查询却报告了自身命中，这说明搜索没有覆盖整个 FASTA，或提前停止了。非常短或低复杂度的序列也可能没有自身命中。

`emapssn_pipeline(action="inspect_file")` 会报告已导入网络中记录的警告；对于尚未导入的 DIAMOND 文件，则报告其搜索程序、版本和限制。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **FASTA 清单**：
     按源顺序读取每条 FASTA 记录，校验其结构，并清理完整的序列标题。清理后的序列标题是查看器中唯一的规范标识。

2. **严格的 BLAST 布局解析**：
     根据所选的明确布局确定各列。字段数不正确、outfmt-7 声明不一致、UTF-8 无效、期望值无效（EFI 比对得分则为比特得分无效）或序列标题未知的行都会失败，并给出实际的行号。在其他布局中，`# Fields:` 注释必须在所选的期望值列上声明期望值（EFI 比对得分则在比特得分列上声明比特得分），且一行列名会被拒绝。不使用任何启发式的期望值检测或首个词元别名。

3. **边解析与得分转换**：
     清理所选的完整 BLAST 序列标题，将其与清单精确匹配，并转换每个有限的非负期望值。零映射为封顶得分 300：
     $$\text{Score} = -\log_{10}(E_{\text{value}} + 10^{-300})$$
     使用 EFI 比对得分时，改为转换每个有限的非负比特得分，两条序列的长度取自 FASTA 清单，结果不做取整：
     $$\text{Score} = \text{bit score} \cdot \log_{10}(2) - \log_{10}(L_{\text{query}} \cdot L_{\text{subject}})$$
     该得分没有上限，不依赖序列集的大小，长序列之间的弱命中其得分为 0 或负数。EFI-EST 没有给出通用的阈值：请从较低的值开始，逐步提高相似性阈值，直到功能已知的序列彼此分开。清理后没有任何残基的序列没有长度，因此涉及它的命中会使导入停止。
     每个查询的不同目标（包括自身命中），以及报告每条序列的查询数，都会被计入搜索完整性检查。

4. **有界去重**：
     写出每段最多 1,000,000 行已解析数据的有序分段，在外部归并，并对每个规范的无向序列对只保留得分最高的比对。最终的序列对严格有序且唯一。

5. **经校验的原子化发布**：
     写入一个 `.partial` HDF5 文件，校验其结构、序列标题、得分的有限性、规范的序列对、排序、唯一性和来源信息，然后以原子方式替换最终输出。没有边的网络也是有效的。

</details>
