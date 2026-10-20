<!-- Translation of Others.md, sha256 a20606a77bb7e07d5b396066e31e7f2115b69db61e8afd2c1d78513c6fac6712 -->
# 双序列 embedding 比对 (`Embedding_PWA.py`)

该脚本使用两条序列的残基级语言模型 embedding 对它们进行比对（alignment）。它计算动态规划（dynamic programming）比对字符串（Needleman-Wunsch 或 Smith-Waterman），并把用户指定的参考序列（reference sequence）残基（residue）位置直接映射到目标序列（target sequence）的位置上，便于比较活性位点和其他特征。

### 输入

#### Embedding 数据库 `INPUT_EMBED`
*   **格式**：元数据优先的 HDF5 embedding 数据库（`.h5`），包含已清理的序列标题（FASTA header）、序列、模型元数据和残基级张量。只要有一条序列按序列标题选取，就需要它。
*   **创建者**：`Generate_Embeddings.py`（embedding 生成工具）。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 参考序列标题 **`REF_HEADER`** | 要在 embedding 数据库中查找的参考序列标题。输入的文本会按“生成 embedding”的同一规范进行序列标题清理（sanitize）。 |
| 目标序列标题 **`TAR_HEADER`** | 要在 embedding 数据库中查找的目标序列标题。输入的文本会按“生成 embedding”的同一规范进行序列标题清理。 |
| 手动参考序列开关 **`MANUAL_REF_SEQ`** | 启用可选的手动参考序列输入框。开启时，即使清理后的序列标题与数据库中存储的序列标题相同，也使用手动输入的序列。默认关闭。 |
| 手动参考序列 **`REF_SEQUENCE`** | 仅在 `MANUAL_REF_SEQ` 开启时使用，并在生成 embedding 之前按规范方式清理；否则按序列标题加载参考序列。 |
| 手动目标序列开关 **`MANUAL_TAR_SEQ`** | 启用可选的手动目标序列输入框。开启时，即使清理后的序列标题与数据库中存储的序列标题相同，也使用手动输入的序列。默认关闭。 |
| 手动目标序列 **`TAR_SEQUENCE`** | 仅在 `MANUAL_TAR_SEQ` 开启时使用，并在生成 embedding 之前按规范方式清理；否则按序列标题加载目标序列。 |
| 高亮映射位置 **`HIGHLIGHT_POSITIONS`** | 以逗号分隔的参考序列残基位置列表（从 1 开始编号），这些位置会被映射到目标序列的比对中并高亮显示。 |
| Embedding 模型 **`EMBEDDING_MODEL`** | 两条序列都为手动输入时使用的模型。可选模型从 `src/resources/pLM_models` 中发现；只要有一条序列来自 embedding 集，就使用该 embedding 集的模型。 |
| 比对模式 **`ALIGNMENT_MODE`** | 要运行的比对模式（“全局（Needleman-Wunsch）”或“局部（Smith-Waterman）”）。 |
| 局部空位罚分 **`LOCAL_GAP_P`** | 局部比对（local alignment）的空位罚分（gap penalty）值。 |
| 全局空位罚分 **`GLOBAL_GAP_P`** | 全局比对（global alignment）的空位罚分值。 |
| 生成报告 **`GENERATE_REPORT`** | 切换是否编制并保存一份完整的 HTML 比对报告，展示残基高亮和得分。 |

### 输出

#### 比对结果和可选的 HTML 报告
*   **控制台输出**：总是打印比对模式和得分、输入长度、比对长度、一致性百分比（percent identity）、已比对的残基字符串、匹配标记，以及所请求的参考序列到目标序列的位置映射。
*   **一致性百分比**：以 `identical/alignment length (percent)` 的形式打印，并附上定义：相同的标准氨基酸数除以比对长度；局部比对的比对长度包括内部空位（gap），全局比对则为包括末端空位在内的全长。`|` 标记每个计入的残基对，`.` 标记其他已比对的残基对，空白表示空位；X、B、Z、J、U 和 O 从不计为相同。
*   **可选文件**：启用 **生成报告**（`GENERATE_REPORT`）时，会在设置的报告目录中写入一个带时间戳的 `PWA_Report_*.html` 文档，内容为相同的比对和高亮映射。关闭此开关时不写入文件。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **Embedding 的提取或生成**：
     直接从 HDF5 清单加载存储的序列文本和张量。输入的序列标题和手动序列会先按规范方式清理。如果只有一条序列是手动输入的，它的 embedding 使用数据库的模型；如果两条都是手动输入的，则使用明确选定的 embedding 模型。

2. **Z-score 矩阵构建**：
     计算归一化（normalization）的残基级得分矩阵：
     $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$

3. **回溯比对**：
     * **全局（Needleman-Wunsch）**：使用 `GLOBAL_GAP_P` 计算递推矩阵。
     * **局部（Smith-Waterman）**：将得分减去 2.0，并使用 `LOCAL_GAP_P` 计算递推矩阵：
       $$\text{Score}_{\text{local}}(a, b) = \text{Score}(a, b) - 2.0$$
     
     回溯沿着每个单元格记录的移动进行，因此显示的路径即为最优路径。得分相同时依次优先匹配、删除、插入，与 `Embedding_SSEARCH.py` 的计算内核顺序相同。

4. **一致性百分比**：
     统计回溯路径上相同的标准氨基酸数 $N_{\text{id}}$，并除以路径长度 $L_{\text{aln}}$：
     $$\text{Identity} = 100 \times \frac{N_{\text{id}}}{L_{\text{aln}}}$$

5. **残基位置映射**：
     对每个（从 1 开始编号的）参考序列高亮位置 $p_{\text{ref}}$，追踪其比对后的索引：
     $$p_{\text{ref}} \to p_{\text{aligned}} \to p_{\text{tar}}$$
     
     这样即可把参考序列中的催化残基或其他特征直接映射到目标序列上。

</details>

---

# Embedding 数据库搜索 (`Embedding_SSEARCH.py`)

该脚本使用残基级语言模型 embedding，用一条查询序列（query sequence）搜索整个数据库。它对数据库中的所有序列并行运行双序列比对（pairwise alignment），按归一化的局部或全局相似性得分对匹配的蛋白质排序，其工作方式类似于 FASTA 软件包中的 ssearch。

### 输入

#### Embedding 数据库 `INPUT_EMBED`
*   **格式**：完整的元数据优先 HDF5 数据库（`.h5`），包含已清理的序列标题、序列和 embedding。
*   **创建者**：`Generate_Embeddings.py` 或其他正在使用的 embedding 写入程序。SSEARCH 不会自动生成缺失的数据库。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 查询序列标题 ID **`QUERY_HEADER`** | 存储在 embedding 数据库中的序列标题。查找前会先清理。 |
| 手动查询序列开关 **`MANUAL_QUERY_SEQ`** | 启用可选的手动查询序列输入框。开启时，查询序列与数据库中序列标题相同的记录保持独立，该记录仍可作为搜索命中（hit）出现。默认关闭。 |
| 手动查询序列 **`QUERY_SEQUENCE`** | 仅在 `MANUAL_QUERY_SEQ` 开启时使用，随后在内存中清理，并用数据库模型的 pLM 插件生成 embedding。 |
| 输出电子表格前缀 **`OUTPUT_NAME`** | 导出的搜索结果电子表格以及可选 FASTA 文件的名称前缀。 |
| 最大数据库命中数 **`TOP_K`** | 输出报告中包含的得分最高的数据库命中的最大数量。 |
| 归一化得分阈值 **`NORM_THRESHOLD`** | 用于排除归一化相似性低于该阈值的命中。设为 'None' 可停用。 |
| 比对模式 **`ALIGNMENT_MODE`** | 搜索所用的比对模式（“全局（Needleman-Wunsch）”或“局部（Smith-Waterman）”）。 |
| 局部空位罚分 **`LOCAL_GAP_P`** | 局部比对的空位罚分值。 |
| 全局空位罚分 **`GLOBAL_GAP_P`** | 全局比对的空位罚分值。 |
| 得分归一化模式 **`NORM_MODE`** | 得分归一化方法（例如比对长度、较短序列长度、较长序列长度、平均序列长度）。局部比对不能使用“比对长度”（`Alignment Length`）。 |
| CPU 工作线程数 **`WORKERS`** | 分配给并行比对计算的 CPU 线程数。 |
| 设备 **`DEVICE_SELECTION`** | 选择自动硬件基准测试，或指定一个具体的 CPU/加速器设备。 |
| 加速器精度 **`ACCELERATOR_PRECISION`** | `automatic_32bit` 在小规模搜索中使用 IEEE FP32，只有目标数达到 4,096 时才会考虑经过验证的 TF32。明确选择 `bf16` 时，矩阵乘法操作数使用 BF16，归一化和后处理使用 FP32，并需要具备相应能力的 CUDA/ROCm、XPU 或 MPS 加速器。此时会打印低精度警告，并针对每种设备/执行变体，在一个单独的、按长度分层的最多 2,048 个目标的样本上报告相对于 FP32 的所选模式长度和原始得分统计。数值差异只要有限，就仅供参考，绝不会因此拒绝 BF16。旧的别名 `auto` 仍然有效。强制使用 TF32 需要 NVIDIA CUDA。 |
| 导出最佳命中 FASTA **`GENERATE_FASTA`** | 切换是否导出一个 FASTA 文件，其中包含前 *K* 个数据库命中的序列。 |

### 输出

#### Embedding 搜索结果
*   **文本报告**：`Report_<name>.txt`，包含参数、一致性百分比的定义以及完整的命中排名表；控制台最多显示前 100 个命中。
*   **Excel 工作簿**：`Report_<name>.xlsx`，包含一个兼容元数据查看器的 `Search Results` 工作表和一个 `Search Parameters` 工作表。第一个工作表遵循 `docs/metadata_template.xlsx`：第 1 行为属性名称，第 2 行为数据类型，A 列为与节点（node）严格匹配所用的精确序列标题。
*   **可选 FASTA**：启用 `GENERATE_FASTA` 时，`Hits_<name>.fasta` 包含查询序列，其后是按排名排列的命中序列。
*   **元数据列**：`Node ID`、`Rank`、`Norm_Score`、`Raw_Score`、`Sequence_Length`、`Alignment_Length` 和 `Percent_Identity`。
*   **一致性百分比**：报告的比对路径上相同的标准氨基酸数，表示为 `Alignment_Length` 的百分比（0–100）。局部比对计入内部空位；全局比对计入包括末端空位在内的全长。X、B、Z、J、U 和 O 从不计为相同。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **查询设置**：
     从 `/sequences` 读取存储的序列。手动开关明确选择来源：关闭时复用存储的序列标题和 embedding；开启时需要一条已清理的手动序列，并通过 `model_name` 记录的模型适配器为其生成 embedding。手动输入的序列标题即使与数据库记录重名，也不会替换或屏蔽该记录。

2. **数据库比对队列**：
     遍历 HDF5 文件中的所有数据库序列 $j$。自动选择设备时，先在一部分目标序列上做一次简短的基准测试，以决定如何运行：
     - 先在少量目标上串行运行 CPU，估算整个搜索的耗时，并据此设定调优预算：该估算的 5%，最多 5 秒。估算不超过 1 秒的搜索直接在 CPU 上运行，不再调优。
     - 随后依次对每个加速器测试标量方案、预期最快的方案、CPU 工作进程池，并在预算允许时测试更多通道（lane）数；在 NVIDIA GPU 上还会测试 TF32。在 CUDA 上，预期最快的方案是分块（tile）方案：查询序列只归一化并上传一次，目标序列只读取一次，按长度分组，并以受显存限制的批次（batch）计算得分；读取线程会在设备为当前目标序列打分的同时预取后续的目标序列，每个批次由一个 CPU 任务完成比对。无论预算如何，该方案的首次测量都会进行，除非搜索太短、试运行的耗时无法得到回报。
     - 每个方案的总耗时由其样本的速度推算到整个搜索，再加上启动耗时。分块方案在同一个分块（tile）内的目标上计时，并先做一次不计时的预热，以免把每个分块的固定开销摊到每几条序列上。最快的方案负责运行搜索，并复用其样本已算出的结果；若有更简单的方案与之相差在 5% 以内（依次为串行 CPU、通道数更少、非分块），则优先选它；该方案失败时改用下一个方案。

3. **多线程动态规划**：
     把比对分配给多进程工作进程。每个工作进程：
     - 计算残基级归一化相似性矩阵：
       $$\text{Score}_j(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$
     - 求解比对得分：
       * **全局（Needleman-Wunsch）**：$$S_{\text{raw}} = \text{NW}(\text{Score}_j, \text{gap}_g)$$
       * **局部（Smith-Waterman）**：$$S_{\text{raw}} = \text{SW}(\text{Score}_j - 2.0, \text{gap}_l)$$

4. **得分归一化**：
     根据 `NORM_MODE` 应用长度归一化因子：
     $$S_{\text{norm}} = \frac{S_{\text{raw}}}{\text{Normalization\_Factor}(L_q, L_j)}$$

5. **一致性百分比**：
     同一次动态规划会统计所选路径上相同的标准氨基酸数 $N_{\text{id}}$；路径长度 $L_{\text{aln}}$ 在 **local** 比对中包括内部空位，在 **global** 比对中包括每一列（含末端空位）：
     $$\text{Identity} = 100 \times \frac{N_{\text{id}}}{L_{\text{aln}}}$$
     由于该路径最大化的是 embedding 相似性而非残基匹配，同一对序列的一致性可能与替换矩阵（substitution matrix）比对得到的不同。

6. **排序与过滤**：
     汇总结果，按 `NORM_THRESHOLD` 过滤，按 $S_{\text{norm}}$ 降序排序，并保留前 $K$ 个命中。

</details>

---

# 基准测试

**运行基准测试** 在一组固定的公开蛋白质序列上为 EMAP-SSN 的耗时计算计时，使不同计算机和不同版本的报告可以相互比较。它在开始前先询问，然后在单独的控制台窗口中运行 `src/resources/benchmark/Run_Benchmark.py`，该窗口显示进度，并在结束时显示报告。

### 输入

#### 内置序列集
*   **序列**：860 条经审阅的 UniProtKB/Swiss-Prot 条目，属于金属 β-内酰胺酶超家族（InterPro IPR001279，版本 2026_03），另有 92 条留作注入（injection）阶段使用。`src/resources/benchmark/README.md` 说明了它们的来源和制作方法。
*   **模型**：ESM-2 8M（`esm2_t6_8m`）。首次运行会把它下载到 Hugging Face 缓存中，约 30 MB；下载不计时。

### 阶段

| 阶段 | 计时内容 |
| :--- | :--- |
| 1. 清理序列 | 对主序列集运行 `Sanitize_Sequences.py`。 |
| 2. Embedding | 使用 ESM-2 8M 运行 `Generate_Embeddings.py`。 |
| 3. 全对全 embedding 比对 | `Align_Similarity_Matrix.py`：369,370 个序列对。 |
| 4. SSN 布局 | 对得分最高的 5% 的边（edge）进行力导向布局。 |
| 5. UMAP 布局 | 用 UMAP 对同一网络进行布局。 |
| 6. 聚类 | Leiden、MCL 和 Jaccard 过滤，与查看器的 `cluster` 命令运行方式相同。 |
| 7. Embedding 数据库搜索 | `Embedding_SSEARCH.py`：用一条查询序列检索主序列集。 |
| 8. 注入新序列 | 使用 92 条留出的序列运行 `Embedding_Injection.py` 和 `Network_Injection.py`。 |
| 9. Embedding MSA | 对 100 条乙二醛酶 II 序列运行 `Embedding_MSA.py`。 |
| 10. BLAST 全对全比对 | `Align_Substitution_Matrix.py`，使用 NCBI BLAST+。 |

所有设备设置都保持为 **自动**，因此每个工具都会进行自己的硬件试验，报告会记录每个工具的选择。本机无法运行的阶段（例如未安装 NCBI BLAST+ 时的 BLAST）会被跳过并注明原因，依赖它的阶段也会被跳过。

### 输出

#### 基准测试报告
*   **文本报告**：基准测试文件夹中的 `Benchmark_Report_<date>_<time>.txt`，可用 **报告文件夹** 打开。报告使用窗口的语言，记录每个阶段的时间、吞吐量、CPU 时间、内存峰值和设备；每一项自动决策；硬件和软件；以及影响这些数值的条件。
*   **数据**：报告旁边有一份包含相同结果的英文 `.json`。
*   **保留**：报告从不删除。同一秒内开始的第二次运行会在文件名后加上 `_2`。

<details markdown="1">
<summary><b>运行方式</b></summary>

1. **独立的文件**：
     一次运行会把它生成的所有文件放在基准测试文件夹的 `temp/` 中，并用一个锁防止同时运行第二个基准测试；它在开始和结束时都会清空 `temp/`。它从不读取或修改已保存的工具设置，也不读取或修改本窗口中设置的目录。

2. **每个阶段一个进程**：
     每个阶段在单独的进程中运行，基准测试从外部测量它：实际用时、CPU 时间，以及该阶段启动的每个进程的内存峰值。

3. **时间和空间**：
     在一台配备 RTX 5070 Ti 和 Core Ultra 7 265KF 的计算机上，一次运行约需 6 分钟；没有 GPU 时用时更长。运行需要约 1 GB 的可用磁盘空间。

4. **停止**：
     在控制台中按 Ctrl+C 会停止运行，并仍为已运行的阶段写出报告。关闭控制台会直接停止运行而不写报告；下一次运行会清理它留下的文件。

5. **比较**：
     只有基准测试规程和序列集都相同时，报告才可比较。同时运行的其他程序、使用电池供电的笔记本电脑，或仍在编译计算内核的首次运行，都会使运行变慢，报告会注明这些情况。

</details>
