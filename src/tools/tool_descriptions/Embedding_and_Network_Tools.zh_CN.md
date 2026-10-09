<!-- Translation of Embedding_and_Network_Tools.md, sha256 7de9298c1d7b66da0ab1dedd15116c75243d01e5a8afa2b2e8ac588fb107fc93 -->
# 🧬 嵌入注入 (`Embedding_Injection.py`)

该脚本把新序列的嵌入注入到已有的 HDF5 嵌入数据库中。它扫描传入的 FASTA 序列列表，对已有的匹配项直接从数据库中取出预先计算的嵌入，只为新增的序列计算嵌入，以节省计算时间。

### 📥 输入

#### 目标嵌入数据库 `INPUT_EMBED`
*   **格式**：完整的元数据优先 HDF5 数据库（`.h5`），包含 `/headers`、`/sequences`、`/embeddings`、`model_name`、`saving_mode`、`num_sequences` 和 `generation_complete`。
*   **生成工具**：`Generate_Embeddings.py`（嵌入生成工具）。

#### 传入的序列集 `INPUT_FASTA`
*   **格式**：FASTA 文件（`.fasta`），包含原有序列以及新追加的目标序列。
*   **生成工具**：用户整理的更新后序列集。
*   **清理**：记录会在内存中自动清理，不做序列标题子串过滤或序列长度过滤。只有记录发生变化时才会打印清理结果。

### ⚙️ 参数

此脚本不需要额外的配置参数。

### 📤 输出

#### 更新后的 HDF5 嵌入数据库
*   **格式**：HDF5（`.h5`）。
*   **说明**：重新编制索引的元数据优先数据库，包含已清理的序列标题和序列，以及新 FASTA 中每条记录的嵌入。只有清理后的序列标题和序列都一致时，才会复用已有的嵌入。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **序列标题清点与差异解析**：
     读取并清理新的输入 FASTA 文件，然后收集所有目标序列标题：
     $$H_{\text{fasta}} = \{h_1, h_2, \dots, h_M\}$$
     
     打开已有的 HDF5 嵌入文件，读取预先计算的序列标题：
     $$H_{\text{exist}} = \{e_1, e_2, \dots, e_N\}$$
     
     用集合差确定需要计算嵌入的新序列子集：
     $$H_{\text{new}} = H_{\text{fasta}} \setminus H_{\text{exist}}$$
     每个已有的序列标题都必须保留，且清理后的序列完全相同；否则注入会在创建输出之前停止。

2. **模型识别与设置**：
     读取已有 HDF5 文件的元数据，识别所用的模型架构和精度（`float16` 或 `float32`）。它加载完全相同的模型（例如 ESM-C），以确保向量一致。

3. **增量嵌入计算**：
     将属于 **H<sub>new</sub>** 的序列片段送入语言模型，计算其残基级嵌入。

4. **同步合并与 HDF5 序列化**：
     按顺序遍历 $H_{\text{fasta}}$。若某个序列标题属于 $H_{\text{exist}}$，就直接从旧文件复制嵌入数据集；若属于 $H_{\text{new}}$，则写入新计算的嵌入张量：
     $$v_{\text{final}}(i) = \begin{cases} v_{\text{exist}}(i) & \text{if } h_i \in H_{\text{exist}} \\ v_{\text{new}}(i) & \text{otherwise} \end{cases}$$
     
     先写入并刷新已清理的 `/headers` 和 `/sequences`，再复制或生成每个嵌入，只有通过最终校验后才设置 `generation_complete=true`。

</details>

---

# 📤 嵌入提取 (`Embedding_Extraction.py`)

该脚本从主 HDF5 数据库中提取部分序列嵌入。提供目标序列标题列表（FASTA 或文本文件）后，它会生成一个更小、经过过滤的 HDF5 嵌入存档，无需运行任何模型计算。

### 📥 输入

#### 源嵌入数据库 `INPUT_EMBED`
*   **格式**：完整的元数据优先 HDF5 嵌入数据库（`.h5`）。
*   **生成工具**：`Generate_Embeddings.py`（嵌入生成工具）。

#### 目标白名单集 `INPUT_FASTA`
*   **格式**：目标白名单 FASTA 文件（`.fasta`），或包含所选序列标题的纯文本文件。
*   **生成工具**：用户定义的子集白名单。
*   **校验**：FASTA 记录的清理方式与生成嵌入时完全相同，其序列必须与源元数据一致。文本列表会清理序列标题，并从源数据库取得序列。

### ⚙️ 参数

此脚本不需要额外的配置参数。

### 📤 输出

#### 提取出的 HDF5 嵌入存档
*   **格式**：HDF5（`.h5`）。
*   **说明**：包含白名单中的嵌入，以及与之一一对应、已清理的 `/headers` 和 `/sequences` 元数据。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **收集目标列表**：
     解析目标序列白名单（来自 FASTA 文件或文本列表），汇总目标序列标题：
     $$H_{\text{target}} = \{t_1, t_2, \dots, t_K\}$$

2. **索引对齐与取交集**：
     遍历主 HDF5 文件的序列标题数据集，过滤掉不在 $H_{\text{target}}$ 中的序列数据集：
     $$H_{\text{extract}} = H_{\text{target}} \cap H_{\text{master}}$$

3. **经校验的数据集提取**：
     读取每个选中的残基级嵌入数组，校验其数据类型、形状、序列长度和共享的特征维度，然后原样写入新数据库，不做模型推理。

4. **元数据序列化**：
     在复制嵌入之前，先写入并刷新所选的序列标题和存储的序列。只有通过数据类型、形状、序列长度和特征维度校验后，输出才会标记为完成。

</details>

---

# 🧬 网络注入 (`Network_Injection.py`)

该脚本执行增量式相似性网络计算。向项目中加入新序列时，它直接从旧的网络缓存中复制所有已有的序列间比对分数，只比对新引入的序列对，以节省时间和计算资源。

### 📥 输入

#### 目标网络文件 `OLD_NETWORK`
*   **格式**：已有的 HDF5 网络数据库文件（`.h5`）。
*   **生成工具**：`Align_Similarity_Matrix.py`（嵌入比对工具）。

#### 更新后的嵌入数据库 `NEW_EMBEDDINGS`
*   **格式**：包含全部嵌入的目标 HDF5 嵌入数据库（`.h5`）。
*   **生成工具**：`Embedding_Injection.py`（嵌入注入工具）。

### ⚙️ 参数

| 参数 | 说明 |
| :--- | :--- |
| 空位罚分 | 直接从输入网络（`OLD_NETWORK`）自动继承。 |
| CPU 工作进程数 **`WORKERS`** | CPU 处理方案使用的工作进程数，也是调优加速器方案时的并发参数。该工具会在具有代表性的待计算序列对上对可用的 CPU/加速器方案做基准测试，必要时依次退回到其他成功的方案。 |
| 处理批大小 **`BATCH_SIZE`** | 每个写入块计算的序列比对数量，可降低内存占用并优化文件写入性能。 |
| 设备 **`DEVICE_SELECTION`** | 选择自动硬件基准测试，或指定一个具体设备来计算新的残基分数矩阵。 |
| 执行模式 **`EXECUTION_MODE`** | `auto` 在支持时比较标量方案与分块方案。`scalar` 将调优和正式计算限制为一次处理一个矩阵的方案。`tiled` 强制在 CUDA/ROCm 或 XPU 上使用内存受限的嵌入分块和填充后的微批次；若没有兼容的加速器，会提前失败。 |
| 主机缓存 **`HOST_CACHE_GB`** | 跨批次保留打包嵌入所用的最大内存（GiB）；`auto` 使用一个安全的内存预算，上限为 128 GiB，`0` 则停用缓存。 |
| 矩阵乘法精度 | 继承自 `OLD_NETWORK`。旧版网络为 IEEE FP32；TF32 网络需要 NVIDIA CUDA，以免复制的边与新计算的边混用精度。BF16 网络需要具备相应能力的加速器。计算新边之前，网络注入会打印低精度警告，并在最多 2,048 个具有代表性的序列对上报告相对于 FP32 的比对长度和原始分数统计；有限的数值差异不会导致拒绝 BF16。 |

### 📤 输出

#### 更新后的 HDF5 比对网络
*   **格式**：HDF5（`.h5`）。
*   **说明**：重新编制索引的比对网络，包含 `/headers`、`/seq_lens`、`/i`、`/j`、`/g_score`、`/g_len`、`/l_score` 和 `/l_len`，以及 `model_name`、`saving_mode`、`gap_penalties`、`embedding_checksum` 和 `matmul_precision` 属性。已有的空位罚分和运算精度均继承自 `OLD_NETWORK`。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **映射设置**：
     设旧网络的序列标题为 $H_{\text{old}} = \{h_1, \dots, h_N\}$，新嵌入的序列标题为 $H_{\text{new}} = \{h'_1, \dots, h'_M\}$。只有当两个端点的序列标题都仍在 $H_{\text{new}}$ 中时，缓存的边才可复用。
     脚本创建一个索引映射字典，把旧索引解析为新索引：
     $$\text{Map}_{\text{old} \to \text{new}}(i) = j \quad \text{such that} \quad h_i = h'_j$$

2. **边的分类**：
     对新网络中所有两两组合 (u, v)（其中 0 ≤ u < v < M）：
     - **情形 1（缓存的序列对）**：若旧网络包含完全相同的序列标题对，则复制其全局/局部分数和长度。
     - **情形 2（新序列对，旧网络完整）**：若任一端点是新的，该序列对将安排进行动态规划比对。
     - **情形 3（新序列对，旧网络稀疏）**：将平均池化嵌入的余弦相似度与可复用旧边中的最低余弦相似度比较。只有达到这一继承阈值的新序列对才会被比对，从而保持稀疏网络的策略。

3. **增量比对**：
     对可用的 CPU/加速器处理方案进行基准测试，并把安排好的新序列对交给成功方案中最优的一个处理。CUDA/ROCm 和 XPU 共享持久的归一化嵌入分块、按长度分桶且填充有界的目标序列，并在执行前预检设备内存。每个序列对都会：
     - 从 HDF5 数据库中取出残基嵌入。
     - 计算归一化分数矩阵：
       $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$
     - 求解全局与局部动态规划比对：
       $$\text{Global Pass} \to \text{NW}(\text{Score}, \text{gap}_g)$$
       $$\text{Local Pass} \to \text{SW}(\text{Score} - 2.0, \text{gap}_l)$$

4. **整合**：
     将复制的分数与新计算的分数合并，并把更新后、重新编制索引的网络数据集（`i`、`j`、`g_score`、`g_len`、`l_score`、`l_len`）写入新的输出文件。

</details>

---

# 📤 网络提取 (`Network_Extraction.py`)

该脚本根据白名单 FASTA 文件从主 HDF5 网络中提取子网络。它只保留两端序列节点都在白名单中的比对连接，并为剩余的边重新编制索引，生成一个干净、自成一体的过滤后子网络。

### 📥 输入

#### 源网络文件 `INPUT_NET`
*   **格式**：主 HDF5 网络数据库文件（`.h5`）。
*   **生成工具**：`Align_Similarity_Matrix.py`（嵌入比对工具）或 `Align_Substitution_Matrix.py` / `Parse_BLAST_Output.py`。

#### 目标白名单集 `INPUT_FASTA`
*   **格式**：白名单序列 FASTA 文件（`.fasta`），包含要保留的节点。
*   **生成工具**：用户定义的子集白名单。

### ⚙️ 参数

此脚本不需要额外的配置参数。

### 📤 输出

#### 提取出的 HDF5 子网络存档
*   **格式**：HDF5（`.h5`）。
*   **嵌入网络结构**：复制源属性，并写入重新编制索引的 `/headers`、`/seq_lens`、`/i`、`/j`、`/g_score`、`/g_len`、`/l_score` 和 `/l_len`。
*   **BLAST/E 值网络结构**：复制源属性，并写入重新编制索引的 `/headers`、`/i`、`/j` 和 `/score`。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **白名单索引**：
     从 FASTA 文件加载目标序列标题白名单 $H_{\text{whitelist}}$。把每个白名单序列标题映射到它在主网络文件中对应的索引：
     $$\text{Map}_{\text{header} \to \text{master\_idx}}(h) = x$$

     然后为子集建立新的索引映射：
     $$\text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(x) = y$$

2. **边过滤**：
     扫描主网络的边 $(i_k, j_k)$。当且仅当两个索引都在白名单中时，才保留这条边：
     $$i_k \in \text{Map}_{\text{master\_idx} \to \text{subset\_idx}} \quad \text{and} \quad j_k \in \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}$$

3. **重新编制索引**：
     对所有保留的边，脚本重新编制源节点和目标节点的索引，使其适应更小的子集矩阵坐标空间：
     $$i'_k = \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(i_k)$$
     $$j'_k = \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(j_k)$$

4. **组装输出**：
     保留检测到的源数据结构：比对网络使用嵌入分数/长度数据集，BLAST/E 值网络使用单个 `score` 数据集。输出文件名取自白名单 FASTA 的基本名称和源网络的 `model_name` 元数据。

</details>
