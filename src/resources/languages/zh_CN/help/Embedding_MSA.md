<!-- Translation of Embedding_MSA.md, sha256 b5f7c5322580423fb82494ac81b6606ac906cb44afc5fbc7e8f0ae154ca756e9 -->
# 嵌入多序列比对 (`Embedding_MSA.py`)

该脚本使用蛋白质语言模型嵌入生成渐进式多序列比对（MSA）。它根据网络分数构建完整的引导树距离矩阵（网络稀疏时用回归插补缺失的值），然后使用考虑空位的动态规划逐步比对各个序列簇。

### 输入

#### 序列集 `INPUT_FASTA`
*   **格式**：标准 FASTA 序列数据库文件（`.fasta`）。
*   **生成工具**：`Sanitize_Sequences.py`（序列清理工具）或用户提供的原始 FASTA。
*   **可选行为**：仅在启用 **使用序列过滤**（`USE_SEQUENCE_FILTER`）时使用。禁用过滤时，界面保存空字符串，比对使用嵌入数据库中存储的序列。

#### 嵌入数据库 `INPUT_EMBED`
*   **格式**：预先计算的 HDF5 嵌入数据库（`.h5`）。
*   **生成工具**：`Generate_Embeddings.py`（嵌入生成工具）。

#### 网络数据库 `INPUT_NETWORK`
*   **格式**：双序列 HDF5 比对网络（`.h5`）。
*   **生成工具**：`Align_Similarity_Matrix.py`（嵌入比对工具）或 `Align_Substitution_Matrix.py` / `Parse_BLAST_Output.py`。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 使用序列过滤 **`USE_SEQUENCE_FILTER`** | 启用时，比对仅限于网络、嵌入数据库和所选 FASTA 文件共有的序列。禁用时无需 FASTA，比对网络与嵌入数据库的交集，并使用嵌入清单中存储的序列。 |
| 显示回归图 **`SHOW_REGRESSION_PLOT`** | 使用稀疏网络时，切换是否显示诊断用的保序回归图。该图展示平均嵌入余弦距离与实际比对分数之间的拟合情况。 |
| 比对评分类型 **`ALIGNMENT_SCORE`** | 选择使用网络中的“global”（全局）还是“local”（局部）连接分数构建引导树。 |
| 分数归一化模式 **`NORMALIZATION_MODE`** | 应用于比对分数的归一化方法（例如 alignment_length、shorter_sequence、longer_sequence、average_sequence）。局部分数不能使用 `alignment_length`。BLAST 网络下此项禁用。 |
| 建树方法 **`TREE_METHOD`** | “UPGMA（快速）”（`UPGMA (Fast)`）使用平均连接层次聚类；“邻接法（慢速）”（`Neighbor-joining (Slow)`）根据同一个完整距离矩阵构建邻接树。所选方法同时用于确定性树和噪声扰动树。 |
| 噪声扰动共识引导树 **`BOOTSTRAP_TREE`** | 启用时，构建若干随机扰动的引导树重复并取平均。这是一种敏感性集成，而不是经典的自展（bootstrap）支持度。为保持向后兼容，沿用原有的设置键名。 |
| 扰动树重复数 **`NUM_TREES`** | 用于构建共识引导树的噪声扰动重复树数量。 |
| 在最终共识中纳入插补的序列对 **`INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS`** | 对于不完整的网络，关闭时只对原本观测到的序列对平均共表型距离，缺失的序列对保留基线回归插补值；开启时，每个序列对都替换为其重复平均后的共表型距离。两种模式下，插补的序列对都会参与每棵重复树的构建。完整网络自动使用完全共识。 |
| 归一化加性噪声尺度 **`NOISE_SCALE`** | 以有效距离范围的比例表示的高斯标准差。例如，`0.02` 施加的加性标准差等于最大引导树距离的 2%。每个观测距离和回归插补距离都会被扰动。 |
| 空位开放罚分 **`GAP_OPEN`** | 空位开放罚分值。数值越负，对开启新空位的惩罚越重。 |
| 空位延伸罚分 **`GAP_EXTEND`** | 空位延伸罚分值。数值越负，对延伸已有空位的惩罚越重。 |
| CPU 工作进程数 **`WORKERS`** | 分配给并行计算噪声扰动引导树的 CPU 进程数。使用 UPGMA 时，每个工作进程每对序列约占用 16 字节（44,000 条序列约 16 GB），因此峰值内存随此设置增长。邻接法的工作进程还会以线程形式共享除两个以外的全部逻辑 CPU，因此单个工作进程就已使用多个核心。 |
| 计算设备 **`DEVICE_SELECTION`** | 为按顺序构建谱（profile）分数矩阵选择 `auto` 或某个可用的 CPU、CUDA、XPU 或 MPS 设备。自动模式会对分数矩阵成本位于第 25、50 和 90 百分位附近的三次真实叶到叶引导树合并进行基准测试，并为整个渐进合并选定一个设备。引导树计算和动态规划回溯始终在 CPU 上进行。 |
| 临时工作目录 **`SAFE_TEMP_DIR`** | 用于缓存中间文件和内存映射矩阵的临时目录。未设置时，缓存建在本次运行的比对文件夹（`MSA_DIR`）中，并在引导树构建完成后删除。 |

### 输出

#### 多序列比对 FASTA 文件
*   **格式**：已比对的 FASTA（`.fasta`）。
*   **结构**：标准 FASTA，包含用空位（`-`）填充至最终比对长度的已比对序列字符串。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **交集过滤**：
     始终取嵌入 HDF5 文件与网络 HDF5 文件中序列标题的交集。若启用 `USE_SEQUENCE_FILTER`，所选 FASTA 文件作为第三个交集来源；否则，序列字符串来自嵌入清单。

2. **网络分数归一化**：
     根据 `NORMALIZATION_MODE` 对网络比对分数进行归一化，以消除序列长度差异的影响。
     * **alignment_length**：$$S_{\text{norm}} = \frac{S_{\text{raw}}}{L_{\text{align}}}$$
     * **shorter_sequence**：$$S_{\text{norm}} = \frac{S_{\text{raw}}}{\min(L_i, L_j)}$$
     * **longer_sequence**：$$S_{\text{norm}} = \frac{S_{\text{raw}}}{\max(L_i, L_j)}$$
     * **average_sequence**：$$S_{\text{norm}} = \frac{S_{\text{raw}}}{\text{mean}(L_i, L_j)}$$

3. **距离矩阵构建**：
     将分数反转以表示距离：
     $$D(i, j) = \max(S_{\text{norm}}) - S_{\text{norm}}(i, j)$$

4. **稀疏网络的保序回归**：
     若网络稀疏，距离矩阵就不完整。此时脚本会：
     - 对残基嵌入做最大池化或平均池化，为所有序列生成全局表示向量 **u<sub>i</sub>**。
     - 计算所有序列两两之间经长度校正的余弦相似度：
       $$\text{Adj}(i, j) = \text{Sim}_{\text{cos}}(i, j) \times \left(\frac{\min(L_i, L_j)}{\max(L_i, L_j)}\right)^P$$
     - 拟合一个**保序回归**函数 $f$，把已知的 $\text{Adj}(i, j)$ 值映射到归一化网络分数 $S_{\text{norm}}(i, j)$。
     - 预测缺失的网络分数：
       $$S_{\text{predicted}}(i, j) = f(\text{Adj}(i, j))$$
       
       这样即可保证距离矩阵完全填满。

5. **噪声扰动共识引导树的构建**：
     使用所选的 `TREE_METHOD`（平均连接 UPGMA 或邻接法）构建引导树。启用 `BOOTSTRAP_TREE` 时，程序使用归一化的加性高斯距离扰动生成 `NUM_TREES` 棵重复树：

     $$D_{\max} = \max(S_{\text{norm}}) + 0.1$$

     $$\sigma_{\text{absolute}} = \text{NOISE\_SCALE} \times D_{\max}$$

     $$D_{\text{perturbed}}(i,j) =
     \operatorname{clip}\left(
     D(i,j) + \mathcal{N}(0,\sigma_{\text{absolute}}^2),
     0,D_{\max}
     \right)$$

     有效范围为 $[0,D_{\max}]$。零表示可能的最近关系。$D_{\max}$ 是最弱或无连接序列对所用的最大距离。截断可防止出现无效的负距离，也可防止观测到的边超过程序的最大距离哨兵值。

     对于稀疏网络，完整的基线矩阵由观测到的网络距离与缺失序列对的回归插补距离组合而成。这一完整基线由每个重复工作进程共享，相同的加性扰动施加于每个距离。因此，插补关系会参与每棵重复树，而不会被替换为 $D_{\max}$。

     最终的平均模式在网络、嵌入与 FASTA 取交集之后确定。完整的诱导网络自动使用全部序列对的共表型共识。对于不完整的诱导网络，**在最终共识中纳入插补的序列对** 只控制最终矩阵：关闭时，观测到的序列对替换为其重复平均共表型距离，缺失的序列对保留基线回归插补值；开启时，每个序列对都替换为其重复平均共表型距离。关闭此开关不会停用插补，也不会把插补的序列对排除在重复树构建之外。停用噪声扰动树时忽略此开关，直接用完整的“观测 + 插补”基线构建确定性树。

     这些重复用于衡量对随机距离扰动的敏感性；它们不是经典的自展重抽样，也不提供经典的自展支持值。

6. **渐进式谱比对**：
     从叶节点到根遍历引导树。在每个内部节点，按以下步骤合并两个子簇（可以是单条序列或比对谱）：
     - 确定手动指定的计算设备；在自动模式下，于引导树构建完成后对具有代表性的真实叶节点合并进行基准测试。基准测试包含分数矩阵在主机与设备之间的传输，但不含 HDF5 加载和 CPU 回溯。
     - 每个叶节点的残基嵌入在进入谱之前先归一化为单位向量。
     - 对簇中每条序列的这些单位向量取平均，空位视为零。因此所得向量的模同时记录了该列的非空位占有率和方向一致性。
     - 根据谱向量的方向计算相互余弦相似度，再用两列的模对每个单元格加权，使支持较弱或不一致的列对比对证据的贡献更小。
     - 使用 `GAP_OPEN` 和 `GAP_EXTEND` 罚分运行动态规划，找出最优路径。
     - 合并比对，并输出最终的 FASTA 多序列比对（MSA）。

</details>

---

# 稀疏 MSA 转换器 (`Sparse_MSA_Converter.py`)

该脚本将多序列比对（MSA）压缩为紧凑的 HDF5 文件。它会验证并清理已比对的 FASTA，将残基字符串转换为 SciPy 压缩稀疏行（CSR）矩阵，写入查找元数据，然后把转换成功的源 FASTA 移入 `Full_Alignments` 子目录。

### 输入

#### MSA 比对文件 `INPUT_FASTA`
*   **格式**：已比对的 FASTA（`.fasta`）。
*   **生成工具**：`Embedding_MSA.py`（嵌入多序列比对工具）或其他外部 MSA 工具（例如 Clustal、MUSCLE）。
*   **说明**：预先计算好的标准多序列比对文件。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 转换所有比对 **`CONVERT_ALL`** | 切换是否转换输入目录中的所有 FASTA 多序列比对。禁用时只转换所选的比对文件。 |

### 输出

#### 压缩稀疏 MSA HDF5 文件
*   **格式**：HDF5（`.h5`），保存在所选比对旁，命名为 `<input_basename>_sparse.h5`。
*   **结构**：
    - `/matrix/data`：非空位条目的整数残基编码一维数组。
    - `/matrix/indices`：CSR 列索引数组。
    - `/matrix/indptr`：映射序列的 CSR 行指针。
    - `/matrix` 属性 `shape`：矩阵的行/列维度。
    - `/headers`：序列标题数组。
    - `/header_map`：完整标题及首个词元标题到行索引的 JSON 映射。
    - `/aa_map` 和 `/int_to_aa`：JSON 残基编码查找表。
    - 根属性 `shape`：比对的总体维度。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **氨基酸映射**：
     将氨基酸字符映射为整数索引（例如 A → 1、R → 2……）。空位字符 `"-"` 不保存（隐式为零）。

2. **压缩稀疏行（CSR）编译**：
     将已比对的序列矩阵转换为 SciPy 压缩稀疏行（CSR）格式：
     $$\text{Matrix} \to (\text{data}, \text{indices}, \text{indptr})$$
     
     使用 `uint8` 数据类型最多可将文件大小压缩 95%。

3. **原子化 HDF5 序列化与源文件归档**：
     将 CSR 数组、标题、映射和形状元数据写入临时 HDF5 文件，并以原子方式发布为 `<input>_sparse.h5`。只有发布成功后，原始 FASTA 才会被移至 `<MSA_DIR>/Full_Alignments/`。启用 `CONVERT_ALL` 时，`MSA_DIR` 中每个顶层 `*.fasta` 文件都按此方式处理。

</details>
