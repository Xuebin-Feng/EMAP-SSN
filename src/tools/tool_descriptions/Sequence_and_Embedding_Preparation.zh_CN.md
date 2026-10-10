<!-- Translation of Sequence_and_Embedding_Preparation.md, sha256 653906db32447349627211b72f6588157a732f11b40d347406c2dcfc1a65755e -->
# 清理序列 (`Sanitize_Sequences.py`)

该脚本清理原始 FASTA 序列数据库，为语言模型生成嵌入做好准备。它会过滤掉长度超出目标范围的序列，排除序列标题中含有特定关键词（例如片段或不完整序列）的序列，替换不安全的非标准字符，并报告序列长度分布。

### 输入

#### 原始序列 FASTA 文件 `INPUT_FASTA`
*   **格式**：从设置的序列集目录中选择的标准蛋白质 FASTA（`.fasta`）。
*   **生成工具**：用户提供的原始蛋白质序列数据库。
*   **结构**：
    ```text
    >Sequence_Header_1 [Optional Description]
    MNSGVSRRQ...
    >Sequence_Header_2
    MKVLLVSDA...
    ```

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 覆盖 **`OVER_WRITE`** | 切换是否用清理后的序列以原子方式替换输入的 FASTA。禁用时，会在同一目录保存一个带 `_sanitized.fasta` 后缀的文件。覆盖模式下拒绝写入空结果。 |
| 启用长度过滤 **`ENABLE_LENGTH_FILTER`** | 切换是否过滤掉不满足最小或最大长度限制的序列。 |
| 最小序列长度 **`MIN_SEQ_LENGTH`** | 保留一条序列所需的最小序列长度（以氨基酸计）。 |
| 最大序列长度 **`MAX_SEQ_LENGTH`** | 允许的最大序列长度。 |
| 按序列标题字符串删除 **`REMOVE_BY_HEADER_STRING`** | 排除序列标题中含有这一子串（区分大小写、完全一致）的序列，例如 `partial`、`fragment` 或 `low quality`。留空即停用；`None` 会被当作字面意义上的搜索词。 |

### 输出

#### 清理后的 FASTA 文件
*   **格式**：标准 FASTA（`.fasta`）。
*   **说明**：包含按规范方式清理并去重的序列。序列标题被处理为安全且全局唯一；序列首尾的杂质被修剪，保留的序列区间内的非残基字符被替换为 `X`。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **序列标题清理**：
     把方括号和花括号转换为圆括号，把 `? * " # % @ $ / \` 替换为下划线，合并连续的下划线，并在保留可读空格的同时规范空白字符：
     $$h_{\text{clean}} = \text{replace\_unsafe}(h_{\text{raw}})$$

2. **序列清理**：
     将序列转换为大写。可接受的残基字母表为 `ACDEFGHIKLMNPQRSTVWYBZJXUO`。第一个与最后一个可接受残基之外的杂质会被修剪；该区间内的无效字符逐个替换为 `X`，保留残基坐标不变。

     随后按清理后的序列去重。相同的序列保留最长的序列标题（结果确定）；清理后序列标题冲突的不同序列会获得避免冲突的数字后缀。

3. **长度与子串过滤**：
     若启用过滤，满足以下条件的序列会被丢弃：
     $$\text{Length}(s_{\text{clean}}) < \text{MIN\_SEQ\_LENGTH} \quad \text{or} \quad \text{Length}(s_{\text{clean}}) > \text{MAX\_SEQ\_LENGTH}$$
     
     区分大小写的序列标题子串过滤作用于规范清理之前的每个原始序列标题：
     $$\text{Substring} \subseteq h_{\text{raw}}$$

4. **写入与诊断**：
     将清理后的序列写入目标 FASTA 文件，然后分析序列长度分布并显示统计信息（平均值、中位数、标准差）。

</details>

---

# 生成嵌入 (`Generate_Embeddings.py`)

该脚本使用预训练的蛋白质语言模型（例如 ESM-2、ESM-C、Ankh、ProtBERT 和 ProstT5）提取序列嵌入。它把残基映射为高维表示向量，并按所选的 `float16` 或 `float32` 精度存入元数据优先的 HDF5 数据库。

### 输入

#### FASTA 文件 `INPUT_FASTA`
*   **格式**：标准 FASTA（`.fasta`），原始的或已清理过的均可。
*   **生成工具**：用户提供的序列集或 `Sanitize_Sequences.py`。
*   **说明**：记录在生成嵌入之前会在内存中自动清理。这种简化的清理不按序列标题文本或序列长度过滤，只有记录发生变化时才打印结果。

### 参数

| 参数 | 说明 |
| :--- | :--- |
| 模型名称 **`MODEL_NAME`** | 要使用的蛋白质语言模型架构，也是写入输出文件名的标签。标识符始终为小写（例如 `esmc_600m`、`esmc_6b`、`esm2_t33_650m`、`esm2_t30_150m`、`ankh_base`、`prot_bert`、`prost_t5`）。 |
| 保存精度 **`SAVING_MODE`** | 在 HDF5 中存储向量所用的数值精度格式（`float16` 或 `float32`）。`float16` 所占的嵌入存储空间约为 `float32` 的一半，但数值精度较低；若下游对数值精度的要求比文件大小更重要，请使用 `float32`。 |
| 计算设备 **`DEVICE_SELECTION`** | `auto` 会在具有代表性的序列长度上对可用、且经安装程序验证的 CPU/加速器候选设备进行基准测试，使用最快的成功设备；若运行时出错，则按排名依次退回。指定某个 CPU、CUDA、XPU 或 MPS 设备时，生成过程固定在该设备上，出错时报告错误，而不会悄悄切换设备。 |

> **共享 Biohub API 访问：** 选择 `esmc_6b`；插件会把这个便于用作文件名的标签映射为 Biohub 的 `esmc-6b-2024-12` API 标识符。首次使用时，`Generate_Embeddings.py` 的终端会提示输入一个隐藏的令牌，并将其存入被 Git 忽略的 `src/resources/Biohub_API.json`。同一个令牌和可选的 `ESM_API_URL` 也供 `esmfold large` 使用；当共享文件和 `ESM_API_KEY` 环境变量都不存在时，其工作进程终端会使用同样的隐藏提示。

> **模型条款：** 模型权重单独下载，并保留其发布者的许可证。特别是，Ankh Base 和 Ankh Large 的权重采用 CC-BY-NC-SA-4.0 许可，程序在访问前要求用户确认。模型清单见 `THIRD_PARTY_LICENSES.md`。这些模型条款不会改变 Apache-2.0 程序源代码的许可。

### 输出

#### HDF5 嵌入数据库
*   **格式**：HDF5（`.h5`）。
*   **结构**：
    - `/headers`：UTF-8 编码的已清理序列标题。
    - `/sequences`：UTF-8 编码的已清理序列，与 `/headers` 一一对应、顺序相同。
    - `/embeddings/{sanitized_header}`：形状为 $L \times D$ 的数据集。
    - 属性 `model_name`、`saving_mode`、`num_sequences` 和 `generation_complete`。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **在内存中清理 FASTA**：
     清理序列标题和序列，删除空记录或重复记录，相同的序列保留最长的序列标题，并分配避免冲突的唯一序列标题。不应用序列标题子串过滤和序列长度过滤。

2. **模型加载与权重缓存**：
     从 Hugging Face 下载并缓存模型权重。加载 Transformer 模型，并对输入进行分词。

3. **硬件目标选择**：
     发现经安装程序验证的 CPU、CUDA、Intel XPU 和 Apple MPS 设备。在 `auto` 模式下，它会在每个可用的候选设备上对具有代表性的已清理序列做基准测试，对成功的设备排名；若生成嵌入失败，可退回到排名中的下一个设备。手动选择的设备会被独占使用。

4. **残基嵌入生成**：
     对每条已清理的序列 $s$，所选的模型适配器会按该模型的要求格式化残基并分词，在不计算梯度的情况下执行推理，并去除该模型特有的特殊标记。ESM 和 Rostlab 系列适配器去除边界标记；Ankh 使用自己的分词器布局，并去除末尾的特殊标记。每个适配器都会校验最终的残基矩阵对每个已清理残基恰好有一行：
     $$E_{\text{residue}} \in \mathbb{R}^{L \times D}$$

5. **HDF5 数据库编制**：
     在生成任何嵌入之前，先写入并刷新已清理的 `/headers` 和 `/sequences`。随后，每个通过校验的残基矩阵都存放在其已清理的序列标题下，并逐个刷新。只有整个数据库通过校验后，`generation_complete` 才会变为 true。

</details>

---

# 嵌入裁剪 (`Embedding_Cropping.py`)

该脚本为裁剪过的或不完整的序列生成嵌入：它直接从已有的全长序列嵌入数据库中切出所需部分，而不是单独为裁剪出的片段生成嵌入。蛋白质语言模型用完整的自注意力上下文计算每个残基的表示，因此直接为短片段生成嵌入，得到的向量会与这些残基在其原生全长序列中得到的向量不同（缺少上下文）。该脚本从不重新运行语言模型——它只读取全长序列的 HDF5 数据库（由 `Generate_Embeddings.py` 生成），为每条裁剪序列切出所需的残基范围，从而保留完整上下文的表示。

### 输入

#### 全长嵌入数据库 `INPUT_EMBED`
*   **格式**：HDF5 嵌入数据库（`.h5`）。
*   **生成工具**：`Generate_Embeddings.py`（嵌入生成工具）。所需的已清理全长序列存放在 `/sequences` 中，无需另外提供全长 FASTA。

#### 裁剪序列集 `CROPPED_FASTA`
*   **格式**：标准 FASTA（`.fasta`）。
*   **说明**：需要生成上下文嵌入的部分序列。记录的清理方式与生成嵌入时完全相同。每个已清理的序列标题都必须出现在 `INPUT_EMBED` 中，每条已清理的序列都必须是所存全长序列的一个完全一致的连续子串。

### 参数

此脚本不需要额外的配置参数——其行为完全由源嵌入数据库和裁剪后的 FASTA 决定。

### 输出

#### HDF5 嵌入数据库
*   **格式**：HDF5（`.h5`），命名为 `{CROPPED_FASTA}_[{model_name}]_embeddings.h5`——其结构与直接对 `CROPPED_FASTA` 运行 `Generate_Embeddings.py` 得到的文件完全相同，因此可直接用作下游工具（`Embedding_PWA.py`、`Embedding_SSEARCH.py`、`Embedding_MSA.py` 等）的输入。
*   **结构**：
    - `/embeddings/{sanitized_header}`：形状为 $L_{\text{crop}} \times D$ 的数据集，从全长嵌入中切出。
    - `/headers`：已解析的裁剪序列标题数组。
    - `/sequences`：与之一一对应的已解析、已清理的裁剪序列数组。
    - 属性 `model_name`、`saving_mode`、`num_sequences` 和 `generation_complete`。

<details markdown="1">
<summary><b>算法详情</b></summary>

1. **序列标题对应**：
     对每条序列标题为 $h$ 的已清理裁剪序列 $s_{\text{crop}}$，在 `INPUT_EMBED` 中找到具有相同序列标题的已存全长序列 $s_{\text{full}}$ 和嵌入矩阵 $E_{\text{full}} \in \mathbb{R}^{L_{\text{full}} \times D}$。缺失的序列标题会被报告并跳过。

2. **一致性检查**：
     校验全长嵌入的行数与全长序列的长度一致：
     $$L_{\text{full}} \overset{?}{=} \text{Length}(s_{\text{full}})$$
     不一致说明源嵌入数据库无效，裁剪会停止，而不会切取过时的数据。

3. **偏移量解析**：
     通过精确子串搜索，找到裁剪序列在其全长母序列中的位置：
     $$\text{offset} = \arg\min \{ i : s_{\text{full}}[i : i+L_{\text{crop}}] = s_{\text{crop}} \}$$
     若裁剪序列出现多次，使用第一次出现的位置并记录警告；若完全找不到，则跳过该序列标题并报告。

4. **保留上下文的切片**：
     由于 $E_{\text{full}}$ 中的残基嵌入已包含完整序列的上下文，切片只是一个简单的索引范围，无需重新计算：
     $$E_{\text{crop}} = E_{\text{full}}[\text{offset} : \text{offset} + L_{\text{crop}}]$$

5. **HDF5 数据库编制**：
     先写入并刷新已解析、已清理的序列标题和序列，再把每个裁剪结果流式写入其序列标题键下，只有通过校验后才把输出标记为完成。

</details>
