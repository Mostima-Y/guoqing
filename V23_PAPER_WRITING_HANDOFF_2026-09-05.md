# V23 / GloBIS-DTI 论文写作交接

更新时间：2026-09-05（Asia/Shanghai）  
用途：供新的论文写作对话完整接管 V23。当前模型开发对话继续探索后续架构，两条工作线不要混写或相互覆盖。

## 0. 新对话的第一条指令

新对话应先完整读取本文件，再读取：

1. `external_benchmark/audits/competition_submission_candidate_preflight.md`
2. `HANDOFF_2026-08-21.md` 中第 46、49、50、52、53、54、55 节
3. `code/external_benchmark_atom_segment_energy_v23.py`
4. `code/external_benchmark_atom_segment_ban_v22.py`
5. 一份 protein-cold、一份 warm 和一份 cross 的 V23 `run_config.json`

在开始正文前，应先建立“主张—证据—文件路径”表。不要根据文件夹名、记忆或对话表述替代结果 JSON。

## 1. 模型身份

建议论文模型名：**GloBIS-DTI**。

建议全称：

> **Global–Local Bilinear Interaction Set Network for Drug–Target Interaction Prediction**

建议标题首选：

> **GloBIS-DTI: Energy-Normalized Global–Local Bilinear Set Interaction for Drug–Target Interaction Prediction**

内部版本号仍为 V23；论文正文使用 GloBIS-DTI，复现说明中写明 GloBIS-DTI 对应内部冻结版本 V23。

完整架构标识：

```text
bn_ln+frozen_bermol_atom_esm_segment_multiglimpse_bilinear_set_energy_normalized_h4_d64_v23
```

模型不是 V18+Atom 集成，也不是 V27/V27.1。GloBIS-DTI 是单一架构；正式五种子结果是同一架构的五次独立训练后做概率平均。

## 2. 一句话研究问题与核心思想

研究问题：在已有全局药物—蛋白表示已经很强时，能否只学习一个受约束的局部 atom×protein-segment 交互残差，在不破坏全局预测器的前提下稳定补充局部匹配信息？

核心回答：冻结 BN_LN 全局预测器和 BerMol token embedding，将药物原子集合与蛋白固定分段集合投影到低维空间，通过四头双线性集合交互提取多种局部匹配模式，再用 zero-up MLP 和无参数的逐样本能量归一化把局部残差以小幅度注入全局交互项。

## 3. 从输入到输出的端到端架构

### 3.1 输入表示

- 全局药物表示：BerMol pooled embedding，形状 `[B, 768]`。
- 全局蛋白表示：ESM2 pooled embedding，形状 `[B, 1280]`。
- 局部药物表示：BerMol radius-1 atom/substructure token IDs，最多 96 个，形状 `[B, T]`，`T <= 96`，带 mask。
- 局部蛋白表示：将已有 ESM2 蛋白表示按序列顺序固定聚合为 16 个 segment token，形状 `[B, 16, 1280]`，带 segment weight/mask。

写作边界：V23 的蛋白局部 token 是**固定序列分段**，不是经过验证的结构域、口袋、残基接触或生物学 motif。不要把 segment 改称 binding site、pocket 或 motif。

### 3.2 冻结全局主干 BN_LN

全局分支使用 seed-和 split-matched 的 `fusion_source_only_v1+protein_norm_bn_ln_v1` 基座。药物和蛋白编码后分别投影为：

```text
x: [B, 256]
y: [B, 256]
```

基础交互为逐维乘积 `x ⊙ y`。全局编码器、fusion 和最终 predictor 均冻结，训练时始终处于 eval 状态。论文可以把 BN_LN 解释为经过既有实验确认的强全局基座，但不要把缩写本身包装为 V23 的新贡献。

### 3.3 去除全局重复信息

局部分支先将 atom embedding 和 segment embedding 投影到 64 维：

```text
atom_projection:  BerMol token embedding -> 64
segment_projection: 1280 -> 64
```

每个药物的 atom token 按 mask 做均值中心化；每个蛋白的 16 个 segment 按有效长度权重做均值中心化。目的是让局部分支学习样本内部差异，避免简单复制 frozen global pooled signal。中心化在 float32 中进行，降低 AMP/fp16 消减误差。

### 3.4 四头双线性集合交互

64 维局部表示拆成四个 head，每头 16 维：

```text
atoms_h:    [B, T, 4, 16]
segments_h: [B, 16, 4, 16]
```

每个 head 使用可学习的对角双线性权重，得到所有 atom×segment pair 的打分：

```text
scores: [B, 4, T, 16]
```

在每个 head 的全部有效 pair 上做 masked softmax，然后对逐元素双线性证据加权汇总，得到：

```text
pair_evidence: [B, 64]
```

四个 head 是多个统计 interaction glimpses，不代表四种已知生物机制。attention 是潜在统计路由，不能解释成实验验证的物理 atom–residue contact。

该 pair pooling 不使用 atom 或 segment 的排列顺序；同时置换 token 和 mask 后输出保持不变。因此更准确的术语是 permutation-invariant bilinear set interaction。

### 3.5 Zero-up 局部残差

pair evidence 进入：

```text
Linear(64, 256) -> LayerNorm(256) -> GELU -> Linear(256, 256)
```

最后一个线性层的 weight 和 bias 初始化为 0，所以初始 raw residual：

```text
delta: [B, 256] = 0
```

初始化时 V23 与对应 BN_LN 基座输出逐点完全相等，审计记录的最大输出误差为 0。第一步主要更新 zero-up 输出层，之后梯度传到上游投影和双线性路由。这是“从已知基座开始做小修正”的工程稳定机制，而不是额外监督信号。

### 3.6 逐样本能量归一化

V23 相对 V22 的唯一核心变化是 bounded residual：

```math
r(\delta)=\frac{\tanh(\delta)}{\sqrt{1+\operatorname{mean}_{j}(\delta_j^2)}}
```

其中均值沿每个样本的 256 个 residual channel 计算。分母没有可学习 gate、threshold 或温度。当 raw residual 能量增大时，整条样本残差会被平滑衰减；在 `delta=0` 时 scale 为 1，保持初始梯度。

局部残差注入全局交互：

```math
z = x\odot y + \epsilon r(\delta)
```

训练 epsilon 固定为 `0.025`。已有完整评测使用在 validation 阶段锁定的 inference epsilon `0.0125`；写作时必须同时披露 trained epsilon 与 inference epsilon，不能让读者误以为二者相同，也不能再次根据 test 扫描 epsilon。

随后仍使用冻结基座的 fusion/predictor 输出最终 logit，概率为 sigmoid(logit)。

### 3.7 参数量

- 冻结全局参数：988,929。
- 冻结 BerMol token 参数：11,919,360。
- V23 可训练局部交互参数：214,080。
- 训练目标只更新新增局部分支；全局基座和 token embedding 不更新。

论文表格应区分 trainable parameters、frozen parameters 和 total referenced parameters，避免只报 214,080 后让人误解为整个推理系统仅有这些参数。

## 4. 训练与模型选择协议

完整 protein-cold / warm V23 协议：

| 项目 | 配置 |
|---|---|
| seeds | 42, 43, 44, 45, 46 |
| epochs | 64 |
| batch size | 256 |
| prediction batch size | 512 |
| optimizer | AdamW |
| local LR | `2e-4` |
| head LR | `2e-5` |
| weight decay | `1e-3` |
| validation interval | 每 2 epoch |
| checkpoint selection | rolling-5 |
| trained epsilon | `0.025` |
| locked inference epsilon | `0.0125` |
| residual-energy regularizer | `0.005 * mean(tanh(raw_delta)^2)` |
| AMP | enabled |

数据规模：

- protein-cold source train：1,735,141 行；validation：212,133 行；source/validation unique protein sequence overlap 为 0。
- warm source train：1,705,458 行；validation：210,786 行；warm 按定义允许实体重叠。
- cross source train：1,044,070 行；validation：102,023 行。

所有五种子 ensemble 均在 probability space 做简单算术平均：

```math
\bar p_i=\frac{1}{5}\sum_{s=42}^{46}\sigma(\ell_{i,s})
```

不平均 logits，不做按 test 表现加权，不混合其他架构。

## 5. 已完成结果与正确叙事

### 5.1 Warm：最稳定的主结果

| 模型（同为五种子） | AUPR | AUROC | Log-loss |
|---|---:|---:|---:|
| BN_LN | 0.896306 | 0.938939 | 0.301708 |
| V23 / GloBIS-DTI | **0.900852** | **0.941696** | **0.294200** |
| 差值 | **+0.004546** | **+0.002757** | **-0.007508** |

五个单种子相对各自 seed-matched BN_LN 均同时提升 AUPR/AUROC，且 log-loss 均改善。可以写“consistent improvements across five seeds”，但统计显著性仍需用保存的逐样本预测做正式检验后再写，不能仅凭五个方向一致就声称 statistically significant。

注意：Original Atom–Segment warm 五种子绝对成绩为 0.905966/0.944704，超过 V23。因而不能写“V23 achieves the highest warm score among every internal candidate”。V23 的优势叙事应放在更广泛的跨域一致性、单架构设计和受控 residual mechanism，而不是所有单项绝对第一。

### 5.2 Protein-cold：基本持平，不应夸大

| 模型（同为五种子） | AUPR | AUROC | Log-loss |
|---|---:|---:|---:|
| BN_LN | 0.635625 | 0.723254 | 待从对应 ensemble JSON 精确引用 |
| V23 / GloBIS-DTI | 0.636763 | 0.723474 | 0.698727 |
| 排序指标差值 | +0.001138 | +0.000220 | — |
| Original Atom–Segment | 0.637057 | 0.722715 | 0.704538 |

相对 Original Atom，V23 为 AUPR `-0.000294`、AUROC `+0.000759`。准确表述是：V23 在严格 protein-cold 上与强基座和 Original Atom 基本持平，未形成有说服力的大幅优势。不要写 significantly outperforms 或 substantial improvement。

V18+Atom 的 0.638532/0.730397 是两个架构、十个 checkpoint 的部署层概率集成，不是 V23 的公平单架构对手，也不能用作 GloBIS-DTI 的核心架构贡献。

### 5.3 Cross-domain：四域排序指标一致小幅提升

以下均为 cross-trained V23 五种子与同批、同 seed 数 BN_LN 五种子比较：

| Domain | BN_LN AUPR | V23 AUPR | ΔAUPR | BN_LN AUROC | V23 AUROC | ΔAUROC | Δlog-loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Davis | 0.301346 | **0.306660** | **+0.005313** | 0.732160 | **0.735119** | **+0.002960** | **-0.009235** |
| KIBA | 0.337175 | **0.340983** | **+0.003809** | 0.600692 | **0.602370** | **+0.001677** | +0.004866 |
| Human | 0.824581 | **0.824924** | **+0.000344** | 0.663363 | **0.664811** | **+0.001448** | +0.042371 |
| ASBench | 0.195179 | **0.196524** | **+0.001345** | 0.659718 | **0.663978** | **+0.004260** | +0.001532 |

可支持的结论：四个外部域的 AUPR 和 AUROC 都呈一致正增益，说明局部 residual 对 ranking discrimination 有一定跨域稳健性。

必须同时写的限制：收益总体较小；只有 Davis 的 log-loss 改善，另外三域恶化，因此不能声称 calibration 或 probability quality 全面改善。Human 的 AUPR 增益尤其接近零。

### 5.4 Warm 权重到 ASBench

| 模型 | AUPR | AUROC | Log-loss |
|---|---:|---:|---:|
| warm BN_LN | 0.191777 | 0.635212 | 0.334995 |
| warm V23 | 0.196529 | 0.643174 | 0.336386 |
| 差值 | +0.004752 | +0.007962 | +0.001391 |

排序指标提升、log-loss 略恶化。它是一个额外迁移诊断，不应与 cross-trained 四域主表混为一套训练协议。

### 5.5 Pseudo-cold 预注册门控历史

V23 早期 300k pseudo-cold 三折相对 BN_LN：

- mean ΔAUPR：`+0.002479`
- mean ΔAUROC：`+0.001417`
- 2/3 folds 同时提升两项排序指标
- 三折 log-loss 全部恶化
- 按当时预注册标准：**gate failed**

后续 full 是用户明确授权的 exploratory full，不得在论文中倒写成“V23 通过严格门控后晋级”。可以把 pseudo-cold 作为开发阶段诊断或补充材料如实披露。

## 6. 建议的论文贡献点

建议保持三点，避免堆砌：

1. **Global-to-local residual refinement**：在冻结的强全局 DTI 表示上只学习小型局部分支，使新增模型从完全复现基座开始优化。
2. **Multi-glimpse bilinear set interaction**：以四头、置换不变的 atom×protein-segment pair pooling 汇总局部统计交互，而不是先将两侧局部 token 各自压成单一向量。
3. **Parameter-free per-sample energy normalization**：根据每个样本 raw residual 的整体能量平滑限制注入幅度，在不加入 gate/threshold 的情况下控制局部分支对全局预测的改写。

第四点可以作为实验贡献而非方法创新：在 warm、protein-cold 和四个 cross-domain 数据集上做同 seed、同模型数的系统评估，并同时报告 ranking 与 calibration 的不一致。

不要把下列内容写成贡献：BN_LN 本身、BerMol、ESM2、固定 16 段、五种子平均、V18+Atom 跨架构集成。

## 7. 论文定位与创新性边界

V23 的创新性属于**组合式、机制明确的架构改进**，不是新的基础模型或新的蛋白结构建模范式。它有希望作为一个独立 DTI 模型投稿到方法/应用导向的普通期刊或会议，但论文能否成立取决于：

- 与公开 DTI 方法的完整公平比较，而不只是内部 BN_LN；
- 对关键组件做清晰消融：全局基座、Original Atom–Segment、V22、V23；
- 报告多种子不确定性/置信区间，而不仅是 ensemble 单点；
- 解释为什么 ranking 改善但部分域 log-loss 恶化；
- 公开或清晰描述数据切分、缓存构建和防泄漏协议。

目前最适合的表述是“a lightweight global–local residual architecture with consistent but modest ranking gains”，而不是 state of the art。是否 SOTA 必须在新的写作对话查阅最新文献和严格复现实验后判断。

## 8. 推荐论文结构

1. Introduction：全局 pooled embedding 丢失局部匹配；直接训练复杂局部分支又容易扰动强基座。
2. Related Work：DTI global encoders、token-level/local interaction、bilinear attention/set pooling、residual adaptation/parameter-efficient refinement。此处需要新对话联网检索正式文献。
3. Method：输入与冻结基座 → centered local tokens → four-head bilinear set interaction → zero-up residual → energy-normalized injection。
4. Experimental Setup：warm/protein-cold/cross-domain；五种子；同 seed BN_LN；AUPR/AUROC/log-loss；防泄漏与缓存 provenance。
5. Results：先 warm 和 cold，再 cross-domain；主表保持同架构/同 seed 数公平口径。
6. Ablation：BN_LN、Original Atom、V22、V23；不要把失败的二十多个版本全部塞入主文。
7. Analysis：残差幅度、跨域 ranking/calibration tension、推理成本。
8. Limitations：固定 segment 不是结构信息；cold 增益很小；外部域 calibration 不稳定；依赖预训练表示。

## 9. 推荐图表

- Figure 1：GloBIS-DTI 总体结构图。左侧 frozen global path，右侧 trainable local path，最后在 `x⊙y` 处受限注入。
- Figure 2：四头 atom×segment pair matrix 及 permutation-invariant pooling。应标注 latent statistical interaction，避免画成真实三维接触图。
- Figure 3：能量归一化曲线或示意，比较 `tanh(delta)` 与 sample-energy-scaled residual 的幅度行为。
- Table 1：数据集/split/样本量/正负比例。
- Table 2：warm 与 protein-cold 的五种子公平比较。
- Table 3：Davis/KIBA/Human/ASBench cross-domain 结果与 Δ。
- Table 4：消融 BN_LN → Original Atom → V22 → V23。

在没有重新统计前，不要虚构标准差、显著性星号、FLOPs 或峰值显存。已有单成员 179,829 行评测约 86.64 秒，但包含 CPU tokenization 和指标计算，不能直接当纯模型吞吐量。

## 10. 需要新写作对话继续完成的工作

1. 查阅最新文献，确认 GloBIS-DTI 名称未与已有模型明显冲突，并建立 Related Work 引用表。
2. 明确目标期刊/会议后反向确定篇幅、格式和实验最低要求。
3. 从保存的五个 seed 结果重建 mean±SD/CI；如需统计检验，只能使用已保存预测并预先定义检验方法。
4. 建立统一主结果表，补齐每个 baseline 的来源、训练协议与 checkpoint 数。
5. 确认 Original DTIAM 与 BN_LN 的关系和论文中使用的正式基线名称。
6. 决定是否把五种子 ensemble 放主表；单模型 mean±SD 与 ensemble 结果必须分开标注。
7. 编写无夸大摘要、贡献段和 limitations。
8. 若准备投稿，单独做代码整理与复现清单；不要直接把当前 `exploratory` 目录名写入论文。

## 11. 禁止性写作错误

- 不得称 V23 为所有内部候选或所有数据集的绝对最佳。
- 不得把 V18+Atom 十 checkpoint 结果归到 V23。
- 不得用十模型跨架构集成与五模型同架构 V23 证明架构优劣。
- 不得把固定 protein segment 称为 pocket、domain、motif 或 residue-level contact。
- 不得把 attention map 当作可解释的物理结合证据。
- 不得隐去 pseudo-cold gate failed，或把 exploratory full 改写成 gate-passed 实验。
- 不得只报 AUPR/AUROC 而隐藏三域 cross log-loss 恶化。
- 不得根据 official test 或未来比赛 hidden test 再调 epsilon、seed 权重或 checkpoint。
- 不得声称 SOTA、novel 或 statistically significant，除非新对话完成相应文献和统计证据。

## 12. 数据集来源与落盘路径

### 12.1 统一数据清单和矩阵

数据来源、resolved path、行数、类别数和特征定义的首要审计依据：

```text
/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/catalog/manifest.json
```

V23 实际训练读取的预计算矩阵位于：

```text
/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/matrices/
```

每个逻辑 split 对应 `{split}_X.npy`、`{split}_y.npy` 和 `{split}.json`。`X` 的前 768 维是 BerMol pooled drug embedding，后 1280 维是 ESM2-t33-650M 最后一层、最多前 1022 residues 的平均 protein embedding。矩阵构建入口：

```text
/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/code/external_benchmark.py build-matrices
```

### 12.2 In-domain warm / protein-cold

这两套 split 来自同一个清理、去冲突后的 in-domain DTI 数据池。上游数据由 BindingDB affinity 记录清洗并二值化；现有构建脚本使用小于 100 nM 为正类的规则。论文正式描述前仍应对照最终数据生成日志，避免把不同历史清洗脚本混成同一版本。

当前 canonical CSV 路径：

```text
warm:
  /mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/train.csv
  /mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/valid.csv
  /mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/test.csv

protein-cold:
  /mnt/home/dachuang/PSICHIC/dataset/in_domain_protein_cold/train.csv
  /mnt/home/dachuang/PSICHIC/dataset/in_domain_protein_cold/valid.csv
  /mnt/home/dachuang/PSICHIC/dataset/in_domain_protein_cold/test.csv
```

manifest 中对应的 resolved、行对齐副本为：

```text
warm:
  /mnt/home/dachuang/dti_in_domain_test/splits/protein_warm_start/{train,valid,test}.csv

protein-cold:
  /mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/{train,valid,test}.csv
```

切分与泄漏检查脚本：

```text
/mnt/home/dachuang/dti_in_domain_test/scripts/make_in_domain_splits.py
```

该脚本合并原始 train/valid、去除完全重复项、删除同一 Ligand–UniProt 出现冲突标签的 pair，再以 seed42 构造 warm 和按 target group 隔离的 protein-cold split。审计摘要：

```text
/mnt/home/dachuang/dti_in_domain_test/reports/dataset_overview.json
/mnt/home/dachuang/dti_in_domain_test/splits/protein_warm_start/stats.json
/mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/stats.json
```

上游 BindingDB 清洗相关脚本（必须在写作时确认哪一个版本实际生成最终原始池）：

```text
/mnt/home/dachuang/PostMidterm_WarmStart_Process/scripts/01_parse_and_clean_bindingdb.py
/mnt/home/dachuang/PostMidterm_WarmStart_Process/scripts/02_remove_data_leakage.py
/mnt/home/dachuang/PostMidterm_WarmStart_Process/scripts/06_split_train_valid.py
/mnt/home/dachuang/DTI_Project/scripts/01_clean_data.py
```

不要在未完成 provenance 串联前笼统声称所有上述脚本都直接参与了当前最终 CSV；其中包含历史/不同处理阶段。最终论文以 manifest、文件 hash 和最终生成日志为准。

2026-09-05 实文件复核：当前 `V2_BindingDB_Final.csv`、PostMidterm 原始 train+valid、Cross 的 train_raw+valid_raw 是相同的五列行多重集合，均 2,144,082 行。In-domain 之后另行去冲突；Cross 不自动继承此去冲突步骤。用当前 V2 Final 按现有 `06_split_train_valid.py + 08_enforce_warm_start.py` 重建早期分配，与保存的原始 train/valid 不一致，不能认定早期全过程已可复现。详见第 12.7 节。

### 12.3 按靶点 ID 排除的 cross-domain source

训练和验证 CSV：

```text
/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold/train.csv
/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold/valid.csv
```

resolved 上游文件：

```text
/mnt/home/dachuang/ColdStart_Process/data/train_sets/train_cold_final.csv
/mnt/home/dachuang/ColdStart_Process/data/train_sets/valid_cold_final.csv
```

构建逻辑先排除 Davis、KIBA、Human 测试 target，再以 UniProt_ID 做 9:1 group split，确保 cross train/valid target 不重叠：

```text
/mnt/home/dachuang/ColdStart_Process/scripts/build_strict_cold_start.py
/mnt/home/dachuang/ColdStart_Process/scripts/re_split_9to1.py
```

其中 `build_strict_cold_start.py` 将 TDC Davis Target_ID 与本地 KIBA/Human 的 pdbid 合并为 blacklist。2026-09-05 复核已用本地 davis.tab 重建 1,308 个 ID，过滤及后续 group split 的记录和行序均与落盘一致。

**已确认脚本缺陷：TDC Davis 的 379 个 Target_ID 是 AAK1、ABL1p 等名称，与 raw source 的 UniProt_ID 直接匹配数为 0；脚本没有先做 accession 映射。不能沿用“标准 UniProt 黑名单”注释，也不能将该结果称为严格 sequence-cold。部分 Davis 靶点可能由 KIBA/Human 黑名单间接排除。** 仍需同时报告实际完整序列 overlap；参见第 12.7 节。

### 12.4 External evaluation domains

| Domain | V23 实际读取路径 | resolved/upstream path | 数据来源说明 |
|---|---|---|---|
| Davis | `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/davis/test.csv` | `/mnt/home/dachuang/ColdStart_Process/data/test_sets/davis_test.csv` | Davis DTI affinity matrix，本地构建为二分类；现有脚本以 Kd `<=100 nM` 为正类。 |
| KIBA | `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/kiba/test.csv` | `/mnt/home/dachuang/ColdStart_Process/data/test_sets/kiba_test.csv` | KIBA interaction matrix，本地构建以 KIBA score `>=12.1` 为正类。 |
| Human | `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/human/test.csv` | `/mnt/home/dachuang/ColdStart_Process/data/test_sets/human_test.csv` | 本地 Human DTI benchmark。正式论文仍需补原始文献和构建 provenance。 |
| ASBench | `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/asbench/test.csv` | `/mnt/home/dachuang/ASBench_Builder/ASBench_KIBA_format.csv` | ASBench 别构 DTI benchmark；当前 1,617 行、147 正例。 |

**当前外部测试子集的直接来源已核验：** 上述 Davis/KIBA/Human 的 Cross test.csv，与 `/mnt/home/dachuang/PostMidterm_WarmStart_Process/data/processed/{davis,kiba,human}_test_warm.csv` 分别 SHA-256 完全一致，包含相同行序与标签。`09_clean_three_datasets.py` 对应这些输出路径，按旧训练集的 InChIKey 第一段匹配和蛋白双向 substring 匹配筛选。此筛选步骤的完整历史执行尚未重建；可以确定它们是早期 warm 处理线留下的外部子集，而不是完整原始 benchmark。Cross source 随后另行剔除靶点 ID，不意味着这些子集对当前 source 仍为 warm。

Davis/KIBA 本地转换脚本：

```text
/mnt/home/dachuang/DTI_Project/data/benchmark/07_build_full_davis.py
/mnt/home/dachuang/DTI_Project/data/benchmark/07_build_full_kiba.py
```

这些是本地处理代码，不等同于正式数据集引用。Davis、KIBA、Human、BindingDB、ASBench 的原始论文、许可证和下载链接需要新论文对话联网查证后写入 Data Availability/References。

### 12.5 局部表示缓存

ESM2 固定 16-segment 缓存：

```text
protein-cold:
  external_benchmark/protein_segments/full_protein_cold_esm2_t33_s16_v1/
warm:
  external_benchmark/protein_segments/full_warm_esm2_t33_s16_v1/
cross:
  external_benchmark/protein_segments/full_cross_esm2_t33_s16_v1/
```

in-domain warm 和 protein-cold segment 都映射到：

```text
/mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/protein.pt
```

cross segment 映射到：

```text
/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold/protein.pt
```

外部域分别使用：

```text
/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/{davis,kiba,human,asbench}/protein.pt
```

BerMol radius-1、最多 96 token 的缓存：

```text
external_benchmark/drug_substructures/full_protein_cold_bermol_r1_t96_v1/
external_benchmark/drug_substructures/full_warm_bermol_r1_t96_v1/
external_benchmark/drug_substructures/full_cross_bermol_r1_t96_v1/
```

缓存构建脚本：

```text
code/build_protein_segment_cache.py
code/build_protein_segment_test_cache.py
code/build_drug_substructure_cache.py
code/build_v23_cross_caches.sh
```

预训练 BerMol 权重：

```text
models/BerMolModel_base.pkl
```

全局蛋白矩阵来源另见 `DTIAM-main/external_benchmark/features/protein_embeddings.sources.json`：采用按优先级补缺的共享 feature table，优先复用 Warm protein.pt；不能与上述分场景局部缓存来源混为一谈。本轮全量核对 CSV→局部 row code 均通过，segment 加权均值与全局 pooled feature 的最大逐蛋白 RMSE 为 4.3864e-5，符合分段缓存 FP16 量化下的近似重建。

### 12.6 V23 运行脚本路径

| 用途 | 脚本 |
|---|---|
| V23 架构/正式训练入口 | `code/external_benchmark_atom_segment_energy_v23.py` |
| V23 pseudo-cold wrapper | `code/external_benchmark_pseudo_cold_energy_v23.py` |
| V23 架构审计 | `code/audit_atom_segment_energy_v23.py` |
| pseudo-cold 三折 launcher | `code/run_pseudo_cold_v23_after_v22_eps001.sh` |
| pseudo-cold gate 汇总 | `code/summarize_v23_pseudo_gate.py` |
| protein-cold full 五种子训练+评测+集成 | `code/run_v23_force_full_five_after_current.sh` |
| warm full 五种子原 launcher | `code/run_v23_warm_full_five_after_v23_cold.sh` |
| cross cache 构建 | `code/build_v23_cross_caches.sh` |
| cross 五种子及外部域评测 | `code/run_v23_cross_and_warm_external_five.sh` |
| 单 checkpoint 评测 | `code/evaluate_atom_segment_test.py --model-variant energy_v23` |
| 同架构五种子概率平均 | `code/ensemble_atom_segment_test_scores.py` |
| 全局矩阵读取/指标基础设施 | `code/external_benchmark_sepsis.py` |
| BN_LN 基座训练入口 | `code/external_benchmark_fusion_methods.py` |

warm 最终有效 checkpoint 来自机器重启后的独立 restart 目录；脚本仍是上述 warm launcher 的协议，最终路径必须使用 `full_warm_energy_normalized_v23_exploratory_restart_20260901`，不要误取中断目录。

### 12.7 2026-09-05 数据来源与脚本实文件复核

详细报告：[第二轮数据来源审计](paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/README_CN.md)。本节结论优先于历史脚本注释；未改动任何训练数据、模型代码或旧结果。

- 11 个相关 split（不含 ASBench）共 5,466,235 行 y 与 CSV 全量逐行一致；固定抽查 2,837 行 X，与 catalog feature 完全一致。原始 CSV 路径、SHA-256、矩阵来源 size/mtime 均匹配。
- Warm/Protein-cold/Cross 的 source/validation atom 和 segment row codes 全量重映射均无差异。
- Cross 原始池过滤以及 9:1 group split 均已重建；Davis 黑名单存在上述 ID namespace 不匹配。
- 当前 Davis/KIBA/Human 是与早期 warm 子集相同的文件；其较早来源文件分别为 30,056/118,083/6,593 行，当前保留 16,569/46,970/2,397 行，不能写为完整原始数据集。
- Davis 加入 pdbid 后回查原始 affinity：未映射、歧义、标签差异均为 0，规则 Kd≤100 nM。KIBA 规则 score≥12.1，仍有 55 行因同一分子字符串/靶点键对应多原始记录而存在标签歧义，须保留原始 ligand/row ID。
- Human 2,397 行都能在本地 TransformerCPI Human 原始 data.txt（6,728 行）中按 SMILES/sequence/label 找到；未从 affinity 重新定阈值。
- 既有完整序列 overlap 仍然存在：Protein-cold train/test 2 条序列、907 条测试行；Cross train/Davis 14 条序列、945 行；Cross train/Human 6 条序列、8 行。
- 当前 V2 原始池已锁定，但其早期 train/valid 分配按现存 06+08 脚本未能复现。BindingDB affinity 清洗的历史版本、覆盖清理顺序与原始数据发布版本仍属缺证部分。

JSON 结果、只读复核脚本、重建 blacklist、数据哈希和原始源码快照均存于 `paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/`。

## 13. 关键文件索引

### 实现

- V23：`code/external_benchmark_atom_segment_energy_v23.py`
- V22 / 四头双线性集合主体：`code/external_benchmark_atom_segment_ban_v22.py`
- Atom–Segment 基类和训练：`code/external_benchmark_atom_segment.py`
- 评测入口：`code/evaluate_atom_segment_test.py --model-variant energy_v23`
- 五种子概率平均：`code/ensemble_atom_segment_test_scores.py`

### 配置示例

- protein-cold：`external_benchmark/ablations/full_energy_normalized_v23_exploratory/protein_cold_energy_normalized_v23_seed42_e64/run_config.json`
- warm：`external_benchmark/ablations/full_warm_energy_normalized_v23_exploratory_restart_20260901/warm_energy_normalized_v23_seed42_e64/run_config.json`
- cross：`external_benchmark/ablations/full_cross_energy_normalized_v23/cross_energy_normalized_v23_seed42_e64/run_config.json`

### 最终五种子结果

- protein-cold：`external_benchmark/test_evaluations/full_energy_normalized_v23_exploratory/ensemble5/completed.json`
- warm：`external_benchmark/test_evaluations/full_warm_energy_normalized_v23_exploratory_restart_20260901/ensemble5/completed.json`
- cross-domain：`external_benchmark/test_evaluations/full_cross_energy_normalized_v23_external/ensemble5_{davis,kiba,human,asbench}_v23/completed.json`
- cross-domain BN_LN 对照：同目录 `ensemble5_{domain}_bn_ln/completed.json`
- warm→ASBench：`external_benchmark/test_evaluations/full_warm_energy_normalized_v23_external/ensemble5_asbench_v23/completed.json`

### 总交接和审计

- 全实验历史：`HANDOFF_2026-08-21.md`
- 比赛候选审计及 checkpoint SHA-256：`external_benchmark/audits/competition_submission_candidate_preflight.md`

## 14. 与当前模型开发线的边界

- V23 文件、checkpoint 和既有结果视为冻结论文证据，不应被后续 V28+ 实验覆盖。
- V27.1 的 representation gate passed，但 performance gate failed，`promotion_decision=do_not_promote`；它不是 V23 的升级版论文结果。
- 当前模型开发对话可以继续探索新架构；只有新架构经过公平协议且形成明确证据后，才在新的、版本化补充文件中讨论是否影响论文主模型。
- 在此之前，论文写作对话以 V23/GloBIS-DTI 为固定研究对象，不随每次新实验临时改变故事。

## 15. 当前交接状态

```text
PAPER_MODEL: V23 / GloBIS-DTI
ARCHITECTURE_FROZEN: yes
CORE_RESULTS_AVAILABLE: yes
PAPER_CLAIM_LEVEL: consistent but modest ranking improvement
STRONGEST_RESULT_AXIS: warm stability and cross-domain ranking consistency
WEAKEST_RESULT_AXIS: protein-cold effect size and external-domain calibration
V18_PLUS_ATOM_IN_MAIN_ARCHITECTURE: no
V27_1_PROMOTED: no
NEXT_OWNER: new paper-writing conversation
```
