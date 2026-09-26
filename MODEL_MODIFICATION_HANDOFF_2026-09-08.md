# DTIAM 第二波模型修改完整日志与交接

更新时间：2026-09-08（Asia/Shanghai）  
工作目录：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev`  
用途：交接给下一位模型，继续模型研究或整理 V23/GloBIS-DTI 论文。  
范围：原始 DTIAM、Sepsis V18、BN_LN、局部交互主线、V2–V27.1、正式结果、消融、审计、论文证据与当前服务器状态。

范围边界：对话中曾出现独立的 `Virtual Cell Mechanism Scaffold` runtime-contract 请求，但当前工作区没有找到 `mechanism_scaffold/runtime/` 或对应 contract 产物。因此本文件不把该独立任务记为已完成，也不将其与 DTI 模型版本混合；下一位模型若接手该方向必须重新核对项目路径和实际 artifact。

## 0. 接手后先读什么

按以下顺序读取，不要只根据文件名猜实验状态：

1. 本文件：当前最终状态、完整版本脉络和未完成事项。
2. `V23_PAPER_WRITING_HANDOFF_2026-09-05.md`：V23 架构、训练、数据、结果和写作边界的细节。
3. `external_benchmark/audits/v23_dtiam_single_model_and_ablation_report_2026-09-06.md`：V23、DTIAM 单模型、集成与消融统一报告。
4. `ARCHITECTURE_DESIGN_LESSONS_V2_V24_2026-09-03.md`：哪些设计有效、哪些设计反复失败及原因。
5. `HANDOFF_2026-08-21.md`：最完整的逐日运行记录，已连续追加到 V26 启动前后。
6. `external_benchmark/ablations/pseudo_cold_motif_anticollapse_v271/gate_summary.json`：V27.1 最终锁定结论。

旧文档中的“当前运行中”“PID”“下一步”均是当时快照。服务器当前状态以本文件第 12 节为准。

## 1. 当前锁定结论

### 1.1 当前论文主模型

当前最适合写成独立单架构论文的模型是 **V23**，建议论文名 **GloBIS-DTI**。

完整架构标识：

`bn_ln+frozen_bermol_atom_esm_segment_multiglimpse_bilinear_set_energy_normalized_h4_d64_v23`

选择 V23 的原因不是它在每个数据集都绝对第一，而是：

- 架构逻辑统一，不依赖跨架构投票；
- warm、protein-cold、cross-domain 均有五次完整全量训练和 held-out 评测；
- warm 五次运行相对同 seed BN_LN 全部 AUPR/AUROC 双升；
- cross-trained 五模型在 Davis、KIBA、Human、ASBench 上相对同批 BN_LN 的 AUPR/AUROC 均双升；
- protein-cold 只取得接近持平的小幅改善，必须如实承认；
- V23 相对 V22 的结构改动明确：对新增 residual 做逐样本能量归一化。

### 1.2 不能写成的结论

- 不能写 V23 在所有场景、所有指标上最好。
- warm 的 Original Atom–Segment 五模型比 V23 更高：`0.905966/0.944704` 对 `0.900852/0.941696`。
- protein-cold 中 V23 相对 BN_LN 只有 `+0.001138/+0.000220` 的 ensemble AUPR/AUROC 增益。
- cross 的 KIBA/Human/ASBench 虽然排序指标改善，但 log-loss 恶化。
- 固定 16 个 protein segments 不是 motif、domain、pocket 或 binding site。
- atom×segment attention 是潜在统计路由，不是真实原子–残基接触。
- V27.1 的 representation gate 虽通过，但 performance gate 失败，禁止晋级。

### 1.3 用户已经明确的偏好

- 评价架构时优先单架构；不接受用 V18+Atom 十 checkpoint 跨架构融合掩盖单架构不足。
- 单种子应与同种子基座比较；五模型应与五模型比较。
- 如果报告“最佳单模型”，同一行 AUPR、AUROC、log-loss 必须来自同一个 checkpoint，不能逐列拼最大值。
- 凡是 test 后才选择模型、epsilon 或权重的结果，必须标为 test-oracle，不能作为正式无偏主结果。
- 新架构应先 source-only/pseudo-cold 门控，再决定是否 full；不得默认继续扩大实验。

## 2. 工程、环境与数据

### 2.1 两套工程

- 原始 DTIAM/AutoGluon：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main`
- 后续神经网络修改与全部证据：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev`
- Python：`/mnt/home/dachuang/conda_envs/DTIAM/bin/python`
- BerMol 权重：`models/BerMolModel_base.pkl`
- 原始矩阵目录：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/matrices`
- 数据 catalog：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/catalog/manifest.json`

该目录不是 Git repository；`git status`/`git diff` 无法提供版本历史。版本追踪依赖独立脚本、`run_config.json`、`completed.json`、checkpoint SHA-256、缓存 metadata 和本系列交接文档。

### 2.2 数据路径

Warm：

- `/mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/train.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/valid.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/in_domain_warm/test.csv`

Protein-cold：

- `/mnt/home/dachuang/PSICHIC/dataset/in_domain_protein_cold/train.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/in_domain_protein_cold/valid.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/in_domain_protein_cold/test.csv`

Cross-domain：

- `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold/train.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold/valid.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/davis/test.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/kiba/test.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/human/test.csv`
- `/mnt/home/dachuang/PSICHIC/dataset/cross_domain_cold_eval_light/asbench/test.csv`

行数：

| Split | Train | Valid | Test |
|---|---:|---:|---:|
| Warm | 1,705,458 | 210,786 | 210,859 |
| Protein-cold | 1,735,141 | 212,133 | 179,829 |
| Cross | 1,044,070 | 102,023 | Davis 16,569 / KIBA 46,970 / Human 2,397 / ASBench 1,617 |

ASBench 原始标签列为 `Y`，构建矩阵时映射为 `classification_label`。

### 2.3 快速筛选与 pseudo-cold

- 禁止再使用 CSV 前 N 行截断作为快速实验；会引入顺序偏差。
- 固定 300k 分层索引位于：`external_benchmark/screen_subsets/protein_label_v1_seed20260805_n300k/`。
- family pseudo-cold folds 由 `code/build_protein_family_folds.py` 生成。
- pseudo-cold 只允许使用 source-training labels 与 pseudo query labels；不应读取 official validation/test。
- 新版本默认门槛：平均 ΔAUPR > 0、平均 ΔAUROC > 0、至少 2/3 folds 两项同时提升、三折 log-loss 不能全部恶化。

## 3. 基线阶段

### 3.1 原始 DTIAM

原始 DTIAM 是 AutoGluon tabular 体系，不是五随机种子神经网络平均。

| 权重 / 测试集 | AUPR | AUROC | Log-loss |
|---|---:|---:|---:|
| warm → warm | 0.850775 | 0.908235 | 0.375164 |
| protein-cold → protein-cold | 0.604970 | 0.721115 | 0.612659 |
| cross → Davis | 0.172170 | 0.653423 | 0.382569 |
| cross → KIBA | 0.283346 | 0.558234 | 0.562174 |
| cross → Human | 0.741023 | 0.560681 | 1.246051 |
| cross → ASBench | 0.197920 | 0.562280 | 0.334291 |
| warm → ASBench | 0.166928 | 0.562923 | 0.319048 |

模型组成：

- warm ensemble 实际为 `1.0 × LightGBMXT`；
- protein-cold 为 LightGBMXT、RandomForestGini、RandomForestEntr 的加权概率平均；
- cross 为 LightGBMXT、RandomForestEntr、XGBoost 的加权概率平均。

### 3.2 Sepsis V18 基线

注意区分 **Sepsis V18** 与后来的 **V18G**：二者不是同一架构。

- 入口：`code/external_benchmark_sepsis.py`
- 标识：`SepsisNetUniversal V18 exact`
- 输入：drug 768、protein 1280、hidden 256；legacy DANN；五个种子 42–46。

五模型结果：

| 权重 / 测试集 | AUPR | AUROC |
|---|---:|---:|
| warm → warm | 0.892023 | 0.936195 |
| protein-cold → protein-cold | 0.631119 | 0.727744 |
| cross → Davis | 0.270454 | 0.713706 |
| cross → KIBA | 0.323544 | 0.580975 |
| cross → Human | 0.807410 | 0.638035 |
| cross → ASBench | 0.193046 | 0.622370 |
| warm → ASBench | 0.215284 | 0.645023 |

### 3.3 BN_LN 全局基座

BN_LN 只在特定 protein fusion projection 位置使用 LayerNorm，其余位置并非全部 LN。

- 架构/训练：`code/external_benchmark_fusion_methods.py`
- 可训练参数：988,929，与 BN_BN 相同；优势不是容量增加。
- full cold validation 五次均值：BN_BN `0.638284/0.738606`，BN_LN `0.641369/0.741021`。
- cold test 五模型：AUPR `0.635625`，AUROC `0.723254`，log-loss `0.684184`。

BN_LN 是后续局部分支的冻结强基座，不是 V23 的原创贡献。

## 4. 修改路线的逐版本日志

### 4.1 早期局部分支与 V2–V7

| 版本/阶段 | 核心修改 | 结果与处理 |
|---|---|---|
| Local Protein V1 | ESM2 固定片段注意力 | 个别相似度区间改善，但 per-protein macro 下降，收益集中在高频蛋白 |
| Local Protein V2 | 冻结局部 adapter/band-pass | grouped 选择后反转，未保留 |
| Original Atom–Segment | BerMol atom token × ESM2 16 segment 双向局部交互残差 | warm 最强；cold AUPR好但 AUROC/校准混合，作为关键基线保留 |
| V2 | 冻结 BN_LN，signed AtomSegment logit residual | 局部信号存在，但跨种子不稳 |
| V3 | hard positive residual | 负区进入零梯度死区，分支塌缩为零 |
| V4 | positive residual + STE | 前向与梯度失配，raw residual 能量暴涨 |
| V5 | train signed、eval 截断负残差 | full validation 好，但 test 收益极小；训练/推理不一致，仅能止损 |
| V6 | 局部作为第四路输入并联合训练 fusion/head | 源域拟合增强，cold test 下降；结论是保持强基座冻结 |
| V7 | 固定 64/16 细粗分段、top-k 稀疏交互 | 小规模有明显信号，full/test AUROC下降；复杂局部结构未转化为稳定泛化 |

### 4.2 门控、全局几何与 normalization 路线 V8–V17

| 版本 | 核心修改 | 结果与处理 |
|---|---|---|
| V8 | 用 `4p(1-p)` 不确定性门控 residual | 不确定不代表局部修正方向正确；pseudo 失败 |
| V9 | global context 控制概率混合 | 没有新增监督，校准自由度过大；未稳定提升 |
| V10 | 全局表示角度/L2 归一化交互 | 丢失有用幅度，full/test 反转 |
| V11 | 固定恢复 25% 能量 | pseudo 好但 full AUPR下降 |
| V12 | raw/angular 自适应几何混合 | 手工能量门双降 |
| V13 | 学习蛋白到药物坐标的残差对齐矩阵 | log-loss改善但 AUPR/AUROC 双降，缺乏对齐监督 |
| V14 | fusion output BN→LN | 破坏全局融合统计，双降 |
| V15 | classifier head BN→LN | AUPR/AUROC交换，不稳定 |
| V16 | classifier BN/LN 固定混合 | screen近零，full明显恶化 |
| V17 | 联合上下文 bottleneck residual | 三折 AUROC升但 AUPR三折全降 |

结论：不要继续扫描全局角度/幅度配方、BN/LN比例或基于置信度的概率门。

### 4.3 任意 hidden-channel 分组路线 V18G–V21

| 版本 | 核心修改 | 结果与处理 |
|---|---|---|
| V18G | 将 learned hidden channels 任意分为四组 | pseudo看似好，full/test绝对性能低；ensemble5 `0.610778/0.717923` |
| V19 | 在 V18G 弱基座叠加 AtomSegment | AUPR略升但 AUROC/log-loss下降，弱基座不能靠小残差修复 |
| V20 | V18G 加低秩跨组通信 | 未修复错误分组前提 |
| V21 | channel shuffle + 二阶段 grouped residual | 进一步退化；ensemble5 `0.600212/0.708740` |

结论：任意 latent channel 没有预定义生物语义，分组只制造信息瓶颈。不要沿此路线继续加通信或 shuffle。

### 4.4 多头集合交互主线 V22–V26

| 版本 | 核心修改 | 门控/全量结果 | 最终状态 |
|---|---|---|---|
| V22 | 4-head、每头16维；有效 atom×segment pair 上 joint softmax；对角双线性路由；zero-up MLP | pseudo 排序有正信号但 log-loss三折全坏；cold ensemble `0.635108/0.720507/0.711274` | cold失败；表示保留供V23修正 |
| V23 | V22 + `tanh(delta)/sqrt(1+mean(delta²))` 逐样本能量归一化 | pseudo仍因三折log-loss全坏而 gate fail；用户授权 exploratory full；warm/cross稳定，cold近持平 | 当前最佳论文单架构 |
| V24 | V23 + 正负双极性 pair evidence | mean ΔAUPR `+0.002004`，ΔAUROC `-0.000485`，1/3双升，log-loss全坏 | gate fail；无full/test |
| V25 | 64 fine + 16 coarse 层级分段，共享权重，固定0.5/0.5 | mean `+0.001484/-0.000595`，2/3双升，log-loss全坏 | gate fail；无full/test |
| V26 | 归一化 Q/K 只控制路由，value保留原幅度 | mean `+0.000859/+0.000315`，仅1/3双升，log-loss全坏 | gate fail；无full/test |

V22 关键代码：`code/external_benchmark_atom_segment_ban_v22.py`。  
V23 关键代码：`code/external_benchmark_atom_segment_energy_v23.py`。  
V24：`code/external_benchmark_atom_segment_dual_polarity_v24.py`。  
V25：`code/external_benchmark_atom_segment_hierarchical_v25.py`。  
V26：`code/external_benchmark_atom_segment_qk_norm_v26.py`。

### 4.5 自适应 motif tokenizer：V27 与 V27.1

#### V27 source-only 审计版

V27 保留 V23 的 BN_LN、药物端、四头双线性集合交互、zero-up、能量归一化和 epsilon，只替换蛋白局部表示：

- 读取逐残基 ESM2 缓存；
- 3/7/15 多尺度一维卷积；
- 16 个 content-attention fine motif tokens；
- 4 个 coarse motif tokens；
- 保留 global protein token；
- 分别计算 atom×fine 与 atom×coarse/global 交互。

实现：`code/external_benchmark_atom_motif_hierarchical_v27.py`。  
架构审计：`code/audit_atom_motif_hierarchical_v27.py`。  
训练动力学烟雾：`external_benchmark/audits/v27_outer0_training_smoke.json`。

V27 没有启动 pseudo-cold 三折。20-step smoke 显示明显表示塌缩：step20 fine-map cosine约 `0.9876`、fine-token cosine约 `0.9968`、coarse-token cosine `1.0`，coarse entropy接近 `ln(16)`，说明不同 query 几乎产生相同 token。原 V27 文件保留，不应覆盖或补跑。

#### V27.1 anti-collapse

V27.1 只修 motif tokenizer：

- multi-scale residue feature 做 mask-aware per-protein centering；
- 16 个 fine query 使用固定重叠软位置锚点，`sigma_f=1/16`；
- coarse pooling 前对 fine tokens 均值中心化；
- 4 个 coarse query 使用软位置锚点，`sigma_c=1/4`；
- 没有加入 diversity loss、entropy loss、Sinkhorn、top-k、额外 gate 或外部注释。

实现：`code/external_benchmark_atom_motif_anticollapse_v271.py`。  
tokenizer审计：`external_benchmark/audits/v27_1_tokenizer_audit.json`。  
20-step smoke：`external_benchmark/audits/v27_1_outer0_training_smoke.json`。  
V27/V27.1对照：`external_benchmark/audits/v27_vs_v27_1_dynamics.json`。

step20 抗塌缩指标：

- fine-map cosine `0.154698 < 0.90`；
- fine-token cosine `-0.043727 < 0.95`；
- coarse-token cosine `-0.228130 < 0.98`；
- coarse entropy `2.453368 < ln(16)-0.10`；
- coarse max weight `0.125645 > 0.08`；
- collective coverage `0.996584 > 0.90`。

因此 tokenizer/representation 审计通过。随后完成 300k pseudo-cold 三折：

| Fold | ΔAUPR | ΔAUROC | ΔLog-loss | Selected epoch |
|---:|---:|---:|---:|---:|
| 0 | +0.004642 | +0.002209 | +0.001100 | 30 |
| 1 | +0.000439 | +0.003571 | +0.017359 | 25 |
| 2 | -0.003051 | -0.004421 | +0.024636 | 10 |
| Mean | **+0.000677** | **+0.000453** | **+0.014365** | — |

结论锁定：

- 2/3 folds 同时提升 AUPR/AUROC；
- representation gate：`passed`；
- performance gate：`failed`，因为三折 log-loss 全部恶化；
- overall gate：`failed`；
- promotion decision：`do_not_promote`；
- 禁止 full、official validation 或 test。

原训练的 epoch 0/2/5/10/20/30 motif 历史因 postprocess exception 丢失，不能重建且不允许伪造。最终 JSON 已标记 `missing_due_to_postprocess_exception`、`recoverable_without_retraining:false`。selected checkpoint representation 已通过只读 repair 重新计算。

最终权威文件：`external_benchmark/ablations/pseudo_cold_motif_anticollapse_v271/gate_summary.json`。

## 5. V23 的实际端到端架构

### 5.1 输入与冻结全局路径

- 全局 drug：BerMol pooled `[B,768]`。
- 全局 protein：ESM2 pooled `[B,1280]`。
- 冻结 BN_LN 基座得到 `x,y ∈ R^[B,256]`。
- 基础交互为 `x⊙y`。
- 最终 frozen fusion 输入为 `[x;y;z] ∈ R^[B,768]`，经 `768→256` fusion 与 `256→512→1` predictor 输出 logit。

### 5.2 局部路径

- 药物：BerMol radius-1 token IDs，最多96个；冻结 token embedding + LayerNorm 后，bias-free `768→64` projection。
- atom projection 后按 token mask 做每个样本均值中心化。
- 蛋白：逐残基 ESM2 按连续序列固定聚合成16个 `[1280]` segment means，保留长度权重和mask。
- protein segments 先按长度权重做样本内中心化，再 bias-free `1280→64` projection。
- 64维拆成4个head，每个head 16维。
- 每个head用可学习 diagonal bilinear 参数计算 `[B,4,T,16]` pair scores。
- 每个head在全部有效 atom×segment pairs 上做 FP32 masked joint softmax。
- 加权汇总逐元素 atom×segment evidence，四头拼接得到 `[B,64]`。
- residual MLP：`Linear(64,256) → LayerNorm → GELU → Linear(256,256)`。
- 最后一层权重与bias初始化为0；初始化时 V23 与冻结基座完全相等。

### 5.3 残差注入

`r(delta) = tanh(delta) / sqrt(1 + mean_j(delta_j²))`

`z = x⊙y + epsilon × r(delta)`

- 训练 epsilon：`0.025`。
- 已完成正式评测的锁定 inference epsilon：`0.0125`。
- 训练 residual-energy regularizer：`0.005 × mean(tanh(raw_delta)²)`。
- V23 trainable local parameters：214,080。
- 冻结 global parameters：988,929。
- 冻结 BerMol token embedding/LN parameters约11,919,360。

推理输出先是 logit；保存的 candidate scores 是 sigmoid 后的 probability。五模型规则是 `mean_s sigmoid(logit_s)`，不能平均 logits。

## 6. V23 正式完整结果

### 6.1 五模型同架构概率平均

| 权重 / 评价集 | AUPR | AUROC | Log-loss |
|---|---:|---:|---:|
| protein-cold → protein-cold | 0.636763 | 0.723474 | 0.698727 |
| warm → warm | 0.900852 | 0.941696 | 0.294200 |
| cross → Davis | 0.306660 | 0.735119 | 0.327204 |
| cross → KIBA | 0.340983 | 0.602370 | 0.633134 |
| cross → Human | 0.824924 | 0.664811 | 1.649616 |
| cross → ASBench | 0.196524 | 0.663978 | 0.334603 |
| warm → ASBench | 0.196529 | 0.643174 | 0.336386 |

### 6.2 每个场景按 AUPR 选择的最佳单次运行

用户要求显示时不提种子，但要保留如下原则：每行三项指标来自同一个 checkpoint；不同场景可能来自不同 checkpoint。这是 test-oracle 描述，不是无偏主结果。

| 权重 / 评价集 | AUPR | AUROC | Log-loss |
|---|---:|---:|---:|
| warm → warm | 0.887975 | 0.933971 | 0.311759 |
| protein-cold → protein-cold | 0.623206 | 0.702063 | 0.845254 |
| cross → Davis | 0.261029 | 0.700280 | 0.360635 |
| cross → KIBA | 0.328301 | 0.594940 | 0.731964 |
| cross → Human | 0.825904 | 0.682101 | 1.842997 |
| cross → ASBench | 0.196740 | 0.677634 | 0.368941 |
| warm → ASBench | 0.186711 | 0.635848 | 0.403795 |

### 6.3 V23 相对 seed-matched BN_LN 五模型

| 场景 | ΔAUPR | ΔAUROC | ΔLog-loss | 判断 |
|---|---:|---:|---:|---|
| protein-cold | +0.001138 | +0.000220 | +0.014543 | 排序近乎持平小升，校准变差 |
| warm | +0.004546 | +0.002757 | -0.007508 | 稳定双升且校准改善 |
| Davis | +0.005313 | +0.002960 | -0.009235 | 最完整跨域改善 |
| KIBA | +0.003809 | +0.001677 | +0.004866 | 排序升、校准降 |
| Human | +0.000344 | +0.001448 | +0.042371 | AUPR提升极小、校准明显降 |
| cross ASBench | +0.001345 | +0.004260 | +0.001532 | 排序双升、校准略降 |
| warm ASBench | +0.004752 | +0.007962 | +0.001391 | 相对自身基座升，但非ASBench绝对最佳 |

## 7. 消融状态

### 7.1 完整五次运行 protein-cold 架构消融

锁定相同179,829行 test，五次对齐运行：

| 方法 | AUPR mean ± SD | AUROC mean ± SD | Log-loss mean ± SD | Ensemble AUPR/AUROC/Log-loss |
|---|---:|---:|---:|---:|
| BN_LN | 0.613050 ± 0.009301 | 0.704646 ± 0.006859 | 0.790616 ± 0.021793 | 0.635625 / 0.723254 / 0.684184 |
| Original Atom–Segment | 0.614295 ± 0.010730 | 0.703384 ± 0.007648 | 0.822734 ± 0.027372 | 0.637057 / 0.722715 / 0.704538 |
| V22 | 0.611206 ± 0.007525 | 0.700942 ± 0.007324 | 0.833311 ± 0.030749 | 0.635108 / 0.720507 / 0.711274 |
| V23 | 0.614239 ± 0.009918 | 0.703904 ± 0.006790 | 0.813018 ± 0.023297 | 0.636763 / 0.723474 / 0.698727 |

可支持：V23 相对 V22 为 `+0.003033/+0.002963/-0.020293`，说明逐样本能量归一化修复了 V22 的主要退化。  
不可支持：V23 对 BN_LN 或 Atom–Segment 在所有指标全面领先。

报告：`paper_evidence/V23_ABLATION_2026-09-06/v23_protein_cold_ablation.md`。

### 7.2 组件消融：只完成 seeds 42/43，n=2

两次运行均值：

| Variant | AUPR | AUROC | Log-loss | 相对Full V23解释 |
|---|---:|---:|---:|---|
| BN_LN | 0.618028 | 0.710641 | 0.779332 | 无局部分支 |
| V22 / 无逐样本能量归一化 | 0.617174 | 0.707968 | 0.815373 | 两项排序下降、log-loss变差 |
| V23 without zero-up | 0.616396 | 0.707340 | 0.813413 | 两项排序下降、log-loss变差 |
| V23 without residual-energy penalty | 0.618926 | 0.709409 | 0.806944 | 排序近似，log-loss比Full V23差0.003544 |
| Full V23 | 0.618906 | 0.709091 | 0.803400 | 参考 |

该实验按计划只保留 seeds42/43；其他已启动但未完成的 partial runs 不使用。不要声称这是五种子稳定性研究。

- 计划：`paper_evidence/V23_ABLATION_2026-09-06/COMPONENT_ABLATION_PLAN.md`
- 两种子汇总：`paper_evidence/V23_ABLATION_2026-09-06/V23_COMPONENT_ABLATION_SEEDS42_43.md`
- no-zero-up 实现：`code/external_benchmark_atom_segment_energy_v23_no_zero_up.py`
- residual-energy-weight=0 通过原训练器参数实现，架构中的能量归一化仍保留。

### 7.3 Cross 外部域局部分支消融

Cross-trained V23 相对将局部分支置零的同批 BN_LN：

- Davis：AUPR/AUROC `+0.005313/+0.002960`，log-loss `-0.009235`；
- KIBA：`+0.003809/+0.001677`，log-loss `+0.004866`；
- Human：`+0.000344/+0.001448`，log-loss `+0.042371`。

报告：`paper_evidence/V23_ABLATION_2026-09-06/V23_CROSS_LOCAL_BRANCH_ABLATION.md`。

### 7.4 新增但尚未统一汇总的 V22 cross 任务

截至2026-09-06晚：

- `external_benchmark/ablations/full_cross_bilinear_set_v22/` 下 seeds42–46 五个 full cross training 均有 `completed.json`；
- `external_benchmark/test_evaluations/full_cross_bilinear_set_v22_external/` 下五个seed的 Davis/KIBA/Human 共15个 `completed.json` 均存在；
- 尚未发现该目录对应的 `ensemble5_<domain>/completed.json` 或最终统一 Markdown；
- 已有汇总脚本 `code/summarize_v22_v23_cross.py`，但它要求先存在 ensemble目录；
- 下一位模型如处理此项，只应基于已保存预测做只读 postprocess/概率平均，不重新训练、不改变checkpoint、不根据结果调参；完成前不要在论文主表引用。

另外，`paper_v23_ablation_warm_v22` 只留下 seeds42/43 的 history，没有完整 `completed.json`，不能作为 warm V22 正式消融。

## 8. DTIAM 单模型审计

2026-09-06 在原始 DTIAM 工程中完成已有模型的只推理审计：45/45完成、exit 0、没有训练。

- 脚本：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/code/evaluate_dtiam_single_models.py`
- wrapper：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/code/run_dtiam_single_model_oracle.sh`
- 汇总：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/test_evaluations/dtiam_single_model_oracle_20260906/summary.json`
- 汇总 SHA-256：`9a345d74d6f9bf829796c26301b81d9995ff48774f51cc6a213e852aec2f602c`

按测试 AUPR 事后选择且每行来自同一模型的上限：

| 场景 | 模型 | AUPR | AUROC | Log-loss |
|---|---|---:|---:|---:|
| warm | LightGBMXT | 0.850775 | 0.908235 | 0.375164 |
| protein-cold | RandomForestGini | 0.609036 | 0.717757 | 0.617969 |
| Davis | ExtraTreesEntr | 0.236655 | 0.690298 | 0.376431 |
| KIBA | RandomForestEntr | 0.294305 | 0.562508 | 0.557527 |
| Human | RandomForestGini | 0.769839 | 0.574693 | 1.167413 |
| cross ASBench | RandomForestEntr | 0.232030 | 0.579907 | 0.341634 |
| warm ASBench | LightGBMXT | 0.166928 | 0.562923 | 0.319048 |

这些结果是 `descriptive_test_oracle_not_for_unbiased_selection`。公平单模型对照应使用 validation leaderboard 预先固定模型；cross 四域统一固定 RandomForestEntr，不能逐域换树模型后声称同一模型。

完整解释见：`external_benchmark/audits/v23_dtiam_single_model_and_ablation_report_2026-09-06.md`。

## 9. 比赛候选与 V27.1 的位置

只读预检：`external_benchmark/audits/competition_submission_candidate_preflight.md`。

条件化选择：

- 隐藏分布未知且只能提交单架构：V23；
- 明确 protein-cold 且允许跨架构 ensemble：V18 + Atom，已有 `0.638532/0.730397`，但用户不接受它作为单架构创新证据；
- 明确 warm：Original Atom–Segment 5-seed，已有 `0.905966/0.944704`；
- V27.1：representation passed，performance failed，`do_not_promote`，不进入提交候选。

比赛规则尚未确认时，不得生成最终提交：还需主指标、hidden split、ensemble许可、checkpoint数量、推理资源、输入输出格式和预训练表示许可。

## 10. 论文证据与架构图

### 10.1 证据包

- `paper_evidence/V23_2026-09-05/README_CN.md`
- `paper_evidence/V23_2026-09-05/BASELINES.md`
- `paper_evidence/V23_2026-09-05/v23_and_backbone_per_seed.csv`
- `paper_evidence/V23_2026-09-05/five_seed_mean_sd.csv`
- `paper_evidence/V23_2026-09-05/ensemble_results.json`
- `paper_evidence/V23_2026-09-05/data_protocol_audit.json`
- `paper_evidence/V23_2026-09-05/result_verification.json`
- `paper_evidence/V23_2026-09-05/file_manifest.json`
- `paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/`
- `V23_PAPER_SOURCE_AUDIT_CN_2026-09-05.md`

### 10.2 架构图产物

目录：`paper_evidence/V23_MECHANISM_FIGURE_2026-09-07/`

已有版本：

- `globis_dti_v23_mechanism_verified.{svg,png,pdf}`
- `globis_dti_v23_mechanism_styled_v2.{svg,png,pdf}`
- `globis_dti_v23_mechanism_visual_v3.{svg,png,pdf}`
- `globis_dti_v23_mechanism_minimal_v4.{svg,png,pdf}`
- `globis_dti_v23_mechanism_reference_v5.{svg,png,pdf}`
- 图注：`globis_dti_v23_mechanism_minimal_v4_caption.md`

生成脚本：

- `code/plot_v23_mechanism_verified.py`
- `code/plot_v23_mechanism_styled.py`
- `code/plot_v23_mechanism_visual.py`
- `code/plot_v23_mechanism_minimal.py`
- `code/plot_v23_mechanism_reference.py`

`globis_dti_v23_mechanism_visual_v3.pdf` 的核心计算图已经与 V23 forward 对齐。仍需在最终图注明确：

1. ESM-2 representations 实际为 frozen/precomputed 缓存；
2. diagonal bilinear 参数用于 pair scoring，最终 pooling 汇总加权 element-wise evidence；
3. 训练 epsilon `0.025`，已有评测 inference epsilon `0.0125`；
4. heatmaps 为示意，不能解释成物理接触。

## 11. 关键代码与改动类别

### 11.1 通用基础与推理

- `code/external_benchmark_sepsis.py`：Sepsis V18/global数据训练基础。
- `code/external_benchmark_fusion_methods.py`：BN/LN及全局融合实验。
- `code/external_benchmark_atom_segment.py`：Atom–Segment基类、训练、能量正则、冻结检查。
- `code/evaluate_atom_segment_test.py`：版本化单checkpoint评测；支持 `energy_v23` 等 variant。
- `code/ensemble_atom_segment_test_scores.py`：同架构多checkpoint概率平均。
- `code/external_benchmark_local_protein.py`：S16 segment lookup/cache相关逻辑。

历史上这些文件曾被运行队列延迟加载，因此当时有“候选进程启动前禁止修改”的约束。目前没有本项目等待启动的进程，但下一位模型启动新后台队列后仍必须遵守：进程真正加载完核心代码前不要修改同一入口。

### 11.2 V22–V27.1 独立版本文件

- V22：`external_benchmark_atom_segment_ban_v22.py`
- V23：`external_benchmark_atom_segment_energy_v23.py`
- V23 no-zero-up：`external_benchmark_atom_segment_energy_v23_no_zero_up.py`
- V24：`external_benchmark_atom_segment_dual_polarity_v24.py`
- V25：`external_benchmark_atom_segment_hierarchical_v25.py`
- V26：`external_benchmark_atom_segment_qk_norm_v26.py`
- V27：`external_benchmark_atom_motif_hierarchical_v27.py`
- V27.1：`external_benchmark_atom_motif_anticollapse_v271.py`

所有路径均位于 `code/`。相应 pseudo wrapper、audit、summarizer、launcher 同样以版本号命名。新版本必须继续采用独立文件，不覆盖 V23–V27.1。

### 11.3 V27/V27.1 缓存安全

V27 全部 family folds 的逐残基 ESM2 cache mapping 已通过 sequence内容和 length-prefixed UTF-8 SHA-256 显式匹配：每fold 300,000行、6,449唯一序列、cache hit 300,000；missing/mismatch/conflict/row-order fallback/drug mismatch均为0。

审计：`external_benchmark/audits/v27_pseudo_cache_coverage.json`。

超过1022 residues的行会按已记录规则截断，每fold 50,030行、624唯一序列。不得把“截断存在”误报成cache mismatch；但正式论文应把这一限制列出。

## 12. 服务器现场状态（2026-09-08）

### 12.1 本项目进程

只读进程审计没有发现仍在运行的 DTIAM/V23/V27/pseudo-cold trainer、evaluator 或 supervisor。`external_benchmark/queues/` 下仍有历史 `.pid` 和 `.log`，但对应任务均已退出；不要只因 PID 文件存在就认定任务仍活着，应使用 `ps -p PID` 二次确认。

### 12.2 GPU

检查时5张卡均有高显存占用或外部进程：GPU0约23.9GB、GPU1约28.8GB、GPU2约21.3GB、GPU3约30.9GB、GPU4约24.2GB。当前占用不是本项目进程。不要抢占、终止或覆盖外部任务；启动新实验前必须同时核对：

- `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader`
- `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`
- `ps` 中PID归属和命令行。

### 12.3 未完成/不应自动继续的任务

- V27.1：已失败，禁止 full/test。
- V25/V26：已失败，禁止 full/test。
- Original V27：只做 source-only smoke，因collapse停止，禁止三折。
- V23 no-zero-up/no-energy-penalty：按计划只有两种子；不要把 partial seeds45/46接着跑，除非用户重新明确扩大范围。
- warm V22 paper ablation：不完整，不自动续跑。
- cross-trained V22：五次训练及15个Davis/KIBA/Human单次评测已完成，但ensemble/最终汇总缺失；仅可做无训练postprocess，且应先获得用户意图后再执行。
- 当前没有自动 NEXT。

## 13. 结果口径与常见陷阱

1. AUPR 对不平衡数据很重要，但 AUROC 与 log-loss 不能因此被隐藏。
2. ensemble结果不是单模型结果；跨架构十模型结果更不能用于证明单架构创新。
3. test-oracle只能做描述性上限。
4. rolling-5 selection score、单个epoch瞬时值、最终selected checkpoint和test结果必须区分。
5. pseudo gate fail 后即使用户授权 exploratory full，也不能改写为“通过预注册门控”。V23就是这种情况。
6. ASBench正例率低，AUPR与AUROC可能明显分歧；不能只挑一个指标。
7. Human 的V23 AUPR较高但log-loss很差，说明ranking与calibration分离。
8. base seed差异远大于局部分支初始化差异；小残差不能救回明显较弱的base。
9. 当前数据并非所有样本都有复合物结构，不应转向假定全样本都有3D contact的架构。
10. 文件路径含 `exploratory` 是provenance名称，不等于可以隐去真实实验门控历史。

## 14. 给下一位模型的建议

### 如果任务是继续写 V23 论文

- 直接以 V23/GloBIS-DTI 为主模型。
- 主消融使用五次运行 `mean ± SD`，不要使用逐域test-oracle作为主证据。
- 外部域可展示V23五模型对seed-matched BN_LN的公平差值。
- 明确 warm绝对最佳内部模型其实是Original Atom–Segment。
- 将固定segments、cold增益很小、外部域校准不稳定、依赖预训练缓存写入Limitations。
- 架构图优先从当前 v3/v4/v5中选择，不要把segments改画成motifs/pockets。

### 如果任务是继续修改模型

- 新版本建议从V23而不是V27.1继续，因为V27.1只解决representation collapse，没有通过性能门控。
- 必须提出一个与V24–V27.1不重复、能解释跨family可靠性或calibration的新归纳偏置。
- 保留：seed-matched frozen BN_LN、局部输入中心化、集合型pairwise交互、zero-up、逐样本残差限制。
- 避免：概率门控、BN/LN比例扫描、全局角度/幅度配方、任意hidden分组、test后epsilon扫描、无监督依据的复杂attention堆叠。
- 先source-only审计和20-step梯度/数值测试，再300k pseudo三折。representation正常并不等于performance可晋级。
- gate无论pass/fail均不得自动读取official validation/test或启动full，除非用户明确授权。

## 15. 最终权威文件索引

- 历史逐日总交接：`HANDOFF_2026-08-21.md`
- 架构失败复盘：`ARCHITECTURE_FAILURE_REVIEW_V2_V21_2026-08-31.md`
- 架构规律总结：`ARCHITECTURE_DESIGN_LESSONS_V2_V24_2026-09-03.md`
- V23论文交接：`V23_PAPER_WRITING_HANDOFF_2026-09-05.md`
- V23数据来源审计：`V23_PAPER_SOURCE_AUDIT_CN_2026-09-05.md`
- V23/DTIAM/消融总报告：`external_benchmark/audits/v23_dtiam_single_model_and_ablation_report_2026-09-06.md`
- 比赛候选预检：`external_benchmark/audits/competition_submission_candidate_preflight.md`
- V23冷启动消融：`paper_evidence/V23_ABLATION_2026-09-06/v23_protein_cold_ablation.md`
- V23组件消融：`paper_evidence/V23_ABLATION_2026-09-06/V23_COMPONENT_ABLATION_SEEDS42_43.md`
- V23跨域局部分支消融：`paper_evidence/V23_ABLATION_2026-09-06/V23_CROSS_LOCAL_BRANCH_ABLATION.md`
- V27缓存审计：`external_benchmark/audits/v27_pseudo_cache_coverage.json`
- V27/V27.1动力学对照：`external_benchmark/audits/v27_vs_v27_1_dynamics.json`
- V27.1最终门控：`external_benchmark/ablations/pseudo_cold_motif_anticollapse_v271/gate_summary.json`

## 16. 交接状态

`CURRENT_BEST_PAPER_ARCHITECTURE = V23 / GloBIS-DTI`  
`V27_1_REPRESENTATION_GATE = passed`  
`V27_1_PERFORMANCE_GATE = failed`  
`V27_1_PROMOTION = do_not_promote`  
`PROJECT_TRAINING_PROCESSES_RUNNING = false`  
`SAFE_AUTOMATIC_NEXT = none`  
`HANDOFF_STATUS = READY`
