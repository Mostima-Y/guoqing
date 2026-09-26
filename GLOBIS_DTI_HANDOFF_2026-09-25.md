# GloBIS-DTI 跨窗口交接（2026-09-25）

工作目录：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev`  
Python：`/mnt/home/dachuang/conda_envs/DTIAM/bin/python`  
状态核查：2026-09-25 约 12:00（Asia/Shanghai）。本目录不是 Git 仓库；以脚本、`run_config.json`、`completed.json`、checkpoint SHA 和产物为准。

## 给下一窗口的任务摘要

用户希望以**自己修改的 V23** 为固定 base model，逐项比较从其他模型借鉴的模块，有效后再组合；GPU 充裕，允许全量试验和顺手看 test。用户要求具体到架构，不能把“仅全局 BN_LN”误称为其 baseline，也不能笼统地说“融合创新模块”。近期用户认为增益有限，要求核查数据标准性与公平性、样本量和有效信号、训练收敛与策略，再参考近期顶会/顶刊，最后考察蛋白结构和几何特征。上述调查已完成并出具报告；新一轮训练尚未开始。

用户明确说过 test 不需要锁定。这允许探索性测试，但反复查看 test 后不能把选出的最优结果当作独立无偏泛化估计。应保留固定 V23 对照和同 seed/同预算比较。

## 目前模型与实验状态

V23 架构入口：`code/external_benchmark_atom_segment_energy_v23.py`。冻结全局 BN_LN：BerMol pooled drug 768 维、ESM2 pooled protein 1280 维，分别投影为 256 维，`x,y,x*y` 进融合 MLP。局部药物是冻结 BerMol embedding+LN 的半径 1 原子子结构 token（至多 96），不是完整上下文化 BerMol 局部 Transformer 输出；局部蛋白是冻结 ESM2 的前 1022 残基均分池化为 16 段。四头、每头 16 维的原子×片段对角双线性路由，在有效配对上 joint softmax。证据经 zero-init 上投影生成残差，`tanh(delta)/sqrt(1+mean(delta²))` 能量归一化后注入全局 `x*y` 融合项。V23 局部可训练参数约 214,080。冻结基座不是无 BerMol 的模型。

近期单改动：V30 加参数自由的双向互注意力上下文和两个初值 0 的门控；V26 只对最终路由 Q/K 做 L2 归一化；V36 组合 V30 与 V26。代码入口：`code/external_benchmark_atom_segment_mutual_v30.py`、`code/external_benchmark_atom_segment_qk_norm_v26.py`、`code/external_benchmark_atom_segment_mutual_qk_norm_v36.py`。最近的全量训练/测试是 V36 seed42/43，测试完成于 2026-09-24 21:17。2026-09-25 约 12:00 核查时没有 GloBIS-DTI 训练进程；消融与测试目录均未出现审计后的新实验。

| 两 seed 平均预测，同一 protein-cold test | AUPR | AUROC | log-loss |
|---|---:|---:|---:|
| 仅全局 BN_LN，诊断参考 | 0.631356 | 0.720866 | 0.716115 |
| **V23，当前对比基线** | **0.632082** | **0.719216** | **0.737328** |
| V23 + 双向互注意力 | 0.633087 | 0.721550 | 0.734539 |
| V23 + 最终路由 Q/K 归一化 | 0.634740 | 0.722163 | 0.735738 |
| V36：两项组合 | 0.636306 | 0.723285 | 0.734462 |

以上是 seed42/43 两成员集成，不可与历史 V23 五成员集成 `0.636763 / 0.723474` 直接比较。V36 相对同条件 V23 是 AUPR +0.004224、AUROC +0.004070；仍是探索性结果，尚无蛋白/家族分组置信区间。验证集表现与 test 排序也不完全一致，切勿称 V36 已正式胜出。`external_benchmark/ablations/full_two_seed_mutual_qk_norm_v36/` 和 `external_benchmark/test_evaluations/full_two_seed_mutual_qk_norm_v36/` 有完整产物；启动脚本 `code/run_full_two_seed_mutual_qk_norm_v36.sh` 给出训练、测试、集成实际参数和缓存路径，其中 GPU UUID 是当时设备，重跑先检查当前 GPU。每个 seed 的 V23/V30/V26/V36 基座 SHA 一致。

## 2026-09-25 数据与训练审计

完整中文报告：`external_benchmark/audits/data_training_20260925/REPORT_CN.md`。可复现脚本：`code/audit_globis_data_training_20260925.py`。数值产物：同目录 `data_summary.json`、`signal_metrics.json`、`training_summary.json`、`matched_configs.json`、八份历史 CSV、`per_protein_test_metrics.csv`、`training_curves.png/pdf`。脚本只读取已有数据、训练历史和保存预测，重跑会覆盖该审计目录下同名审计产物；如要保留快照先另开输出目录。`code/audit_data_training_20260925.py` 是早期仅 77 字节的空壳，不是正式脚本。

Protein-cold：训练 1,735,141 对、676,322 正例、6,761 独立蛋白序列；验证 212,133 对、862 序列；测试 179,829 对、843 序列。训练每条蛋白样本数中位数 12，前 100 条蛋白占训练行数 38.43%；测试前 100 条占 81.98%。这种集中度说明整体微平均指标可能由高频靶点驱动，不能把 173 万对当作同等独立的生物样本。

原始字符串层面，三组内部没有重复 Ligand+Protein 输入对或冲突标签，矩阵标签逐行吻合。历史审计中抽样特征行和局部/全局缓存行号亦通过。这不等于完成 canonical SMILES、所有特征逐元素、蛋白同源性和原始 assay 审核。

训练/测试在 UniProt ID 上不重叠，但有两条完全相同蛋白序列，训练 59 行、测试 907 行。当前是 ID cold，不是严格序列/同源 cold。仅删除 59 行训练数据后剩 1,735,082 对，但模型/基座必须重训才能声称剔除这些信息；只在已有预测中去掉 907 测试行是敏感性分析。该分析中 V23 AUPR/AUROC `0.632761/0.719392`，V36 `0.637006/0.723523`，模块增益未消失。测试有 95,544 行药物在训练出现，这符合 protein-cold 场景，不能说双冷启动。

历史来源审计：Davis 排除名单曾用 Target_ID 名称与 UniProt accession 比较，379 个名称匹配 0；实际训练/Davis 有 14 条相同序列、945 测试行，Human 有 6 条、8 行。当前 Davis/KIBA/Human 外测是挑选子集，不应称完整标准公开测试集；KIBA 有 55 行歧义冲突。部分早期候选数据处理脚本会移除亲和力 `>`/`<`，混合 Kd/Ki/IC50/EC50 取最小值并用 100 nM 二值化。是否所有最终行经过该脚本尚未证明，现有二分类 CSV 不能完整恢复 assay 溯源。原始审计在 `paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/`。

训练配置：AdamW LR 0.0002 恒定，weight decay 0.001，batch256，64 epoch，每 2 epoch 验证一次，`patience=99` 在此预算内无效。选模使用验证 AUPR 5 次滚动窗口。训练/验证残差 epsilon 0.025，当前 test epsilon 0.0125。Q/K 归一化 seed42/43 选中 epoch20/6，末轮验证 AUPR 已明显下降而训练 BCE 继续下降；组合选中 epoch48/52，seed43 也出现末轮回落。不能一概延长训练。V36 的蛋白片段边缘注意力熵约 2.764，接近均匀 16 段的 ln16=2.773，但这不证明联合原子×片段交互完全均匀。

在 416 条有正负两类测试样本的蛋白上，逐蛋白等权宏平均 V23 AUPR/AUROC `0.594276/0.698155`，V36 `0.598360/0.698834`：宏 AUROC 仅 +0.000679。随机排序 AP 的近似参考是正例率 0.3881；模型显然有预测信号，但没有科学依据估计“捕获了百分之多少生物机制”。局部分支的 log-loss 比仅全局差，值得检查概率校准。

## 输入与结构实验现状

局部蛋白只取前 1022 残基并压成 16 段。训练有 645 条更长序列，涉及 277,734 对（16.01%）。CSV 中训练结构路径 5,606 个，文件存在 2,511 个；只是存在性检查，尚未逐条核对序列、残基映射和 pLDDT。药物缓存 train+valid 联合唯一分子约 119.5 万，max96 截断约 0.93% 的分子，17 个无效分子以 UNK 代替。

**已有结构尝试，避免重复声称首创：** 历史对 1,000 条 accession 拉取 AlphaFold DB v6，969 份精确序列匹配结构可用（488 train、481 valid），跑了 fpocket。口袋统计特征、配体条件口袋注意力、口袋残基 ESM 池化均未稳定改善；结构覆盖训练约 10.1 万行、验证约 12.3 万行，覆盖差异曾制造表观增益。详情见 `HANDOFF_2026-08-21.md`。新实验应做残基级几何图或结构预训练表示，并先在相同结构覆盖子集与序列模型对比。独立药物构象和蛋白结构没有共同坐标系，不能直接算可信跨分子接触距离；真实接触需要对接或复合物结构。

近期文献和对应建议已整理在审计报告中：DataSAIL（Nature Communications 2025）用于相似性控制划分；ScopeDTI（Nature Communications 2025）提示靶点级标签偏斜和数据整理；SaProt（ICLR 2024）、PSC-CPI（AAAI 2024）、GearNet（ICLR 2023）分别提示结构词表、多尺度序列结构对比学习、残基几何图。DrugCLIP（NeurIPS 2023）是口袋与分子对比预训练，任务与当前二分类不同。阅读原论文再落实方案，避免把论文报告值直接与当前分割指标比较。

## 建议接手顺序

1. **先消除评估条件不一致。** 从现有 V23/V36 checkpoint 用 epsilon=0.025、0.0125（必要时 0）重评同一验证/测试数据；记录同 seed、同集成、同选中 checkpoint。`code/evaluate_atom_segment_test.py` 是测试入口，先读参数帮助和原运行脚本；新输出目录，保留旧产物。此项能在不重训情况下判断 test 改 epsilon 的影响。
2. **建立更稳的误差画像。** 以蛋白为单位做配对 bootstrap 区间和每蛋白指标，按样本数、正例率、序列长度、是否有训练同源蛋白分层。先做 canonical SMILES 与 MMseqs2 等序列同源核查；历史 ESM 余弦伪 family folds 不能当真正同源审核。追溯标签 assay/单位/不等号。若重建数据或划分，冻结全局基座及 V23 对照必须相应重训。
3. **训练策略单因素全量实验。** 以 V23 固定架构和数据，对照恒定 LR 与衰减调度、真正可触发的早停、蛋白均衡混合采样、同蛋白内排序辅助损失。每次一个因素、同 seed/预算；训练策略可能影响概率校准，AP/AUROC/log-loss 都要记录。
4. **输入分辨率单因素实验。** 先 16→64 蛋白片段保持前1022范围；再另测全长窗口以分辨截断效应。药物端另测上下文化 BerMol 局部表示或2D图消息传递。需要重建对应缓存时另存目录，避免覆盖 V23 缓存。
5. **结构试验。** 先建 sequence hash→AlphaFold结构→残基匹配→pLDDT/缺失 mask 的 manifest，做相同覆盖样本对照；再测试 SaProt 冻结结构表示或残基几何图，之后考虑全体缺失回退及药物3D。几何图可用序列边、空间 kNN/接触边、距离 RBF、方向/二面角等特征，接回原有 V23 局部分支，保持其他结构不变。

用户偏好前期大量对比实验，算力充裕；不要把上述方案仅作为建议停下。如果新窗口受托继续，应先核查当时 GPU、文件状态和实验目录，再实际运行可复现对照，并把新结果追加到新交接记录。

## 关键入口

- 最新诊断：`external_benchmark/audits/data_training_20260925/REPORT_CN.md`。
- 数据来源证据：`paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/README_CN.md`、`verification_summary.json`。
- 原 V23 长交接：`MODEL_MODIFICATION_HANDOFF_2026-09-08.md`、`V23_PAPER_WRITING_HANDOFF_2026-09-05.md`。
- 历史结构尝试：`HANDOFF_2026-08-21.md`。
- V23、V30、V26、V36 架构和运行入口：本文件前述 `code/` 脚本。
- 原始矩阵：`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/matrices`。
- 当前 cold CSV：`/mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/{train,valid,test}.csv`。
- V36 完整产物：`external_benchmark/ablations/full_two_seed_mutual_qk_norm_v36/` 与 `external_benchmark/test_evaluations/full_two_seed_mutual_qk_norm_v36/`。

最后强调：本交接中的“当前无训练进程”是 2026-09-25 约 12:00 的快照。新窗口先看进程、日志和 `completed.json`，不要按文档时点推断实时状态。
