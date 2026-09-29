# 稿件与实验数据审查报告

> 审查对象：`dasfaa_paper_data_package.zip`（`draft/main.tex`、`template/references.bib`、`data/*.csv`、`figures/*`）
> 审查日期：2026-09-28

## 0. 结论

问题不只是"写得烂"。**这份稿子目前不能投，也不值得在文字上打磨**：

1. **参考文献大面积捏造**：2025–2026 年的新文献几乎全部是编造的作者和标题，包括 CLOC 本身。投出去轻则被当成不严谨，重则算学术不端。
2. **四条核心结论里有三条被它自己的数据否定**：
   - "依赖微观相位同步"（Finding 2）；
   - "Top-K 优于随机、超过 32 个通道后饱和"（Finding 5）；
   - "接近 Kalman oracle 的 gap closure"（Finding 4）。
3. **E2、E3、E5 的实现各有 bug**，需要修复后重跑（§3）。
4. **计划里的必做项缺失**：E0（官方基准的收益分解）和全部对照模型（Chronos-2、PatchTST、iTransformer、CLOC）都没有做，也没有任何置信区间。

好消息是：E1 的数据本身看起来可用，而且有两个真实、有意思的发现（§4）；E2 的逐窗口原始数据也还在，我已经用正确的方式重新算了一遍。

**处理顺序**：先换参考文献，再修 bug、重跑，最后写文字。文字我已经把不依赖重跑结果的部分（引言、相关工作、框架、实验设置）重写了一版（`main_rewrite.tex`）。

---

## 1. 最严重：参考文献捏造（必须全部替换）

下表是本次对话中我逐条核实过的条目。修正后的版本见 `references_fixed.bib`。

| key | 稿件里写的 | 实际 |
|---|---|---|
| song2026cloc | Song, Yang; Zhang, Wei; Liu, Ming; Wang, Jian | **Song, Yuhao; Liu, Yiyuan; Wang, Chang; Li, Haiyan**，"Controlled Local–Global Channel Communication for Multivariate Long-Term Forecasting"，Mathematics 14(17):3132，DOI 10.3390/math14173132 |
| abdelmalak2025lookback | Abdelmalak, **Mina**; Chen, Haoxuan…，"…the **Myth** of Multivariate…" | **Abdelmalak, Ibram; Madhusudhanan, Kiran; Choi, Jungmin; Kloetergens, Christian; Yalavarit, Vijaya Krishna; Stubbemann, Maximilian; Schmidt-Thieme, Lars**，"Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased Is Time Series Forecasting?"，PAKDD 2026 |
| liu2026falconx | Liu, Yuxuan; Chen, Hao…，"Beyond Semantic Collapse…" | **Liu, Yiding; Hu, Yifan; Xia, Hongjie; Liu, Peiyuan; Chen, Hongzhou; Dai, Xilin; Dong, Zewei; Yang, Jiangming**，"Falcon-X: A Time Series Foundation Model for Heterogeneous Multivariate Modeling" |
| ansari2025chronos2 | 标题 "…From Univariate to Multivariate…with Group Attention" | "Chronos-2: From Univariate to **Universal** Forecasting" |
| liu2025moirai2 | 标题 "…without Cross-Variate…" | "Moirai 2.0: When Less Is More for Time Series Forecasting" |
| garg2026toto2 | Garg, Smith, Taylor, Patel（编造） | "Toto 2.0: Time Series Forecasting Enters the Scaling Era"，arXiv 2605.20119（作者待从 arXiv 页面补全） |
| carlsen2026tirex2 | Carlsen, Schmidt, Hansen（编造） | Podest, Patrick; Pichler, Marco; Bürger, Elias; others，"TiRex-2: Generalizing TiRex to Multivariate Data and Streaming" |
| rmisc2026 | Chen, Zhao, Sun（编造） | Sun, Qian; Tian, Yong-Ming; others，"RMISC: A Large-scale Real-world Multivariate Corpus for Time Series Foundation Models" |
| fevbench2025 | He, Zhang, Shen, Lin（编造） | "fev-bench: A Realistic Benchmark for Time Series Forecasting"（作者待核实，第一作者应为 Shchur） |
| gifteval2024 | Garza, Mergenthaler-Canseco…（这是 Nixtla 的人，张冠李戴） | Aksu, Taha; others，"GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation" |
| timebench2026 | Wang, Chen, Long，"TIME: A Leak-Free Benchmark…" | Qiao, Zhongzheng; Pan, Sheng; Wang, Anni; others，"It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks"，ICML 2026 |
| cpiri2026 | Zhang, Zhao, Li（编造） | Xu, Jiyuan; Zhang, Wenyu; Jing, Xin; Chen, Shuai; Zhang, Shuai; Nie, Jiahao，"CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting"，ICLR 2026 |
| lift2024 | Wu, Liu, Long（编造） | "Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators"，ICLR 2024（作者待核实） |
| mtsf_survey2025 | 作者编造 | "A Comprehensive Survey of Deep Learning for Multivariate Time Series Forecasting: A Channel Strategy Perspective"，arXiv 2502.10721（作者待核实） |
| kim2021revin | Kim, Taesung; **Choi, Jongmin; Bhoo…**（后三位不对） | 作者需按 ICLR 2022 论文核对（第一作者 Kim, Taesung 正确；发表年份为 2022） |
| timesfm3_blog2026 | 作者写成 {Google Research} | Jain, Ayush; Sen, Rajat（博客署名） |

**规则**：这个 bib 里**没有一条可以不经核对就用**，包括那些经典文献。所有条目一律从 DBLP、arXiv 或出版社页面导出 BibTeX，禁止让 AI 凭记忆生成。

---

## 2. 被它自己数据否定的结论

### 2.1 Finding 2："依赖微观相位同步"——数字对不上，结论也反了

稿件写的是：Weather 上 I3 的 MSE 为 0.1601、R=−1.18；I4 的 MSE 为 0.2284、R=1.33。`e2_interventions_grouped.csv` 里实际是：I3 为 **0.2052**，I4 为 **0.2048**，两者几乎一样。稿件里的数字在数据包里找不到出处。

另外，稿件里的 R 是"逐窗口比值的平均"。当某个窗口的 MV 和 UV 误差几乎相等时，分母接近 0，这个平均就会爆炸：Weather I3 的逐窗口 R 标准差为 44，最小值为 −1318。正确做法是**先对误差求和，再求比值**（方案 §3.3 的定义就是这样）。

我用逐窗口原始数据重新算了一遍（按窗口 bootstrap 2000 次，给出 95% 置信区间；结果见 `e2_recomputed.csv`）。Weather 上的多变量收益为 B=12.4% [3.1, 24.1]，各干预的依赖度 R 如下：

| 干预 | R | 95% CI | 含义 |
|---|---|---|---|
| I1 通道内部打乱时间顺序 | 0.97 | [0.53, 3.10] | 收益基本被抹掉 |
| N4 换成独立噪声 | **1.42** | [0.88, 3.64] | 比单变量还差：模型被干扰 |
| I3 按整周对齐换成另一时间段 | 0.48 | [0.11, 1.31] | 抹掉约一半 |
| I4 IAAFT 相位随机化 | 0.46 | [0.13, 1.30] | 抹掉约一半 |
| I2 不对齐换成另一时间段 | **0.18** | [−0.15, 0.75] | 大部分收益还在 |
| I6 平移 24 步 | **0.15** | [−0.27, 0.48] | 大部分收益还在 |
| N2 / N3 冗余副本 | 约 0.05 | 包含 0 | 无影响 |
| P1 领先指标（16 步 / 32 步） | −0.71 / −1.32 | 均小于 0 | 正对照成立：额外带来大幅收益 |

**正确的解读几乎和稿件相反**：只要其他通道是"看起来真实的同类信号"，即使和目标在时间上错开（I2、I6），Weather 的大部分收益仍然保留；只有把其他通道变成白噪声式的信号（I1、N4）才会失去收益。这更符合方案里的"池化而非读取"（H2），而不是"微观相位同步"。

两点注意：
- I2（不对齐）保留的收益反而比 I3（对齐）多，这与预期不符，而且置信区间重叠。这说明目前的样本量（100 个窗口 × 16 个目标）不足以区分两者，需要扩大样本后重跑。
- Electricity 在 E2 这批样本上 B=3.4% [−2.8, 8.6]，置信区间包含 0；ETTh1 上 B 为负。在收益不显著的数据集上，R 没有意义（分母可能为负或接近 0），稿件却照样解读了。

### 2.2 Finding 5："Top-K 优于随机，超过 32 个通道后饱和"——两句都不成立

- **Top-K 并不优于随机**：Electricity 上，K≥4 时随机选择在每一档 K 都最好（K=32：随机 0.13395，Top-K 0.13445，Bottom-K 0.13553）。稿件把随机的 0.1340 写成"Top-K 胜过它"。
- **"饱和"是实现造成的假象**：K=64、128 时，三种策略的 MSE 与 K=32 **完全相同，直到小数点后 5 位**。原因是这些实验走了官方评测器的 32 通道切块，目标所在的那一块始终是同样的 32 个通道，模型从来没有看到过超过 32 个通道。延迟从 53 ms 翻倍到 108 ms、再到 217 ms，也只是切块的数量从 1 块变成 2 块、4 块，不是什么"硬件不连续性"。
- 方案 §5.6 明确要求：K>32 时必须绕开评测器，直接调用 `TimesFM3Forecaster`。

### 2.3 Finding 4：Gap closure——oracle 本身是错的

在 39 个合成配置中：
- 有 **30 个配置里，模型的误差低于"理论最优"的多变量 Kalman oracle**；
- 有 **8 个配置里，多变量 oracle 比单变量 oracle 还差**。

在参数正确的线性高斯模型上，这两种情况都不可能出现。所以 oracle 的实现有错误（常见原因：oracle 和模型的误差不在同一尺度上，例如一个用了标准化数据、一个没用；oracle 用的参数与生成数据的参数不一致；多步预测公式有误）。因此所有 GC 数值（包括 −109、25.9、"closing 152% of the gap"）都不能用。

### 2.4 其他与数据或事实不符的地方

- **N1 通道置换**：稿件说"在所有数据集上 S<8.5e-8"。实际在 Electricity 上，置换后 MSE 变化了 −1.8%。原因是 N>32 时，打乱列顺序会改变目标所在的 32 通道块的成员，这已经不是纯粹的置换。N1 必须在固定的通道组内部打乱。
- **Wilcoxon 检验**：稿件写了 p<10⁻⁴、p<10⁻⁵，但数据包里没有任何检验结果，E1 也没有逐窗口数据。要么补算，要么删掉。
- **正对照 P1**：稿件说"16 步领先指标使误差下降 18.6%"。实际 16 步是 −10.0%，32 步才是 −18.6%。
- **Solar 的解释**："derails long-term rollouts / multi-patch autoregressive decoding"。TimesFM-3 是单次前向、非自回归的，这个解释不成立。
- **数据划分写反了**："ETT 和 Weather 用 7:1:2，Electricity、Solar、Traffic 用 12:4:4 个月"。实际是 ETT 用 12/4/4 个月，其余数据集用 7:1:2。
- **硬件写错**：稿件写"独立工作站，AMD Ryzen 9，PyTorch 2.4，CUDA 12.4"。实际是 AutoDL 云实例，Xeon Platinum 8352V，PyTorch 2.8，CUDA 12.8。
- **"Contiguous Patch Masking"不是分词方式**，而是解码方式：把预测区间作为掩码 token 接在上下文后面，一次前向算完。
- **"9,600 distinct evaluation windows"**：实际是每个数据集均匀抽取 300 个窗口（未用步长 1），8 个数据集 × 4 个 horizon 的配置数合计 9,600。正文必须如实写"每个数据集 300 个抽样窗口"。
- **"guaranteed zero-shot clean testbeds"**：GiftEvalPretrain 不含 ETT，只能说明"根据公开清单推断未见过"，不能说"保证"。
- **相关工作写错**：
  - Toto 2.0 不是非对称注意力，而是交替的时间注意力和全变量注意力，结构与 TimesFM-3 相同；非对称的是 TiRex-2。
  - Moirai 2.0 放弃跨变量注意力的原因，稿件写成"to avoid negative transfer / scalability"，原文没有这样说，只说目前不支持跨变量预测。
- **实践指南没有实验支撑**：ρ<0.35 的"谱门控阈值"、"节省 50% 计算"、"按 32 对齐 batch"都没有对应的实验，而且 32 对齐这一条正是建立在 §2.2 的假象之上。

---

## 3. 需要修复并重跑的实验

| 实验 | 问题 | 修复方式 |
|---|---|---|
| E2 | ① R 用逐窗口比值的平均；② N>32 时 N1 改变了分组；③ 样本太少（100 个窗口 × 16 个目标），收益不显著的数据集上 R 无意义 | ① R 按"先求和、再求比值"计算，并按窗口 bootstrap 给出置信区间；② 每个目标的上下文组先固定（官方同块的 31 个通道），所有干预都在这个组内进行；③ 高维数据集按方案用 64 个分层抽样的目标、至少 150 个窗口；B 的置信区间包含 0 的数据集只报告 Δe，不报告 R |
| E3 | oracle 错误 | 先写单元测试：在同一组数据上，① 多变量 oracle 的 MSE ≤ 单变量 oracle 的 MSE；② 用真实参数的 oracle 在大样本上的 MSE 等于理论预测误差方差（Riccati 方程的稳态解）；③ oracle 和模型在同一尺度上计算误差。通过测试之后再重跑 G1–G6 |
| E5 | K>32 的实验走了评测器切块；Top-K 结论写反 | K>32 时直接调用 `TimesFM3Forecaster.predict_batch`（不经过评测器）；延迟也改成测"一次前向放 K 个变量"的实际耗时 |
| E1 | 数据可用，但缺置信区间和检验；Solar 的结果需要诊断 | 输出逐窗口误差，补 bootstrap 置信区间和 Wilcoxon 检验。Solar 的均值收益为 −7.7%，但**中位数收益为 +7.6%**（H=96），说明退化集中在少数窗口：把逐窗口的 ΔMSE 与"窗口起点的时刻（夜间/日出/日落）"和"上下文通道的标准差"画在一起。怀疑夜间大量为 0 的通道经过逐通道归一化后数值被放大，再通过变量注意力扩散到目标通道 |
| E0 | 没有做 | 方案里的第一个判断节点，必须补 |
| E4 | 没有做 | 至少补 Chronos-2（开/关 group attention）和 PatchTST / iTransformer / CLOC 在 E1 上的数值；否则审稿人会问"只测一个模型能说明什么" |
| 数值检查 | WQL 数值在 2–4.4 之间，不像常规定义下的量级 | 核对 WQL 的计算公式（是否忘了除以 Σ\|y\|，或者多乘了分位数的个数） |

---

## 4. 目前数据能支持的发现（重跑后需要再确认）

1. **多变量收益因数据集而异，而且不总是正的**（E1，每个数据集 300 个抽样窗口，4 个 horizon 取平均）：Weather +4.5%，ETTh2 +6.6%，Electricity +2.5%，ETTh1 +1.6%，Traffic +1.0%，ETTm2 约 0，ETTm1 −1.7%，**Solar −19.5%**。
2. **Solar 的退化集中在少数窗口，而且随 horizon 加剧**：均值收益从 H=96 的 −7.7% 到 H=720 的 −32.0%；但中位数收益从 +7.6% 到 −18.1%。先查清机制，再写解释。
3. **与 CLOC 选出的工作点并不完全一致**：CLOC 在 ETT 上选择完全抑制通道通信，但 TimesFM-3 在 ETTh2 上有 +6.6% 的收益。这可以作为 H1 的部分反例写进论文（零样本模型与有监督模型的行为不同），是个有价值的点。
4. **Weather 上的收益更像是"池化"而不是"读取"**（§2.1），而且**模型容易被噪声通道干扰**（N4：R=1.42）。
5. **正对照成立**：领先指标（16 步 / 32 步）在 ETTh1（−10.9% / −21.9%）和 Weather（−10.0% / −18.6%）上都显著降低误差；Electricity 上 32 步显著（−6.8%），16 步（−2.5%）的置信区间包含 0。这说明模型"有能力"读取跨通道信息，标准数据上收益有限的原因在于数据本身。

---

## 5. 写作和图表的问题

**文字**
- 过度宣称："prove"、"catastrophic"、"strictly"、"guarantee"、"first comprehensive"、"causal mechanism discovery"。诊断性研究应当写"indicates"、"is consistent with"，并给出置信区间。
- Findings 只是在罗列数字，没有和假设、对照组联系起来，也没有写"这说明了什么、还有什么其他解释"。
- 用 R 去解释收益为负或不显著的数据集（ETTh1、Electricity）。
- 数据集名称大小写混乱（electricity / Electricity）；用"domain bimodality"这种说法，但 8 个数据集并没有呈现双峰分布。

**表格**
- 32 行的主表占了整整一页，信息密度很低。改成两张小表：① 每个数据集在 4 个 horizon 上平均的 MSE（UV / MV）加上 B 及其置信区间；② B 按"数据集 × horizon"排成矩阵（8×4）。
- 加粗规则不统一（MSE 和 MAE 分开加粗，两者冲突时读者不知道该看哪个）。

**图**
- 所有图里都有 "Fig. 1:" 这样的标题（LNCS 的图题应该写在 caption 里）。
- 图 1：32 个点的标签严重重叠；每个点一种颜色，但颜色没有任何含义；象限分界线 S=0.05 是随手画的；图例缺失。应当按数据集着色、用不同标记表示 horizon，或者只画数据集层面的 8 个点并加上置信区间。
- 图 2：把 3 个数据集混在一起求平均（包括 B<0 的 ETTh1），而且与 caption 说的"on Weather, ETTh1, Electricity"不符；没有误差棒。应当按数据集分面，只对 B 显著的数据集画 R，并加上置信区间。
- 图 3、图 4：建立在错误的 oracle 和切块假象之上，重跑之后重画。

---

## 6. 给 agent 的修复 prompt

```text
这份稿件与数据经过审查，发现以下必须修复的问题（审查报告全文见 review.md）。请按顺序处理，每一步完成后报告，不要跳步，也不要改写任何结论性的文字。

1. 参考文献：删除现有 references.bib，改用 review 附带的 references_fixed.bib。其中标了 VERIFY 的条目，逐条到 arXiv / DBLP / 出版社页面导出 BibTeX 替换，报告每条的来源链接。禁止凭记忆生成任何 bib 字段。

2. E2 重跑：
   - 每个目标的上下文组先固定（N≤32 用全部通道；N>32 用官方同块的 31 个通道），所有干预（包括 N1）都只在这个组内进行；
   - Electricity / Solar / Traffic 按"目标与其他通道平均 |相关系数|"的分位数分层抽 64 个目标，至少 150 个窗口；
   - R = (ΣMSE_π − ΣMSE_MV) / (ΣMSE_UV − ΣMSE_MV)，先求和再求比值；按窗口 bootstrap 2000 次给出 95% 置信区间；
   - B 的置信区间包含 0 的数据集，只报告 Δe = ΣMSE_π − ΣMSE_MV，不报告 R；
   - 输出逐窗口、逐目标的原始误差 CSV。

3. E3：先为 Kalman oracle 写单元测试，必须全部通过：
   (a) 多变量 oracle 的 MSE ≤ 单变量 oracle 的 MSE（大样本下）；
   (b) 用真实参数的 oracle，在 10^5 步样本上的 h 步预测 MSE 与 Riccati 稳态解给出的理论值相差 < 2%；
   (c) oracle 与模型的误差在同一尺度上计算（同样的标准化方式）。
   测试通过后重跑 G1–G6，并确认"模型 MSE 低于多变量 oracle"的情况不再出现（允许的偏差只能来自抽样误差，且要能用置信区间解释）。

4. E5：K>32 时直接调用 TimesFM3Forecaster.predict_batch，不经过 TimesFM3Evaluator 的 32 通道切块；延迟测量"一次前向放 K 个变量"的真实耗时。先检查 K=64 与 K=32 的 MSE 是否不再完全相同。

5. E1：输出逐窗口误差；补 bootstrap 95% 置信区间和 Wilcoxon 符号秩检验（Holm 校正）。诊断 Solar：把逐窗口 ΔMSE（MV−UV）与窗口起点的时刻、上下文各通道的标准差画在一起，报告退化是否集中在夜间或日出/日落附近的窗口。

6. 核对 WQL 的计算公式，给出代码行号和一个手算的小例子。

7. 补 E0（方案 §5.1）和 E4 中的 Chronos-2（开/关 group attention）以及 PatchTST、iTransformer、CLOC 在 E1 上的结果。

所有数字必须来自这次重跑生成的文件；报告里的每个数字都要注明来自哪个 CSV 的哪一行。
```
