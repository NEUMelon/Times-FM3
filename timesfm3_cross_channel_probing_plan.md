# 实验方案：Does Variate Attention Pay Off? Probing Cross-Channel Use in Multivariate Time-Series Foundation Models

> 版本 v1 · 2026-09-27 · 首投 DASFAA 2027（CCF-B，截稿 2026-11-25 23:59 AoE，16 页 LNCS **含参考文献和附录**，双盲）
> 代码事实均基于 `google-research/timesfm` 仓库 commit `e31dadd`（2026-09-15）与 HF `google/timesfm-3.0-pytorch` 的 `config.json`。动笔前请重新拉取并核对。

---

## 0. 一页总览

**核心论点**：TimesFM-3 的"多变量模式更强"是一个聚合排名结论。本文把它拆开：在同一个模型内部，通过开关模式、干预输入、可控合成结构和通道分组，分别回答**有没有收益**（benefit）、**是否真的读取了其他通道**（reliance / sensitivity）、**在什么结构下能读、什么结构下会被误导**，以及**高维数据该给它多少个、哪些通道**。

**调研中发现、会直接改变实验设计的事实**（详见 §1.1）：

1. **官方评测器每次前向最多只放 32 个变量**（`_MAX_VARIATES_PER_FORWARD = 32`，checkpoint 配置 `max_variates: 32`）。超过 32 个目标时按**列顺序**每 32 个切一块，最后一块用开头的通道**平铺补齐**。所以 Traffic（862 通道）在官方流程里是 27 个任意分组，并不是"每个通道都能看到其他所有通道"。原计划的"显存可能不够、需要分组"其实不是瓶颈（显存很宽裕），**真正的问题是预训练和官方流程的变量数上限都是 32**。
2. **变量注意力没有位置编码**（`use_rope_var: false`），对通道顺序置换等变。所以"打乱通道顺序"只能当负对照（实现正确性检查），不能当干预手段；有效的干预必须破坏**通道内容与时间的对齐**。
3. **在网络内部，"其他目标通道"和"过去协变量"完全等价**（`patch_is_target` 对两者都为 True，没有类型嵌入），只有"已知未来协变量"有前瞻 token。因此官方报告的多变量收益可能主要来自已知未来协变量任务，而不是多目标之间的跨通道信息。这个拆分本身就是本文的第一个实验（E0）。
4. **官方只公布了多变量模式的逐任务结果**（fev-bench 100 个任务、GIFT-Eval 97 个配置全部是 `use_variate_attention=true`），单变量模式只在博客里有一个聚合排名点。逐任务的收益分解目前是空白。
5. **官方基准里的"多变量"大多是低维的**：GIFT-Eval 的多变量配置最多 21 个变量；fev-bench 35 个多目标任务中 26 个没有动态协变量，14 个超过 32 个目标（最多 100 个），走的是切块流程。Electricity（321）、Solar（137）、Traffic（862）这种高维场景没有被官方的多变量结论覆盖。
6. **数据泄漏**：公开的 GiftEvalPretrain 清单包含 `traffic_hourly`（与 LTSF Traffic 同源的 862 个 PeMS 传感器）、PEMS03/04/07/08、PEMS_BAY、LOS_LOOP、LargeST（largest_2017–2021），不含 ETT。LTSF Traffic 极可能在 TimesFM-3 的预训练里出现过，需要单独处理（§4.6）。
7. **已发表的 CLOC 论文（Mathematics 2026, 14(17):3132）正文里没有合成数据生成器**。本文不能只写"沿用 CLOC 的生成器"，必须给出完整定义（§5.4 已给出可直接替换的形式化版本）。
8. **最接近的已有工作是 Falcon-X（arXiv 2605.27286）**：它报告 Chronos-2 在 GIFT-Eval 上开/关 group attention 表现几乎一样。必须引用并说明区别（§1.2、§10）。

**贡献（草案）**

- C1 一套同一模型内部的诊断协议：收益 / 依赖度 / 敏感度三个指标，加上正负对照、日历对齐交换、相位随机化替代序列。
- C2 （据本次调研）首个对 TimesFM-3 的系统性探测，并与 Chronos-2、Moirai-1.1、Toto-2.0 对照：多变量模式在哪里有用，在哪里"闲置"，在哪里有害。
- C3 带 oracle 的可控合成实验：量化零样本 TSFM 实际提取了多少"理论上可用"的跨通道信息（gap closure），并与有监督 CD 模型对比。
- C4 面向实践：高维数据的通道分组策略、精度-成本权衡、基于验证集的模式选择规则（契合 DASFAA 的数据管理受众）。

**实验总表**

| 编号 | 内容 | 回答 | 优先级 | RTX 4090 估算 |
|---|---|---|---|---|
| S | 实现正确性检查 S1–S7 | 协议可信 | P0 | < 1 h |
| E0 | 官方基准上的单/多变量收益分解（fev-bench、GIFT-Eval、TIME） | RQ1 | P0 | 6–8 h |
| E1 | LTSF 8 个数据集单/多变量对比 + 上下文长度扫描 | RQ1, H3 | P0（主表）/ P1（扫描） | 8–10 h |
| E2 | 干预测试（含正负对照） | RQ2 | P0 | 约 9 h |
| E3 | 合成数据 G1–G7 + oracle gap closure | RQ3 | P0（G1–G4, G5）/ P1（G6–G7） | 约 2 h + 有监督训练约 2 h |
| E4 | 对照模型（Chronos-2、Moirai-1.1、Toto-2.0；PatchTST、iTransformer、CLOC） | RQ5 | P0（Chronos-2 + 有监督）/ P1（Moirai、Toto） | TSFM 约 1 天；有监督 1–2 天 |
| E5 | 通道数量 K 与选择策略 + 官方切块的副作用 + 延迟 | RQ4 | P0（E5a/E5b）/ P1（E5c/E5d） | 约 12 h |
| E6 | 逐层变量注意力敲除、注意力与相关性对齐 | RQ2 机制 | P2 | 约 2 h |
| E7 | 收益可预测性回归 + 模式选择规则 | 实用结论 | P1 | 分析为主 |

**可行性判断**：硬件定为单张 RTX 4090（24 GB）。全部实验合计约 4–5 个 GPU·天（其中 TimesFM-3 约 2 天），算力不再是瓶颈，时间主要花在写代码和分析上。如果从 9/28 开始，可以在 11/8 冻结实验、留两周半写作。原计划"实验 2–3 周"只够完成 P0；全部做完需要 5–6 周（§8）。4090 多出来的余量建议用在：P1 全部做完、主要随机干预的种子从 3 个加到 5 个、高维数据集的干预实验也改用步长 1。

---

## 1. 调研：前沿现状与本文定位

### 1.1 TimesFM-3 代码级事实核查

| 事项 | 事实 | 来源 | 对实验的含义 |
|---|---|---|---|
| 发布与规模 | 2026-08-31 发布，330M 参数；权重使用 TimesFM Non-Commercial License v1.0 | Google Research 博客、HF 模型卡 | 学术用途没问题；论文里不要做可商用的表述 |
| 技术报告 | 截至 2026-09-27 没有单独的技术报告，HF 和 GitHub 仍引用 2023 年的 TimesFM 论文（arXiv 2310.10688） | HF 模型卡、GitHub README | 目前没有官方逐任务消融；需每周监测（§12） |
| 层结构 | 20 层，每层依次为：因果时间注意力（时间轴 RoPE）→ 变量注意力（非因果）→ FFN；d=1280，16 头，FFN 隐层 1280 | `torch/transformer.py` `MixingTransformer`；`config.json` | 共 20 个变量注意力子层（不是隔层交替），逐层敲除（E6）有 20 个位置可选 |
| 变量位置编码 | `use_rope_var: false` | `config.json` | 变量注意力对通道顺序置换等变 → 通道重排只能做负对照 S3 |
| 目标与协变量 | 目标和过去协变量的 `patch_is_target` 都为 True；只有已知未来协变量有前瞻 patch | `torch/model.py` 第 585 行附近 | 多目标 ≡ 过去协变量；E0 需要把"多目标"和"已知未来协变量"两种收益拆开 |
| 变量数上限 | 评测器 `_MAX_VARIATES_PER_FORWARD = 32`；`config.json` 中 `max_variates: 32` | `torch/evaluator.py` 第 29 行 | 预训练很可能最多只见过 32 个变量；K>32 是分布外（E5） |
| 官方切块规则 | 超过 32 个时按列顺序每 32 个一块；末块用开头的通道平铺补齐；协变量随机子采样到 ≤31 个（seed 42） | `torch/evaluator.py` 第 141–230 行 | 分组取决于数据集的列顺序；末块混入重复通道（E5c） |
| 评测默认值 | `use_symmetric_averaging=True`（对 x 和 −x 各推理一次再平均，计算量翻倍），`make_positive=True`（预测截断为 ≥0） | `torch/evaluator.py` 第 54–55 行 | **LTSF 的数据是标准化过的，必须设 `make_positive=False`**，否则负值会被截断 |
| 解码方式 | 单次前向（Contiguous Patch Masking）；预测区间以 32 步 patch 的掩码 token 追加；时间注意力是因果的 | `torch/model.py` decode | H=720 的前 96 步理论上应等于 H=96 的预测；验证后可一次前向覆盖 4 个 horizon（S5） |
| 数值精度 | fp32，SDPA 注意力；上下文上限 15,360 | `timesfm3_forecaster.py` | 4090 显存足够；如需 bf16 要自己加 autocast，并验证与 fp32 的误差 |
| 单变量模式 | `TimesFM3Evaluator.predict_batch(univariate=True)` 把每个通道拆成独立序列（v=1），变量注意力作用在单个 token 上 | `torch/evaluator.py` | 这才是博客里的 "univariate mode"；`use_variate_attention=False` 会跳过整个子层，属于分布外的结构敲除，只放在 E6 |
| 公开结果 | 仓库只有多变量模式的逐任务结果：fev-bench 100 个任务，GIFT-Eval 97 个配置 | `timesfm3-usage/benchmarks/*/…csv` | E0 的单变量模式逐任务结果需要自己跑 |
| 官方多变量规模 | GIFT-Eval 43 个多变量配置，最多 21 个变量；fev-bench 35 个多目标任务（26 个无动态协变量、9 个有），其中 14 个超过 32 个目标（最多 100 个） | 同上 CSV（本地统计） | 官方多变量结论基本来自低维任务；高维场景未验证 |
| 预训练数据 | GiftEvalPretrain（剔除与 fev-bench 重叠的部分）+ Wikipedia 页面浏览量 + Google Trends + 合成与增强数据 | HF 模型卡 | 见 §4.6 泄漏审计 |

### 1.2 多变量 TSFM 与"跨通道信息是否有用"的最新证据

| 工作 | 时间 | 关键内容 | 与本文的关系/区别 |
|---|---|---|---|
| Chronos-2（arXiv 2510.15821） | 2025-10 | group attention 在同一 patch 索引上跨组内序列做注意力，显存 O(V)；多变量能力来自合成 "multivariatizer"（同期变换、序贯依赖）；自述最大收益在带协变量的任务上 | 报告的是整体收益，没有做同一模型内的干预和结构可控实验 |
| **Falcon-X（arXiv 2605.27286）** | 2026-06 | 称 GIFT-Eval 上 Chronos-2 开/关 group attention 表现几乎相同；group attention 对完全不相似的输入产生几乎相同的注意力图（"semantic collapse"） | **最接近的已有观察**。但它只是作为提出新模型的动机，只用了一个基准，没有干预、正负对照、合成结构或分组分析，也没有涉及 TimesFM-3 |
| Moirai（arXiv 2402.02592） | 2024 | any-variate attention：把所有变量展平成一条序列，用两个可学习的二元偏置区分同/异变量，对变量置换等变；Small/Base/Large 为 14M/91M/311M | 计算量对 N·T 是平方级；高维数据必须分组 |
| Moirai 2.0（arXiv 2511.11698） | 2025-11 | 明确"不支持跨变量预测，把多变量当作独立单变量任务的集合" | 旁证：连 Moirai 的后续版本都放弃了跨变量注意力 |
| Toto 2.0（arXiv 2605.20119） | 2026 | 4M–2.5B 五个规模；交替的时间轴因果注意力和变量轴全注意力（与 TimesFM-3 同构）；论文称最优配置是把变量注意力放在最后一层；据论文，预训练只用 Datadog 内部数据和合成数据（仓库 README 提到的 Chronos 公开数据子集可能对应 1.0 版本，使用前需核实） | 与 TimesFM-3 结构同构，且很可能不含公开数据 → 理想的**无泄漏对照**；313M 版本规模接近 |
| TiRex-2（arXiv 2607.01204） | 2026-07 | 非对称分组注意力：目标可以读协变量，协变量不能读目标 | 可选对照；说明"让所有通道互相读"未必是最优设计 |
| RMISC（arXiv 2607.06504） | 2026-07 | 用真实多变量数据替代合成多变量化，可提升 Chronos-2 等模型的泛化 | 暗示仅靠合成数据学到的跨变量能力有限，支持本文的动机 |
| fev-bench（arXiv 2509.26468） | 2025-09 | 100 个任务，46 个含协变量，35 个多变量；原生多变量模型在多变量任务上更好 | 这是**跨模型**比较，混入了模型本身的差异；本文做的是同一模型内部的开关 |
| TIME（arXiv 2602.12147，ICML 2026） | 2026 | 50 个全新数据集、98 个任务，专为无泄漏零样本评测设计 | E0 的无泄漏补充集 |
| 应用侧证据 | 2026 | 电力需求（arXiv 2602.05390）：外生特征收益不稳定；金融（arXiv 2605.21504）：相关序列有益，混入不相关市场会降低精度；Żukowska 等（arXiv 2409.13530）：ETTm2/Weather 上通道独立反而更好 | 支持"收益有条件、会被干扰"的假设 |
| 可解释性 | 2026 | Dissecting Chronos（arXiv 2603.10071）用稀疏自编码器分析单变量 Chronos | 尚未见到对 TSFM 变量注意力做逐层敲除或干预的工作 |

### 1.3 CI/CD 之争（有监督模型）

- **Han 等（TKDE 2024, 36(11):7129–7142；arXiv 2304.05206）**：通道独立（CI）容量低，但对训练/测试分布漂移更稳健；提出 PRReg。本文在**不在目标数据上训练**的预训练模型里检验同样的权衡：漂移（G3）和干扰通道。
- **Abdelmalak 等（PAKDD 2026；arXiv 2502.09683）**：lookback 必须逐任务在 {96, 192, 336, 512, 720} 中调优，否则 CI/CD 的排名可能反转；7 个标准数据集的 Granger F 统计量均值只有 1.23–2.46，6 个 ODE 系统达到 35–335；标准数据集上 CI 在 28 次比较中赢 21 次，ODE 上 CD 在 24 次中赢 18 次；没有涉及基础模型。本文沿用它的 lookback 调优协议，并把"上下文长度"作为 TSFM 的扫描维度（H3），把 ODE 类强依赖作为合成对照（G7）。
- **CLOC（Song 等，Mathematics 2026, 14(17):3132，DOI 10.3390/math14173132）**：把通道通信作为可选的工作点 (g, K, α, β)；在验证集上选择的结果是 ETT 全抑制（CLOC-0），Electricity/Solar 局部+全局，Weather 只用很小的全局摘要，Traffic 以局部为主；lookback 统一为 720。本文检验：TimesFM-3 的多变量收益模式是否与 CLOC 选出的工作点一致（H1）。
- **CPiRi（ICLR 2026；arXiv 2601.20318）**：CD 模型容易过拟合通道顺序。**LIFT（ICLR 2024；arXiv 2401.17548）**：领先指标（lead-lag）是跨通道信息的主要来源之一。**综述（arXiv 2502.10721）**：从通道策略角度梳理 MTSF。

### 1.4 空白与本文定位

已有工作要么是**跨模型比较**（fev-bench 等，混入了模型本身的差异），要么是**有监督模型**的 CI/CD 研究（Han、PAKDD、CLOC），要么是**顺带一提的观察**（Falcon-X）。目前还没有人在**同一个预训练多变量 TSFM 内部**，用干预和可控结构系统地回答：它有没有读其他通道、读到的是什么（真实的同步变化，还是共享的日历/季节相位）、什么时候会被误导，以及高维数据怎样分组。TimesFM-3 刚发布、没有技术报告，而且官方评测流程里有 32 变量切块这个没人讨论过的细节，时机合适。

### 1.5 需要特别处理的调研发现

1. **合成生成器需要重新完整定义**：CLOC 已发表的版本里没有合成实验。§5.4 给出了四种生成器的形式化定义，如果你们手上的 CLOC 代码与之不同，**以代码为准，但论文里要写全公式和参数**。
2. **COI 与匿名**：CLOC 的作者单位是东北大学（沈阳），DASFAA 2027 在沈阳举办，大会主席也来自东北大学。如果作者团队与该单位有关，投稿时要按 COI 规则申报同单位的 PC 成员；正文只用第三人称引用 CLOC（"Song et al. [x] propose…"），代码使用匿名仓库。
3. **Falcon-X 必须引用**，并在引言里正面说明区别。
4. **TimesFM-3 技术报告可能在投稿前发布**：应对方案见 §12。

---

## 2. 研究问题与可检验假设

- **RQ1（收益）**：多变量模式相比单变量模式，在标准基准上有没有收益？收益集中在哪类任务（多目标 vs 协变量、低维 vs 高维、短 vs 长上下文）？
- **RQ2（是否在读）**：模型的预测是否依赖其他通道**与时间对齐的内容**？读到的是真实的同步变化，还是共享的季节相位或尺度信息？
- **RQ3（结构）**：在已知结构的合成数据上，它能提取多少理论上可用的跨通道信息？结构漂移造成的虚假相关会不会误导它？
- **RQ4（多少、哪些通道）**：给多少个其他通道、按什么规则选，收益最大？超过 32 个变量会怎样？官方的列顺序切块有没有副作用？
- **RQ5（对照）**：上述行为在 Chronos-2、Moirai-1.1、Toto-2.0 和有监督的 CD 模型里是否一致？

| 假设 | 预测 | 证伪条件 | 实验 |
|---|---|---|---|
| H1 数据依赖 | ETT 上多变量收益 ≈0 或为负；Electricity/Solar/Traffic 为正；模式与 CLOC 选出的工作点一致 | ETT 上出现显著且稳定的正收益，或高维数据集上没有收益 | E1 |
| H2 读到的是"池化"而非"同步变化" | 在标准数据集上，把其他通道换成**日历对齐**的别的时间段后，多变量收益大部分仍然保留 | 日历对齐交换后收益被完全破坏（R≈1） | E2 |
| H3 替代历史 | 上下文越长，多变量收益越小（其他通道在补偿自身历史的不足） | 收益随上下文长度不变或增大 | E1 扫描 |
| H4 数量非单调 | 收益随 K 先升后降；按相关性挑选优于随机和列顺序切块；K>32 时注意力被稀释、收益下降 | 收益随 K 单调上升，且 K>32 无退化 | E5 |
| H5 被漂移误导 | 在结构漂移数据（G3）上，当变化点位于预测起点时，多变量模式比单变量模式更差，且上下文里的相关性越强越差 | 多变量模式在漂移下始终不差于单变量模式 | E3 |
| H6 能力存在 | 正对照：加入一个领先指标通道（其值等于目标未来值加噪声）时，误差大幅下降 → 说明模型**能**读跨通道信息；如果标准数据上依赖度很低，原因在数据而不在能力 | 正对照也没有收益（说明变量注意力无法传递这类信息，此时需先排查实现） | E2 |
| H7 滞后 | 滞后 ℓ ≤ 32（一个 patch 内）的领先关系比跨多个 patch 的滞后更容易被利用 | 各滞后下 gap closure 没有差异 | E3（G5） |

---

## 3. 度量框架

### 3.1 记号

窗口 $w$，目标通道 $i$，预测长度 $H$。$\hat y^{\text{MV}}_{w,i}$、$\hat y^{\text{UV}}_{w,i}$、$\hat y^{\pi}_{w,i}$ 分别是多变量模式、单变量模式、以及在干预 $\pi$（只改其他通道，目标通道本身不动）下的预测；$e(\cdot)$ 为 MSE（标准化尺度），并同时报告 MAE、MASE 和 WQL。

### 3.2 收益（Benefit）

$$B = \frac{\sum_{w,i} e^{\text{UV}}_{w,i} - \sum_{w,i} e^{\text{MV}}_{w,i}}{\sum_{w,i} e^{\text{UV}}_{w,i}}$$

$B>0$ 表示多变量模式有帮助。同时报告按窗口的中位数相对变化，防止少数极端窗口主导结果。

### 3.3 依赖度（Reliance，收益归因比）

$$R_\pi = \frac{e^{\pi} - e^{\text{MV}}}{e^{\text{UV}} - e^{\text{MV}}}$$

- $R_\pi \approx 1$：干预抹掉了全部多变量收益 → 收益来自被 $\pi$ 破坏的那部分信息。
- $R_\pi \approx 0$：干预后收益仍在 → 收益并不来自被 $\pi$ 破坏的信息。
- $R_\pi > 1$：干预后比单变量还差 → 模型被损坏的通道**带偏**了。
- 只在 $B$ 显著的（数据集，horizon）上报告 $R$；$B$ 不显著时改为直接报告 $\Delta e = e^{\pi}-e^{\text{MV}}$，避免分母接近 0。

### 3.4 敏感度（Sensitivity，与精度无关）

$$S_\pi = \frac{\operatorname{mean}_{w,i,h}\,|\hat y^{\pi} - \hat y^{\text{MV}}|}{\sigma_i},\qquad S_{\text{UV}} = \frac{\operatorname{mean}\,|\hat y^{\text{UV}} - \hat y^{\text{MV}}|}{\sigma_i}$$

$\sigma_i$ 是训练集上的标准差。$S$ 衡量模型**是否在读**其他通道，而不管读了之后预测变好还是变差。

### 3.5 四象限分类（论文的核心图）

以 $S_{\text{UV}}$ 为横轴、$B$ 为纵轴，每个点是（模型，数据集）：

| | $B$ 显著为正 | $B\approx 0$ | $B$ 显著为负 |
|---|---|---|---|
| **$S$ 高** | 利用（exploits） | 在读但没用（noisy reading） | 被干扰（distracted） |
| **$S$ 低** | 理论上很少出现（若出现，先检查实现和显著性阈值） | 闲置（idle） | — |

### 3.6 合成数据上的 oracle gap closure

对线性高斯生成器（G1–G5），用真实参数的 Kalman 滤波构造两个 oracle：只观测目标通道的最优预测器 $e^{\text{UV}}_{\star}$，以及观测全部通道的最优预测器 $e^{\text{MV}}_{\star}$。

$$\mathrm{GC} = \frac{e^{\text{UV}}_{\text{model}} - e^{\text{MV}}_{\text{model}}}{e^{\text{UV}}_{\star} - e^{\text{MV}}_{\star}}$$

GC≈1 表示提取了全部可用的跨通道信息；GC≈0 表示忽略；GC<0 表示受损。这个指标把"数据里本来有多少跨通道信息"和"模型用到了多少"分开，是本文区别于以往 CI/CD 对比的关键。

### 3.7 统计检验

- **配对设计**：所有比较都在同一批窗口、同一批通道上进行。
- **置信区间**：按窗口做分块 bootstrap（块长 = 一个季节周期对应的窗口数，重抽样 2000 次），给出 95% 置信区间。
- **显著性**：按窗口聚合的误差做 Wilcoxon 符号秩检验，对（数据集 × horizon × 条件）用 Holm–Bonferroni 校正。
- **随机干预**：每种随机干预至少 3 个随机种子，报告种子间的标准差。
- **实际意义阈值**：相对 MSE 变化小于 1% 视为"无实际差异"，即使统计上显著。

---

## 4. 通用实验协议

### 4.1 数据集

| 数据集 | 通道 N | 频率 | 长度 | 划分 | CLOC 选出的工作点 | TimesFM-3 泄漏风险 |
|---|---|---|---|---|---|---|
| ETTh1 / ETTh2 | 7 | 1h | 17,420 | 12/4/4 个月 | 全抑制 | 低（GiftEvalPretrain 不含 ETT） |
| ETTm1 / ETTm2 | 7 | 15min | 69,680 | 12/4/4 个月 | 全抑制 | 低 |
| Weather | 21 | 10min | 52,696 | 7:1:2 | 极小的全局摘要 | 低（预训练里的 `weather` 是 Monash 的澳大利亚数据，不是 Jena） |
| Solar-Energy | 137 | 10min | 52,560 | 7:1:2 | 局部 + 全局 | 低（预训练里的 `solar_power` 是另一份数据；GIFT-Eval 的 solar 测试集已被剔除） |
| Electricity | 321 | 1h | 26,304 | 7:1:2 | 局部 + 全局 | 低（预训练里只有 elecdemand / 澳大利亚电力需求） |
| Traffic | 862 | 1h | 17,544 | 7:1:2 | 以局部为主 | **高**（`traffic_hourly` 与之同源） |

注：泄漏风险一列是根据公开的 GiftEvalPretrain 清单推断的。Google 实际使用的子集（"剔除与 fev-bench 重叠的数据集"之后）没有公开，而且还有合成和增强数据，所以只能作为推断，Traffic 需按 §4.6 做逐值比对。

基准任务集（E0）：fev-bench（100 个任务）、GIFT-Eval（43 个多变量配置）、TIME（98 个任务，专门设计为无泄漏）。

可选的"新鲜数据"（P2）：2025–2026 年新采集的 PeMS 或 ENTSO-E 多区域负荷数据，时间晚于所有模型的预训练截止点，用来排除 Traffic 上的记忆效应。

### 4.2 窗口、预测长度与上下文

- **预测长度**：{96, 192, 336, 720}。先通过 S5 验证 H=720 预测的前缀与短 horizon 的预测一致，然后每个窗口只做一次 H=720 的前向，再截取；H<720 标准协议里多出来的尾部窗口单独补跑。
- **窗口步长**：主表（E1 的单/多变量对比）在全部 8 个数据集上用**步长 1**（与 LTSF 协议一致，4090 上约 5–6 h）。扫描和干预实验在 Solar/Electricity/Traffic 上**均匀抽取 300 个（或 150 个）窗口**，ETT/Weather 抽 500 个；并在一个数据集上抽查"抽样结论与步长 1 一致"。
- **上下文**：主结果用 512（与 TSFM 常见设置一致），稳健性检查用 2048；E1 扫描 {96, 192, 336, 512, 720, 1024, 2048, 4096}。有监督基线的 lookback 按 PAKDD 协议在验证集上调优（§5.5）。TSFM 同时报告"固定 512"和"验证集上最优上下文"两种结果，保证公平。

### 4.3 输入尺度与模型设置（容易踩坑的地方）

1. 按 LTSF 协议用训练集的均值和标准差标准化，指标在标准化尺度上计算。TimesFM-3 自带逐变量的 RevIN 和线性去趋势，输入标准化后的值没有问题。
2. **必须设 `make_positive=False`**（评测器默认是 True，会把标准化后的负值截断成 0）。
3. `use_symmetric_averaging`：主表保持官方默认值 True，**所有模式、所有条件保持一致**。扫描实验可以关掉以省一半计算量，前提是先在两个数据集上验证关掉后单/多变量的差值不变。
4. 固定版本：TimesFM 仓库 commit、HF checkpoint 的 revision、各对照模型的版本号；打开 `torch.use_deterministic_algorithms`，记录 CUDA 和 PyTorch 版本。
5. 所有随机干预和随机分组使用固定的种子列表，结果写入以（模型，数据集，条件，种子）为键的缓存（§7.2）。

### 4.4 指标

- 点预测：MSE、MAE（标准化尺度，与 LTSF 文献可比）。
- 概率预测：WQL（9 个分位数）、10%–90% 区间覆盖率；另报 MASE，以便和 fev-bench / GIFT-Eval 对照。
- 诊断指标：B、R、S、GC（§3）。

### 4.5 正确性检查（第 1 周完成，全部写进附录或匿名仓库）

| 编号 | 检查 | 通过标准 |
|---|---|---|
| S1 | 用官方脚本复现 fev-bench 和 GIFT-Eval 的若干任务 | 与仓库 CSV 中的数值相差小于 1e-3（相对） |
| S2 | 多变量模式下只输入 1 个通道 ≡ 单变量模式 | 最大绝对差小于 1e-5 |
| S3 | 组内通道顺序打乱 | 预测只差一个同样的置换（误差小于 1e-5） |
| S4 | 其他通道作为"目标"输入 vs 作为"过去协变量"输入 | 目标通道的预测一致（验证 §1.1 的等价性） |
| S5 | H=720 预测的前 96/192/336 步 vs 分别用对应 H 预测 | 一致（验证后才能复用） |
| S6 | 同一 batch 中其他样本的内容、长度是否影响本样本的预测 | 不影响 |
| S7 | 重复运行的确定性 | 结果逐位一致，或差异小于 1e-6 |

### 4.6 数据泄漏审计

1. **Traffic**：从 GiftEvalPretrain 下载 `traffic_hourly`，与 LTSF 的 `traffic.csv` 逐值比对。如果一致（很可能一致），在论文里把 Traffic 标为"可能见过"，主要结论以其他数据集为准；Traffic 的结果用 Toto-2.0（据其论文不含公开数据）和新鲜数据（P2）作交叉验证。
2. **对照模型**：Chronos-2 的训练数据是"Chronos 数据集的子集，剔除了与 GIFT-Eval 测试部分重叠的数据"（HF 模型卡）；Moirai-1.1 用的 LOTSA 很可能也收录了 Monash 的 `traffic_hourly`（需按其数据清单核实）。逐个模型、逐个数据集填一张泄漏表，放进论文。
3. 泄漏会影响绝对精度，但同一模型内部的干预对比仍然有意义；需要额外说明的是，记忆可能**放大**多变量收益（模型可能记住了通道之间的联合模式），这正是 Traffic 需要交叉验证的原因。

---

## 5. 实验详细设计

### 5.1 E0：官方基准上的收益分解（P0，第 1 周，也是继续做下去的判断依据）

**目的**：把博客里"多变量模式再进一步"的聚合结论，拆成"多目标之间的跨通道收益"和"协变量收益"。

**设计**：对每个任务做 2×2 析因，外加一个协变量细分：

| 条件 | 其他目标通道 | 协变量 |
|---|---|---|
| UV | 无（每个目标单独推理） | 无 |
| MV-targets | 有 | 无 |
| UV+cov | 无 | 有 |
| MV-full（官方） | 有 | 有 |

- 带协变量的任务再把"过去协变量"和"已知未来协变量"分开开关，检验收益是否主要来自已知未来协变量的前瞻 token。
- 注意：评测器的 `univariate=True` 会**同时丢掉协变量**，所以 UV+cov 条件需要自己写一个包装：逐个目标调用 forecaster，同时传入协变量。
- 任务分层：（a）26 个多目标、无动态协变量的任务；（b）9 个多目标、有动态协变量的任务；（c）33 个单目标、有动态协变量的任务；（d）若干纯单变量任务（MV 与 UV 按构造应该完全相同，作为额外的正确性检查）。（a）中 14 个超过 32 个目标的任务单独列出，因为它们走切块流程。
- GIFT-Eval 的 43 个多变量配置（ETT、jena_weather、bitbrains、bizitobs）做 UV vs MV；TIME 做同样的对比，作为无泄漏的补充。

**报告**：fev-bench 的 skill score 与 win rate（1000 次 bootstrap 的 95% 置信区间），按层 a–d 分别给出；逐任务 SQL 的 MV-vs-UV 散点图。

**判断节点（10/4）**：
- 如果层（a）的收益显著且可观：叙事为"收益真实但有条件"，重点放在 E2/E3，解释收益从哪里来。
- 如果层（a）的收益接近 0、主要收益在协变量任务上：叙事为"官方的多变量收益主要来自协变量，多目标之间的变量注意力基本闲置"，E2/E3 用来解释为什么闲置。
- 两种结果都能写成论文，因为本文本质上是诊断性研究，不依赖某一个特定结果。

### 5.2 E1：LTSF 主实验 + 上下文长度扫描（P0）

- **主表**：8 个数据集 × 4 个 horizon × {UV, MV}，上下文 512，窗口步长 1。高维数据集上的 MV 使用**官方协议**（列顺序 32 通道切块）；其他分组方式放在 E5。同一张表里放入 E4 的对照模型。
- **上下文扫描**：{96, 192, 336, 512, 720, 1024, 2048, 4096} × {UV, MV}，每个数据集 300 个窗口。画出 B 随上下文长度的变化曲线，检验 H3（与 PAKDD 的 lookback 论点呼应）。
- **horizon 趋势**：B 随 H 的变化。跨通道信息如果主要帮助估计当前状态，收益应该集中在短 horizon。
- **与 CLOC 的对照**：把每个数据集的 B 与 CLOC 选出的工作点（全抑制 / 局部 / 全局）放在一起，检验 H1。

### 5.3 E2：干预测试（P0）

**以目标为中心的设计**：干预只改"上下文通道"，目标通道本身保持原样，只评估目标通道的误差。

- ETT、Weather：对每个目标 i，一组输入就是全部 N 个通道，干预作用在其余 N−1 个通道上。
- Solar、Electricity、Traffic：按"目标与其他通道平均 |相关系数|"的分位数分层抽 64 个目标通道；每个目标的上下文集合 = 它在官方切块中的同块 31 个通道（这样未干预时的 MV 正好等于官方协议）。再用"最相关的 31 个通道"作为上下文集合重复一遍（E5 的最佳分组），检验在最有利的分组下 H2 是否仍然成立。

**干预清单**

| 编号 | 干预 | 保留了什么 | 破坏了什么 | 预期 / 用途 |
|---|---|---|---|---|
| N1 | 组内通道顺序打乱 | 全部 | 无 | 负对照，必须完全不变（S3） |
| N2 | 加入目标本身的一个副本 | 全部 | 无新信息 | 负对照；也直接检验官方切块"末块用重复通道补齐"的影响 |
| N3 | 加入目标自身的滞后副本 x(t)=y(t−k) | 全部 | 无新信息 | 负对照：只有冗余信息时预测不应改变 |
| N4 | 额外加入与原通道均值、方差匹配的独立噪声通道 | 原有通道 | 无 | 干扰物测试：理想情况下 S 很小、B 不下降；否则就是模型被干扰的直接证据 |
| I1 | 上下文通道**内部**的时间点随机打乱 | 边缘分布 | 全部时间结构 | 最强的破坏，但输入会变得"不自然"（分布外），只作参考 |
| I2 | 换成同一通道在另一时间段的片段（与原窗口相距 ≥30 天，且**不**按日/周相位对齐） | 各通道自身的动态 | 与目标的对齐、日历相位 | 检验收益是否依赖时间对齐 |
| I3 | 同 I2，但时间偏移取整周的倍数（**日历对齐**） | 各通道自身的动态 + 日/周相位 | 具体的同步变化（天气事件、事故等） | **关键干预**：I3 后收益仍在 → 模型用的是共享的季节相位（H2） |
| I4 | 每个上下文通道独立做 IAAFT 相位随机化替代 | 各通道的频谱、自相关和边缘分布 | 通道间的相位关系 | 比 I1 更"自然"的去对齐干预 |
| I5 | 换成另一个数据集的通道（标准化、截取成相同长度） | 真实时序的外观 | 全部相关信息 | 分布外干扰物 |
| I6 | 上下文通道循环平移 δ ∈ {1, 4, 24, 168} 步（小时数据；10 分钟数据相应换算） | 各通道自身的动态 | 细粒度的对齐（δ=24/168 时日/周相位保留） | 画出 S 和 R 随 δ 的曲线，看模型读取对齐的"时间精度" |
| P1 | 用领先指标替换一个上下文通道：x(t)=y_i(t+k)+ε，k ∈ {1, 8, 32, 96}，信噪比 ∈ {∞, 10, 1} | — | — | **正对照**：误差应大幅下降；如果不降，说明变量注意力无法传递这类信息（先排查实现） |

**报告**：各数据集的 R_π 条形图（带置信区间）；S 和 R 随 δ 的曲线；正对照的误差随 k 和信噪比的变化；干预对 WQL 和区间覆盖率的影响（干扰通道可能让区间变宽或变窄）。

**解读示例**：如果 I2 的 R≈1 而 I3 的 R≈0，说明模型从其他通道读到的是日/周相位，而不是具体的同步变化，也就是"池化"而非"读取"；如果 I4 的 R≈1，说明模型确实依赖通道间对齐的线性结构。

### 5.4 E3：合成数据（P0：G1–G5；P1：G6–G7）

**统一构造**：每个通道 = 自身分量 + 跨通道分量 + 观测噪声，全部为线性高斯状态空间模型，便于用 Kalman 滤波计算 oracle。

$$x_i(t)=\sqrt{1-\gamma}\,\tilde u_i(t)+\sqrt{\gamma}\,\tilde c_i(t)+\sigma_\varepsilon\,\varepsilon_i(t)$$

- $u_i$：自身分量，AR(2)（随机抽取的平稳系数）加周期 P ∈ {24, 168} 的季节项（二维旋转状态，随机相位）；$\tilde u_i$、$\tilde c_i$ 按理论标准差缩放到单位方差。
- $\gamma$ ∈ {0, 0.25, 0.5, 0.75}：跨通道分量的占比（耦合强度）；$\sigma_\varepsilon$ ∈ {0.1, 0.5}。
- 保持自身可预测性不变，只改变跨通道信息量，这样 B 和 GC 的变化可以归因于结构本身。

| 生成器 | 跨通道分量 $c(t)$ | 扫描参数 | 检验什么 |
|---|---|---|---|
| G1 全局低秩 | $c(t)=\Lambda f(t)$，$f_k(t)=0.95 f_k(t-1)+\eta_k(t)$，$\Lambda$ 为按行归一化的高斯矩阵 | 秩 r ∈ {1, 2, 4, 8}；N ∈ {8, 32}（另加 128，测 K>32） | 能否利用其他通道估计共享因子 |
| G2 局部块 | 通道分成大小为 b 的块，每块有自己的因子，块间独立 | b ∈ {2, 4, 8}；通道顺序：块连续排列 vs 随机打乱（N=128 时与官方切块交互） | 能否找到块结构；列顺序切块会不会切断块 |
| G3 结构漂移 | $\Lambda(t)=\Lambda_1$（$t<\tau$），$\Lambda_2=\sqrt{1-\delta}\Lambda_1+\sqrt{\delta}\Lambda'$（$t\ge\tau$）；另设一个符号翻转变体 | δ ∈ {0, 0.25, 0.5, 1}；τ ∈ {上下文中点, 预测起点} | τ=预测起点时，上下文里的相关性对预测区间是误导性的，检验 H5 |
| G4 组合 | G1 + G2 + G3（δ=0.5） | 少量代表性配置 | 接近真实数据的混合结构 |
| G5 领先滞后 | $y(t)=a\,x_j(t-\ell)+\text{自身分量}+\text{噪声}$ | ℓ ∈ {1, 8, 16, 32, 64, 96, 192}；a 取 3 档 | 滞后与 patch 长度（32）的关系，检验 H7 |
| G6 零结构 | γ=0，通道相互独立 | N ∈ {8, 32, 128} | 纯干扰基线，B 应为 0 |
| G7 混沌 ODE | Lorenz-96：$\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F$，F=8 | N ∈ {8, 16, 32}；观测噪声 2 档 | 非线性强耦合（对应 PAKDD 的 ODE 数据）；没有 Kalman oracle，用域内训练的 CD 模型作经验上界 |

**oracle**：G1、G2、G5、G6 用真实参数的 Kalman 滤波，分别计算只观测目标通道和观测全部通道的最优预测，得到 GC（§3.6）。G3 同时计算"不知道变化点"的朴素 oracle：模型在漂移下的损失（$e^{\text{MV}}-e^{\text{UV}}$）要和朴素 oracle 的同类损失比较，区分"数据本身不可预测"和"模型被误导"。

**规模**：每个配置 3 个生成器种子 × 200 个窗口，上下文 512，H ∈ {96, 336}；约 140 个配置，TimesFM-3 合计约 5 GPU·h。

**有监督对照**：挑 24 个代表性配置，在每个配置的 2 万步训练序列上训练 PatchTST（CI）、iTransformer（CD）、CLOC 和 CLOC-0，在同一过程的测试段上评估。它们给出"在域内训练时能利用多少结构"的参考，与零样本 TSFM 对比。

**主图**：GC 热图（生成器 × γ），TimesFM-3 / Chronos-2 / Toto-2.0 / 有监督 CD 模型并排；G3 的"多变量损失 vs δ"曲线；G5 的 GC 随 ℓ 的变化曲线。

### 5.5 E4：对照模型（P0：Chronos-2 + 有监督模型；P1：Moirai、Toto-2.0）

| 模型 | 规模 | 跨通道机制 | MV / UV 开关方式 | 备注 |
|---|---|---|---|---|
| TimesFM-3 | 330M | 每层都有全变量注意力，无位置编码 | 评测器的 `univariate` 参数，或自写包装 | 主角 |
| Chronos-2 | 120M | 同一 patch 索引上的 group attention | 所有通道作为同一个条目的多个目标（同组）vs 每个通道单独成为一个条目 | Apache-2.0；上下文最长 8192 |
| Moirai-1.1-R（base 91M；可选 large 311M） | 同左 | any-variate attention（展平 + 二元变量偏置） | 一次输入全部变量 vs 每个变量单独输入 | 计算量对 N·T 为平方级，高维时必须用与 TimesFM-3 相同的 32 通道分组 |
| Toto-2.0-313m（可选 22m，看规模效应） | 313M | 交替的时间因果注意力 / 变量全注意力 | 输入形状 (batch, n_variates, T) 中 n_variates=N vs 1 | 据论文预训练不含公开时序 → 无泄漏对照（需核实） |
| TimesFM-2.5（可选） | 200M | 无（单变量） | — | 看 TimesFM-3 的单变量模式本身是否已经更强 |
| PatchTST | 小 | CI | — | 有监督 |
| iTransformer | 小 | CD（变量作为 token） | — | 有监督；测试时同样做 E2 干预 |
| CLOC / CLOC-0 | 小 | 可控的局部 + 全局通信 / 零通信端点 | CLOC vs CLOC-0 就是有监督版本的 MV vs UV | 第三人称引用；测试时同样做 E2 干预 |
| DLinear（可选） | 极小 | CI | — | 便宜的下界 |

**公平性**
- 所有 TSFM 在高维数据上使用**完全相同的分组**（相同的 32 通道划分、相同的 64 个抽样目标、相同的干预种子）。
- 有监督模型的 lookback：先在 H=96 上用验证集从 {96, 336, 720} 中选（算力允许就用 PAKDD 的完整集合 {96, 192, 336, 512, 720}），再固定用于其他 horizon；最终配置跑 3 个种子。Traffic 上的 PatchTST 需要把 batch size 调小（CI 会把 batch 放大 862 倍），显存主要被激活值占用。
- 有监督 CD 模型同样做 E2 的干预：可以检验 CLOC 在 ETT 上选择零通信时 R 是否按构造为 0（一致性检查），也可以对比"在域内训练过的 CD 模型"与"零样本 TSFM"的依赖度谁更高。
- P1 的 TSFM 使用精简协议：E1 主表 + E2（I2、I3、I4、P1）+ E3（G1、G3、G5）+ E5a（只在 Electricity 上）。

### 5.6 E5：给多少个通道、给哪些通道（P0）

原计划里的"Traffic 862 通道可能超显存"并不成立（K=862、上下文 512、H=720 时一次前向约 3.4 万个 token，开启对称平均后翻倍；在 SDPA 下，fp32 峰值显存仍只有几 GB，24 GB 的 4090 绰绰有余）。真正的问题是**预训练和官方流程的变量数上限都是 32**，所以这一节改为回答"分组"问题。

**E5a 划分设计**（每个 K 都相当于跑一次完整的 MV 推理）
- K ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, N}；分组策略：列顺序切块（官方）、随机（3 个种子）、基于训练集 |相关系数| 的均衡聚类。
- 数据集：Solar、Electricity、Traffic（可选 fev-bench 中 boomlet 那几个 35–100 个目标的任务）。
- K>32 时绕过评测器的切块，直接调用 `TimesFM3Forecaster`。

**E5b 以目标为中心的选择**（64 个抽样目标）
- 上下文集合 = 最相关的 K−1 个 / 最不相关的 K−1 个 / 随机 K−1 个；K ∈ {2, 4, 8, 16, 32}。

**E5c 官方切块的副作用**
- 列顺序敏感性：把列顺序随机打乱 5 次再按官方规则切块，看数据集级 MSE 的方差（如果模型不读其他通道，方差应为 0）。
- 末块补齐：末块用重复通道补齐 vs 不补齐（直接用较少的通道推理），看末块通道的预测变化多大。

**E5d 成本**：4090 上每个窗口的延迟、峰值显存、吞吐量（序列/秒）随 K 的变化，画出精度-成本的帕累托前沿。这部分对 DASFAA 的数据管理受众很有价值。

**附带的注意力诊断**：通过 hook 记录变量注意力的熵随 K 的变化，检验"K>32 时注意力被稀释"（H4）。

### 5.7 E6：机制分析（P2，篇幅不够时只放一张小图）

- **逐层敲除**：在某些层把变量注意力的掩码改成只看自己（对角掩码，相当于该层退化为单变量计算），分别敲除单层、前 ℓ 层、后 ℓ 层，看 B 还剩多少，定位跨通道信息从哪几层进入。可以和 Toto-2.0"变量注意力放在最后一层最好"的发现对照。
- **注意力与相关性的对齐**：目标到各通道的注意力权重（对头、层、patch 取平均）与上下文窗口内 |相关系数| 的 Spearman 相关。
- **注意力坍缩检验**（对应 Falcon-X 的说法）：比较真实输入与 I5（跨数据集替换）下注意力矩阵的余弦相似度；如果两者几乎一样，说明注意力并没有按内容分配。
- 说明：注意力权重不等于解释（Jain & Wallace, NAACL 2019），所以这部分只作为干预结果的补充证据。

### 5.8 E7：什么时候有用——可预测性分析与模式选择规则（P1）

- **特征**（只用上下文窗口内可计算的量，部署时也能用）：目标与其他通道的平均/最大 |相关系数|；上下文前半段和后半段相关系数的变化（稳定性）；上下文协方差矩阵第一主成分的解释方差比例（低秩程度）；主周期上的谱相干（共享季节性）；N/K；训练集层面的 Granger F（沿用 PAKDD 的做法）。
- **模型**：梯度提升树或线性混合模型（数据集作随机效应），预测每个窗口的相对收益；按"留一个数据集"做交叉验证。
- **模式选择规则**：（a）数据集级：在验证集上选 MV 或 UV（与 CLOC 在验证集上选工作点的做法一致）；（b）窗口级：预测收益超过阈值才用 MV。在测试集上报告实际收益，并与"总用 MV""总用 UV""事后最优"对比。
- 这部分给出一条可以直接用的实践建议，适合 DASFAA 的受众。

---

## 6. 预期结果与对应的论文叙事

本文是诊断性研究，**不依赖某一种特定结果**。下面列出几种可能的结果组合，以及各自对应的写法：

| 结果模式 | 对应叙事 | 论文主张 |
|---|---|---|
| A：E0 层（a）收益≈0，E1 上 B 普遍很小，E2 中各干预的 S 都很低 | 闲置 | 官方的多变量收益主要来自协变量；在标准 LTSF 数据上变量注意力基本闲置，现有基准衡量不出跨通道能力 → 本文的诊断协议就是贡献 |
| B：高维数据集上 B>0，但 I3（日历对齐交换）后收益仍大部分保留 | 池化而非读取 | 模型把其他通道当作共享季节相位/尺度的先验，而不是读取具体的同步变化 |
| C：I2/I4 抹掉了收益，合成数据上 GC 较高 | 真的在读 | 把重点放在失败模式上：漂移（G3）、K>32、列顺序切块 → 给出分组和模式选择的指南 |
| D：ETT 上 B<0，G3 上多变量模式被误导 | 零样本版本的容量-稳健性权衡 | 与 Han 等的结论对应：零样本模型也会被不稳定的相关性带偏 |
| E：正对照 P1 也没有收益 | 能力缺失（需先排除实现错误） | 以 patch 对齐的变量注意力传递滞后信息的能力有限；用 Toto-2.0 / Chronos-2 交叉验证后，这本身就是一个重要发现 |

实际结果很可能是几种模式的混合（例如 ETT 属于 D、Electricity 属于 B、合成数据属于 C），四象限图（§3.5）正好可以把它们放在一张图里。

---

## 7. 算力预算与工程实现

### 7.1 GPU 时间估算（RTX 4090）

估算方法：每个"变量 token"的稠密计算量约为 2 × 330M = 0.66 GFLOPs；上下文 512 + H=720 时每个变量约 40 个 token；开启对称平均时 ×2。有效吞吐的保守假设：RTX 3080 在 fp32 下约 12 TFLOPS，RTX 4090 约为它的 2.5 倍（约 30 TFLOPS）。有监督小模型用不满 GPU，按 1.5–2 倍计。**第一周请实测一次吞吐，再按比例校正下表。**

| 实验 | 主要开销 | 3080 估算 | **4090 估算** |
|---|---|---|---|
| S1–S7 | 少量窗口 | < 2 h | < 1 h |
| E0 | fev-bench 官方多变量一遍在 Google 的 GPU 上约 2.5 h（仓库 CSV 中 `inference_time_s` 之和）；4 个条件 × 相关子集，外加 GIFT-Eval 和 TIME | 15–20 h | 6–8 h |
| E1 主表 | 8 个数据集 × {MV, UV} × 步长 1（3080 上 Traffic 单个条件约 2.9 h，Electricity 约 1.8 h，Solar 约 1.6 h） | 约 13.6 h | 约 5.5 h |
| E1 扫描 | 8 种上下文 × 2 种模式 × 每个数据集 300 个窗口 | 约 12 h | 约 5 h |
| E2 | 以目标为中心的设计，约 15 次运行（8 类干预，随机干预各 3 个种子） | 约 23 h | 约 9 h |
| E3 | 约 140 个配置，N 多数 ≤32 | 约 5 h | 约 2 h |
| E5 | 划分设计 + 以目标为中心 | 约 29 h | 约 12 h |
| E6 | 抽样窗口，需要显式计算注意力权重 | 约 5 h | 约 2 h |
| **TimesFM-3 小计** | | 约 100 h | **约 40 h ≈ 2 天** |
| 其他 TSFM（精简协议） | Chronos-2 较便宜；Toto-313m 与 TimesFM-3 相当；Moirai 需分组 | 2–3 天 | 约 1 天 |
| 有监督基线 | 3 个模型 × 8 个数据集 × lookback 调优 + 4 个 horizon + 3 个种子；外加合成数据 | 2–4 天 | 1–2 天 |
| **合计** | | 9–11 GPU·天 | **4–5 GPU·天** |

**如果时间紧**：扫描类实验关闭对称平均（先验证）；E6 砍掉。

### 7.1b 4090 的环境与使用设置

- **软件版本**：官方公布的 fev-bench 结果是在 `torch==2.9.1+cu129` 下跑出来的（见仓库 CSV 的 `framework_version` 字段）。建议用同一个 PyTorch/CUDA 版本，这样 S1 复现官方数值最容易对上。4090 是 Ada 架构（sm_89），FlashAttention-2、bf16、TF32 都支持。
- **精度设置**：主结果保持纯 fp32（`torch.backends.cuda.matmul.allow_tf32 = False`，`torch.backends.cudnn.allow_tf32 = False`），与官方推理一致；4090 在纯 fp32 下已经足够快。如果之后想在扫描实验里开 TF32 提速，先在两个数据集上确认 MV−UV 的差值不受影响，并把设置写进论文。整个项目只有一种卡，配对比较不会受硬件差异影响。
- **长时间运行**：连续跑几天时，可以用 `nvidia-smi -pl 350` 把功耗上限从 450 W 降到 350 W 左右。性能损失通常只有个位数百分比，温度和噪音会低很多。第一周实测吞吐时顺便测一下限功耗前后的差别。
- **整机建议**：电源 1000 W、ATX 3.x，用原生的 12V-2x6（12VHPWR）线直连显卡，插头要插到底、不要贴着插头急弯（这是 4090 供电接口烧毁的主要原因）；内存 64 GB（32 GB 也能跑，但同时做数据生成和分析会紧）；NVMe SSD 2 TB（HF 模型缓存、基准数据集和预测缓存，见 §7.2）；CPU 8 核以上即可，推理不吃 CPU，有监督训练的数据加载需要几个核。
- 如果买到的是 4090D，性能大约低 5%–10%，对这个项目没有实质影响；24 GB 显存已经足够，不需要改装的 48 GB 版本。

### 7.2 代码结构建议

```
probe/
  data.py            # LTSF 加载、标准化、窗口索引（全部由种子决定）
  models/
    base.py          # 统一接口: forecast(ctx[N,T], horizon, groups) -> quantiles[N,H,9]
    timesfm3.py      # 包装 TimesFM3Forecaster / Evaluator；univariate、分组、make_positive=False
    chronos2.py  moirai.py  toto2.py
  interventions.py   # N1–N4, I1–I6, P1；输入 ctx，返回被修改的 ctx 与元数据（不使用掩码，见 §11）
  grouping.py        # 列顺序切块 / 随机 / 聚类 / top-k / bottom-k
  synth/
    generators.py    # G1–G7（线性高斯部分写成状态空间形式）
    oracle.py        # Kalman 滤波：只观测目标 vs 观测全部通道
  hooks.py           # 变量注意力权重采集；逐层"只看自己"的敲除
  metrics.py         # MSE/MAE/MASE/WQL/覆盖率 + B/R/S/GC + 分块 bootstrap
runs/                # 每个实验一个 yaml 配置
cache/               # 以 hash(模型, 数据集, 窗口, 条件, 种子) 为键的 parquet（存储策略见下方"实现要点"）
analysis/            # 从 cache 生成全部图表，不重跑模型
```

**实现要点**
- **分两级缓存，不要对步长 1 的主表存完整分位数预测**：以 E1 主表为例，8 个数据集、步长 1、H=720、9 个分位数、fp32，每个条件就要约 145 GB（Traffic 一个数据集约 62 GB）。
  - 步长 1 的实验：只存逐窗口、逐通道、按 horizon 分段（96/192/336/720）的误差汇总（MSE、MAE、WQL 和覆盖率），每个条件约 0.3 GB。
  - 抽样窗口（300/500 个）：存完整的分位数预测，用 fp16，每个条件约 5 GB。之后新增指标、画图、算敏感度 S 都用这一级，不用重跑模型。
  - E1 扫描、E5 这类条件数很多的实验，只存中位数预测（fp16，体积再降到约 1/9）加误差汇总。
  - 按这个策略，包括对照模型在内，整个项目的缓存约 100–200 GB，2 TB 的 SSD 足够。
- **逐层敲除**：变量注意力只看自己时，softmax 只有一个元素、结果为 1，所以该子层的输出等于 `out_proj(value_proj(x))`（`v_norm` 为 none）。可以对指定层的 `MixingTransformer.forward` 打补丁，比构造对角掩码更简单也更快。
- **采集注意力权重**：官方实现用 SDPA，不返回权重；分析时对被 hook 的层改为显式计算 softmax(QKᵀ)，并且只在抽样窗口上做。
- **K>32**：直接调用 `TimesFM3Forecaster.predict_batch`（不走评测器的切块），同时手动设置 `use_symmetric_averaging`、`make_positive` 等参数，使其与评测器的设置一致。

---

## 8. 时间表（2026-09-28 至 11-25）与优先级

| 周 | 日期 | 任务 | 里程碑 |
|---|---|---|---|
| W1 | 9/28–10/4 | 环境、数据、TimesFM-3 包装；S1–S7；E0 | **10/4 判断节点**：根据 E0 确定叙事方向（§5.1） |
| W2 | 10/5–10/11 | E1 主表 + 上下文扫描；泄漏审计；有监督基线开始在后台调 lookback | E1 主表完成 |
| W3 | 10/12–10/18 | E2 全部干预与正负对照；合成生成器与 oracle 实现、单元测试 | 四象限图初版 |
| W4 | 10/19–10/25 | E3（TimesFM-3 + 有监督）；E5a/E5b | GC 热图、K 曲线 |
| W5 | 10/26–11/1 | E4 其他 TSFM（精简协议）；有监督 CD 模型上的 E2；E5c/E5d；E7 | 全部 P0 + 大部分 P1 完成 |
| W6 | 11/2–11/8 | 补缺口；可选 E6；定稿全部图表 | **11/8 冻结实验** |
| W7 | 11/9–11/15 | 写完整初稿 | 初稿 |
| W8 | 11/16–11/24 | 内部审阅、修改、匿名化检查、匿名代码仓库；**11/24 提交**（留一天余量） | 提交 |

- **P0**（必须）：S、E0、E1 主表、E2（N1–N4、I2、I3、I4、P1）、E3（G1–G5）、E4（Chronos-2 + PatchTST/iTransformer/CLOC）、E5a/E5b。
- **P1**（尽量）：E1 扫描、E2（I1、I5、I6）、E3（G6、G7）、E4（Moirai、Toto-2.0）、E5c/E5d、E7。
- **P2**（可选）：E6、新鲜数据、TiRex-2、TimesFM-2.5。
- 如果到 10/25 进度落后一周以上：P1 只保留 Toto-2.0 和 E5c，其余移到 ECML-PKDD 版本。

---

## 9. 论文结构（16 页 LNCS，含参考文献和附录）

DASFAA 的 16 页**包括附录**，所以没有额外的附录空间；补充材料放在匿名代码仓库里。摘要不超过 200 词。

| 部分 | 页数 | 内容 |
|---|---|---|
| 1 引言 | 1.5 | 开头：Moirai 2.0 放弃跨变量注意力，而 TimesFM-3 宣称多变量模式更强，Falcon-X 又观察到 Chronos-2 开关 group attention 几乎没有差别 → 提出本文的问题和贡献 |
| 2 相关工作 | 1.25 | 多变量 TSFM；CI/CD 之争；TSFM 评测与可解释性 |
| 3 诊断框架 | 2 | B / R / S / GC；干预清单与正负对照；合成生成器与 oracle |
| 4 实验设置 | 1.25 | 模型、数据、泄漏表、32 变量上限与分组协议、统计方法 |
| 5 结果 | 6 | 5.1 收益从哪里来（E0 + E1）；5.2 它在读什么（E2）；5.3 可控结构（E3）；5.4 给多少、给哪些通道（E5 + 成本）；对照模型的结果贯穿各小节 |
| 6 讨论与实践建议 | 1 | E7 的模式选择规则；局限性 |
| 7 结论 | 0.5 | |
| 参考文献 | 约 2 | 约 35 篇 |

**图表清单**（7 个以内）：图 1 四象限图（teaser）；表 1 E0 分层收益；表 2 LTSF 主表（含全部对照模型）；图 2 干预依赖度条形图 + δ 曲线；图 3 合成数据 GC 热图 + G3 曲线；图 4 收益随 K 的曲线（3 种策略 × 3 个数据集）+ 延迟；图 5 上下文长度扫描。

---

## 10. 必须引用的文献及区别说明（可直接改写进相关工作）

以下英文句子是草稿，引用编号按最终参考文献调整。

- **TimesFM-3** [Google Research blog 2026; GitHub; HF] 与 **TimesFM** [Das et al., ICML 2024, arXiv 2310.10688]：
  "TimesFM-3 extends the decoder-only TimesFM with full variate attention in every layer and reports that its multivariate mode improves the average rank on GIFT-Eval, fev-bench and TIME. The per-task decomposition of this gain and the mechanism behind it have not been reported; we provide both."
- **Chronos-2** [Ansari et al., arXiv 2510.15821]：
  "Chronos-2 shares information within groups of series through group attention and acquires multivariate behavior from synthetically multivariatized data; its reported gains are largest on covariate-informed tasks. We complement such aggregate evaluations with within-model interventions."
- **Moirai** [Woo et al., ICML 2024, arXiv 2402.02592] 与 **Moirai 2.0** [Liu et al., arXiv 2511.11698]：
  "Moirai flattens all variates into a single sequence with any-variate attention, whereas its successor Moirai 2.0 forecasts each variate independently, which suggests that the value of cross-variate modeling in pretrained forecasters remains unsettled."
- **PAKDD 2026** [Abdelmalak et al., arXiv 2502.09683]：
  "Abdelmalak et al. show that the apparent superiority of channel-independent models on standard benchmarks stems from weak inter-channel dependence and untuned lookback windows. They study supervised models trained per dataset; we ask the analogous question for a single pretrained model at inference time, where channel use can be switched off and intervened on without retraining. We adopt their lookback-tuning protocol for our supervised baselines and treat context length as an explicit factor for the foundation models."
- **Han et al.** [TKDE 2024, 36(11):7129–7142]：
  "Han et al. attribute the robustness of channel independence to a capacity–robustness trade-off under distribution drift. Our structural-drift generator tests whether a zero-shot model exhibits the same trade-off without any training on the target data."
- **CLOC** [Song et al., Mathematics 2026, 14(17):3132]（第三人称）：
  "Song et al. propose CLOC, which selects a channel-communication budget per dataset on validation data and find that suppression is preferable on ETT while local–global communication helps on Electricity and Solar-Energy. We test whether a pretrained model's multivariate gains follow the same dataset-dependent pattern, and use CLOC as a supervised reference whose zero-communication endpoint provides an exact CI/CD toggle."
- **Falcon-X** [Liu et al., arXiv 2605.27286]（新增，必须引用）：
  "Liu et al. report that Chronos-2 performs nearly identically on GIFT-Eval with and without group attention, and that its attention maps are similar across dissimilar inputs. They use this observation to motivate a new architecture; we turn it into a systematic diagnostic, apply it to TimesFM-3 and three other models, and characterize when cross-variate attention helps, idles or hurts."

**建议同时引用**：Toto 2.0（arXiv 2605.20119）、TiRex-2（arXiv 2607.01204）、RMISC（arXiv 2607.06504）、fev-bench（arXiv 2509.26468）、GIFT-Eval（arXiv 2410.10393）、TIME（arXiv 2602.12147）、Contiguous Patch Masking（arXiv 2505.23719）、Żukowska 等（arXiv 2409.13530）、CPiRi（arXiv 2601.20318）、LIFT（arXiv 2401.17548）、通道策略综述（arXiv 2502.10721）、PatchTST、iTransformer、IAAFT 替代序列（Schreiber & Schmitz 1996）、Attention is not Explanation（Jain & Wallace 2019）。

**匿名化检查**：正文不出现"our previous work"；CLOC 与其他文献一样按第三人称引用；代码用匿名仓库（如 anonymous.4open.science）；PDF 元数据清空作者信息；致谢删除。

---

## 11. 与 1 月"通道缺失"论文的边界

- **本文**：推理时**所有通道都可用**；变量的是**通道内容**（干预）和**有意选择的上下文集合**（分组）。
- **1 月论文**：推理时通道**缺失或不可用**；研究鲁棒性和补全方法。
- **容易重叠的地方**：E5 的 K 扫描和 E2 的"换成噪声"在形式上可能被看成"只给部分通道"或"通道损坏"。处理方式：
  1. E5 统一表述为"上下文集合设计"（主动挑选给模型哪些通道），不写成"随机删掉通道后看性能下降"。
  2. 本文**不使用任何基于掩码（mask）的干预**。TimesFM-3 支持掩码输入，但"掩码"语义上等同于缺失，留给 1 月论文。
  3. 代码分支、随机种子、结果缓存完全分开；两篇论文不共享任何表格或图。如果 1 月论文也需要同一数据集上的 MV/UV 基线，用它自己的协议（不同的上下文长度或窗口集合）重新跑，不复用本文的数字。

---

## 12. 风险与应对

| 风险 | 可能性 | 应对 |
|---|---|---|
| 截稿前 Google 发布 TimesFM-3 技术报告，内含单/多变量逐任务消融 | 中 | 每周一检查 arXiv（搜索 "TimesFM-3"）、GitHub `timesfm3-usage` 的提交记录和 HF 模型卡。如果发布了：E0 改成"复现 + 按层拆分"，把重点移到 E2/E3/E5（技术报告一般不会做干预、合成结构和分组分析），并引用该报告 |
| 出现同期的 TimesFM-3 探测论文 | 中 | 尽快完成 P0；DASFAA 的 CFP 没有写明 arXiv 政策，如果想挂预印本占位，先写信向程序委员会主席确认 |
| 所有数据集上 B 都≈0 | 中 | 诊断性研究本身不依赖正向结果（§6）；正对照 P1 和合成数据 GC 可以证明"能力存在但没有被用上" |
| 正对照 P1 失败 | 低 | 先检查实现（S4、horizon 掩码、领先指标的时间对齐）；用 Chronos-2 和 Toto-2.0 交叉验证；如果仍然失败，这本身就是关键发现 |
| 审稿人质疑 Traffic 的泄漏 | 高（如果不处理） | 按 §4.6 做逐值比对，放出泄漏表，用 Toto-2.0 和新鲜数据交叉验证 |
| 算力不够 | 低（已改用 4090） | 严格按 P0/P1/P2 取舍；扫描实验关闭对称平均；必要时临时租一张云 GPU 并行跑有监督基线 |
| 审稿人认为超出 DASFAA 范围 | 中 | 强调数据管理视角：高维传感器数据的通道分组、精度-成本权衡、部署时的模式选择规则；CFP 明确列出 "Data streams and time series" 和 "Machine learning in databases" |
| TimesFM 仓库更新，评测器的切块逻辑改变 | 低 | 固定 commit `e31dadd`，在论文里写明版本 |
| 权重许可 | 低 | 仅学术使用、不再分发权重；Chronos-2 和 Toto 为 Apache-2.0 |
| 16 页含附录，放不下 | 高 | 按 §9 的页数分配；次要结果放匿名仓库 |
| 转投 ECML-PKDD / CIKM | — | ECML-PKDD 同为 LNCS 格式，改动小；CIKM 用 ACM 格式，页数规则不同，需要重排。日期和格式以两者正式 CFP 为准 |

---

## 13. 调研来源

**TimesFM-3**
- [Google Research 博客：TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [GitHub：google-research/timesfm](https://github.com/google-research/timesfm)（本文核查的 commit：`e31dadd`，2026-09-15）
- [HF 模型卡：google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)，[config.json](https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/config.json)
- [TSFM.ai 对 TimesFM-3 的分析](https://tsfm.ai/blog/timesfm-3-multivariate-zero-shot-forecasting)
- [MarkTechPost 报道](https://www.marktechpost.com/2026/08/31/google-ai-releases-timesfm-3-a-330m-parameter-zero-shot-foundation-model-for-multivariate-time-series-forecasting/)

**多变量 TSFM 与相关证据**
- [Chronos-2（arXiv 2510.15821）](https://arxiv.org/abs/2510.15821)，[HF：amazon/chronos-2](https://huggingface.co/amazon/chronos-2)
- [Falcon-X（arXiv 2605.27286）](https://arxiv.org/abs/2605.27286)
- [Moirai（arXiv 2402.02592）](https://arxiv.org/abs/2402.02592)
- [Moirai 2.0（arXiv 2511.11698）](https://arxiv.org/abs/2511.11698)
- [Toto 2.0（arXiv 2605.20119）](https://arxiv.org/html/2605.20119v1)，[DataDog/toto](https://github.com/datadog/toto)
- [TiRex-2（arXiv 2607.01204）](https://arxiv.org/html/2607.01204)
- [RMISC（arXiv 2607.06504）](https://arxiv.org/html/2607.06504v1)
- [fev-bench（arXiv 2509.26468）](https://arxiv.org/abs/2509.26468)
- [GIFT-Eval（arXiv 2410.10393）](https://arxiv.org/html/2410.10393v2)，[GiftEvalPretrain 数据集](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- [TIME（arXiv 2602.12147）](https://arxiv.org/abs/2602.12147)
- [电力需求与外生特征（arXiv 2602.05390）](https://arxiv.org/abs/2602.05390)
- [金融多变量 Chronos-2（arXiv 2605.21504）](https://arxiv.org/abs/2605.21504)
- [Towards Long-Context TSFMs（arXiv 2409.13530）](https://arxiv.org/html/2409.13530v1)
- [Dissecting Chronos（arXiv 2603.10071）](https://arxiv.org/abs/2603.10071)

**CI/CD 与通道依赖**
- [PAKDD 2026：Channel Dependence, Limited Lookback Windows…（arXiv 2502.09683）](https://arxiv.org/abs/2502.09683)，[Springer 版本](https://link.springer.com/chapter/10.1007/978-981-92-1462-4_46)
- [Han 等：The Capacity and Robustness Trade-off（arXiv 2304.05206）](https://arxiv.org/abs/2304.05206)，[代码](https://github.com/hanlu-nju/channel_independent_MTSF)
- [CLOC：Controlled Local–Global Channel Communication（Mathematics 2026）](https://www.mdpi.com/2227-7390/14/17/3132)
- [CPiRi（arXiv 2601.20318）](https://arxiv.org/abs/2601.20318)
- [LIFT：Learning from Leading Indicators（arXiv 2401.17548）](https://arxiv.org/abs/2401.17548)
- [通道策略综述（arXiv 2502.10721）](https://arxiv.org/html/2502.10721)

**会议**
- [DASFAA 2027 研究论文征稿](https://dasfaa2027.github.io/CallforResearchTrack/list.htm)，[组织委员会](https://dasfaa2027.github.io/orgcmte/list.htm)

