<div align="center">

# Times-FM3 Probing Suite
### Does Variate Attention Pay Off? Probing Cross-Channel Information Use in Multivariate Time-Series Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Paper: DASFAA 2027](https://img.shields.io/badge/Paper-DASFAA%202027-success.svg)](paper/main.pdf)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**English**](#english) | [**中文说明**](#中文说明)

</div>

---

<a name="english"></a>
## English Overview

This repository provides the official implementation, diagnostic probing benchmarks, experimental suites, and publication artifacts for the paper:

> **Does Variate Attention Pay Off? Probing Cross-Channel Information Use in Multivariate Time-Series Foundation Models**  
> *Under review at DASFAA 2027 (Springer LNCS).*

In next-generation zero-shot time-series foundation models (TSFMs) such as Google's 330M-parameter **TimesFM-3**, cross-channel attention is introduced to model inter-variate dependencies. However, whether and how these models genuinely utilize inter-channel information has remained an open question. 

This repository provides a rigorous **within-model diagnostic probing framework**: on identical test windows and fixed model weights, we probe TimesFM-3 across 8 standard benchmarks (9,600 window-horizon pairs), apply 16 target-centric causal interventions, benchmark against analytical Kalman filter oracles on controlled synthetic VAR/state-space systems, and map the accuracy-latency Pareto frontier across channel budgets $K \in \{1..128\}$.

### Key Discoveries

1. **Domain Bimodality (RQ1)**: Multivariate ($\text{MV}$) mode provides statistically significant gains on physically coupled systems (Weather $+8.04\%$, ETTh2 $+7.20\%$, Electricity $+2.60\%$), but severely degrades on high-frequency noise domains (Solar-Energy $-31.95\%$).
2. **Calendar vs. Micro-Phase Synchronization (RQ2)**: Replacing context channels with week-aligned history preserves half the multivariate gain ($R=0.50$ on Electricity), whereas unaligned history makes the forecast substantially worse than univariate ($R=2.06$). Shifting history by 1 hour has virtually no effect ($R=0.02$), demonstrating that the model relies on calendar consistency rather than micro-step temporal alignment.
3. **Information Ceiling vs. Analytical Kalman Oracles (RQ3)**: On synthetic linear Gaussian systems with ground-truth cross-channel Granger causality, zero-shot TimesFM-3 closes at most **$29\%$** of the gap between univariate and optimal multivariate Kalman oracles.
4. **Channel Budget & Evaluator Anatomy (RQ4)**: In the reference evaluator, channels outside the target's block of 32 leave forecasts completely unchanged (identical down to $10^{-5}$ MSE) while inference latency scales linearly from $55\text{ ms}$ to $217\text{ ms}$. Stratified correlation-based Top-$K$ selection capped at $K=32$ provides optimal Pareto efficiency.

---

<a name="中文说明"></a>
## 中文说明

本项目为论文《**Does Variate Attention Pay Off? Probing Cross-Channel Information Use in Multivariate Time-Series Foundation Models**》（投稿 DASFAA 2027，Springer LNCS）的官方开源代码、实验探针套件、复现脚本与数据包。

针对以 **TimesFM-3**（3.3 亿参数）为代表的引入跨通道注意力（Cross-Channel Attention）的多变量时间序列大模型，我们构建了首个**模型内因果探针诊断框架（Within-Model Diagnostic Probing）**：
- **E1 标准长时预测基准测试**：8 大真实世界数据集（ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Solar-Energy）$\times$ 4 个预测步长（96, 192, 336, 720），共 9,600 个测试窗口；
- **E2 目标中心因果干预分析**：设计 16 种干预策略（时序打乱、周期对齐/非对齐历史替换、IAAFT 相位随机化、平移扰动、独立噪声替换、领先指标注入等），精准解构模型收益来源；
- **E3 受控合成动力学系统与 Kalman 理论上界**：对比分析多变量/单变量卡尔曼最优滤波 Oracle，量化零样本模型对交叉通道耦合的 Gap Closure；
- **E5 通道预算与切块机制**：系统评估通道预算 $K \in [1..128]$ 并在 RTX 4090 上实测准确度-延迟帕累托边界，揭示官方评测器 32 通道切块机制的实际行为。

---

## Repository Structure / 仓库目录结构

```text
Times-FM3/
├── analysis/                     # Table & plot generation scripts
│   └── generate_paper_tables_and_plots.py
├── data/                         # Data provenance & benchmark markers
│   └── PROVENANCE.md             # Experimental provenance & run tracking
├── experiments/                  # Experiment runners
│   ├── run_e1_ltsf.py            # RQ1: Main LTSF benchmark suite
│   ├── run_e2_interventions.py   # RQ2: Target-centric causal interventions
│   ├── run_e3_synthetic.py       # RQ3: Synthetic VAR & Kalman oracle evaluation
│   ├── run_e5_grouping.py        # RQ4: Channel budget & grouping strategies
│   └── diag_p1_solar.py          # Solar & P1 diagnostic probes
├── paper/                        # Complete DASFAA 2027 paper LaTeX source & PDF
│   ├── main.tex                  # 15-page final manuscript (Springer LNCS)
│   ├── main.pdf                  # Compiled PDF
│   ├── references.bib            # Fully verified bibliography
│   ├── figures/                  # Figures 1–4 (PDF & PNG)
│   ├── data/                     # Clean experimental CSV summaries
│   └── scripts/                  # Figure generation script (make_figures.py)
├── paper_figures/                # Vector publication figures & LaTeX tables
│   ├── fig1_four_quadrant.pdf/.png
│   ├── fig2_interventions.pdf/.png
│   ├── fig3_synthetic_gc.pdf/.png
│   ├── fig4_channel_budget.pdf/.png
│   └── table2_ltsf_main.tex
├── probe/                        # Core probing library
│   ├── config.py                 # Hyperparameters & paths
│   ├── data.py                   # Unified dataset loading & windowing
│   ├── grouping.py               # Channel budget & selection algorithms
│   ├── hooks.py                  # PyTorch model intervention hooks
│   ├── interventions.py          # Causal intervention implementations
│   ├── metrics.py                # Stationary block bootstrap, Holm-Bonferroni CI
│   ├── models/                   # TimesFM-3 forecaster wrapper
│   │   ├── base.py
│   │   └── timesfm3.py
│   └── synth/                    # Synthetic generators & Kalman/DARE oracles
│       ├── generators.py
│       └── oracle.py
├── results/                      # Full experimental results (CSVs & logs)
│   ├── e1_ltsf_sampled.csv       # E1 32 benchmark configurations
│   ├── e2_interventions_summary.csv
│   ├── e3_synthetic_summary.csv
│   ├── e5_grouping_summary.csv
│   └── diag_p1_solar_summary.csv
├── scripts/                      # Utility, verification & benchmarking scripts
├── tests/                        # Unit tests (pytest)
├── pyproject.toml                # Build system configuration
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

---

## Quick Start & Installation / 快速上手

### 1. Clone & Environment Setup

```bash
git clone https://github.com/NEUMelon/Times-FM3.git
cd Times-FM3

# Create and activate conda environment
conda create -n timesfm python=3.10 -y
conda activate timesfm

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation with Unit Tests

```bash
pytest tests/ -v
```

All tests for causal interventions, metrics, stationary block bootstrap, and analytical Kalman DARE oracles should pass cleanly.

---

## Reproducing the Experiments / 实验复现

### E1: Long-Term Time Series Forecasting Benchmark (RQ1)
Runs both univariate ($\text{UV}$) and multivariate ($\text{MV}$) inference across 8 datasets and 4 prediction horizons:
```bash
python experiments/run_e1_ltsf.py --windows 300 --horizons 96 192 336 720
```

### E2: Target-Centric Causal Interventions (RQ2)
Applies 16 interventions on context channels (keeping the target variate intact) across 100 evaluation windows:
```bash
python experiments/run_e2_interventions.py --windows 100 --horizon 96
```

### E3: Synthetic Systems & Kalman Filter Oracles (RQ3)
Evaluates TimesFM-3 against exact analytical Kalman filter oracles on controlled VAR(1)/VAR(2) and state-space systems:
```bash
python experiments/run_e3_synthetic.py --num_series 50
```

### E5: Channel Budget & Grouping Strategies (RQ4)
Maps the accuracy-latency curve for $K \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ across Random, Top-$K$, and Bottom-$K$ strategies:
```bash
python experiments/run_e5_grouping.py --windows 50 --targets 64
```

---

## Generating Figures & Tables / 生成图表

To regenerate all publication figures and LaTeX tables directly from experimental CSV results:

```bash
# Generate Figures 1-4 for the paper
python paper/scripts/make_figures.py

# Generate Table 2 and comprehensive plots
python analysis/generate_paper_tables_and_plots.py
```

Generated outputs will be saved to `paper/figures/` and `paper_figures/`.

---

## Compiling the Paper / 编译论文

The complete 15-page Springer LNCS paper is included in `paper/`:

```bash
cd paper
python compile_paper.py
```

This compiles `main.tex` with `pdflatex` and `bibtex` cleanly without overfull hboxes or missing citations. The compiled PDF is located at [paper/main.pdf](paper/main.pdf).

---

## Data Provenance & Reproducibility / 数据来源说明

All numbers, tables, and figures reported in the manuscript are strictly verified against the CSVs in `results/` and `paper/data/`. Detailed run parameters and commit provenance are documented in [data/PROVENANCE.md](data/PROVENANCE.md).

- **E1 Baseline**: 300 test windows per dataset, $H \in \{96, 192, 336, 720\}$, context length $L=512$, seed 42.
- **E2 Interventions**: 100 windows, $H=96$, amplitude SNR 10 for lead indicators.
- **E5 Grouping**: 50 windows $\times$ 64 stratified targets (3,200 forecasts per strategy and $K$).
- **Hardware**: Evaluated on NVIDIA GeForce RTX 4090 GPUs (PyTorch 2.8, CUDA 12.8).

---

## Citation

If you find this work, codebase, or diagnostic probing protocols helpful, please cite our paper:

```bibtex
@inproceedings{timesfm3_probing2027,
  title     = {Does Variate Attention Pay Off? Probing Cross-Channel Information Use in Multivariate Time-Series Foundation Models},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the International Conference on Database Systems for Advanced Applications (DASFAA)},
  year      = {2027},
  publisher = {Springer}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
