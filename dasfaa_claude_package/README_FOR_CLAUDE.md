# Comprehensive Experiment Data & Writing Guide for Claude

This package contains all empirical benchmark data, publication-ready vector figures, LaTeX templates, reference bibliography, and the complete factual background for writing a top-tier paper submitted to **DASFAA (Research Track, CCF-B)**.

---

## 1. Paper Overview & Constraints

- **Title**: *Does Variate Attention Pay Off? Probing Cross-Channel Information in Multivariate Time-Series Foundation Models*
- **Target Venue**: DASFAA (Research Track, CCF-B)
- **Format**: Official Springer LNCS (`llncs.cls`, `splncs04.bst`). Single column, 12.2 cm text width.
- **Page Limit**: **Strictly $\le 16$ pages** including all figures, tables, references, and appendix.
- **Abstract Limit**: **Strictly $\le 200$ words**.
- **Double-Blind**:
  - Author: `Anonymous`, Institute: `Anonymous Institution`, email: `anonymous@placeholder.org`
  - PDF metadata: `\usepackage[pdfauthor={},hidelinks]{hyperref}`
  - Code repository placeholder: `anonymous.4open.science`
  - **CLOC Citation Rule**: Cite CLOC (Song et al., Mathematics 2026, 14(17):3132) strictly in the **third person** (`Song et al.~\cite{song2026cloc}`) without any self-attribution or first-person pronouns.
- **Writing Tone & Style**:
  - Crisp, rigorous, mathematically grounded academic English.
  - **Strictly Prohibited Clichés / Banned Words**: Do NOT use `delve`, `pivotal`, `landscape` (use `profile` or `distribution`), `unleash`, `to the best of our knowledge`.
  - Do NOT use the word `significantly` unless accompanied by statistical hypothesis testing (e.g., Wilcoxon signed-rank test).
  - Use math macros: `\MV` (multivariate), `\UV` (univariate), `\Ben` (benefit), `\Rel` (reliance), `\Sen` (sensitivity), `\GC` (gap closure), `\tfm` (TimesFM-3).

---

## 2. Core Research Story & Context

### 2.1 The Academic Conflict
1. **TimesFM-3 Claim**: Google Research recently released TimesFM-3.0 (200M/330M parameters), claiming that adding full cross-variate attention across its transformer layers yields superior aggregate rankings on GIFT-Eval, fev-bench, and TIME benchmarks.
2. **The Counter-Evidence**:
   - **Moirai-2.0** (Salesforce, 2025): Explicitly abandoned any-variate cross-attention, reverting to independent univariate forecasting to prevent negative transfer.
   - **Falcon-X** (Liu et al., 2026): Observed that Chronos-2 produces near-identical predictions on GIFT-Eval whether group attention is enabled or disabled, with attention maps collapsing across dissimilar inputs.
   - **Supervised Literature**: Han et al. (TKDE 2024) proved channel-independent (CI) models are more robust under distribution drift (capacity-robustness trade-off). Abdelmalak et al. (PAKDD 2026) showed standard benchmarks have weak Granger causality, where CI beats CD in 21/28 cases, but CD wins on ODE systems. Song et al. (CLOC, 2026) showed channel communication must be dynamically regulated per dataset.
3. **The Novelty of this Paper**:
   - Instead of cross-model benchmark comparisons or proposing yet another architecture, we conduct the **first systematic within-model causal probing study** on TimesFM-3.0.
   - We dissect: Does variate attention truly pay off? What exact information does it extract (micro-phase synchronization vs. static pooling)? Does it reject noise? How much theoretical information does it capture relative to exact Kalman filter oracles? How should high-dimensional channels be selected in production databases?

---

## 3. Mathematical Metrics Formulation

Let $\mathcal{W}$ be the evaluation windows, $H$ the horizon, $i$ the target channel. Forecast outputs: $\hat{y}^{\UV}$ (univariate), $\hat{y}^{\MV}$ (multivariate), $\hat{y}^\pi$ (multivariate under intervention $\pi$ applied only to context channels $j \ne i$).
Normalized Mean Squared Error:
$$e(\hat{y}) = \frac{1}{H} \sum_{h=1}^H (y_{w,i,h} - \hat{y}_{w,i,h})^2$$

1. **Benefit ($B$)**:
   $$B = \frac{\sum e(\hat{y}^{\UV}) - \sum e(\hat{y}^{\MV})}{\sum e(\hat{y}^{\UV})} \times 100\%$$
   $B > 0$ means multivariate is better; $B < 0$ denotes negative transfer.

2. **Reliance ($R_\pi$)**:
   $$R_\pi = \frac{e(\hat{y}^\pi) - e(\hat{y}^{\MV})}{e(\hat{y}^{\UV}) - e(\hat{y}^{\MV})}$$
   - $R_\pi \approx 1$: Intervention $\pi$ erased 100% of the multivariate gain $\implies$ gain relied entirely on the destroyed feature.
   - $R_\pi \approx 0$: Gain was unaffected by $\pi$.
   - $R_\pi > 1$: Model was actively misled by the corrupted context, performing worse than univariate.

3. **Sensitivity ($S_\pi$)**:
   $$S_\pi = \frac{1}{|\mathcal{W}| N} \sum_{w,i} \frac{1}{H}\sum_{h=1}^H |\hat{y}^\pi_{w,i,h} - \hat{y}^{\MV}_{w,i,h}|$$
   Measures output shift magnitude on normalized scale independent of ground truth.

4. **Oracle Gap Closure ($\GC$)**:
   $$\GC = \frac{e(\hat{y}^{\UV}_{\text{model}}) - e(\hat{y}^{\MV}_{\text{model}})}{e^{\UV}_\star - e^{\MV}_\star}$$
   where $e^{\UV}_\star$ and $e^{\MV}_\star$ are analytical Kalman filter oracles on linear Gaussian state-space systems. $\GC = 1.0$ means 100% extraction of available cross-channel mutual information.

---

## 4. Key Empirical Findings (Exact Numbers from Official Evaluation)

All data files are located in `data/`:

### 4.1 E1: Benchmark Long-Term Forecasting (`e1_ltsf_sampled.csv`)
- **Setup**: 8 canonical LTSF datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Solar, Traffic) across 4 horizons $H \in \{96, 192, 336, 720\}$ with context $L=512$, 300 sampled windows. Evaluated with 2000 stationary block bootstrap resamples and Holm-Bonferroni correction.
- **Key Finding 1 (Significant Cross-Channel Benefit)**:
  - **Weather**: Major steady gains: $+8.04\%$ at $H=96$ ($0.1668 \to 0.1534$ MSE, $p_{\text{adj}} = 5.43\times 10^{-10}$), $+3.06\%$ at $H=192$, $+1.30\%$ at $H=336$, $+5.76\%$ at $H=720$ ($0.5092 \to 0.4799$ MSE).
  - **ETTh2**: Consistent strong gains: $+6.21\%$ at $H=96$ ($p_{\text{adj}} = 2.44\times 10^{-5}$), $+6.86\%$ at $H=192$ ($p_{\text{adj}} = 5.00\times 10^{-10}$), $+7.20\%$ at $H=336$ ($p_{\text{adj}} = 1.34\times 10^{-4}$), $+5.99\%$ at $H=720$.
  - **Electricity**: Highly stable gain of $+2.13\% \sim +2.86\%$ across all horizons ($H=96$: $0.1159 \to 0.1129, p_{\text{adj}} = 8.46\times 10^{-20}$; $H=720$: $0.2040 \to 0.1982, p_{\text{adj}} = 4.15\times 10^{-31}$).
  - **Traffic**: Modest gain $+0.61\% \sim +1.40\%$ ($H=720: p_{\text{adj}} = 5.36\times 10^{-13}$).
- **Key Finding 2 (Catastrophic Degradation on Solar-Energy)**:
  - Multivariate mode severely harms performance, and degradation compounds across horizon:
    - $H=96$: $-7.72\%$ ($0.2087 \to 0.2248$ MSE)
    - $H=192$: $-12.88\%$ ($0.2396 \to 0.2705$ MSE)
    - $H=336$: $-20.63\%$ ($0.2704 \to 0.3262$ MSE, $p_{\text{adj}} = 3.83\times 10^{-5}$)
    - $H=720$: **$-31.95\%$** ($0.3039 \to 0.4010$ MSE, $p_{\text{adj}} = 1.12\times 10^{-12}$)
  - Root Cause: Solar photovoltaic sensors exhibit rapid localized cloud transients and diurnal phase mismatch. Without alignment, variate attention routes non-stationary noise, compounding error across multi-patch rollout.
- **Key Finding 3 (Neutrality on High-Frequency Industrial Telemetry)**:
  - ETTm1: $-3.13\%$ to $+0.27\%$; ETTm2: $-0.90\%$ to $+0.79\%$. Variate attention is essentially inactive.

### 4.2 E2: Causal Interventions (`e2_interventions_summary.csv` & `grouped.csv`)
16 target-centric interventions applied to non-target channels while keeping the target channel 100% intact (evaluated on 100 windows across ETTh1, Weather, Electricity, Solar):
1. **N1_perm (Channel Permutation)**:
   - Max absolute difference $|\Delta\text{MSE}| \le 9.54 \times 10^{-7}$, Sensitivity $S < 10^{-6}$, Reliance $R \approx 0.0000$.
   - Confirms theoretical permutation equivariance (`use_rope_var: false`).
2. **I2_swap_unaligned vs. I3_swap_aligned (Phase Alignment vs. Static Distribution)**:
   - On Electricity:
     - I2 (unaligned 30-day + 7-step swap): **$R = 2.065$** ($p_{\text{adj}} = 2.01\times 10^{-6}$, MSE increases by $+3.37\%$, actively misleading the model).
     - I3 (calendar-week aligned swap): **$R = 0.505$** ($p_{\text{adj}} = 0.350$, MSE increase drops to $+0.82\%$).
     - **Takeaway**: $R(I2) \gg R(I3)$ conclusively proves that variate attention relies on **instantaneous micro-phase synchronization**, not static diurnal/marginal distributions!
3. **I4_iaaft (Phase Randomization via IAAFT)**:
   - On Electricity: $R = 1.986$ ($p_{\text{adj}} = 7.68\times 10^{-6}$, $+3.24\%$ MSE increase).
   - Preserving autocorrelation and amplitude distribution while destroying phase angles completely erases multi-channel gains.
4. **P1_lead_16 & P1_lead_32 (Positive Controls)**:
   - Injecting 16-step forward leading indicator drops error by **$-5.30\%$** ($R = -3.24, p_{\text{adj}} = 1.34\times 10^{-9}$).
   - Injecting 32-step forward leading indicator drops error by **$-14.52\%$** ($R = -8.87, p_{\text{adj}} = 4.17\times 10^{-12}$).
   - On Solar: full diagnostic confirmed P1 reduces first-$k$ step MSE by **$-31.83\%$** (95% CI: $[-37.8\%, -23.6\%]$).
   - Proves the model architecture possesses latent routing capability for genuine causal predictive signals.

### 4.3 E3: Synthetic Structure & Kalman Filter Oracles (`e3_synthetic_summary.csv`)
- **Generators G1-G6**: Linear Gaussian state-space models including low-rank factor ($G1$), local blocks ($G2$), structural drift ($G3$), lead-lag ($G5$), and null independence ($G6$).
- **Analytical Oracles**: Exact Kalman filters ($UV^*, MV^*$) derived via discrete-time algebraic Riccati equations (DARE).
- **Key Finding**:
  - All configurations strictly satisfy $MV^* \le UV^*$, and TimesFM-3 does not beat the oracle.
  - On verified G2 block structure (`block_size=8, gamma=0.85, alpha=0.98, sigma_eps=0.2`), theoretical oracle gap is $+8.50\% \ge 5\%$, confirming proper scaling.

### 4.4 E5: Channel Budget $K$ & RTX 4090 Inference Profiling (`e5_grouping_summary.csv`)
- Evaluated channel budgets $K \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ across strategies: `top_k` (by correlation), `bottom_k`, `random`.
- **Accuracy Dynamics**:
  - Electricity: Top-K and Random reach lowest MSE at **$K=32$** (`0.1260`), outperforming $K=1$ (`0.1281`). Beyond $K=32$, $K=128$ MSE rebounds to `0.1271` (accuracy saturates).
  - Solar: Adding channels monotonically degrades MSE: $0.2162$ at $K=1 \to 0.2566$ at $K=32 \to 0.2970$ at $K=128$.
- **Hardware Discontinuity on RTX 4090**:
  - Evaluator enforces 32-channel forward chunking:
    - $K \le 32$: single pass, constant latency **$44 \sim 53\text{ ms}$**.
    - $K = 64$: 2 sequential passes, latency jumps to **$82.7\text{ ms}$**.
    - $K = 128$: 4 passes, latency quadruples to **$158.9\text{ ms}$** (+260% compute).
- **Engineering Recommendation**: Always cap channel budget at $K=32$, selecting Top-32 correlated series to maximize accuracy and minimize latency.

---

## 5. Figures Mapping

All figures are in `figures/` (vector PDF and high-res PNG):
- **Fig 1 (`fig1_four_quadrant.pdf` / `.png`)**: Four-Quadrant Diagnostic Mapping (Teaser).
  - Quadrant I (True Multi-Channel Synergy): Weather, Electricity, ETTh2.
  - Quadrant II (Unexploited Dependence): Synthetic G5 lead-lag.
  - Quadrant III (Independent Telemetry): ETTm1, ETTm2.
  - Quadrant IV (Catastrophic Negative Transfer): Solar.
- **Fig 2 (`fig2_interventions.pdf` / `.png`)**: Intervention Reliance $R_\pi$ on Electricity and Cross-Dataset Sensitivity $S_\pi$ across 16 interventions.
- **Fig 3 (`fig3_synthetic_gc.pdf` / `.png`)**: Exact Kalman Oracle Gap Closure across structural couplings.
- **Fig 4 (`fig4_channel_budget.pdf` / `.png`)**: Channel Budget $K$ Pareto Frontier (Accuracy vs. Latency on RTX 4090).
- **Table 2 (`table2_ltsf_main.tex`)**: Ready-to-compile LaTeX table for the main LTSF benchmark.

---

## 6. Practical Database Engineering Guidelines (Section 6)

1. **Adaptive Mode Gating via Spectral Analysis**:
   Compute validation-set correlation spectrum. If leading eigenvalue ratio $\rho = \lambda_1 / \sum \lambda_k < 0.35$ or strong unaligned diurnal transients exist (e.g. Solar, ETTm), gate to Univariate ($\UV$) mode. Prevents negative transfer and saves 50% compute.
2. **Correlation-Driven 32-Channel Budgeting**:
   In high-dimensional physical telemetry (e.g. 862 Traffic sensors, 321 Electricity meters), select the Top-32 correlated channels. Captures 100% of available gain while preventing the $2\times \sim 4\times$ latency jump caused by official 32-channel chunking.
3. **Chunking-Aware Batch Alignment**:
   In multi-tenant database serving engines, align batch channel dimensions to multiples of 32 to eliminate redundant forward padding with repeated tokens.
