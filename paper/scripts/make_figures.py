"""Generate all paper figures from the CSV files in ../data (vector PDF, LNCS text width 12.2 cm).

Run:  python3 scripts/make_figures.py   (from the paper root)
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- style: Computer Modern to match LNCS body text ----
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TEXTW = 12.2 / 2.54  # inches
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"            # validated categorical slots 1-3
HORIZON_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, steps 250/400/550/700
MARKERS = ["o", "s", "^", "D"]
NAMES = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2", "weather": "Weather",
         "electricity": "Electricity", "solar": "Solar-Energy", "traffic": "Traffic"}


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ======================================================================
# Fig. 1  E1: benefit B with 95% CI, dataset x horizon
# ======================================================================
e1 = pd.read_csv(DATA / "e1_ltsf_sampled.csv")
order = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic", "solar"]
fig, ax = plt.subplots(figsize=(TEXTW, 2.55))
hs = [96, 192, 336, 720]
for j, h in enumerate(hs):
    sub = e1[e1.horizon == h].set_index("dataset").loc[order]
    y = np.arange(len(order)) + (j - 1.5) * 0.17
    sig = sub.B_significant.values.astype(bool)
    ax.errorbar(sub.Benefit_B_pct, y, xerr=[sub.Benefit_B_pct - sub.B_lo, sub.B_hi - sub.Benefit_B_pct],
                fmt="none", ecolor=HORIZON_RAMP[j], elinewidth=0.9, capsize=0)
    ax.scatter(sub.Benefit_B_pct[sig], y[sig], s=16, marker=MARKERS[j], color=HORIZON_RAMP[j],
               edgecolor=HORIZON_RAMP[j], lw=0.8, zorder=3, label=f"$H={h}$")
    ax.scatter(sub.Benefit_B_pct[~sig], y[~sig], s=16, marker=MARKERS[j], facecolor="white",
               edgecolor=HORIZON_RAMP[j], lw=0.8, zorder=3)
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([NAMES[d] + ("$^\\dagger$" if d == "traffic" else "") for d in order])
ax.invert_yaxis()
ax.set_xlabel("Benefit $B$ of \\textsc{mv} over \\textsc{uv} mode (%)".replace("\\textsc{mv}", "MV").replace("\\textsc{uv}", "UV"))
grid(ax)
ax.legend(ncol=4, loc="lower left", frameon=False, handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(0, 1.0))
ax.text(1.0, 1.02, "filled: 95% CI excludes 0", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7)
fig.savefig(FIG / "fig1_benefit.pdf")
plt.close(fig)

# ======================================================================
# Fig. 2  E2: (a) reliance R on Electricity & Weather, (b) positive control P1
# ======================================================================
e2 = pd.read_csv(DATA / "e2_interventions_summary.csv")
ivs = [("N2_dup", "N2 copy of target"), ("N3_lag_16", "N3 lagged copy"), ("I6_shift_1step", "I5 shift 1 step"),
       ("I6_shift_1h", "I5 shift 1 h"), ("I3_swap_aligned", "I3 week-aligned swap"),
       ("I2_swap_unaligned", "I2 unaligned swap"), ("I4_iaaft", "I4 IAAFT surrogate"),
       ("I6_shift_1d", "I5 shift 1 day"), ("I1_shuffle", "I1 time shuffle"), ("N4_noise", "N4 Gaussian noise")]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXTW, 2.7), gridspec_kw={"width_ratios": [1.55, 1]})
XMAX = 4.0
for k, (ds, col, mk, off) in enumerate([("electricity", C1, "o", -0.16), ("weather", C2, "s", 0.16)]):
    sub = e2[e2.dataset == ds].set_index("intervention")
    for i, (iv, _) in enumerate(ivs):
        r, lo, hi = sub.loc[iv, ["R", "R_lo", "R_hi"]]
        y = i + off
        ax.plot([lo, min(hi, XMAX)], [y, y], color=col, lw=0.9)
        if hi > XMAX:
            ax.annotate("", xy=(XMAX + 0.12, y), xytext=(XMAX - 0.05, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=5))
        ax.scatter(min(r, XMAX), y, marker=mk, s=15, color=col, zorder=3,
                   label=NAMES[ds] if i == 0 else None)
for x in (0, 1):
    ax.axvline(x, color=MUTED, lw=0.6, ls="-" if x == 0 else "--")
ax.set_yticks(range(len(ivs)))
ax.set_yticklabels([lbl for _, lbl in ivs])
ax.invert_yaxis()
ax.set_xlim(-1.2, XMAX + 0.3)
ax.set_xlabel("Reliance $R_\\pi$ (1 = benefit erased)")
ax.set_title("(a) Reliance on context properties", fontsize=8, loc="left")
grid(ax)
ax.legend(frameon=False, loc="upper right", handletextpad=0.2)

pdat = [("ETTh1", "ETTh1"), ("weather", "Weather"), ("solar", "Solar"), ("electricity", "Electr.")]
for j, (iv, col, mk) in enumerate([("P1_lead_16", C1, "o"), ("P1_lead_32", C2, "s")]):
    for i, (ds, _) in enumerate(pdat):
        row = e2[(e2.dataset == ds) & (e2.intervention == iv)].iloc[0]
        y = i + (j - 0.5) * 0.3
        bx.plot([row.dPI_lo, row.dPI_hi], [y, y], color=col, lw=0.9)
        bx.scatter(row.dPI_pct, y, marker=mk, s=15, color=col, zorder=3, label=f"$k={iv.split('_')[-1]}$" if i == 0 else None)
bx.axvline(0, color=MUTED, lw=0.6)
bx.set_yticks(range(len(pdat)))
bx.set_yticklabels([n for _, n in pdat])
bx.invert_yaxis()
bx.set_xlabel("$\\Delta$MSE vs. MV mode (%)")
bx.set_title("(b) Leading-indicator control P1", fontsize=8, loc="left")
grid(bx)
bx.legend(frameon=False, loc="lower left", handletextpad=0.2)
fig.tight_layout(w_pad=1.0)
fig.savefig(FIG / "fig2_interventions.pdf")
plt.close(fig)

# ======================================================================
# Fig. 3  E3: lead-lag generator G5 - available vs. realized cross-channel gain
# ======================================================================
e3 = pd.read_csv(DATA / "e3_synthetic_summary.csv")
g5 = e3[e3.generator == "G5"].sort_values("lag")
oracle_gain = (g5.MSE_UV_Oracle - g5.MSE_MV_Oracle) / g5.MSE_UV_Oracle * 100
model_gain = g5.Benefit_B_pct
fig, ax = plt.subplots(figsize=(TEXTW * 0.62, 2.0))
x = np.arange(len(g5))
w = 0.36
ax.bar(x - w / 2 - 0.02, oracle_gain, w, color=C1, label="Kalman oracle")
ax.bar(x + w / 2 + 0.02, model_gain, w, color=C2, label="TimesFM-3")
for xi, (og, mg, gc) in enumerate(zip(oracle_gain, model_gain, g5.Gap_Closure_GC)):
    if np.isfinite(gc):
        ax.text(xi, max(og, mg) + 1.0, f"GC$\\,=\\,${gc:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
g6 = e3[(e3.generator == "G6") & (e3.N == 8)].Benefit_B_pct.iloc[0]
ax.axhline(g6, color=MUTED, lw=0.7, ls="--")
ax.text(len(g5) - 0.45, g6, "G6", ha="left", va="center", fontsize=6.5, color=MUTED, clip_on=False)
ax.set_xticks(x)
ax.set_xticklabels([f"{int(l)}" for l in g5.lag])
ax.set_xlabel("Lag $\\ell$ of the leading channel (steps)")
ax.set_ylabel("MV gain over UV (%)")
ax.axhline(0, color=MUTED, lw=0.6)
grid(ax, "y")
ax.legend(frameon=False, loc="upper left")
fig.savefig(FIG / "fig3_leadlag.pdf")
plt.close(fig)

# ======================================================================
# Fig. 4  E5: (a,b) MSE change vs. K=1 per dataset, (c) latency
# ======================================================================
e5 = pd.read_csv(DATA / "e5_grouping_summary.csv")
fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.05), gridspec_kw={"width_ratios": [1, 1, 0.9]})
strat = [("top_k", "top-$K$ corr.", C1, "o"), ("random", "random", C2, "s"), ("bottom_k", "bottom-$K$ corr.", C3, "^")]
for ax, ds in zip(axs[:2], ["electricity", "solar"]):
    sub = e5[e5.dataset == ds]
    base = sub[sub.K == 1].MSE.iloc[0]
    for s, lbl, col, mk in strat:
        ss = sub[sub.strategy == s].sort_values("K")
        ax.plot(ss.K, (ss.MSE / base - 1) * 100, color=col, lw=1.2, marker=mk, ms=3.2, label=lbl)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(32, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 8, 32, 128])
    ax.set_xticklabels(["2", "8", "32", "128"])
    ax.set_xlabel("Channel budget $K$")
    ax.set_title(f"({'a' if ds == 'electricity' else 'b'}) {NAMES[ds]}", fontsize=8, loc="left")
    grid(ax, "y")
axs[0].set_ylabel("MSE change vs. $K=1$ (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.05), handlelength=1.6)
lat = e5[e5.dataset == "electricity"].groupby("K").latency_median_ms.median()
off = pd.read_csv(DATA / "e5_official_chunked_from_log.csv")
lat_off = off[off.dataset == "electricity"].groupby("K").latency_ms.median()
axs[2].plot(lat_off.index, lat_off.values, color=C2, lw=1.2, marker="s", ms=3.2, label="official")
axs[2].plot(lat.index, lat.values, color=C1, lw=1.2, marker="o", ms=3.2, label="single pass")
axs[2].legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1, fontsize=6.5)
axs[2].set_xscale("log", base=2)
axs[2].set_xticks([2, 8, 32, 128])
axs[2].set_xticklabels(["2", "8", "32", "128"])
axs[2].set_ylim(0, 245)
axs[2].set_xlim(0.8, 200)
axs[2].set_xlabel("Channel budget $K$")
axs[2].set_ylabel("Latency per call (ms)")
axs[2].set_title("(c) Latency, RTX 4090", fontsize=8, loc="left")
grid(axs[2], "y")
fig.tight_layout(w_pad=0.8)
fig.savefig(FIG / "fig4_budget.pdf")
plt.close(fig)
print("figures written to", FIG)
