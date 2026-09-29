"""
P1（领先指标正对照）诊断脚本 —— 独立实现，用来核对 probe/interventions.py 里的 P1。

放到 /root/autodl-tmp/DASFAA/experiments/ 下运行：
    cd /root/autodl-tmp/DASFAA
    python3 experiments/diag_p1_solar.py --dataset solar --windows 100 --horizon 96
    # 其他数据集对照：
    python3 experiments/diag_p1_solar.py --dataset weather --windows 100
    python3 experiments/diag_p1_solar.py --dataset electricity --windows 100

做四件事：
  1. 单元测试：自己构造的 P1 通道（SNR=∞）必须满足
       x[-k:] == y_true[:k]  且  x[:-k] == target_ctx[k:]
     并与 probe.interventions.apply_p1_leading_indicator 的输出逐位比较（若可导入）。
  2. 扩大样本：100 个窗口 × 分层抽样目标，k ∈ {16, 32}，SNR ∈ {∞, 10}。
  3. 按 horizon 拆分误差：第 1..k 步、第 k+1..H 步分别计算 MSE。
  4. 按起报时刻（夜间 / 过渡 / 白天）分组；并打印训练集的日内均值曲线，用于核对时刻换算。

输出：
  results/diag_p1_<dataset>_raw.csv      逐 (窗口, 目标, 条件) 的误差
  results/diag_p1_<dataset>_summary.csv  汇总 + 按窗口 bootstrap 的 95% CI
  results/diag_p1_<dataset>_diurnal.csv  训练集日内均值曲线（仅 10min 数据）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import MODEL_DIR, PRIMARY_SEED, RESULTS_DIR  # noqa: E402
from probe.data import load_dataset  # noqa: E402
from probe.models.timesfm3 import TimesFM3Wrapper  # noqa: E402

BLOCK = 32


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def fixed_group(t_idx: int, n: int):
    """与 E2 相同的固定分组：N<=32 用全部通道；否则取官方 32 通道块。"""
    if n <= BLOCK:
        return list(range(n)), t_idx
    start = (t_idx // BLOCK) * BLOCK
    grp = list(range(start, min(start + BLOCK, n)))
    return grp, t_idx - start


def stratified_targets(train: np.ndarray, n_targets: int, seed: int):
    """按与其他通道的平均 |相关系数| 分四档抽样；排除所在块只有 1 个通道的目标。"""
    n = train.shape[1]
    valid = [i for i in range(n) if len(fixed_group(i, n)[0]) > 1]
    if len(valid) <= n_targets:
        return valid
    c = np.corrcoef(train, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 0.0)
    score = np.abs(c).mean(axis=1)
    order = [i for i in np.argsort(score) if i in set(valid)]
    rng = np.random.RandomState(seed)
    picked = []
    for q in np.array_split(np.array(order), 4):
        m = min(len(q), n_targets // 4)
        picked.extend(rng.choice(q, size=m, replace=False).tolist())
    return sorted(picked)


def build_p1(target_ctx: np.ndarray, y_true: np.ndarray, k: int, snr, rng):
    """x(t) = y(t+k) + eps，在上下文坐标系中：x = concat(target_ctx[k:], y_true[:k]) + eps。
    噪声方差 = Var(目标上下文) / SNR；SNR=inf 时不加噪声。"""
    x = np.concatenate([target_ctx[k:], y_true[:k]]).astype(np.float32)
    if np.isfinite(snr):
        sig_var = float(np.var(target_ctx))
        x = x + rng.normal(0.0, np.sqrt(sig_var / snr), size=x.shape).astype(np.float32)
    return x


def replace_slot(grp_len: int, t_pos: int) -> int:
    """被领先指标替换的上下文通道：组内第一个非目标通道（固定规则，便于复现）。"""
    return 0 if t_pos != 0 else 1


def diurnal_regime(hour: float) -> str:
    if hour < 5.0 or hour >= 20.0:
        return "night"
    if hour < 8.0 or hour >= 17.0:
        return "twilight"
    return "day"


def window_bootstrap(df: pd.DataFrame, col_base: str, col_new: str, n_boot=2000, seed=0):
    """按窗口聚合后重抽样：返回 (改变百分比, lo, hi)。改变 = (Σnew − Σbase)/Σbase。"""
    w = df.groupby("window_idx")[[col_base, col_new]].sum()
    b, n = w[col_base].to_numpy(), w[col_new].to_numpy()
    point = (n.sum() - b.sum()) / b.sum() * 100
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(w), len(w))
        boots.append((n[idx].sum() - b[idx].sum()) / b[idx].sum() * 100)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--targets", type=int, default=64)
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = ap.parse_args()

    H, L = args.horizon, args.context
    ks = [16, 32]
    snrs = [np.inf, 10.0]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    ds = load_dataset(args.dataset, context_length=L, horizon=H)
    n = ds.num_variates
    train = ds.norm_values[: ds.border2s[0]]
    targets = stratified_targets(train, args.targets, args.seed)
    windows = ds.get_sampled_window_indices(num_samples=args.windows, seed=args.seed)
    print(f"[{args.dataset}] N={n}, targets={len(targets)}, windows={len(windows)}, H={H}")

    # ---------- 时刻核对（10 分钟数据：每天 144 步） ----------
    freq = str(ds.info.get("freq", "")).lower()
    steps_per_day = 144 if "10" in freq and "min" in freq else (96 if "15" in freq else 24)
    prof = np.array([train[s::steps_per_day].mean() for s in range(steps_per_day)])
    pd.DataFrame({"step_of_day": np.arange(steps_per_day), "mean_norm_value": prof}).to_csv(
        RESULTS_DIR / f"diag_p1_{args.dataset}_diurnal.csv", index=False
    )
    low = np.where(prof <= prof.min() + 0.05 * (prof.max() - prof.min()))[0]
    print(f"日内均值最低的步（step_of_day，接近最小值 5% 以内）：{low.min()}–{low.max()}"
          f"（若为 10min 数据，对应 {low.min()/6:.1f}h–{low.max()/6:.1f}h；夜间应落在这里）")

    # ---------- 1. 单元测试 ----------
    w0 = windows[0]
    ctx0, tgt0 = ds.get_window(w0)
    t0 = targets[0]
    grp0, tp0 = fixed_group(t0, n)
    for k in ks:
        x = build_p1(ctx0[t0], tgt0[t0, :H], k, np.inf, rng)
        assert np.array_equal(x[-k:], tgt0[t0, :k].astype(np.float32)), "x[-k:] != y_true[:k]"
        assert np.array_equal(x[:-k], ctx0[t0, k:].astype(np.float32)), "x[:-k] != target_ctx[k:]"
    print("单元测试 1 通过：自建 P1 通道对齐正确（SNR=∞）。")
    try:
        from probe.interventions import apply_p1_leading_indicator
        g = ctx0[grp0].copy()
        for k in ks:
            out = apply_p1_leading_indicator(g, tgt0[t0, :H], tp0, lead_k=k, snr=1e12, seed=args.seed)
            ctx_new, meta = out if isinstance(out, tuple) else (out, {})
            tpos = meta.get("target_pos", tp0) if isinstance(meta, dict) else tp0
            assert np.array_equal(ctx_new[tpos], g[tp0]), "官方 P1 修改了目标通道"
            ref = build_p1(g[tp0], tgt0[t0, :H], k, np.inf, rng)
            # 找到被替换或新增的那一行
            diffs = [np.max(np.abs(ctx_new[r] - ref)) for r in range(ctx_new.shape[0])]
            best = int(np.argmin(diffs))
            print(f"  官方 P1(k={k}): 行数 {g.shape[0]}→{ctx_new.shape[0]}，"
                  f"与自建通道最接近的是第 {best} 行，最大绝对差 {diffs[best]:.3e}，meta={meta}")
            if diffs[best] > 1e-3:
                print("  !!! 官方 P1 与自建 P1 不一致（SNR≈∞ 下差异 > 1e-3），请检查 interventions.py 的平移方向和对齐。")
    except Exception as e:  # noqa: BLE001
        print(f"  （未能对照官方 P1：{type(e).__name__}: {e}）")

    # ---------- 2–4. 主实验 ----------
    conds = [("MV", None, None)] + [(f"P1_k{k}_snr{'inf' if not np.isfinite(s) else int(s)}", k, s)
                                   for k in ks for s in snrs]
    rows = []
    t_start = time.time()
    for wi, w in enumerate(windows):
        ctx, tgt = ds.get_window(w)
        global_t = ds.border1s[2] + w + L  # 预测起点（与现有 Solar 诊断的约定一致；请用上面的日内曲线核对）
        hour = ((global_t % steps_per_day) / steps_per_day) * 24.0
        regime = diurnal_regime(hour) if steps_per_day == 144 else "n/a"

        from collections import defaultdict
        batches_by_n = defaultdict(list)
        meta_by_n = defaultdict(list)
        uv_batch = []
        for t in targets:
            grp, tp = fixed_group(t, n)
            g = ctx[grp].astype(np.float32).copy()
            y = tgt[t, :H]
            uv_batch.append(g[tp : tp + 1])
            n_vars = len(grp)
            for name, k, s in conds:
                gi = g.copy()
                if k is not None:
                    gi[replace_slot(len(grp), tp)] = build_p1(g[tp], y, k, s, rng)
                    assert np.array_equal(gi[tp], g[tp])
                batches_by_n[n_vars].append(gi)
                meta_by_n[n_vars].append((t, tp, name, k))

        preds = []
        meta_rows = []
        for n_vars in sorted(batches_by_n.keys()):
            p_n, _ = WRAP.forecast(batches_by_n[n_vars], horizon=H, univariate=False, return_quantiles=False)
            preds.extend(p_n)
            meta_rows.extend(meta_by_n[n_vars])

        preds_uv, _ = WRAP.forecast(uv_batch, horizon=H, univariate=True, return_quantiles=False)
        uv_err = {t: preds_uv[j][0] for j, t in enumerate(targets)}

        for (t, tp, name, k), p in zip(meta_rows, preds):
            y = tgt[t, :H]
            yhat = p[tp]
            kk = k if k is not None else 32  # MV 行按 k=32 的切分也记一份，便于对照
            rows.append({
                "dataset": args.dataset, "window_idx": w, "hour": hour, "regime": regime,
                "target_idx": t, "cond": name,
                "mse_all": float(np.mean((yhat - y) ** 2)),
                "mse_1_16": float(np.mean((yhat[:16] - y[:16]) ** 2)),
                "mse_17_H_k16": float(np.mean((yhat[16:] - y[16:]) ** 2)),
                "mse_1_32": float(np.mean((yhat[:32] - y[:32]) ** 2)),
                "mse_33_H_k32": float(np.mean((yhat[32:] - y[32:]) ** 2)),
                "mse_uv_all": float(np.mean((uv_err[t] - y) ** 2)),
            })
        if (wi + 1) % 10 == 0:
            print(f"  [{wi+1}/{len(windows)}] {time.time()-t_start:.0f}s", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / f"diag_p1_{args.dataset}_raw.csv", index=False)

    # ---------- 汇总：相对 MV 的误差变化（负数 = P1 让误差下降） ----------
    base = raw[raw.cond == "MV"].set_index(["window_idx", "target_idx"])
    out = []
    for name, k, s in conds[1:]:
        cur = raw[raw.cond == name].set_index(["window_idx", "target_idx"])
        m = base.join(cur, lsuffix="_mv", rsuffix="_p1").reset_index()
        seg_first = f"mse_1_{k}"
        seg_rest = f"mse_{k+1}_H_k{k}"
        for regime in ["all"] + (sorted(m["regime_mv"].unique()) if steps_per_day == 144 else []):
            mm = m if regime == "all" else m[m["regime_mv"] == regime]
            if mm["window_idx"].nunique() < 5:
                continue
            res = {"cond": name, "regime": regime, "n_windows": mm["window_idx"].nunique()}
            for label, col in [("all", "mse_all"), (f"steps_1_{k}", seg_first), (f"steps_{k+1}_H", seg_rest)]:
                p, lo, hi = window_bootstrap(mm, f"{col}_mv", f"{col}_p1")
                res[f"dMSE_pct_{label}"] = round(p, 2)
                res[f"CI_{label}"] = f"[{lo:.2f}, {hi:.2f}]"
            out.append(res)
    # 也给出 MV 相对 UV 的收益，作为背景
    pmv, lomv, himv = window_bootstrap(raw[raw.cond == "MV"], "mse_uv_all", "mse_all")
    summary = pd.DataFrame(out)
    summary.to_csv(RESULTS_DIR / f"diag_p1_{args.dataset}_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print(f"[{args.dataset}] MV 相对 UV 的误差变化：{pmv:+.2f}% [{lomv:.2f}, {himv:.2f}]（负数 = MV 更好）")
    print("P1 相对 MV 的误差变化（负数 = 领先指标让误差下降），按窗口 bootstrap 95% CI：")
    print(summary.to_string(index=False))
    print("=" * 100)
    print("解读：\n"
          "  - SNR=∞ 时 steps_1_k 仍不下降 → 模型无法利用完美的领先信息（真实现象，照实记录）；\n"
          "  - steps_1_k 下降但 steps_k+1_H 上升 → 模型读到了信息，但后续预测被带偏；\n"
          "  - SNR=∞ 正常而 SNR=10 失败 → 问题出在噪声构造（例如夜间也加了噪声）。")


if __name__ == "__main__":
    WRAP = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device="cuda")
    main()
