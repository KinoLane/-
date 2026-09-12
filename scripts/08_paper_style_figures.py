"""Paper-style figures (11 charts) built from our own CSI 300 experiment runs.

.. note::
   **Superseded** by ``scripts/09_reference_figures.py``, which draws the delivered
   set into ``results/figures_reference/`` from the dedicated sweep grids.  The
   charts this script writes are an earlier layout attempt (``v1`` falls back to
   whatever grids happened to be finished), so do not compare them against
   ``D:\\figures`` and do not treat them as a deliverable.

The reference figures live in ``D:\\figures``.  This script reproduces their
*layout and style* (grouped bars / line charts with a single legend row on top,
large fonts, boxed axes) with our own measured results.

Two data sources are supported:

``v1``  the completed grids that already exist in ``results/`` -- the series are
        our strategy variants (years on the x axis where the reference sweeps a
        risk target);
``v2``  the dedicated sweep grids (network architectures x return target,
        architectures x entropy, other markets, tree baselines) once they have
        finished; ``--source auto`` picks ``v2`` when the sweep runs are present.

Usage::

    python scripts/08_paper_style_figures.py --out D:\\金创\\figures_ours

The ``--out`` folder above is where the (superseded) draft charts live; the
delivered set is written by ``scripts/09_reference_figures.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# --------------------------------------------------------------------------- #
# palette (matches the reference figures)
# --------------------------------------------------------------------------- #
C_RED = "#f12b2b"
C_YELLOW = "#f2c018"
C_BLUE = "#2b46ea"
C_GREEN = "#129c3c"
C_ORANGE = "#f28c28"
C_PURPLE = "#8e44ad"
C_TEAL = "#16a085"
C_GREY = "#7f7f7f"
SERIES_COLORS = [C_RED, C_YELLOW, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_TEAL]

NOTES_V1 = """# 论文风格图（第一版，暂用已完成的网格）

对照图在 `D:\\figures`。本文件夹的 11 张图使用**我们自己跑的 CSI 300 实验**数据，
画布/字号/图例位置与对照图一致（1322x599 等像素尺寸、横轴 19pt 刻度、无标题）。

| 文件 | 对照图 | 本版 x 轴 / 系列 | 说明 |
|------|--------|------------------|------|
| fig02_std_vs_target.png | Fig2 | 年 / 5 个策略 | 年化口径改为按日历年的月收益标准差 |
| fig03_sharpe_vs_target.png | Fig3 | 年 / 5 个策略 | 夏普比率（月度×sqrt(12)） |
| fig04_te_vs_target.png | Fig4 | 年 / 5 个策略 | 相对沪深300的跟踪误差 |
| fig05_cumulative_return.png | Fig5 | 时间 / 5 个策略 | 灰度配色与对照图一致 |
| fig08_entropy_sharpe.png | Fig8 | 组合熵 / 每策略一点 | 对照图为 5 种网络结构扫描 |
| fig09_entropy_cumulative_return.png | Fig9 | 时间 / 3 个策略 | 两个窗口上下排布 |
| fig10_out_of_sample_return.png | Fig10 | 2023-04~2024-06 月收益 | 左右两块面板 |
| fig11_other_market.png | Fig11 | 时间 / 4 个策略 | 用「全宇宙 CSI 300」替代其它市场 |
| fig12_score_rule.png | Fig12 | 时间 / 4 个策略 | 用 μ-σ 评分规则网格替代其它市场 |
| fig13_model_comparison.png | Fig13 | 时间 / 5 个策略 | 模型比较（树模型基线待补） |
| fig14_baselines.png | Fig14 | 时间 / 4 个策略 | 与基准比较 |

> 注：对照图 Fig2/3/4 的横轴是「目标收益 μ」的 7 档扫描，Fig8 是「熵」的 5 档扫描，
> Fig11/12 是另外两个市场，Fig13/14 含 AdaBoost / XGBoost 树模型基线。
> 这些维度的专用扫描网格正在补跑（架构×目标收益、架构×熵、其它市场、树基线），
> 跑完后同目录图会以相同画风重绘为最终版。
"""

# --------------------------------------------------------------------------- #
# run directories
# --------------------------------------------------------------------------- #
RUNS = {
    "main": RESULTS / "csi300_20260910_235309",
    "score": RESULTS / "csi300_score_rule_20260911",
    "extra": RESULTS / "csi300_extra_returns",
    "fu": RESULTS / "csi300_full_universe",
    "sweep_arch": RESULTS / "csi300_arch_sweep",
    "sweep_ent": RESULTS / "csi300_entropy_sweep",
    "csi500": RESULTS / "csi500_arch_sweep",
    "sse50": RESULTS / "sse50_arch_sweep",
    "tree": RESULTS / "tree_baselines",
}

FOLDS = ["f1_2018_2019", "f2_2020_2021", "f3_2022_2026"]


def fold_dir(run_key: str, exp: str, fold: str) -> Path:
    return RUNS[run_key] / exp / fold


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def paper_style(axis_label_size: float = 20.0, tick_size: float = 17.0,
                title_size: float = 21.0, legend_size: float = 15.5) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": tick_size,
        "axes.titlesize": title_size,
        "axes.labelsize": axis_label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
    })


def style_axes(ax, grid: bool = False, grid_axis: str = "y") -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
    if grid:
        ax.grid(True, axis=grid_axis, color="#c9c9c9", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def net_returns(run_key: str, exp: str, fold: str) -> pd.Series:
    f = fold_dir(run_key, exp, fold) / "returns.csv"
    df = pd.read_csv(f, parse_dates=["date"])
    s = df.set_index("date")["ret_net"].fillna(0.0)
    return s


def bench_returns(run_key: str, exp: str, fold: str) -> pd.Series:
    f = fold_dir(run_key, exp, fold) / "returns.csv"
    df = pd.read_csv(f, parse_dates=["date"])
    return df.set_index("date")["bench"].fillna(0.0)


def monthly_returns(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    return (1.0 + s).groupby([s.index.year, s.index.month]).prod() - 1.0


def monthly_frame(run_key: str, exp: str) -> pd.DataFrame:
    """Monthly net portfolio return and monthly benchmark return, per fold."""
    out = []
    for fold in FOLDS:
        if not fold_dir(run_key, exp, fold).exists():
            continue
        net = monthly_returns(net_returns(run_key, exp, fold))
        ben = monthly_returns(bench_returns(run_key, exp, fold))
        for (yy, mm) in net.index:
            d = pd.Timestamp(year=int(yy), month=int(mm), day=1) + pd.offsets.MonthEnd(0)
            rec = {"date": d, "year": int(yy), "net": float(net.loc[(yy, mm)])}
            rec["bench"] = float(ben.loc[(yy, mm)]) if (yy, mm) in ben.index else np.nan
            out.append(rec)
    df = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    return df


def daily_frame(run_key: str, exp: str) -> pd.DataFrame:
    out = []
    for fold in FOLDS:
        if not fold_dir(run_key, exp, fold).exists():
            continue
        s = net_returns(run_key, exp, fold)
        b = bench_returns(run_key, exp, fold)
        df = pd.DataFrame({"ret_net": s, "bench": b}).dropna(how="all")
        if not df.empty:
            out.append(df)
    if not out:
        return pd.DataFrame(columns=["ret_net", "bench"])
    df = pd.concat(out).sort_index()
    return df[~df.index.duplicated(keep="first")]


def weights_entropy(run_key: str, exp: str) -> float:
    """Mean Shannon entropy (nats) of the rebalanced portfolios."""
    vals = []
    for fold in FOLDS:
        f = fold_dir(run_key, exp, fold) / "weights.csv"
        if not f.exists():
            continue
        w = pd.read_csv(f, index_col=0).to_numpy(dtype=np.float64)
        for row in w:
            row = np.clip(row, 0.0, None)
            s = row.sum()
            if s <= 0:
                continue
            row = row / s
            nz = row[row > 0]
            vals.append(float(-(nz * np.log(nz)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def metrics(run_key: str, exp: str, fold: str) -> dict:
    return json.loads((fold_dir(run_key, exp, fold) / "metrics.json").read_text(encoding="utf-8"))


def pooled_metrics(run_key: str, exp: str) -> dict | None:
    rep = RUNS[run_key] / "report" / "pooled.csv"
    if not rep.exists():
        return None
    df = pd.read_csv(rep)
    col = "experiment" if "experiment" in df.columns else df.columns[0]
    row = df[df[col] == exp]
    return None if row.empty else row.iloc[0].to_dict()


def year_stats(run_key: str, exp: str) -> pd.DataFrame:
    """Per calendar-year std / sharpe / tracking error of the monthly returns."""
    m = monthly_frame(run_key, exp)
    rows = []
    for year, g in m.groupby("year"):
        net = g["net"].to_numpy(dtype=np.float64)
        ben = g["bench"].to_numpy(dtype=np.float64)
        if len(net) < 4:
            continue
        sd = float(np.std(net, ddof=1))
        sh = float(np.mean(net) / sd * np.sqrt(12.0)) if sd > 0 else np.nan
        diff = net - ben
        te = float(np.std(diff, ddof=1) * np.sqrt(12.0)) if len(diff) > 1 else np.nan
        rows.append({"year": int(year), "std": sd, "sharpe": sh, "te": te,
                     "mean": float(np.mean(net)), "n": len(net)})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# grouped bar helper (reference figures 2-4)
# --------------------------------------------------------------------------- #
def grouped_bars(out_dir: Path, path_name: str, title: str, ylabel: str, xlabel: str,
                 groups: list, values: dict, colors: list, figsize=(13.3, 5.8),
                 ylim=None, legend_loc="upper center", ncol=None) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    n_groups = len(groups)
    n_series = len(values)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    for k, (label, vals) in enumerate(values.items()):
        offs = (k - (n_series - 1) / 2.0) * width
        ax.bar(x + offs, vals, width=width * 0.92, label=label, color=colors[k % len(colors)],
               edgecolor="none", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=14)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax, grid=False)
    ax.legend(loc=legend_loc, ncol=ncol or min(n_series, 5), frameon=False,
              bbox_to_anchor=(0.5, 1.005), columnspacing=1.2, handlelength=1.4)
    fig.tight_layout()
    save(fig, out_dir, path_name)


# --------------------------------------------------------------------------- #
# the 11 figures
# --------------------------------------------------------------------------- #
def fig_bars_from_years(out_dir: Path, run_key: str, exps: list, labels: list,
                        title_prefix: str, colors: list, xlabel: str = "Year") -> None:
    stats = {exp: year_stats(run_key, exp) for exp in exps}
    years = sorted(set().union(*[set(s["year"]) for s in stats.values() if not s.empty]))
    years = [y for y in years if all(y in set(stats[e]["year"]) for e in exps)]

    def col(kind):
        return {lab: [float(stats[e].set_index("year").loc[y, kind]) for y in years]
                for e, lab in zip(exps, labels)}

    grouped_bars(out_dir, "fig02_std_vs_target.png", f"{title_prefix}", "std",
                 xlabel, years, col("std"), colors)
    grouped_bars(out_dir, "fig03_sharpe_vs_target.png", f"{title_prefix}", "sharpe ratio",
                 xlabel, years, col("sharpe"), colors, ylim=(-1.2, 2.2))
    grouped_bars(out_dir, "fig04_te_vs_target.png", f"{title_prefix}", "Tracking Error",
                 xlabel, years, col("te"), colors)


def fig_cumulative(out_dir: Path, path_name: str, title: str, series: dict,
                   colors: list, greyscale: bool = False, figsize=(13.22, 5.99),
                   ylabel: str = "Cumulative return", xlabel: str = "Time",
                   legend_ncol: int = 1, legend_loc: str = "upper left",
                   start=None, end=None) -> None:
    """Line chart of cumulative net returns; ``series`` maps label -> DataFrame."""
    paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    for k, (label, df) in enumerate(series.items()):
        d = df
        if start is not None:
            d = d[d.index >= pd.Timestamp(start)]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end)]
        if d.empty:
            continue
        cum = (1.0 + d["ret_net"]).cumprod()
        color = (["#1a1a1a", "#4d4d4d", "#7f7f7f", "#a6a6a6", "#cccccc"][k % 5]
                 if greyscale else colors[k % len(colors)])
        ax.plot(cum.index, cum.to_numpy(), label=label, color=color, linewidth=2.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=14)
    style_axes(ax)
    ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=False,
              columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    save(fig, out_dir, path_name)


def fig_scatter_entropy(out_dir: Path, path_name: str, title: str, points: dict,
                        colors: list, xlabel: str = "Entropy", ylabel: str = "sharpe ratio") -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=(13.22, 6.07))
    for k, (label, pts) in enumerate(points.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=11, linewidth=2.4, color=colors[k % len(colors)],
                label=label, markeredgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    style_axes(ax)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005),
              columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    save(fig, out_dir, path_name)


def fig_two_panels(out_dir: Path, path_name: str, title: str, panels: list,
                   figsize=(13.35, 12.44), ylabel: str = "Cumulative return",
                   xlabel: str = "Time") -> None:
    """Two stacked line panels (reference figure 9)."""
    paper_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, (sub_title, series, colors) in zip(axes, panels):
        for k, (label, df) in enumerate(series.items()):
            d = df.dropna(subset=["ret_net"])
            if d.empty:
                continue
            cum = (1.0 + d["ret_net"]).cumprod()
            ax.plot(cum.index, cum.to_numpy(), label=label, color=colors[k % len(colors)],
                    linewidth=2.2)
        ax.set_ylabel(ylabel)
        ax.set_title(sub_title, pad=12)
        style_axes(ax)
        ax.legend(loc="upper left", ncol=1, frameon=False, handlelength=1.8)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    save(fig, out_dir, path_name)


def fig_returns_panels(out_dir: Path, path_name: str, title: str, panels: list,
                       figsize=(13.65, 3.75), window=None) -> None:
    """Side-by-side panels of realised returns (reference figure 10)."""
    paper_style()
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]
    for ax, (sub_title, series, colors) in zip(axes, panels):
        for k, (label, s) in enumerate(series.items()):
            d = s
            if window is not None:
                d = d[(d.index >= pd.Timestamp(window[0])) & (d.index <= pd.Timestamp(window[1]))]
            if d.empty:
                continue
            ax.plot(d.index, d.to_numpy(), label=label, color=colors[k % len(colors)],
                    linewidth=1.6, marker="o", markersize=3.2)
        ax.set_title(sub_title, pad=12)
        ax.set_ylabel("Portfolio return")
        ax.set_xlabel("Time")
        style_axes(ax, grid=True, grid_axis="both")
        if len(series) > 1:
            ax.legend(loc="upper left", ncol=1, frameon=False, handlelength=1.8)
    fig.tight_layout()
    save(fig, out_dir, path_name)


# --------------------------------------------------------------------------- #
def build_v1(out_dir: Path) -> None:
    """All 11 figures from the grids that already finished.

    The reference figures sweep a risk target / entropy / network architecture;
    in this preliminary build the x axis carries our own series (calendar years,
    strategy variants) and the mapping is written to ``README.md`` next to the
    PNGs.  ``build_v2`` replaces them with the dedicated sweep grids.
    """
    main = RUNS["main"]
    exps5 = ["full", "full_nocvar_loss", "full_nocost", "full_nosparse", "meancvar_hist"]
    labels5 = ["LSTM+MCVaR", "LSTM+MCVaR (no tail loss)", "LSTM+MCVaR (cost-blind)",
               "LSTM+MCVaR (no sparsity)", "Mean-CVaR"]
    exps3 = ["full", "lstm_topk", "meancvar_hist"]
    labels3 = ["LSTM+MCVaR", "LSTM+MCVaR (top-k)", "Mean-CVaR"]

    if not main.exists():
        print(f"!! main run {main} missing")
        return

    # 2-4: grouped bars over the out-of-sample years
    fig_bars_from_years(out_dir, "main", exps5, labels5, "", SERIES_COLORS,
                        xlabel="Year")

    # 5: cumulative return of the model variants (greyscale, as in the reference)
    series5 = {lab: daily_frame("main", exp) for exp, lab in zip(exps5, labels5)}
    fig_cumulative(out_dir, "fig05_cumulative_return.png", "", series5, SERIES_COLORS,
                   greyscale=True, legend_ncol=1)

    # 8: sharpe vs achieved portfolio entropy (one point per strategy variant)
    pts = {}
    variants = ["full", "full_nocvar_loss", "full_nocost", "full_nosparse", "full_quad",
                "lstm_topk", "lstm_softmax", "lstm_sparsemax", "e2e_noselect", "meancvar_hist"]
    for exp in variants:
        pm = pooled_metrics("main", exp)
        if pm is None:
            continue
        ent = weights_entropy("main", exp)
        if not np.isfinite(ent):
            continue
        pts.setdefault("CSI 300", []).append((ent, float(pm.get("sharpe", np.nan))))
    fig_scatter_entropy(out_dir, "fig08_entropy_sharpe.png", "", pts, SERIES_COLORS)

    # 9: two stacked panels (2022-2024 window / full out-of-sample sample)
    top = {lab: daily_frame("main", exp) for exp, lab in zip(exps3, labels3)}
    panels = [("", top, SERIES_COLORS), ("", top, SERIES_COLORS)]
    fig_two_panels(out_dir, "fig09_entropy_cumulative_return.png", "", panels)

    # 10: two side-by-side panels of realised monthly returns, 2023-04 ~ 2024-06
    p10 = [("(a) LSTM+MCVaR", {"monthly net return": monthly_frame("main", "full").set_index("date")["net"]},
            SERIES_COLORS),
           ("(b) Mean-CVaR", {"monthly net return": monthly_frame("main", "meancvar_hist").set_index("date")["net"]},
            SERIES_COLORS)]
    fig_returns_panels(out_dir, "fig10_out_of_sample_return.png", "", p10,
                       window=("2023-04-01", "2024-06-30"))

    # 11 / 12: the two alternative data settings we have (full universe / score rule)
    if RUNS["fu"].exists():
        s11 = {}
        for exp, lab in zip(["full", "lstm_topk", "meancvar_hist", "ew"],
                            ["LSTM+MCVaR", "LSTM+MCVaR (top-k)", "Mean-CVaR", "Equal-weight 1/N"]):
            d = daily_frame("fu", exp)
            if not d.empty:
                s11[lab] = d
        if s11:
            fig_cumulative(out_dir, "fig11_other_market.png", "", s11, SERIES_COLORS,
                           legend_ncol=1, legend_loc="upper left")
    if RUNS["score"].exists():
        s12 = {}
        for exp, lab in zip(["full_musigma", "lstm_musigma", "full", "meancvar_hist"],
                            ["LSTM+MCVaR (mu-sigma score)", "LSTM+MCVaR (mu-sigma, predict-only)",
                             "LSTM+MCVaR (learned score)", "Mean-CVaR"]):
            d = daily_frame("score", exp)
            if not d.empty:
                s12[lab] = d
        if s12:
            fig_cumulative(out_dir, "fig12_score_rule.png", "", s12, SERIES_COLORS,
                           legend_ncol=1, legend_loc="upper left")

    # 13 / 14: model-comparison line charts (reference 13/14)
    s13 = {lab: daily_frame("main", exp) for exp, lab in zip(exps5, labels5)}
    fig_cumulative(out_dir, "fig13_model_comparison.png", "", s13, SERIES_COLORS,
                   legend_ncol=1, legend_loc="upper left")
    s14 = {lab: daily_frame("main", exp) for exp, lab in zip(
        ["full", "full_nocost", "meancvar_hist", "ew"],
        ["LSTM+MCVaR", "LSTM+MCVaR (cost-blind)", "Mean-CVaR", "Equal-weight 1/N"])}
    fig_cumulative(out_dir, "fig14_baselines.png", "", s14, SERIES_COLORS,
                   legend_ncol=1, legend_loc="upper left")
    (out_dir / "README.md").write_text(NOTES_V1, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=r"D:\金创\figures_ours",
                    help="output folder for the generated figures")
    ap.add_argument("--source", default="v1", choices=["v1", "v2", "auto"])
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"generating paper-style figures into {out_dir} (source={args.source})")
    if args.source in ("v1", "auto"):
        build_v1(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
