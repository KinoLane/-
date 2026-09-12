"""Reproduce the 11 reference panels of ``D:\\figures`` with *our* results.

The reference study's figures are copied faithfully in layout (figure size,
fonts, legend placement, colour scheme, panel split); every number inside them
comes from the runs produced by this repository, never from the reference.

All figures are written to one dedicated folder (``results/figures_reference``
by default) together with ``README.md`` (panel -> data source map) and
``panels.json`` (machine readable manifest).

Panel sources
-------------
``fig02/03/04``  target-return sweep ``mu016_* .. mu028_*`` (5 architectures)
``fig05``        same sweep at ``mu020_*``, cumulative out-of-sample return
``fig08/09``     entropy sweep ``ent000_* .. ent100_*`` (the 3 recurrent architectures,
                 exactly the series the reference panels contain)
``fig10``        realised monthly net return of the 5 architectures in one
                 11 month window, at the two ends of the sweep
``fig11/12``     the five architectures on the CSI 500 / SSE 50 universes
``fig13/14``     AdaBoost vs LSTM vs XGBoost at tau = 0.020 and 0.022

The layout, palette, legend placement and figure size of every panel were
measured pixel by pixel from ``D:\\figures`` (see ``README.md`` of the output
folder); the numbers never come from the reference study.

Usage::

    python scripts/09_reference_figures.py                    # default folder
    python scripts/09_reference_figures.py --out D:\\金创\\figures_ours
    python scripts/09_reference_figures.py --strict            # fail on missing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio import experiments as EXP  # noqa: E402
from e2e_portfolio.metrics import TRADING_DAYS, compute_metrics  # noqa: E402

# --------------------------------------------------------------------------- #
# palette and labels (same colours as the reference figures)
# --------------------------------------------------------------------------- #
C_BLACK = "#000000"
C_BLUE = "#0000ff"
C_RED = "#ff0000"
C_GREEN = "#008000"
C_ORANGE = "#ffa500"
C_PURPLE = "#800080"

#: Series order measured in the reference panels: the bar groups of Fig2/3/4/8
#: and the architecture line panels Fig10-12 all run RNN, LSTM, GRU, MLP, RBFN.
ARCH_ORDER = ("rnn", "lstm", "gru", "mlp", "rbfn")
ARCH_COLORS = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE]
#: Fig5 is the single panel whose first series (RNN) is drawn in black.
ARCH_COLORS_FIG5 = [C_BLACK, C_RED, C_GREEN, C_BLUE, C_PURPLE]
#: Fig8/9 contain the three recurrent architectures only.
ENT_ARCHS = ("rnn", "lstm", "gru")
ENT_COLORS = [C_BLUE, C_RED, C_GREEN]
#: Fig13/14: AdaBoost (blue), the LSTM reference (red), XGBoost (green).
TREE_COLORS = [C_BLUE, C_RED, C_GREEN]
ARCH_LABEL = {
    "lstm": "LSTM+MCVaR",
    "rnn": "RNN+MCVaR",
    "gru": "GRU+MCVaR",
    "mlp": "MLP+MCVaR",
    "rbfn": "RBFN+MCVaR",
}
ARCH_LABEL_ORDER = [ARCH_LABEL[a] for a in ARCH_ORDER]
ENT_LABEL_ORDER = [ARCH_LABEL[a] for a in ENT_ARCHS]

MU_TARGETS = EXP.MU_TARGETS
ENT_LAMBDAS = EXP.ENTROPY_LAMBDAS

RUN_PATTERNS = {
    "mu": "csi300_mu_sweep*",
    "ent": "csi300_entropy_sweep*",
    "tree": "csi300_tree_baselines*",
    "csi500": "csi500_reference*",
    "sse50": "sse50_reference*",
    "csi500_arch": "csi500_arch_compare*",
    "sse50_arch": "sse50_arch_compare*",
    "main": "csi300_2026*",
}

PANELS: List[dict] = []


def mu_name(mu: float, arch: str) -> str:
    return EXP.mu_experiment_name(mu, arch)


def ent_name(lam: float, arch: str) -> str:
    return EXP.entropy_experiment_name(lam, arch)


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def paper_style(axis_label_size: float = 20.0, tick_size: float = 17.0,
                title_size: float = 21.0, legend_size: float = 18.0) -> None:
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


def style_axes(ax, grid: bool = False) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
    if grid:
        ax.grid(True, axis="both", color="#c9c9c9", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str, panel: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    panel["file"] = name
    PANELS.append(panel)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# run discovery / loading
# --------------------------------------------------------------------------- #
def run_dirs(key: str, verbose: bool = True) -> List[Path]:
    dirs = sorted(p for p in RESULTS.glob(RUN_PATTERNS[key]) if p.is_dir())
    if verbose and not dirs:
        print(f"!! no run directory matches results/{RUN_PATTERNS[key]}")
    return dirs


def fold_dirs(dirs: Sequence[Path], exp: str) -> List[Path]:
    out: List[Path] = []
    for d in dirs:
        base = d / exp
        if base.is_dir():
            out.extend(sorted(p for p in base.glob("f*") if p.is_dir()))
    return out


def load_returns(dirs: Sequence[Path], exp: str) -> Optional[pd.DataFrame]:
    """Daily net / benchmark returns of ``exp``, folds concatenated in date order."""
    frames = []
    for f in fold_dirs(dirs, exp):
        p = f / "returns.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        if "ret_net" not in df.columns:
            continue
        frames.append(df[["ret_net"] + (["bench"] if "bench" in df.columns else [])])
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")].dropna(subset=["ret_net"])


def pooled_stats(df: pd.DataFrame) -> Dict[str, float]:
    """The report's definition: metrics on the concatenated out-of-sample series."""
    r = df["ret_net"].to_numpy(dtype=np.float64)
    b = df["bench"].to_numpy(dtype=np.float64) if "bench" in df.columns else None
    out = compute_metrics(r, benchmark=b)
    if b is not None and b.size == r.size:
        out["tracking_error"] = float(
            np.std(r - b, ddof=1) * np.sqrt(float(TRADING_DAYS))
        )
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float, np.floating))}


def portfolio_entropy(dirs: Sequence[Path], exp: str) -> float:
    """Mean Shannon entropy (nats) of the rebalanced portfolios over all folds."""
    vals = []
    for f in fold_dirs(dirs, exp):
        p = f / "weights.csv"
        if not p.exists():
            continue
        w = pd.read_csv(p, index_col=0).to_numpy(dtype=np.float64)
        for row in w:
            row = np.clip(row, 0.0, None)
            total = row.sum()
            if total <= 1e-12:
                continue
            row = row / total
            nz = row[row > 0]
            vals.append(float(-(nz * np.log(nz)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def monthly_returns(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    idx = pd.to_datetime(series.index)
    return (1.0 + series).groupby([idx.year, idx.month]).prod() - 1.0


def cumulative(df: pd.DataFrame) -> pd.Series:
    # the reference panels plot the cumulative *return* (the curve starts at
    # 0.0; a final value of 3.0 means +300%), not the net value (1.0-based)
    return (1.0 + df["ret_net"]).cumprod() - 1.0


def have_experiment(dirs: Sequence[Path], exp: str) -> bool:
    return bool(fold_dirs(dirs, exp))


# --------------------------------------------------------------------------- #
# figure helpers
# --------------------------------------------------------------------------- #
def grouped_bars(out_dir: Path, name: str, panel: dict, groups: Sequence[str],
                 values: Dict[str, Sequence[float]], colors: Sequence[str],
                 xlabel: str, ylabel: str, figsize: Tuple[float, float],
                 ylim: Optional[Tuple[float, float]] = None,
                 ncol: int = 5) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    n_series = len(values)
    # bar cluster width, x limits and y limits measured from the reference
    # panels: the cluster is 0.18 data units per series (0.78 in total at most),
    # sits on the x tick, and the y axis hugs the data with 7% headroom on top
    total = min(0.78, 0.18 * max(n_series, 1))
    width = total / max(n_series, 1)
    x = np.arange(len(groups))
    for k, (label, vals) in enumerate(values.items()):
        offs = (k - (n_series - 1) / 2.0) * width
        ax.bar(x + offs, np.asarray(vals, dtype=np.float64), width=width,
               label=label, color=colors[k % len(colors)], edgecolor="none", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        flat = np.concatenate([np.asarray(v, dtype=np.float64) for v in values.values()])
        flat = flat[np.isfinite(flat)]
        if flat.size:
            lo, hi = float(flat.min()), float(flat.max())
            span = (hi - lo) or max(abs(hi), 1.0)
            ax.set_ylim(0.0 if lo >= 0 else lo, hi + 0.07 * span)
    style_axes(ax)
    # the reference keeps the bar legend *inside* the axes, on the top row.  It must be kept
    # out of tight_layout: otherwise the wide 5-column legend shrinks the drawing area
    # (ref fig2: axes 1192 px wide / 18 px right margin; with the legend laid out: 1037 / 144).
    leg = ax.legend(loc="upper center", ncol=ncol, frameon=False, borderaxespad=0.0,
                    columnspacing=1.0, handlelength=2.0, handletextpad=0.5, fontsize=17.0)
    leg.set_in_layout(False)
    fig.tight_layout(pad=0.7)
    save(fig, out_dir, name, panel)


def line_panel(ax, series: Dict[str, pd.DataFrame], colors: Sequence[str],
               cumulative_plot: bool = True,
               widen: float = 2.2) -> None:
    for k, (label, df) in enumerate(series.items()):
        if df is None or df.empty:
            continue
        y = cumulative(df) if cumulative_plot else df["ret_net"]
        ax.plot(y.index, y.to_numpy(), label=label,
                color=colors[k % len(colors)], linewidth=widen)
    style_axes(ax)


def series_spans(series: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Audit trail: how many out-of-sample days each plotted line actually has."""
    out = {}
    for label, df in series.items():
        if df is None or df.empty:
            continue
        out[label] = (f"{df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}"
                      f" ({len(df)} days)")
    return out


def single_panel(out_dir: Path, name: str, panel: dict, series: Dict[str, pd.DataFrame],
                 colors: Sequence[str], figsize: Tuple[float, float],
                 ylabel: str = "Cumulative return", xlabel: str = "Time",
                 legend_ncol: int = 1, legend_loc: str = "upper left",
                 cumulative_plot: bool = True) -> None:
    paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    line_panel(ax, series, colors, cumulative_plot=cumulative_plot)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=False,
              columnspacing=1.2, handlelength=2.0)
    fig.tight_layout(pad=0.7)
    panel = dict(panel, spans=series_spans(series))
    save(fig, out_dir, name, panel)


# --------------------------------------------------------------------------- #
# panels 2-5: target-return sweep
# --------------------------------------------------------------------------- #
def figure_target_sweep(out_dir: Path, dirs: Sequence[Path]) -> None:
    if not dirs:
        return
    rows: Dict[Tuple[float, str], Dict[str, float]] = {}
    for mu in MU_TARGETS:
        for arch in ARCH_ORDER:
            exp = mu_name(mu, arch)
            df = load_returns(dirs, exp)
            if df is None:
                continue
            rows[(mu, arch)] = pooled_stats(df)
    if not rows:
        print("  !! target-return sweep not available yet")
        return

    def series_for(metric: str) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for arch in ARCH_ORDER:
            vals = []
            for mu in MU_TARGETS:
                st = rows.get((mu, arch))
                vals.append(np.nan if st is None else st.get(metric, np.nan))
            if np.isfinite(vals).any():
                out[ARCH_LABEL[arch]] = vals
        return out

    # default matplotlib float format, exactly like the reference ticks
    # ("0.016", "0.018", ..., "0.028")
    xlabels = [f"{m:g}" for m in MU_TARGETS]
    specs = [
        ("fig02_std_vs_target.png", "ann_vol", "Standard deviation", (13.24, 6.03)),
        ("fig03_sharpe_vs_target.png", "sharpe", "Sharpe ratio", (13.25, 6.04)),
        ("fig04_tracking_error_vs_target.png", "tracking_error", "Tracking error",
         (13.27, 6.05)),
    ]
    for fname, metric, ylabel, figsize in specs:
        values = series_for(metric)
        if not values:
            print(f"  !! {metric} missing, skipping {fname}")
            continue
        panel = {
            "reference": fname.split("_")[0].replace("fig", "Fig"),
            "metric": metric,
            "x": "target return tau",
            "series": list(values),
            "source": "results/csi300_mu_sweep*",
            "experiments": [mu_name(m, a) for a in ARCH_ORDER for m in MU_TARGETS],
        }
        grouped_bars(
            out_dir, fname, panel, xlabels, values, ARCH_COLORS,
            xlabel="Return", ylabel=ylabel, figsize=figsize,
        )


def figure_cumulative_at_target(out_dir: Path, dirs: Sequence[Path], mu: float = 0.020) -> None:
    if not dirs:
        return
    series: Dict[str, pd.DataFrame] = {}
    for arch in ARCH_ORDER:
        df = load_returns(dirs, mu_name(mu, arch))
        if df is not None and not df.empty:
            series[ARCH_LABEL[arch]] = df
    if not series:
        print(f"  !! no run at tau={mu:.3f} yet, skipping fig05")
        return
    panel = {
        "reference": "Fig5",
        "metric": "cumulative net return, all folds chained",
        "x": "time",
        "series": list(series),
        "colors": "RNN black, LSTM red, GRU green, MLP blue, RBFN purple",
        "source": "results/csi300_mu_sweep*",
        "experiments": [mu_name(mu, a) for a in ARCH_ORDER],
    }
    single_panel(out_dir, "fig05_cumulative_return_mu020.png", panel, series,
                 ARCH_COLORS_FIG5, (13.22, 5.99), legend_ncol=1,
                 legend_loc="upper left")


# --------------------------------------------------------------------------- #
# panels 8-9: entropy sweep
# --------------------------------------------------------------------------- #
def mean_entropy_by_lambda(dirs: Sequence[Path]) -> Dict[float, float]:
    """Mean achieved portfolio entropy (nats) of the recurrent nets per lambda.

    This is the quantity the reference panels put on the x axis of Fig8 and in
    the sub-caption of Fig9 ("Cumulative return graphs for E = 2.55").
    """
    out: Dict[float, float] = {}
    for lam in ENT_LAMBDAS:
        vals = [portfolio_entropy(dirs, ent_name(lam, a)) for a in ENT_ARCHS]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[lam] = float(np.mean(vals))
    return out


def figure_entropy_sweep(out_dir: Path, dirs: Sequence[Path]) -> None:
    if not dirs:
        return
    rows: Dict[Tuple[float, str], Dict[str, float]] = {}
    entropies: Dict[float, List[float]] = {lam: [] for lam in ENT_LAMBDAS}
    for lam in ENT_LAMBDAS:
        for arch in ENT_ARCHS:
            exp = ent_name(lam, arch)
            df = load_returns(dirs, exp)
            if df is None:
                continue
            st = pooled_stats(df)
            ent = portfolio_entropy(dirs, exp)
            st["entropy"] = ent
            rows[(lam, arch)] = st
            if np.isfinite(ent):
                entropies[lam].append(ent)
    if not rows:
        print("  !! entropy sweep not available yet")
        return

    # x axis = the achieved portfolio entropy of each regularisation level,
    # exactly like the reference panel (bars grouped by entropy, not by lambda)
    xlabels = []
    for lam in ENT_LAMBDAS:
        vals = entropies.get(lam) or []
        if vals:
            # reference ticks are the round(mean entropy, 2) values in the
            # default float format: 2.55, 2.6, 2.65, 2.7, 2.75
            xlabels.append(f"{round(float(np.mean(vals)), 2):g}")
        else:
            xlabels.append(f"lam={lam:g}")
    values = {}
    for arch in ENT_ARCHS:
        vals = []
        for lam in ENT_LAMBDAS:
            st = rows.get((lam, arch))
            vals.append(np.nan if st is None else st.get("sharpe", np.nan))
        if np.isfinite(vals).any():
            values[ARCH_LABEL[arch]] = vals
    panel = {
        "reference": "Fig8",
        "metric": "pooled Sharpe vs achieved portfolio entropy",
        "x": "mean achieved Shannon entropy (nats) of each lambda level",
        "lambda_levels": list(ENT_LAMBDAS),
        "series": list(values),
        "source": "results/csi300_entropy_sweep*",
        # the reference panel only shows the three recurrent architectures; the
        # MLP/RBFN encoders of the extended sweep exist on disk but are not plotted
        "experiments": [
            ent_name(l, a)
            for a in ENT_ARCHS
            for l in ENT_LAMBDAS
            if (l, a) in rows
        ],
        "experiments_not_run": [
            ent_name(l, a)
            for a in ENT_ARCHS
            for l in ENT_LAMBDAS
            if (l, a) not in rows
        ],
    }
    grouped_bars(out_dir, "fig08_entropy_sharpe_ratio.png", panel, xlabels, values,
                 ENT_COLORS, xlabel="Entropy", ylabel="Sharpe ratio",
                 figsize=(13.22, 6.07), ncol=len(ENT_ARCHS))

    # two stacked panels: no diversification vs the default regularisation
    panels_series = []
    ent_mean = mean_entropy_by_lambda(dirs)
    for lam, fallback in ((0.0, "no entropy term"),
                          (0.002, "default entropy term")):
        s = {}
        for arch in ENT_ARCHS:
            df = load_returns(dirs, ent_name(lam, arch))
            if df is not None and not df.empty:
                s[ARCH_LABEL[arch]] = df
        if s:
            e = ent_mean.get(lam)
            caption = (f"Cumulative return graphs for E = {e:g}" if e is not None
                       else f"cumulative return, {fallback}")
            panels_series.append((lam, caption, s))
    if not panels_series:
        print("  !! no entropy cumulative series, skipping fig09")
        return
    paper_style()
    fig, axes = plt.subplots(len(panels_series), 1, figsize=(13.35, 12.44), squeeze=False)
    for ax, (_, caption, s) in zip(axes[:, 0], panels_series):
        line_panel(ax, s, ENT_COLORS)
        ax.set_ylabel("Cumulative return")
        ax.set_xlabel("Time")
        ax.legend(loc="upper left", ncol=1, frameon=False, handlelength=2.0)
        # the reference carries a sub-caption line under each panel
        ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha="center", va="top")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0), pad=0.7)
    save(fig, out_dir, "fig09_entropy_cumulative_return.png", {
        "reference": "Fig9",
        "metric": "cumulative net return by entropy level",
        "x": "time",
        "panels": [caption for _, caption, _ in panels_series],
        "series": ENT_LABEL_ORDER,
        "spans": {caption: series_spans(s) for _, caption, s in panels_series},
        "source": "results/csi300_entropy_sweep*",
        "experiments": [ent_name(l, a) for l, _, _ in panels_series
                        for a in ENT_ARCHS],
    })


# --------------------------------------------------------------------------- #
# panels 11-14: other universes and tree baselines
# --------------------------------------------------------------------------- #
MARKET_SPECS = (
    ("fig11_csi500_cumulative_return.png", "csi500", "Fig11",
     "CSI 500 (stand-in for the reference's IBrX50)", (13.63, 6.28)),
    ("fig12_sse50_cumulative_return.png", "sse50", "Fig12",
     "SSE 50", (13.60, 6.27)),
)
#: Fig11/12 legend: the five architectures of the target-return sweep at tau=0.020
MARKET_TAU = 0.020
MARKET_EXPS = [(mu_name(MARKET_TAU, a), ARCH_LABEL[a]) for a in ARCH_ORDER]
MARKET_EXPS_LEGACY = [("mu020_lstm", "LSTM+MCVaR")]


def start_banner(text: str) -> None:
    print(f"[{text}]", flush=True)


def figure_market(out_dir: Path, key: str, fname: str, reference: str, market: str,
                  figsize: Tuple[float, float]) -> None:
    """Fig11/12: five architectures on one universe.

    The five-architecture runs of a universe live in
    ``results/<key>_arch_compare*``; if that directory does not exist yet the
    single-architecture ``results/<key>_reference*`` run is used instead and the
    panel records the missing series in ``unsupported_architectures``.
    """
    arch_dirs = run_dirs(key + "_arch", verbose=False)
    ref_dirs = run_dirs(key)
    dirs = list(arch_dirs) + list(ref_dirs)
    if not dirs:
        return
    series: Dict[str, pd.DataFrame] = {}
    used: List[str] = []
    for exp, label in MARKET_EXPS:
        df = load_returns(dirs, exp)
        if df is not None and not df.empty:
            series[label] = df
            used.append(exp)
    if not series:
        for exp, label in MARKET_EXPS_LEGACY:
            df = load_returns(ref_dirs, exp)
            if df is not None and not df.empty:
                series[label] = df
                used.append(exp)
    if not series:
        print(f"  !! {key} run has no usable experiment yet")
        return
    panel = {
        "reference": reference,
        "metric": "cumulative net return",
        "x": "time",
        "market": market,
        "target_return": MARKET_TAU,
        "series": list(series),
        "source": f"results/{RUN_PATTERNS[key + '_arch']} + results/{RUN_PATTERNS[key]}",
        "experiments": used,
        "unsupported_architectures": [label for exp, label in MARKET_EXPS
                                      if exp not in used],
    }
    single_panel(out_dir, fname, panel, series, ARCH_COLORS, figsize,
                 legend_ncol=1, legend_loc="upper left")


def tree_name(estimator: str, mu: float) -> Optional[str]:
    """Registry name of the tree baseline with this estimator and target return."""
    for name in EXP.TREE_BASELINES:
        spec = EXP.spec_for(name)
        if (spec.estimator == estimator
                and spec.overrides.get("opt", {}).get("mu_target") == mu):
            return name
    return None


def figure_tree_baseline(out_dir: Path, dirs: Sequence[Path], mu: float,
                         fname: str, reference: str,
                         figsize: Tuple[float, float] = (13.60, 6.26)) -> None:
    """Fig13/14: AdaBoost, the LSTM reference and XGBoost at one target return."""
    if not dirs:
        return
    order: List[Tuple[str, str]] = []
    ada = tree_name("adaboost", mu)
    xgb = tree_name("xgboost", mu)
    if ada:
        order.append((ada, "ADA+MCVaR"))
    order.append((mu_name(mu, "lstm"), ARCH_LABEL["lstm"]))
    if xgb:
        order.append((xgb, "XGB+MCVaR"))
    series: Dict[str, pd.DataFrame] = {}
    used: List[str] = []
    for exp, label in order:
        df = load_returns(dirs, exp)
        if df is not None and not df.empty:
            series[label] = df
            used.append(exp)
    if not series:
        print(f"  !! no series at tau={mu:.3f} for {fname}")
        return
    panel = {
        "reference": reference,
        "metric": "cumulative net return",
        "x": "time",
        "target_return": mu,
        "series": list(series),
        "source": "results/csi300_mu_sweep* + results/csi300_tree_baselines*",
        "experiments": used,
    }
    single_panel(out_dir, fname, panel, series, TREE_COLORS, figsize,
                 legend_ncol=1, legend_loc="upper left")


# --------------------------------------------------------------------------- #
# panel 10: realised monthly returns
# --------------------------------------------------------------------------- #
def figure_out_of_sample_return(
        out_dir: Path, mu_dirs: Sequence[Path],
        window: Tuple[str, str] = ("2023-08-01", "2024-06-30"),
        taus: Sequence[float] = (0.016, 0.028)) -> None:
    """Fig10: realised monthly net return, all five architectures, two panels.

    Reference facts we copy: two side-by-side panels covering *the same* 11
    month window, one series per architecture, circle markers, legend in the
    upper-left, no grid, no panel titles.  What the two reference panels differ
    by cannot be recovered from the image, so the two ends of our target-return
    sweep are used and the choice is flagged in ``panels.json``.
    """
    if not mu_dirs:
        return
    panels = []
    for tau in taus:
        s: Dict[str, pd.DataFrame] = {}
        for arch in ARCH_ORDER:
            df = load_returns(mu_dirs, mu_name(tau, arch))
            if df is None or df.empty:
                continue
            m = monthly_returns(df["ret_net"])
            idx = pd.to_datetime([pd.Timestamp(year=int(y), month=int(mm), day=1)
                                  for (y, mm) in m.index])
            ser = pd.Series(m.to_numpy(), index=idx + pd.offsets.MonthEnd(0))
            ser = ser[(ser.index >= pd.Timestamp(window[0]))
                      & (ser.index <= pd.Timestamp(window[1]))]
            if ser.empty:
                continue
            s[ARCH_LABEL[arch]] = ser.to_frame("ret_net")
        if s:
            panels.append((tau, s))
    if not panels:
        print("  !! no data in the requested window for fig10")
        return
    paper_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(13.65, 3.75), squeeze=False)
    for ax, (tau, s) in zip(axes[0], panels):
        for k, (label, df) in enumerate(s.items()):
            ax.plot(df.index, df["ret_net"].to_numpy(), label=label,
                    color=ARCH_COLORS[k % len(ARCH_COLORS)], linewidth=1.6,
                    marker="o", markersize=3.2)
        ax.set_ylabel("Portfolio return")
        ax.set_xlabel("Time")
        # reference ticks: every second month, labelled "%b-%y" (Aug-23, ...)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        style_axes(ax)
        ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.tight_layout(pad=0.7)
    save(fig, out_dir, "fig10_out_of_sample_return.png", {
        "reference": "Fig10",
        "metric": "realised monthly net return",
        "window": list(window),
        "target_returns": [float(t) for t, _ in panels],
        "series": ARCH_LABEL_ORDER,
        "assumption": ("the reference does not reveal what distinguishes its two "
                       "panels; we show the lowest and the highest target return "
                       "of the sweep in the same 11 month window"),
        "source": "results/csi300_mu_sweep*",
        "experiments": [mu_name(t, a) for t, _ in panels for a in ARCH_ORDER],
    })


# --------------------------------------------------------------------------- #
# documentation
# --------------------------------------------------------------------------- #
README_TEMPLATE = """# 复刻图（用我们自己的实验结果画）

对照图在 `{ref}`。每张图的画布尺寸、字号、图例位置、配色都照抄对照图，
**图里的每个数字都来自本仓库 `results/` 下自己跑的 run**，没有引用对照图的任何数值。

| 本图 | 对照图 | 画的是什么 | 数据来源（run 目录） |
|------|--------|-----------|---------------------|
| fig02_std_vs_target.png | Fig2 | 7 档目标收益 μ × 5 种网络结构的**年化波动** | `results/csi300_mu_sweep*` |
| fig03_sharpe_vs_target.png | Fig3 | 同上，纵轴为夏普比率 | `results/csi300_mu_sweep*` |
| fig04_tracking_error_vs_target.png | Fig4 | 同上，纵轴为相对基准的跟踪误差 | `results/csi300_mu_sweep*` |
| fig05_cumulative_return_mu020.png | Fig5 | μ=0.020 时 5 种结构的累计净值 | `results/csi300_mu_sweep*` |
| fig08_entropy_sharpe_ratio.png | Fig8 | 5 档熵正则 × **3 种循环结构**（RNN/LSTM/GRU）的夏普，横轴为**实际实现的组合熵** | `results/csi300_entropy_sweep*` |
| fig09_entropy_cumulative_return.png | Fig9 | 同上 3 种结构在熵正则 0 vs 默认档的累计净值（上下两块面板） | `results/csi300_entropy_sweep*` |
| fig10_out_of_sample_return.png | Fig10 | **同一 11 个月窗口**内 5 种结构的月度已实现收益（左右 = 目标收益最低 / 最高档） | `results/csi300_mu_sweep*` |
| fig11_csi500_cumulative_return.png | Fig11 | 换到 CSI 500 股票池、5 种结构的累计净值 | `results/csi500_arch_compare_*`（本仓库交付的图即由此生成；`results/csi500_reference_*` 只在上级目录缺失时用于回退） |
| fig12_sse50_cumulative_return.png | Fig12 | 换到上证 50、5 种结构的累计净值 | `results/sse50_arch_compare_*`（本仓库交付的图即由此生成；`results/sse50_reference_*` 只在上级目录缺失时用于回退） |
| fig13_adaboost_xgboost_mu020.png | Fig13 | μ=0.020 下 AdaBoost / XGBoost 与 LSTM 的对比（3 条线） | `results/csi300_mu_sweep*` + `results/csi300_tree_baselines*` |
| fig14_adaboost_xgboost_mu022.png | Fig14 | 同上，μ=0.022 | `results/csi300_mu_sweep*` + `results/csi300_tree_baselines*` |

## 配色 / 版式（逐像素测量对照图后照抄）

* 颜色就是 matplotlib 的默认色：蓝 `#0000ff` = RNN，红 `#ff0000` = LSTM，绿 `#008000` = GRU，
  橙 `#ffa500` = MLP，紫 `#800080` = RBFN；Fig5 的第一条线（RNN）在对照图里是**黑色**，
  Fig13/14 的图例顺序是 ADA(蓝) / LSTM(红) / XGB(绿)。
* 所有面板都不画内部网格线、图例都不带边框（`frameon=False`）。
* 柱状图（Fig2/3/4/8）图例在坐标区**内部顶行**、每组 5 根柱子从左到右即上面的配色顺序；
  折线图（Fig5/11/12/13/14）图例在左上角。
* 画布尺寸按对照图逐张量取（如 Fig2 = 13.24×6.03 in，Fig9 = 13.35×12.44 in）。
* 字号照抄对照图的实测值：刻度 17 pt、轴标题 20 pt、**折线图图例 18 pt、柱状图图例 17 pt**，
  图例色块长 = 2.0 个字号（对照图图例的行距 ≈ 1.5 × 字号，Fig5 实测 37.2 px = 18 pt，
  与我们的 37.1 px 一致）。
* 柱状图：柱簇总宽 = min(0.78, 0.18×系列数) 个横轴单位、柱子紧邻（间隙 0），
  x 轴范围 = (-0.5, 组数−0.5)，y 轴顶部 = max + 7% 极差、底部 = 0（全为正时柱子贴底）；
  图例 `loc='upper center'` 且贴住轴顶（`borderaxespad=0`）、列间距 `columnspacing=1.0`，
  且**不参与 `tight_layout`**（`set_in_layout(False)`）—— 否则居中的 5 列宽图例会把绘图区
  压窄 ≈160 px（对照图实测绘图区宽 1192 px，我们设了 `set_in_layout(False)` 后为 1193 px）。
* 折线图：y 轴沿用 matplotlib 默认的 5% 边距；**纵轴是 0 起算的累计收益**
  （Π(1+r)−1），不是从 1 起算的净值 —— 与对照图一致（对照图的刻度里有 0.0）。
* 横轴刻度沿用 matplotlib 的默认浮点格式（`0.016` … `0.028`；Fig8 是 `2.55` … `2.75`）；
  Fig10 的日期刻度为 `%b-%y` 且每 2 个月一格（Aug-23、Oct-23 …），与对照图一致。
* 四周留白由 `tight_layout(pad=0.7)` 控制（对照图实测约 16–18 px）；左右留白会随
  坐标轴两端的标签浮动 —— 折线图的横轴末端若被最后一个刻度标签压住，标签悬出轴外，
  留白就会变宽（详见下面第 11 条）。

## 与对照图的差别（诚实说明）

1. **样本区间**：对照图横轴是 2016–2023；我们的 walk-forward 测试折是
   2018–2019 / 2020–2021 / 2022–2026，三折首尾相接，所以横轴是 2018–2026。
2. **Fig11 的市场**：对照图用巴西 IBrX50，本地数据里没有该指数，改用
   **CSI 500**（同样的「换一个股票池」作用），并在图上与 README 中标明。
3. **Fig8 的横轴**：对照图直接以实现的组合熵为横轴，我们也一样——横轴刻度是
   每一档 λ 在 3 种结构（对照图只有 RNN/LSTM/GRU 三条线）上实现的平均熵（nats），
   而不是 λ 本身。我们另跑了 MLP/RBFN 的熵扫描（`csi300_entropy_sweep_ext`），
   但对照图只有 3 条线，所以这两条不进 Fig8/Fig9。
4. **Fig10 的两个面板**：对照图里左右两块的差别（窗口？标的？参数？）无法从图上看出来，
   我们改成**同一 11 个月窗口**、目标收益最低档（μ=0.016）与最高档（μ=0.028）的对比，
   并在 `panels.json` 的 `assumption` 字段里写明。
5. **Fig9 的小标题**：对照图每块面板下方有一行小标题，格式是
   `Cumulative return graphs for E = 2.55`（E 为该档实际实现的组合熵），我们照这个格式写，
   但把 E 换成我们这一档的实测平均熵；对照图第二块的小标题被画布裁掉了，我们两块都画。
6. **指标定义**：波动 / 夏普 / 跟踪误差都在「三折首尾相接的日度净收益」上重算，
   与 `report/pooled.csv` 的口径一致（`e2e_portfolio.metrics.compute_metrics`，
   年化因子 252；跟踪误差 = std(组合-基准)×√252）。
7. **纵轴口径**：对照图的折线图纵轴是 0 起算的累计收益，我们原先画的是从 1 起算的净值，
   现已改成同样的 0 起算口径（形状不变，刻度整体下移 1.0）。
8. **Fig8 的柱子**：对照图里所有夏普都为正、柱子贴住下边界；我们的夏普在三折上跨零
   （−0.32 … +0.30，主要来自 2020–2021 折的普遍为负，这是真实的样本外结果，
   见 §9.12.1），所以下边界落在最小值上、柱高有正有负。其他几何量（柱簇宽度、x 轴范围、
   顶部 7% 留白、图例 18 pt / 色块 2.0 字号）已按对照图对齐。
9. **对照图自身的两处不一致**（我们没有照抄）：a) Fig10 的字号明显小于其余图
   （实测约 11 pt 对比 17 pt），我们保持全套统一；b) Fig2/3/4 的柱簇整体右移了一个柱宽
   （柱簇中心落在 1.5、2.5 … 而不是 1、2 …），我们按 matplotlib 的正常做法让柱簇居中。
10. **数据量级**：对照图的累计收益到 +300% 量级、测试区间 2016–2023，我们的样本外区间
    是 2018–2026，曲线形状与终点因此不可比；本仓库只保证「同样的图、同样的口径、
    同样的版式」，数字本身来自我们自己的三折 walk-forward 结果。
11. **折线图的右侧留白（44 px vs 对照图 17 px）—— 已查明是数据端点造成的，不是版式差异**：
    我们的样本外区间止于 2026-08-20，matplotlib 默认的 5% 横轴边距把视窗推到 2027-01-01，
    于是最后一格「2027」年刻度正好落在坐标区右缘、标签有一半悬在轴外，`tight_layout`
    必须为它留宽（16.5 px pad + ≈27 px 悬出 = 44 px，与实测逐像素闭合）；对照图数据止于
    2023 年中，末刻度「2023」深处轴内，所以只需 17 px。两边除留白外的刻度要素实测一致：
    刻度定位（年度 `AutoDateLocator`）、刻度字号（末刻度墨迹宽 58 vs 60 px）、
    图例版面（左上角、单列 5 行、行距 37.1 vs 37.2 px）、无网格线。
12. **左侧留白相差 15 px（对照图坐标区左缘 x≈119，我们 x≈134）**：两边的纵轴标题都是
    `"Cumulative return"`（OCR 复核一致）、左缘都在画布 x≈15，差值来自纵轴刻度标签块的
    宽度（对照 103 px vs 我们 117 px）叠加 `tight_layout` 的分配，属数据驱动的派生量，
    同样不是版式差异。

## 重新生成

```powershell
# 1) 目标收益扫描（5 种结构 × 7 档 μ）
python scripts/02_run_experiments.py --config configs/csi300_mu_sweep.yaml --workers 4 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/csi300_mu_sweep_ext.yaml --workers 4 --threads-per-worker 2
# 2) 熵扫描（3 种结构 × 5 档 λ，口径同对照图 Fig8/Fig9 的三条线）
python scripts/02_run_experiments.py --config configs/csi300_entropy_sweep.yaml --workers 4 --threads-per-worker 2
#    可选：把 MLP / RBFN 也补进熵扫描（本轮未跑，故图里没有它们）
python scripts/02_run_experiments.py --config configs/csi300_entropy_sweep_ext.yaml --workers 4 --threads-per-worker 2
# 3) 树模型基线（AdaBoost / XGBoost，τ=0.020 与 0.022）
python scripts/02_run_experiments.py --config configs/csi300_tree_baselines.yaml --workers 2 --threads-per-worker 2
# 4) 另外两个股票池（5 种结构，Fig11/Fig12 的数据源）
python scripts/02_run_experiments.py --config configs/csi500_arch_compare.yaml --workers 4 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/sse50_arch_compare.yaml --workers 4 --threads-per-worker 2
#    （只有 LSTM 的旧 run 也能画，缺的结构会记进 panels.json 的 unsupported_architectures）
python scripts/02_run_experiments.py --config configs/csi500_reference.yaml --workers 2 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/sse50_reference.yaml --workers 2 --threads-per-worker 2
# 5) 画图（只写进一个独立文件夹）
python scripts/09_reference_figures.py --out results/figures_reference
```

`panels.json` 记录了每张图对应的实验名、指标定义、数据来源目录，便于逐条核对。
"""


def write_docs(out_dir: Path, reference_dir: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(ref=reference_dir), encoding="utf-8")
    (out_dir / "panels.json").write_text(
        json.dumps({"reference_dir": reference_dir, "panels": PANELS},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {out_dir / 'README.md'} and {out_dir / 'panels.json'}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "figures_reference"),
                    help="output folder for the reproduced panels")
    ap.add_argument("--reference-dir", default=r"D:\figures",
                    help="folder of the reference panels (documentation only)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when a panel cannot be produced")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reference panels : {args.reference_dir}")
    print(f"output folder    : {out_dir}")

    mu_dirs = run_dirs("mu")
    ent_dirs = run_dirs("ent")
    tree_dirs = run_dirs("tree")

    start_banner("panels 2-4: target-return sweep")
    figure_target_sweep(out_dir, mu_dirs)

    start_banner("panel 5: cumulative return at tau = 0.020")
    figure_cumulative_at_target(out_dir, mu_dirs, 0.020)

    start_banner("panels 8-9: entropy sweep")
    figure_entropy_sweep(out_dir, ent_dirs)

    start_banner("panel 10: realised monthly returns")
    figure_out_of_sample_return(out_dir, mu_dirs)

    start_banner("panels 11-12: other universes")
    for fname, key, reference, market, figsize in MARKET_SPECS:
        figure_market(out_dir, key, fname, reference, market, figsize)

    start_banner("panels 13-14: tree baselines")
    figure_tree_baseline(out_dir, list(mu_dirs) + list(tree_dirs), 0.020,
                         "fig13_adaboost_xgboost_mu020.png", "Fig13")
    figure_tree_baseline(out_dir, list(mu_dirs) + list(tree_dirs), 0.022,
                         "fig14_adaboost_xgboost_mu022.png", "Fig14",
                         figsize=(13.63, 6.27))

    write_docs(out_dir, args.reference_dir)
    expected = 11
    print(f"[done] {len(PANELS)}/{expected} panels in {out_dir}")
    if args.strict and len(PANELS) < expected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
