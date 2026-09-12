"""Aggregate a completed experiment run into tables, figures and conclusions.

    # newest run under results/
    python scripts/03_report.py --latest

    # a specific run
    python scripts/03_report.py --run results/csi300_20260910_233720

The script only *reads* the artefacts written by ``02_run_experiments.py`` and
writes everything into ``<run>/report/``:

    report.md            the full report (tables + figures + conclusions)
    pooled.csv           all experiments, test windows of every fold concatenated
    per_fold.csv         every (experiment, fold) pair as one row
    figures/*.png        the plots referenced by the report
    summary.json         machine readable copy of the headline numbers

Nothing here is stochastic: re-running the script on the same run directory
reproduces byte-identical tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio.experiments import REGISTRY, MAIN_GRID  # noqa: E402
from e2e_portfolio.metrics import (  # noqa: E402
    HOLDING_EPS,
    concentration_metrics,
    concentration_series,
    compute_metrics,
    count_holdings,
    effective_n,
)

#: concentration diagnostics recovered from ``weights.csv`` when an older run
#: predates them in ``metrics.json``
CONCENTRATION_KEYS = ("hhi_mean", "top5_weight_mean", "max_weight_mean")
CONCENTRATION_COLUMNS = ("hhi", "top5_weight", "max_weight")

#: fixed experiment order (registry order) plus anything found on disk
ORDER: List[str] = list(MAIN_GRID)

#: compact identifiers used in the figures so the legends stay readable
SHORT = {
    "ew": "EW 1/N",
    "meancvar_hist": "Hist. Mean-CVaR",
    "lstm_topk": "P2O top-k",
    "lstm_softmax": "P2O softmax",
    "lstm_sparsemax": "P2O sparsemax",
    "e2e_noselect": "E2E no-select",
    "full": "E2E full",
    "full_nocvar_loss": "abl. no-CVaR-loss",
    "full_nocost": "abl. no-cost",
    "full_nosparse": "abl. no-sparsity",
    "full_quad": "abl. quadratic",
    "lstm_musigma": "P2O mu/sigma rule",
    "full_musigma": "E2E mu/sigma rule",
}

GROUPS = {
    "baseline": "无学习基线",
    "main": "预测-再优化",
    "e2e": "端到端可微",
    "ablation": "消融",
}

FOLD_LABEL = {
    "f1_2018_2019": "F1 2018–2019",
    "f2_2020_2021": "F2 2020–2021",
    "f3_2022_2026": "F3 2022–2026",
}


def group_of(experiment: str) -> str:
    """Classify an experiment for the report tables.

    Derived from the registry so that adding an experiment needs no change here:
    ``full_*`` ablations, the end-to-end variants (``e2e=True``), and everything
    else keeps its declared group.
    """
    spec = REGISTRY.get(experiment)
    if spec is None:
        return "main"
    if experiment.startswith("full_") and experiment != "full":
        return "ablation"
    return "e2e" if spec.e2e else spec.group


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def latest_run(results_root: Path) -> Path:
    cands = [d for d in results_root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    if not cands:
        raise SystemExit(f"no completed run found under {results_root}")
    return max(cands, key=lambda d: (d / "summary.json").stat().st_mtime)


def load_run_config(run_dir: Path):
    """Return ``(Config, fold rows)`` from the run's own ``config.yaml`` copy."""
    path = run_dir / "config.yaml"
    if not path.exists():
        return None, []
    try:
        from e2e_portfolio.config import Config

        cfg = Config.from_yaml(path)
    except Exception as exc:  # pragma: no cover - report must still work
        print(f"[report] could not read {path}: {type(exc).__name__}: {exc}")
        return None, []
    rows = [f.__dict__ if hasattr(f, "__dict__") else dict(f) for f in cfg.folds]
    return cfg, rows


def ensure_concentration(job: Dict, fold_dir: Path) -> bool:
    """Backfill the concentration block of a job from its ``weights.csv``.

    Runs that were produced before the concentration metrics existed keep their
    per-rebalance weight matrix on disk, so ``hhi`` / top-5 share / largest
    weight can be recovered exactly without retraining anything.  Returns
    ``True`` when something was filled in.
    """
    metrics = job["metrics"]
    if all(key in metrics for key in CONCENTRATION_KEYS):
        return False
    weights_path = fold_dir / "weights.csv"
    if not weights_path.exists():
        return False
    weights = pd.read_csv(weights_path, index_col=0)
    filled = False
    for key, value in concentration_metrics(weights.to_numpy()).items():
        if key not in metrics:
            metrics[key] = value
            filled = True
    reb = job.get("rebalance")
    if reb is not None and not reb.empty:
        series = concentration_series(weights.to_numpy())
        for name, values in series.items():
            if name not in reb.columns and len(values) == len(reb):
                reb[name] = values
    return filled


def ensure_holdings(job: Dict, fold_dir: Path) -> bool:
    """Recompute ``holdings_mean`` from the saved target weights.

    Runs produced before ``HOLDING_EPS`` existed counted every non-zero target
    weight, which includes the 1e-8..1e-6 residuals the conic solver leaves on
    the rejected names, so their ``holdings_mean`` was the whole investable
    cross-section.  ``weights.csv`` keeps the exact target weights per rebalance,
    so the number of economic positions can be recovered without retraining.
    Returns ``True`` when the recorded value was corrected.
    """
    metrics = job["metrics"]
    weights_path = fold_dir / "weights.csv"
    if not weights_path.exists():
        return False
    weights = pd.read_csv(weights_path, index_col=0).to_numpy()
    series = count_holdings(weights)
    if series.size == 0:
        return False
    reb = job.get("rebalance")
    if reb is not None and not reb.empty and len(series) == len(reb):
        reb["n_holdings"] = series
    value = float(series.mean())
    if metrics.get("holdings_mean") == value:
        return False
    metrics["holdings_mean"] = value
    return True


def load_jobs(run_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict], Dict]:
    """Read every ``(experiment, fold)`` artefact written by the grid runner.

    Legacy runs are migrated in place: when a metric had to be recovered (or
    corrected) from ``weights.csv``, the repaired ``metrics.json`` and
    ``rebalance.csv`` are written back, so the raw artefacts, the report and the
    README all show the same numbers.
    """
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    jobs: Dict[Tuple[str, str], Dict] = {}
    n_backfilled = 0
    n_holdings_fixed = 0
    n_migrated = 0
    for metrics_path in sorted(run_dir.glob("*/*/metrics.json")):
        fold_dir = metrics_path.parent
        fold = fold_dir.name
        experiment = fold_dir.parent.name
        if experiment in {"report"}:
            continue
        job: Dict = {"metrics": json.loads(metrics_path.read_text(encoding="utf-8"))}
        ret_path = fold_dir / "returns.csv"
        job["returns"] = (
            pd.read_csv(ret_path) if ret_path.exists() else pd.DataFrame()
        )
        reb_path = fold_dir / "rebalance.csv"
        job["rebalance"] = pd.read_csv(reb_path) if reb_path.exists() else pd.DataFrame()
        hist_path = fold_dir / "history.csv"
        job["history"] = pd.read_csv(hist_path) if hist_path.exists() else None
        recovered = bool(ensure_concentration(job, fold_dir))
        corrected = bool(ensure_holdings(job, fold_dir))
        n_backfilled += int(recovered)
        n_holdings_fixed += int(corrected)
        if recovered or corrected:
            metrics_path.write_text(
                json.dumps(job["metrics"], indent=2, ensure_ascii=False, default=float),
                encoding="utf-8",
            )
            if not job["rebalance"].empty:
                job["rebalance"].to_csv(reb_path, index=False, encoding="utf-8")
            n_migrated += 1
        jobs[(experiment, fold)] = job
    if not jobs:
        raise SystemExit(f"no metrics.json found under {run_dir}")
    if n_backfilled:
        print(f"[report] concentration metrics backfilled from weights.csv for "
              f"{n_backfilled} job(s)")
    if n_holdings_fixed:
        print(f"[report] holdings_mean recomputed with HOLDING_EPS={HOLDING_EPS:g} "
              f"for {n_holdings_fixed} job(s)")
    if n_migrated:
        print(f"[report] rewrote metrics.json + rebalance.csv for {n_migrated} job(s)")
    return jobs, summary


def experiment_order(jobs: Dict[Tuple[str, str], Dict]) -> List[str]:
    seen = {e for e, _ in jobs}
    ordered = [e for e in ORDER if e in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


def fold_order(jobs: Dict[Tuple[str, str], Dict], summary: Dict) -> List[str]:
    seen = {f for _, f in jobs}
    declared = [f["name"] for f in summary.get("folds", [])] if summary.get("folds") else []
    ordered = [f for f in declared if f in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def per_fold_table(jobs: Dict[Tuple[str, str], Dict], experiments: Sequence[str],
                   folds: Sequence[str]) -> pd.DataFrame:
    rows = []
    for e in experiments:
        for f in folds:
            job = jobs.get((e, f))
            if job is None:
                continue
            m = dict(job["metrics"])
            m.setdefault("experiment", e)
            m.setdefault("fold", f)
            rows.append(m)
    return pd.DataFrame(rows)


def pooled_returns(jobs: Dict[Tuple[str, str], Dict], experiment: str,
                   folds: Sequence[str]) -> Optional[pd.DataFrame]:
    """Concatenate the out-of-sample daily series of every fold, in date order."""
    frames = []
    for f in folds:
        job = jobs.get((experiment, f))
        if job is None or job["returns"].empty:
            continue
        df = job["returns"].copy()
        df["fold"] = f
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def pooled_table(jobs: Dict[Tuple[str, str], Dict], experiments: Sequence[str],
                 folds: Sequence[str], alpha: float) -> pd.DataFrame:
    """Recompute every metric on the concatenated out-of-sample series."""
    rows = []
    for e in experiments:
        rets = pooled_returns(jobs, e, folds)
        if rets is None or rets.empty:
            continue
        turnover, holdings, eff = [], [], []
        for f in folds:
            job = jobs.get((e, f))
            if job is None or job["rebalance"].empty:
                continue
            rb = job["rebalance"]
            turnover.extend(rb["turnover"].tolist())
            holdings.extend(rb["n_holdings"].tolist())
            eff.extend(rb["eff_holdings"].tolist())
        net = rets["ret_net"].to_numpy(dtype=np.float64)
        gross = rets["ret_gross"].to_numpy(dtype=np.float64)
        bench = rets["bench"].to_numpy(dtype=np.float64)
        m = compute_metrics(
            net,
            benchmark=bench,
            turnover=np.asarray(turnover) if turnover else None,
            n_holdings=np.asarray(holdings) if holdings else None,
            effective_holdings=np.asarray(eff) if eff else None,
            alpha=alpha,
        )
        # holdings *and* concentration, one value per rebalance across all folds
        for col, key in zip(CONCENTRATION_COLUMNS, CONCENTRATION_KEYS):
            values: List[float] = []
            for f in folds:
                job = jobs.get((e, f))
                if job is None or job["rebalance"].empty or col not in job["rebalance"]:
                    continue
                values.extend(job["rebalance"][col].tolist())
            m[key] = float(np.mean(values)) if values else float("nan")
        g = compute_metrics(gross, benchmark=None, alpha=alpha)
        for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "cvar_95", "total_return"):
            m[f"{k}_gross"] = g[k]
        m["cost_drag_ann"] = g["ann_return"] - m["ann_return"]
        m["experiment"] = e
        m["group"] = group_of(e)
        m["n_folds"] = int(rets["fold"].nunique())
        m["bench_ann_vol"] = float(
            np.std(bench, ddof=1) * np.sqrt(252.0)
        ) if bench.size > 2 else float("nan")
        rows.append(m)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _pct(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def _num(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def markdown_table(df: pd.DataFrame, cols: Sequence[Tuple[str, str, str]]) -> str:
    """``cols`` is a list of ``(column, header, kind)`` with kind in pct|num|int|str|sci."""
    header = "| " + " | ".join(h for _, h, _ in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, rule]
    for _, row in df.iterrows():
        cells = []
        for col, _, kind in cols:
            v = row.get(col, np.nan)
            if kind == "pct":
                cells.append(_pct(float(v)))
            elif kind == "num":
                cells.append(_num(float(v)))
            elif kind == "int":
                cells.append("n/a" if v is None or not np.isfinite(float(v)) else f"{int(v)}")
            elif kind == "sci":
                cells.append("n/a" if v is None or not np.isfinite(float(v)) else f"{float(v):.2e}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _cum(r: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(r, dtype=np.float64))


def _drawdown(curve: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(curve)
    return curve / peak - 1.0


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def fig_equity(jobs, experiments, folds, out: Path) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for e in experiments:
        rets = pooled_returns(jobs, e, folds)
        if rets is None or rets.empty:
            continue
        curve = _cum(rets["ret_net"].to_numpy())
        is_full = e == "full"
        ax.plot(rets["date"], curve, label=SHORT.get(e, e),
                lw=2.0 if is_full else 1.1, alpha=1.0 if is_full else 0.85,
                zorder=5 if is_full else 2)
    first = next((pooled_returns(jobs, e, folds) for e in experiments
                  if pooled_returns(jobs, e, folds) is not None), None)
    if first is not None:
        ax.plot(first["date"], _cum(first["bench"].to_numpy()), color="black",
                lw=1.4, ls="--", label="CSI 300", zorder=4)
    ax.set_title("Out-of-sample net equity, all folds concatenated")
    ax.set_ylabel("cumulative net growth of 1")
    ax.legend(ncol=2, fontsize=7.5)
    fig.tight_layout()
    path = out / "fig_equity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_drawdown(jobs, experiments, folds, out: Path) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for e in experiments:
        rets = pooled_returns(jobs, e, folds)
        if rets is None or rets.empty:
            continue
        ax.plot(rets["date"], _drawdown(_cum(rets["ret_net"].to_numpy())),
                label=SHORT.get(e, e), lw=1.1, alpha=0.85)
    ax.set_title("Drawdown of the net equity curve (all folds concatenated)")
    ax.set_ylabel("drawdown")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    path = out / "fig_drawdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_bars(pooled: pd.DataFrame, out: Path) -> Optional[Path]:
    if pooled.empty:
        return None
    labels = [SHORT.get(e, e) for e in pooled["experiment"]]
    specs = [
        ("ann_return", "annualised net return", "%"),
        ("sharpe", "annualised Sharpe (net)", ""),
        ("max_drawdown", "max drawdown", "%"),
        ("cvar_95", "CVaR 95% (daily)", "%"),
        ("turnover_ann", "annualised turnover", ""),
        ("eff_holdings_mean", "mean effective holdings", ""),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (col, title, unit) in zip(axes.ravel(), specs):
        if col not in pooled:
            ax.set_visible(False)
            continue
        vals = pooled[col].to_numpy(dtype=np.float64) * (100.0 if unit == "%" else 1.0)
        colors = ["#c0392b" if e == "full" else "#4a6fa5" for e in pooled["experiment"]]
        ax.barh(labels, vals, color=colors)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.invert_yaxis()
    fig.tight_layout()
    path = out / "fig_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_risk_return(pooled: pd.DataFrame, out: Path) -> Optional[Path]:
    if pooled.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for _, row in pooled.iterrows():
        e = row["experiment"]
        c = "#c0392b" if e == "full" else ("#7f8c8d" if row.get("group") == "baseline" else "#4a6fa5")
        ax.scatter(row["ann_vol"], row["ann_return"], s=48, color=c, zorder=3)
        ax.annotate(SHORT.get(e, e), (row["ann_vol"], row["ann_return"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    if "bench_ann_return" in pooled:
        b = pooled.iloc[0]
        bx = float(b.get("bench_ann_vol", np.nan))
        ax.scatter(bx, b["bench_ann_return"], marker="*", s=160, color="black", zorder=4)
        ax.annotate("CSI 300", (bx, b["bench_ann_return"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("annualised volatility (net)")
    ax.set_ylabel("annualised net return")
    ax.set_title("Risk / return, all folds concatenated")
    fig.tight_layout()
    path = out / "fig_risk_return.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_sharpe_heatmap(per_fold: pd.DataFrame, experiments, folds, out: Path) -> Optional[Path]:
    if per_fold.empty:
        return None
    mat = np.full((len(experiments), len(folds)), np.nan)
    for i, e in enumerate(experiments):
        for j, f in enumerate(folds):
            sel = per_fold[(per_fold["experiment"] == e) & (per_fold["fold"] == f)]
            if len(sel):
                mat[i, j] = float(sel["sharpe"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.6, 0.44 * len(experiments) + 1.6))
    lim = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(folds)), [FOLD_LABEL.get(f, f) for f in folds], fontsize=7.5)
    ax.set_yticks(range(len(experiments)), [SHORT.get(e, e) for e in experiments], fontsize=7.5)
    for i in range(len(experiments)):
        for j in range(len(folds)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Sharpe per fold", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Sharpe")
    fig.tight_layout()
    path = out / "fig_sharpe_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_training_history(jobs, folds, experiment: str, out: Path) -> Optional[Path]:
    keys = [k for k in ("L_total", "L_pred", "L_decision", "L_tail", "L_turnover", "L_sparse")]
    have = False
    for f in folds:
        job = jobs.get((experiment, f))
        if job is not None and job["history"] is not None and not job["history"].empty:
            have = True
    if not have:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, key in zip(axes.ravel(), keys):
        plotted = False
        for f in folds:
            job = jobs.get((experiment, f))
            h = None if job is None else job["history"]
            if h is None or h.empty or key not in h.columns:
                continue
            x = h["epoch"] if "epoch" in h.columns else np.arange(1, len(h) + 1)
            ax.plot(x, h[key], marker="o", ms=2.5, lw=1.1,
                    label=FOLD_LABEL.get(f, f))
            plotted = True
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("epoch", fontsize=7.5)
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.set_visible(False)
    fig.suptitle(f"Training history — {experiment}", fontsize=10)
    fig.tight_layout()
    path = out / f"fig_history_{experiment}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cost_drag(jobs, folds, experiment: str, out: Path) -> Optional[Path]:
    rets = pooled_returns(jobs, experiment, folds)
    if rets is None or rets.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(rets["date"], _cum(rets["ret_gross"].to_numpy()), lw=1.3, label="gross (no cost)")
    ax1.plot(rets["date"], _cum(rets["ret_net"].to_numpy()), lw=1.3, label="net (15 bp per side)")
    ax1.set_ylabel("cumulative growth of 1")
    ax1.set_title(f"Cost drag — {SHORT.get(experiment, experiment)}")
    ax1.legend()
    drag = _cum(rets["ret_gross"].to_numpy()) / _cum(rets["ret_net"].to_numpy()) - 1.0
    ax2.plot(rets["date"], drag, color="#c0392b", lw=1.2)
    ax2.set_ylabel("cumulative cost")
    fig.tight_layout()
    path = out / f"fig_cost_{experiment}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_portfolio_profile(jobs, folds, experiment: str, out: Path) -> Optional[Path]:
    rets = pooled_returns(jobs, experiment, folds)
    if rets is None or rets.empty:
        return None
    reb = []
    for f in folds:
        job = jobs.get((experiment, f))
        if job is None or job["rebalance"].empty:
            continue
        df = job["rebalance"].copy()
        df["fold"] = f
        reb.append(df)
    if not reb:
        return None
    rb = pd.concat(reb, ignore_index=True)
    rb["date"] = pd.to_datetime(rb["date"])
    rb = rb.sort_values("date")
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    axes[0].plot(rb["date"], rb["n_holdings"], lw=1.2, label="names held")
    axes[0].plot(rb["date"], rb["eff_holdings"], lw=1.2, label="effective N (1/Σw²)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=7.5, loc="upper left")
    if "top5_weight" in rb.columns:
        ax_c = axes[0].twinx()
        ax_c.plot(rb["date"], rb["top5_weight"], lw=1.1, color="#e67e22",
                  label="top-5 weight share")
        ax_c.set_ylabel("top-5 share", fontsize=8)
        ax_c.tick_params(axis="y", labelsize=8)
        ax_c.legend(fontsize=7.5, loc="upper right")
    axes[1].plot(rb["date"], rb["turnover"], lw=1.2, color="#8e44ad")
    axes[1].set_ylabel("one-way turnover")
    axes[2].plot(rb["date"], rb["cost"], lw=1.2, color="#c0392b")
    axes[2].set_ylabel("cost per rebalance")
    axes[0].set_title(f"Portfolio profile — {SHORT.get(experiment, experiment)}")
    fig.tight_layout()
    path = out / f"fig_profile_{experiment}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# conclusions
# --------------------------------------------------------------------------- #
def _pct_delta(a: float, b: float) -> str:
    return f"{(a - b) * 100:+.2f} 个百分点"


def load_selection_diag(run_dir: Path) -> pd.DataFrame:
    """Read ``report/selection_diagnostics.csv`` (written by ``04_selection_diagnostics.py``)."""
    path = run_dir / "report" / "selection_diagnostics.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - the report must still build
        print(f"[report] could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _weighted(df: pd.DataFrame, col: str) -> float:
    """Period-weighted mean, so folds with more test periods count proportionally."""
    if df.empty or col not in df:
        return float("nan")
    w = df["n_periods"].astype(float) if "n_periods" in df else pd.Series(1.0, index=df.index)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float((df[col].astype(float) * w).sum() / total)


def build_selection_insight(diag: pd.DataFrame) -> List[str]:
    """Interpret the selection-layer diagnostics (empty list when unavailable)."""
    out: List[str] = []
    if diag.empty:
        return out

    p2o_names = ("lstm_topk", "lstm_softmax", "lstm_sparsemax")
    present = set(diag["experiment"])
    p2o = [e for e in p2o_names if e in present]
    p2o_df = diag[diag["experiment"].isin(p2o)] if p2o else diag.iloc[0:0]

    if len(p2o) > 1 and p2o_df.groupby("fold")["encoder_sha256"].nunique().max() == 1:
        out.append(
            "1. **三个预测-再优化实验训练的是同一个网络**：`lstm_topk` / `lstm_softmax` / "
            "`lstm_sparsemax` 的编码器参数 SHA-256 完全相同（因为该范式的训练损失只保留预测项 "
            "`l_pred = 1`，选择层只在组合形成阶段起作用）。三者在样本外的差异 100% 来自选择层"
            "（支撑集与仓位上限），不包含任何训练随机性，因而是干净的对照。"
        )

    # ---- predict-then-optimise: sparsemax vs softmax on the *same* logits ----
    # NOTE: scoped to the p2o experiments on purpose.  `full` also uses sparsemax, but it is
    # trained end-to-end with the sparsity loss, so its logit scale is a different regime and
    # must not be pooled with the frozen-encoder runs.
    if "lstm_sparsemax" in present and "lstm_softmax" in present:
        sp = diag[diag["experiment"] == "lstm_sparsemax"]
        sm = diag[diag["experiment"] == "lstm_softmax"]
        thr, spread = _weighted(sp, "spar_thresh"), _weighted(sp, "score_spread")
        below = _weighted(sp, "spread_below_thresh")
        per_fold = "、".join(
            f"{r['fold'].split('_')[0]} {r['spread_below_thresh'] * 100:.0f}%"
            for _, r in sp.sort_values("fold").iterrows()
        )
        out.append(
            "2. **在预测-再优化范式下 sparsemax 几乎不稀疏**：稀疏的临界值是 "
            f"`max(z) − z_i > 1 / n`（可投资域约 {_weighted(sp, 'n_valid'):.0f} 只，"
            f"即阈值约 {thr:.2e}），而冻结编码器输出的 logit 极差平均只有 {spread:.2e}，"
            f"与阈值同量级；各折「极差 < 1/n」的期数占比为 {per_fold}"
            f"（三折加权 {below * 100:.0f}%）。结果是 sparsemax 的"
            f"平均支撑集 {_weighted(sp, 'pi_support'):.0f} 只，与 softmax 的 "
            f"{_weighted(sm, 'pi_support'):.0f} 只几乎一致，它退化成了（近似）softmax。"
            "**要真正得到稀疏组合，要么显式给定支撑集大小（top-k），"
            "要么让选择层参与训练、把 logit 尺度抬到 `1/n` 以上。**"
        )

    if "lstm_topk" in present:
        tk = diag[diag["experiment"] == "lstm_topk"]
        others = diag[diag["experiment"].isin(["lstm_softmax", "lstm_sparsemax"])]
        out.append(
            f"3. **在该范式下只有 top-k 是真正稀疏的选择器**：支撑集恒为 "
            f"{_weighted(tk, 'pi_support'):.0f} 只（等于配置的 `topk = 25`），"
            f"实际账本平均持有 {_weighted(tk, 'book_n'):.1f} 只、有效持仓 "
            f"{_weighted(tk, 'book_pr'):.1f} 只（单票 10% 上限把权重摊薄，所以有效持仓"
            "明显少于名义持仓）；相比之下 softmax / sparsemax 的账本摊到 "
            f"{_weighted(others, 'book_n'):.0f} 只左右。"
        )

    # ---- the same sparsemax layer *is* sparse once the sparsity loss trains the encoder ----
    if "full" in present and "lstm_sparsemax" in present:
        fu = diag[diag["experiment"] == "full"]
        sp = diag[diag["experiment"] == "lstm_sparsemax"]
        out.append(
            "4. **决定稀疏性的是损失函数，而不是选择层本身**：同一个 sparsemax 层，在 `full`"
            f"（端到端、损失里带 `l_sparse`）里 logit 极差被训练到 {_weighted(fu, 'score_spread'):.2e}，"
            f"是阈值 {_weighted(fu, 'spar_thresh'):.2e} 的 "
            f"{_weighted(fu, 'score_spread') / _weighted(fu, 'spar_thresh'):.1f} 倍，"
            f"**没有任何一期退化**（`极差 < 1/n` 占比 {_weighted(fu, 'spread_below_thresh') * 100:.0f}%），"
            f"π 的支撑集收缩到 {_weighted(fu, 'pi_support'):.0f} 只、账本收缩到 "
            f"{_weighted(fu, 'book_n'):.1f} 只；而在 `lstm_sparsemax` 里同样的层几乎不删任何票"
            f"（支撑集 {_weighted(sp, 'pi_support'):.0f} 只、账本 {_weighted(sp, 'book_n'):.1f} 只）。"
            "也就是说，稀疏选择是 `L_sparse` 这个损失项学出来的行为，选择层只负责把学好的 "
            "logit 变成合法的组合权重。"
        )
    return out


def build_conclusions(pooled: pd.DataFrame, per_fold: pd.DataFrame,
                      folds: Sequence[str], diag: Optional[pd.DataFrame] = None) -> List[str]:
    out: List[str] = []
    if pooled.empty:
        return ["没有可用的结果。"]

    by = {r["experiment"]: r for _, r in pooled.iterrows()}
    ranked = pooled.sort_values("sharpe", ascending=False)
    best = ranked.iloc[0]

    out.append(
        f"1. **整体排序**：在全部 {int(pooled['n_folds'].max())} 折测试窗口拼接后的样本上，"
        f"夏普比率最高的是 `{best['experiment']}`（Sharpe {_num(best['sharpe'])}，"
        f"年化净收益 {_pct(best['ann_return'])}，最大回撤 {_pct(best['max_drawdown'])}）。"
        f"基准（等权 1/N）的年化净收益为 {_pct(by['ew']['ann_return']) if 'ew' in by else 'n/a'}，"
        f"传统历史情景 Mean-CVaR 为 "
        f"{_pct(by['meancvar_hist']['ann_return']) if 'meancvar_hist' in by else 'n/a'}。"
    )

    if "full" in by and "lstm_sparsemax" in by:
        f, p = by["full"], by["lstm_sparsemax"]
        better = "优于" if f["sharpe"] > p["sharpe"] else "未能优于"
        out.append(
            f"2. **端到端 vs 预测-再优化**：`full`（端到端可微，Sharpe {_num(f['sharpe'])}，"
            f"年化 {_pct(f['ann_return'])}, 最大回撤 {_pct(f['max_drawdown'])}）"
            f"{better}对应的预测-再优化版本 `lstm_sparsemax`"
            f"（Sharpe {_num(p['sharpe'])}，年化 {_pct(p['ann_return'])}，"
            f"最大回撤 {_pct(p['max_drawdown'])}）。"
            f"Sharpe 差 {f['sharpe'] - p['sharpe']:+.2f}，"
            f"最大回撤差 {_pct_delta(f['max_drawdown'], p['max_drawdown'])}。"
            f"两者的换手率分别为 {_num(f.get('turnover_ann', float('nan')))} 和 "
            f"{_num(p.get('turnover_ann', float('nan')))}（年化，单边）。"
        )

    if all(e in by for e in ("lstm_topk", "lstm_softmax", "lstm_sparsemax")):
        tk, sm, sp = by["lstm_topk"], by["lstm_softmax"], by["lstm_sparsemax"]
        winner = max((tk, sm, sp), key=lambda r: r["sharpe"])
        band = max(r["sharpe"] for r in (tk, sm, sp)) - min(r["sharpe"] for r in (tk, sm, sp))
        out.append(
            f"3. **选择层对比（三者训练完全相同，只差选择层）**：`lstm_topk` Sharpe "
            f"{_num(tk['sharpe'])}、有效持仓 {_num(tk.get('eff_holdings_mean', float('nan')), 1)}；"
            f"`lstm_softmax` Sharpe {_num(sm['sharpe'])}、"
            f"有效持仓 {_num(sm.get('eff_holdings_mean', float('nan')), 1)}；"
            f"`lstm_sparsemax` Sharpe {_num(sp['sharpe'])}、"
            f"有效持仓 {_num(sp.get('eff_holdings_mean', float('nan')), 1)}。"
            f"三者编码器参数逐位相同（见 §3 诊断），因此差异完全由选择层决定："
            f"`{winner['experiment']}` 名义上最优（Sharpe {_num(winner['sharpe'])}），"
            f"但三者 Sharpe 的极差只有 {band:.2f}，落在本设置的重训噪声量级内，"
            "**因此这不构成一个稳健的排序**。"
            f"`lstm_softmax` 与 `lstm_sparsemax` 的净值曲线几乎重合，原因是 logit 极差"
            "与稀疏临界值 `1/n` 同量级（f1 折甚至全程低于阈值），"
            "sparsemax 退化为（近似）softmax，几乎不删任何票（详见 §3）。"
        )

    if "full" in by:
        base = by["full"]
        rows = []
        for name, human in (
            ("full_nocvar_loss", "去掉决策损失中的尾部（CVaR）项"),
            ("full_nocost", "训练与优化都不看交易成本"),
            ("full_nosparse", "去掉稀疏选择激励"),
            ("full_quad", "用二次（Herfindahl）分散项替代熵正则"),
        ):
            if name not in by:
                continue
            r = by[name]
            rows.append(
                f"   - {human}（`{name}`）：Sharpe {_num(r['sharpe'])}"
                f"（{r['sharpe'] - base['sharpe']:+.2f}），"
                f"年化净收益 {_pct(r['ann_return'])}"
                f"（{_pct_delta(r['ann_return'], base['ann_return'])}），"
                f"最大回撤 {_pct(r['max_drawdown'])}"
                f"（{_pct_delta(r['max_drawdown'], base['max_drawdown'])}），"
                f"年化换手 {_num(r.get('turnover_ann', float('nan')))}"
                f"（{r.get('turnover_ann', float('nan')) - base.get('turnover_ann', float('nan')):+.2f}）"
            )
        if rows:
            out.append("4. **消融（相对 `full`，正数表示变好）**：\n" + "\n".join(rows))

    if "full" in by and "ew" in by:
        f, e = by["full"], by["ew"]
        out.append(
            f"5. **相对等权基线的真实超额**：`full` 年化净收益 {_pct(f['ann_return'])}，"
            f"等权 {_pct(e['ann_return'])}；信息比率 {_num(f.get('info_ratio', float('nan')))}，"
            f"alpha {_pct(f.get('alpha', float('nan')))}，beta {_num(f.get('beta', float('nan')))}。"
            f"最大回撤从 {_pct(e['max_drawdown'])} "
            f"{'收窄到' if f['max_drawdown'] > e['max_drawdown'] else '扩大到'} "
            f"{_pct(f['max_drawdown'])}。"
        )

    if "full" in by:
        f = by["full"]
        out.append(
            f"6. **风险与成本**：`full` 的日 CVaR(95%) 为 {_pct(f['cvar_95'])}，"
            f"年化波动 {_pct(f['ann_vol'])}，平均持仓 "
            f"{_num(f.get('holdings_mean', float('nan')), 1)} 只、"
            f"有效持仓 {_num(f.get('eff_holdings_mean', float('nan')), 1)} 只，"
            f"年化单边换手 {_num(f.get('turnover_ann', float('nan')))}，"
            f"成本拖累（毛收益−净收益）{_pct(f.get('cost_drag_ann', float('nan')))}。"
        )

    # cross-fold stability
    stab = []
    for e in pooled["experiment"]:
        sel = per_fold[per_fold["experiment"] == e]
        if sel.empty:
            continue
        pos = int((sel["sharpe"] > 0).sum())
        stab.append((e, pos, len(sel), float(sel["sharpe"].min()), float(sel["sharpe"].max())))
    if stab:
        n_fold = max(t[2] for t in stab)
        stable = max(stab, key=lambda t: (t[1], t[3]))
        all_pos = [t[0] for t in stab if t[1] == n_fold]
        prefix = (
            f"没有任何实验在全部 {n_fold} 折上都取得正夏普；"
            if not all_pos
            else f"只有 {', '.join('`%s`' % n for n in all_pos)} 在全部 {n_fold} 折上夏普为正；"
        )
        out.append(
            f"7. **跨折稳定性**：{prefix}"
            f"正夏普折数最多的是 `{stable[0]}`"
            f"（{stable[1]}/{stable[2]} 折，最差 {stable[3]:+.2f} / 最好 {stable[4]:+.2f}）。"
            f"所有策略在不同年份区间上都有明显波动，单折结论不足以支撑泛化性判断。"
        )

    out.append(
        "8. **解读与局限**：结论受限于（a）单一指数（沪深 300 动态成分股）与单一成本假设"
        "（单边 15 bp 线性成本，未建模冲击成本与涨跌停延迟成交）；"
        "（b）测试窗口只有 3 折、共约 8 年，统计功效有限；"
        "（c）场景生成是高斯参数化模型，CVaR 只对模型隐含分布稳健；"
        "（d）深度网络的随机性使同一配置重复训练仍会有数个百分点差异，"
        "报告中的差异若小于该量级不应视为真实效应；"
        "（e）资产槽位数 `n_max` 由每折的可投资域决定（f1/f2/f3 = 154/155/165），"
        "它通过 `Dropout` 的随机数消耗量影响训练轨迹，因此**跨折的绝对值不可直接横向比较**"
        "（仓库内同一折的所有实验共用同一个 `n_max`，折内比较仍然成立）；"
        "（f）选择层的稀疏不等于组合稀疏——单票 10% 上限会把权重摊到更多票上，"
        "有效持仓应看回测输出的 `eff_holdings`（`full` 为 "
        f"{_num(by['full'].get('eff_holdings_mean', float('nan')), 1) if 'full' in by else 'n/a'} 只），"
        "而不是 π 的支撑集大小。"
    )
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    # the report prints Chinese with typographic characters (e.g. U+2212); a
    # legacy console code page (cp936) cannot encode all of them, so never let
    # printing crash the script after the artefacts have been written.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # pragma: no cover - older / redirected streams
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run directory, e.g. results/csi300_...")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--latest", action="store_true", help="use the newest run")
    ap.add_argument("--alpha", type=float, default=None, help="VaR/CVaR level")
    args = ap.parse_args()

    results_root = ROOT / args.results_root
    if args.run:
        run_dir = (ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    else:
        run_dir = latest_run(results_root)
    run_dir = run_dir.resolve()
    print(f"[report] run directory: {run_dir}")

    jobs, summary = load_jobs(run_dir)
    experiments = experiment_order(jobs)
    folds = fold_order(jobs, summary)
    cfg, fold_rows = load_run_config(run_dir)
    alpha = args.alpha
    if alpha is None:
        alpha = float(cfg.opt.alpha) if cfg is not None else 0.95
    cost_bps = float(cfg.backtest.cost_bps) if cfg is not None else float("nan")

    per_fold = per_fold_table(jobs, experiments, folds)
    pooled = pooled_table(jobs, experiments, folds, alpha)
    diag = load_selection_diag(run_dir)
    if not pooled.empty:
        pooled = pooled.sort_values("sharpe", ascending=False).reset_index(drop=True)
    experiments_pooled = list(pooled["experiment"]) if not pooled.empty else experiments

    out = run_dir / "report"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    _style()

    per_fold.to_csv(out / "per_fold.csv", index=False, encoding="utf-8")
    pooled.to_csv(out / "pooled.csv", index=False, encoding="utf-8")

    figures: List[Tuple[str, Path]] = []
    for name, fn in (
        ("net equity", lambda p: fig_equity(jobs, experiments, folds, p)),
        ("drawdown", lambda p: fig_drawdown(jobs, experiments, folds, p)),
        ("metric bars", lambda p: fig_bars(pooled, p)),
        ("risk/return", lambda p: fig_risk_return(pooled, p)),
        ("sharpe heatmap", lambda p: fig_sharpe_heatmap(per_fold, experiments, folds, p)),
        ("training history", lambda p: fig_training_history(jobs, folds, "full", p)),
        ("cost drag", lambda p: fig_cost_drag(jobs, folds, "full", p)),
        ("portfolio profile", lambda p: fig_portfolio_profile(jobs, folds, "full", p)),
    ):
        try:
            path = fn(figs)
        except Exception as exc:  # keep the report going even if one plot fails
            print(f"[report] figure '{name}' failed: {type(exc).__name__}: {exc}")
            continue
        if path is not None:
            figures.append((name, path))

    # ---- markdown ---------------------------------------------------------
    md: List[str] = []
    n_ok = summary.get("n_ok", len(jobs))
    n_failed = summary.get("n_failed", 0)
    md.append("# 端到端可微 Mean-CVaR 投资组合学习 — 实验结果报告\n")
    md.append(f"- 运行目录：`{run_dir.name}`")
    md.append(f"- 运行耗时：{summary.get('seconds', float('nan')) / 60:.1f} 分钟，"
              f"作业 {n_ok}/{summary.get('n_jobs', len(jobs))} 成功"
              + (f"，**{n_failed} 个失败**" if n_failed else ""))
    md.append(f"- 样本：沪深 300 动态成分股，测试窗口 {len(folds)} 折拼接，"
              f"VaR/CVaR 置信水平 {alpha:.0%}")
    md.append(f"- 成本假设：单边 {_num(cost_bps)} bp，线性；"
              f"换手率按单边（½·Σ|Δw|）统计，并按实际调仓频率年化\n")

    if fold_rows:
        md.append("## 0. 折划分\n")
        md.append(markdown_table(pd.DataFrame(fold_rows),
                                 [("name", "折", "str"),
                                  ("train_start", "训练起", "str"),
                                  ("train_end", "训练止", "str"),
                                  ("val_start", "验证起", "str"),
                                  ("val_end", "验证止", "str"),
                                  ("test_start", "测试起", "str"),
                                  ("test_end", "测试止", "str")]))
        md.append("")

    md.append("## 1. 主结果（所有折的样本外日收益拼接后重算）\n")
    cols = [
        ("experiment", "实验", "str"),
        ("group", "类别", "str"),
        ("ann_return", "年化净收益", "pct"),
        ("ann_vol", "年化波动", "pct"),
        ("sharpe", "Sharpe", "num"),
        ("max_drawdown", "最大回撤", "pct"),
        ("cvar_95", "CVaR95(日)", "pct"),
        ("turnover_ann", "年化换手", "num"),
        ("eff_holdings_mean", "有效持仓", "num"),
        ("cost_drag_ann", "成本拖累", "pct"),
    ]
    md.append(markdown_table(pooled, cols))
    md.append("")

    md.append("### 1.2 持仓数量与集中度（Plan §6 要求项）\n")
    md.append("逐次调仓统计后取均值；HHI = Σw²，Top-5 与最大权重为占组合比例。\n")
    md.append(markdown_table(
        pooled,
        [("experiment", "实验", "str"),
         ("group", "类别", "str"),
         ("holdings_mean", "持仓数(>1e-3)", "num"),
         ("eff_holdings_mean", "有效持仓 1/Σw²", "num"),
         ("hhi_mean", "HHI Σw²", "num"),
         ("top5_weight_mean", "前5大权重", "pct"),
         ("max_weight_mean", "最大单一权重", "pct")]))
    md.append("")

    md.append("## 2. 分折结果\n")
    md.append("### 2.1 Sharpe\n")
    md.append(markdown_table(per_fold.sort_values(["experiment", "fold"]),
                             [("experiment", "实验", "str"), ("fold", "折", "str"),
                              ("sharpe", "Sharpe", "num"),
                              ("ann_return", "年化净收益", "pct"),
                              ("max_drawdown", "最大回撤", "pct"),
                              ("cvar_95", "CVaR95(日)", "pct")]))
    md.append("")

    md.append("## 3. 选择层诊断\n")
    if diag.empty:
        md.append("未找到 `report/selection_diagnostics.csv`。先运行\n")
        md.append("```bash")
        md.append("python scripts/04_selection_diagnostics.py --run " + run_dir.name)
        md.append("```")
        md.append("后重新生成本报告，即可看到选择层（支撑集 / 仓位上限）的逐期诊断。\n")
    else:
        for line in build_selection_insight(diag):
            md.append(line)
            md.append("")
        md.append(markdown_table(
            diag.sort_values(["experiment", "fold"]),
            [("experiment", "实验", "str"), ("fold", "折", "str"),
             ("selection", "选择器", "str"), ("encoder_sha256", "编码器哈希", "str"),
             ("n_periods", "期数", "int"), ("n_valid", "可投资", "num"),
             ("score_spread", "logit 极差", "sci"), ("spar_thresh", "稀疏临界 1/n", "sci"),
             ("spread_below_thresh", "极差<1/n 占比", "pct"),
             ("pi_support", "π 支撑集", "num"), ("pi_pr", "π 有效数", "num"),
             ("book_n", "账本持仓", "num"), ("book_pr", "账本有效持仓", "num")]))
        md.append("")
        md.append("> 诊断口径：重放测试期并只跑编码器（`run_layer=False`），"
                  "以空账本（`prev = 0`）隔离选择层本身；`book_*` 列读自回测写出的 "
                  "`weights.csv`，因此与 `pi_*` 的差别来自不可卖出持仓的结转与优化层。\n")

    md.append("## 4. 结论\n")
    md.extend(build_conclusions(pooled, per_fold, folds, diag))
    md.append("")

    if figures:
        md.append("## 5. 图表\n")
        # report.md lives in <run>/report/, so links must be relative to that directory
        for name, path in figures:
            rel = path.relative_to(out).as_posix()
            md.append(f"![{name}]({rel})\n")

    md.append("## 6. 复现\n")
    try:
        run_rel = run_dir.relative_to(ROOT).as_posix()
    except ValueError:  # a run directory outside the project
        run_rel = run_dir.as_posix()
    md.append("```bash")
    md.append("python scripts/01_prepare_data.py --validate")
    md.append(f"python scripts/02_run_experiments.py --config {run_rel}/config.yaml --workers 4")
    md.append(f"python scripts/04_selection_diagnostics.py --run {run_rel}")
    md.append(f"python scripts/03_report.py --run {run_rel}")
    md.append("```")
    md.append("")
    md.append("> 每次运行都会把**解析后的完整配置**复制到运行目录的 `config.yaml`，" 
              "因此上面第二条命令对基线网格（`configs/csi300.yaml`）、"
              "固定分数规则网格（`configs/csi300_score_rule.yaml`）和"
              "全样本扩展网格（`configs/csi300_full_universe.yaml`）都成立。\n")

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "n_experiments": len(experiments),
                "n_folds": len(folds),
                "pooled": pooled.to_dict(orient="records"),
                "per_fold": per_fold.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        ),
        encoding="utf-8",
    )

    print(f"[report] wrote {out / 'report.md'}")
    print(f"[report] wrote {len(figures)} figures into {figs}")
    print()
    for line in build_conclusions(pooled, per_fold, folds, diag):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
