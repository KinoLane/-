"""Markdown tables of README §9.9-§9.13 straight from the runs on disk.

Every number is recomputed from the out-of-sample ``returns.csv`` /
``weights.csv`` of the sweep runs with the report's own metric definition
(:func:`e2e_portfolio.metrics.compute_metrics` over the concatenated folds), so
the tables quoted in the README can be refreshed with one command::

    python scripts/10_summary_tables.py                     # print to stdout
    python scripts/10_summary_tables.py --out results/logs/summary_tables.md

A missing run or fold is printed as ``-`` instead of an invented number, and
``--check`` only reports how many jobs of each group are still incomplete.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DASH = "-"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIG = _load("reference_figures", "09_reference_figures.py")
EXP = FIG.EXP
#: the three walk-forward folds every run shares
FOLDS = ("f1_2018_2019", "f2_2020_2021", "f3_2022_2026")

_CACHE: Dict[Tuple[str, str], Optional[dict]] = {}


def stats(key: str, exp: str) -> Optional[dict]:
    """Pooled out-of-sample metrics of ``exp`` in the runs matching ``key``."""
    cache_key = (key, exp)
    if cache_key not in _CACHE:
        dirs = FIG.run_dirs(key, verbose=False)
        df = FIG.load_returns(dirs, exp)
        _CACHE[cache_key] = FIG.pooled_stats(df) if df is not None else None
    return _CACHE[cache_key]


def complete(key: str, exp: str) -> bool:
    dirs = FIG.run_dirs(key, verbose=False)
    return all(any((d / exp / f / "metrics.json").exists() for d in dirs) for f in FOLDS)


def cell(key: str, exp: str, metric: str, pct: bool = False, nd: int = 2) -> str:
    s = stats(key, exp)
    if not s or metric not in s:
        return DASH
    value = float(s[metric])
    if value != value:  # NaN
        return DASH
    return f"{value * 100:.{nd}f}" if pct else f"{value:.{nd}f}"


def ent(key: str, exp: str, nd: int = 2) -> str:
    dirs = FIG.run_dirs(key, verbose=False)
    if not complete(key, exp):
        return DASH
    value = FIG.portfolio_entropy(dirs, exp)
    return DASH if value != value else f"{value:.{nd}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# §9.9 target-return sweep (reference figures 2-4)
# --------------------------------------------------------------------------- #
def target_tables() -> List[str]:
    header = ["encoder"] + [f"$\\tau$={t:.3f}" for t in EXP.MU_TARGETS]
    rows: Dict[str, List[List[str]]] = {"ann_vol": [], "sharpe": [], "tracking_error": []}
    for metric, acc in rows.items():
        for arch in FIG.ARCH_ORDER:
            if not complete("mu", EXP.mu_experiment_name(EXP.MU_TARGETS[0], arch)):
                continue
            acc.append([FIG.ARCH_LABEL[arch]] + [
                cell("mu", EXP.mu_experiment_name(t, arch), metric, pct=(metric != "sharpe"))
                for t in EXP.MU_TARGETS
            ])
    if not rows["ann_vol"]:
        return ["_(no target-return run finished yet)_"]
    return [
        "**Annualised volatility (%)**\n\n" + table(header, rows["ann_vol"]),
        "**Sharpe ratio**\n\n" + table(header, rows["sharpe"]),
        "**Tracking error vs. CSI 300 (%)**\n\n" + table(header, rows["tracking_error"]),
    ]


# --------------------------------------------------------------------------- #
# §9.10 entropy sweep (reference figures 8-9)
# --------------------------------------------------------------------------- #
def entropy_tables() -> List[str]:
    archs = ("lstm", "gru", "rnn")
    header = ["encoder"] + [f"$\\lambda$={l:.3f}" for l in EXP.ENTROPY_LAMBDAS]
    out = []
    for title, fn, pct in (
        ("**Sharpe ratio vs. requested diversification**",
         lambda a, l: cell("ent", EXP.entropy_experiment_name(l, a), "sharpe"), False),
        ("**Achieved portfolio entropy (nats)**",
         lambda a, l: ent("ent", EXP.entropy_experiment_name(l, a)), False),
        ("**Annualised return (%)**",
         lambda a, l: cell("ent", EXP.entropy_experiment_name(l, a), "ann_return", pct=True), True),
    ):
        rows = [[FIG.ARCH_LABEL[a]] + [fn(a, l) for l in EXP.ENTROPY_LAMBDAS] for a in archs]
        out.append(title + "\n\n" + table(header, rows))
    return out


# --------------------------------------------------------------------------- #
# §9.11 tree baselines (reference figures 13-14)
# --------------------------------------------------------------------------- #
def tree_table() -> str:
    header = ["target", "forecaster", "ann. return %", "ann. vol %", "Sharpe",
              "max DD %", "tracking error %"]
    rows: List[List[str]] = []
    for mu in EXP.TREE_MU_TARGETS:
        candidates: List[Tuple[str, str, str]] = []  # (key, experiment, label)
        for name in EXP.TREE_BASELINES:
            if float(EXP.spec_for(name).overrides["opt"]["mu_target"]) == float(mu):
                label = "AdaBoost+MCVaR" if name.startswith("ada") else "XGBoost+MCVaR"
                candidates.append(("tree", name, label))
        for arch in ("lstm", "gru", "rnn"):
            candidates.append(("mu", EXP.mu_experiment_name(mu, arch), FIG.ARCH_LABEL[arch]))
        for key, name, label in candidates:
            rows.append([
                f"{mu:.3f}", label,
                cell(key, name, "ann_return", pct=True),
                cell(key, name, "ann_vol", pct=True),
                cell(key, name, "sharpe"),
                cell(key, name, "max_drawdown", pct=True),
                cell(key, name, "tracking_error", pct=True),
            ])
    return table(header, rows)


# --------------------------------------------------------------------------- #
# §9.12 cross-market runs (reference figures 11-12)
# --------------------------------------------------------------------------- #
def market_table() -> str:
    header = ["universe", "strategy", "ann. return %", "ann. vol %", "Sharpe",
              "max DD %", "tracking error %"]
    plan = [
        ("CSI 300", "main", ("ew", "meancvar_hist", "full")),
        ("CSI 500", "csi500", ("ew", "meancvar_hist", "mu020_lstm", "full")),
        ("SSE 50", "sse50", ("ew", "meancvar_hist", "mu020_lstm", "full")),
    ]
    rows: List[List[str]] = []
    for market, key, experiments in plan:
        for name in experiments:
            rows.append([
                market, name,
                cell(key, name, "ann_return", pct=True),
                cell(key, name, "ann_vol", pct=True),
                cell(key, name, "sharpe"),
                cell(key, name, "max_drawdown", pct=True),
                cell(key, name, "tracking_error", pct=True),
            ])
    return table(header, rows)


# --------------------------------------------------------------------------- #
# §9.12b the five encoders on the two other universes (figures 11-12)
# --------------------------------------------------------------------------- #
def market_arch_table() -> str:
    """Five encoders at tau = 0.020 on CSI 500 and SSE 50, one row per cell."""
    tau = min(t for t in EXP.MU_TARGETS if t >= 0.020)
    header = ["universe", "encoder", "ann. return %", "ann. vol %", "Sharpe",
              "max DD %", "tracking error %"]
    rows: List[List[str]] = []
    for market, key in (("CSI 500", "csi500_arch"), ("SSE 50", "sse50_arch")):
        for arch in FIG.ARCH_ORDER:
            name = EXP.mu_experiment_name(tau, arch)
            rows.append([
                market, FIG.ARCH_LABEL[arch],
                cell(key, name, "ann_return", pct=True),
                cell(key, name, "ann_vol", pct=True),
                cell(key, name, "sharpe"),
                cell(key, name, "max_drawdown", pct=True),
                cell(key, name, "tracking_error", pct=True),
            ])
    return table(header, rows)


# --------------------------------------------------------------------------- #
# progress check
# --------------------------------------------------------------------------- #
def groups() -> List[Tuple[str, str, List[str]]]:
    """``(label, run key, experiments)`` for every group the figures need."""
    return [
        ("mu_sweep (fig 2-5, 13-14)", "mu", [
            EXP.mu_experiment_name(mu, arch)
            for mu in EXP.MU_TARGETS for arch in EXP.SWEEP_ARCHS
        ]),
        ("entropy_sweep (fig 8-9)", "ent", [
            EXP.entropy_experiment_name(lam, arch)
            for lam in EXP.ENTROPY_LAMBDAS for arch in ("lstm", "gru", "rnn")
        ]),
        ("tree_baselines (fig 13-14)", "tree", list(EXP.TREE_BASELINES)),
        ("csi500_reference (fig 11)", "csi500",
         ["ew", "meancvar_hist", "mu020_lstm", "full"]),
        ("sse50_reference (fig 12)", "sse50",
         ["ew", "meancvar_hist", "mu020_lstm", "full"]),
        ("csi500_arch_compare (fig 11)", "csi500_arch", [
            EXP.mu_experiment_name(0.020, arch) for arch in FIG.ARCH_ORDER
        ]),
        ("sse50_arch_compare (fig 12)", "sse50_arch", [
            EXP.mu_experiment_name(0.020, arch) for arch in FIG.ARCH_ORDER
        ]),
    ]


def progress() -> str:
    rows = []
    for label, key, experiments in groups():
        missing = [e for e in experiments if not complete(key, e)]
        done = len(experiments) - len(missing)
        rows.append([label, f"{done}/{len(experiments)}",
                     ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")])
    return table(["group", "experiments", "still incomplete"], rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="also write the tables here")
    ap.add_argument("--check", action="store_true",
                    help="only report the per-group progress")
    args = ap.parse_args()

    parts = ["# Sweep tables (generated by scripts/10_summary_tables.py)\n",
             "## Progress\n", progress()]
    if not args.check:
        parts += [
            "\n## §9.9 target-return sweep\n",
            "\n\n".join(target_tables()),
            "\n## §9.10 entropy sweep\n",
            "\n\n".join(entropy_tables()),
            "\n## §9.11 AdaBoost / XGBoost baselines\n",
            tree_table(),
            "\n## §9.12 other universes\n",
            market_table(),
            "\n## §9.12b five encoders on the two other universes\n",
            market_arch_table(),
        ]
    text = "\n".join(parts) + "\n"
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[written] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
