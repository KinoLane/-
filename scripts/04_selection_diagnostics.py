"""Diagnose what each selection layer really feeds into the optimisation layer.

    # newest completed run (a run is complete when results/<run>/summary.json exists)
    python scripts/04_selection_diagnostics.py --latest

    # a specific run, a subset of experiments
    python scripts/04_selection_diagnostics.py --run results/csi300_20260910_235309 \
        --experiments lstm_topk,lstm_softmax,lstm_sparsemax

Motivation
----------
``lstm_softmax`` and ``lstm_sparsemax`` produced almost identical out-of-sample
metrics in this study.  This script makes that observation checkable instead of
anecdotal: it reloads every saved ``model.pt`` of the selection experiments and
replays the *test* periods through the encoder, recording what the selection
layer hands to the Mean-CVaR layer.

Per period it reports

    n_valid       investable names of the period (clean book, no carried holdings)
    score_std     std of the selection logits over the valid names
    score_spread  mean - min of those logits
    spar_thresh   1 / n_valid -- the simplex projection keeps *full* support while
                  score_spread < spar_thresh, which is exactly when sparsemax
                  degenerates into softmax
    pi_support    names with pi > 1e-6, i.e. the support the selector really uses
    pi_pr         1 / sum(pi^2): the effective number of selected names
    cap_budget    sum_i cap_i *before* the widening, i.e. what the selection can
                  fund on its own; ``< 1`` means the deficit had to be widened
    deficit       max(0, 1 - cap_budget): the budget handed to the next-best names
    cap_sum       sum_i cap_i after the widening (>= 1 keeps the layer feasible)
    cap_at_max    names whose cap already equals y_max (a binding position limit)
    book_n        names with realised weight > 1e-3 in the backtest weights.csv
    book_pr       effective number of holdings of that book

The replay is deliberately done with ``prev = 0`` (a clean rebalance): it isolates
the selection stage from the holdings that were carried over because they could
not be sold.  ``book_n`` / ``book_pr`` are read from the artefacts written by the
backtester, so the difference between them and ``pi_support`` is the effect of
the carried holdings plus the layer itself.

Writes ``<run>/<exp>/<fold>/selection_diag.csv`` (per period) and
``<run>/report/selection_diagnostics.csv`` (one row per experiment/fold; rows
written by earlier invocations for other experiments are kept).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio.config import Config  # noqa: E402
from e2e_portfolio.dataset import build_dataset  # noqa: E402
from e2e_portfolio.experiments import (  # noqa: E402
    REGISTRY,
    experiment_config,
    n_max_slots,
    spec_for,
)
from e2e_portfolio.metrics import HOLDING_EPS  # noqa: E402
from e2e_portfolio.models import E2EPortfolioModel  # noqa: E402
from e2e_portfolio.selection import participation_ratio, raw_caps  # noqa: E402
from e2e_portfolio.train import build_period_tensors  # noqa: E402

#: experiments whose selection stage is worth dissecting
DEFAULT_EXPERIMENTS: Tuple[str, ...] = (
    "lstm_topk",
    "lstm_softmax",
    "lstm_sparsemax",
    "full",
)

#: a weight is treated as held above this size; the shared constant keeps the
#: diagnostics and ``metrics.json["holdings_mean"]`` on the same definition
WEIGHT_EPS = HOLDING_EPS

#: pi is treated as "selected" above this size
PI_EPS = 1e-6


def latest_run(results_root: Path) -> Path:
    cands = [d for d in results_root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    if not cands:
        raise SystemExit(f"no completed run found under {results_root}")
    return max(cands, key=lambda d: (d / "summary.json").stat().st_mtime)


def encoder_hash(state: Dict[str, object]) -> str:
    """SHA-256 of the encoder parameters (first 16 hex chars).

    The predict-then-optimise variants share one loss (``l_pred = 1``, everything
    else 0), so their encoders must come out bit-identical: the digest is the
    evidence that the reported differences are caused by the selection layer
    alone and not by three different training runs.
    """
    h = hashlib.sha256()
    for key in sorted(k for k in state if str(k).startswith("encoder")):
        h.update(np.ascontiguousarray(state[key].detach().numpy()).tobytes())
    return h.hexdigest()[:16]


def load_model(
    fold_dir: Path, cfg: Config, n_features: int
) -> Tuple[E2EPortfolioModel, str, str]:
    """Rebuild the architecture from the fold's own config and load ``model.pt``."""
    import torch

    model = E2EPortfolioModel(n_features, cfg.model, cfg.opt, opt_layer=None)
    state = torch.load(fold_dir / "model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    note = ""
    if missing or unexpected:
        note = f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
    model.eval()
    return model, note, encoder_hash(state)


def book_stats(weights_csv: Path, date: str) -> Tuple[float, float]:
    """``(n_held, effective_n)`` of the realised book on ``date`` (NaN when absent)."""
    if not weights_csv.exists():
        return float("nan"), float("nan")
    w = pd.read_csv(weights_csv, index_col=0)
    key = str(date)
    if key not in w.index:
        return float("nan"), float("nan")
    row = w.loc[key].to_numpy(dtype=np.float64)
    row = np.clip(row, 0.0, None)
    n_held = float((row > WEIGHT_EPS).sum())
    pr = float(participation_ratio(torch_from(row)).item()) if row.sum() > 0 else float("nan")
    return n_held, pr


def torch_from(arr: np.ndarray):
    import torch

    return torch.as_tensor(arr, dtype=torch.float64)


def diagnose(
    run_dir: Path,
    experiments: Sequence[str],
    folds: Optional[Sequence[str]] = None,
    threads: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    import torch

    torch.set_num_threads(max(1, int(threads)))

    base_cfg = Config.from_yaml(run_dir / "config.yaml")
    ds = build_dataset(base_cfg, verbose=False)
    n_features = len(ds.feature_names)
    fold_by_name = {f.name: f for f in base_cfg.folds}

    rows: List[Dict[str, object]] = []
    for exp in experiments:
        spec = spec_for(exp)
        if spec.kind != "nn":
            if verbose:
                print(f"[diag] skip {exp}: kind={spec.kind} has no selection layer")
            continue
        cfg = experiment_config(base_cfg, exp)
        for fold in base_cfg.folds:
            if folds and fold.name not in folds:
                continue
            fold_dir = run_dir / exp / fold.name
            if not (fold_dir / "model.pt").exists():
                if verbose:
                    print(f"[diag] skip {exp}/{fold.name}: no model.pt")
                continue
            model, note, digest = load_model(fold_dir, cfg, n_features)
            if note and verbose:
                print(f"[diag] {exp}/{fold.name}: state_dict note -> {note}")

            train_p = ds.periods_between(fold.train_start, fold.train_end, pad_horizon=False)
            val_p = ds.periods_between(fold.val_start, fold.val_end, pad_horizon=False)
            test_p = ds.periods_between(fold.test_start, fold.test_end, pad_horizon=True)
            n_max = n_max_slots(cfg, [*train_p, *val_p, *test_p], dataset=ds)

            zeros = np.zeros(ds.panel.num_tickers, dtype=np.float64)
            per_period: List[Dict[str, object]] = []
            for p in test_p:
                batch = build_period_tensors(ds, p, zeros, n_max)
                with torch.no_grad():
                    out = model(
                        batch.x,
                        batch.mask,
                        batch.valid,
                        y_prev=batch.prev,
                        y_floor=batch.floor,
                        run_layer=False,
                    )
                valid = batch.valid.cpu().numpy()
                n_valid = int(valid.sum())
                score = out.score.cpu().numpy()[valid].astype(np.float64)
                pi = out.pi.cpu().numpy()[valid].astype(np.float64)
                cap = out.cap.cpu().numpy()[valid].astype(np.float64)
                y_eff = float(max(cfg.opt.y_max, 1.0 / max(n_valid, 1)))
                book_n, book_pr = book_stats(fold_dir / "weights.csv", p.date)
                # budget the selection can fund on its own, before the widening
                raw = raw_caps(out.pi, cfg.opt.y_max, batch.valid).cpu().numpy()
                cap_budget = float(raw[valid].sum())

                per_period.append(
                    {
                        "experiment": exp,
                        "fold": fold.name,
                        "date": str(p.date),
                        "selection": cfg.model.selection,
                        "n_valid": n_valid,
                        "n_max": int(n_max),
                        "score_std": float(score.std()),
                        "score_spread": float(score.mean() - score.min()),
                        "spar_thresh": 1.0 / max(n_valid, 1),
                        "pi_support": int((pi > PI_EPS).sum()),
                        "pi_pr": float(participation_ratio(torch_from(pi)).item()),
                        "cap_budget": cap_budget,
                        "deficit": max(0.0, 1.0 - cap_budget),
                        "cap_sum": float(cap.sum()),
                        "cap_at_max": int((cap > y_eff - 1e-9).sum()),
                        "book_n": book_n,
                        "book_pr": book_pr,
                    }
                )

            df = pd.DataFrame(per_period)
            df.to_csv(fold_dir / "selection_diag.csv", index=False, encoding="utf-8")
            summary = {
                "experiment": exp,
                "fold": fold.name,
                "selection": cfg.model.selection,
                "encoder_sha256": digest,
                "n_periods": len(df),
                "n_valid": float(df["n_valid"].mean()),
                "score_std": float(df["score_std"].mean()),
                "score_spread": float(df["score_spread"].mean()),
                "spar_thresh": float(df["spar_thresh"].mean()),
                "spread_below_thresh": float((df["score_spread"] < df["spar_thresh"]).mean()),
                "pi_support": float(df["pi_support"].mean()),
                "pi_pr": float(df["pi_pr"].mean()),
                "cap_budget": float(df["cap_budget"].mean()),
                "fill_periods": float((df["deficit"] > 0).mean()),
                "cap_sum": float(df["cap_sum"].mean()),
                "book_n": float(df["book_n"].mean()),
                "book_pr": float(df["book_pr"].mean()),
            }
            rows.append(summary)
            if verbose:
                print(
                    f"[diag] {exp:16s} {fold.name:14s} enc={digest} n={summary['n_valid']:6.1f} "
                    f"spread={summary['score_spread']:.2e} <1/n={summary['spar_thresh']:.2e} "
                    f"({summary['spread_below_thresh'] * 100:5.1f}% of periods) "
                    f"pi_support={summary['pi_support']:6.1f} pi_pr={summary['pi_pr']:6.1f} "
                    f"cap_budget={summary['cap_budget']:6.3f} "
                    f"fill={summary['fill_periods'] * 100:4.1f}% "
                    f"book={summary['book_n']:5.1f} ({summary['book_pr']:5.1f})"
                )

    agg = pd.DataFrame(rows)
    if not agg.empty:
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / "selection_diagnostics.csv"
        # the script is usually invoked on a subset of the experiments, so the
        # already diagnosed rows are kept: invoking it twice must not delete
        # evidence produced by the first call
        done = [(r.experiment, r.fold) for r in agg.itertuples()]
        keep = pd.DataFrame()
        if out.exists():
            old = pd.read_csv(out)
            keep = old[[(e, f) not in done for e, f in zip(old["experiment"], old["fold"])]]
        merged = pd.concat([keep, agg], ignore_index=True)
        merged = merged.sort_values(["experiment", "fold"], kind="stable").reset_index(drop=True)
        merged.to_csv(out, index=False, encoding="utf-8")
    return agg


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:  # a legacy console code page cannot encode every typographic character
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # pragma: no cover
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: --latest)")
    ap.add_argument("--latest", action="store_true", help="use the newest completed run under results/")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--experiments", type=str, default=",".join(DEFAULT_EXPERIMENTS))
    ap.add_argument("--folds", type=str, default="", help="comma separated fold names (default: all)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)

    run_dir = args.run
    if run_dir is None:
        run_dir = latest_run(args.results_dir)
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "config.yaml").exists():
        raise SystemExit(f"{run_dir} is not a run directory (no config.yaml)")

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    unknown = [e for e in experiments if e not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]

    print(f"[diag] run {run_dir}")
    agg = diagnose(run_dir, experiments, folds=folds or None, threads=args.threads)
    if agg.empty:
        print("[diag] nothing to diagnose (no model.pt found)")
        return 1

    print()
    print("每个 (实验, 折) 的选择层诊断（测试期均值）")
    show = agg[
        [
            "experiment",
            "fold",
            "selection",
            "encoder_sha256",
            "n_periods",
            "n_valid",
            "score_spread",
            "spar_thresh",
            "spread_below_thresh",
            "pi_support",
            "pi_pr",
            "cap_budget",
            "fill_periods",
            "book_n",
            "book_pr",
        ]
    ].copy()
    for col in ("n_valid", "pi_support", "pi_pr", "book_n", "book_pr"):
        show[col] = show[col].map(lambda v: f"{v:6.1f}")
    for col in ("score_spread", "spar_thresh"):
        show[col] = show[col].map(lambda v: f"{v:8.2e}")
    show["cap_budget"] = show["cap_budget"].map(lambda v: f"{v:6.3f}")
    show["spread_below_thresh"] = show["spread_below_thresh"].map(lambda v: f"{v * 100:5.1f}%")
    show["fill_periods"] = show["fill_periods"].map(lambda v: f"{v * 100:5.1f}%")
    print(show.to_string(index=False))
    print()
    print(f"[diag] wrote {run_dir / 'report' / 'selection_diagnostics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
