"""Inference-time control for the deficit-fill rule of the selection layer.

The fill rule lives *inside* the differentiable model, so changing it also changes
the training trajectory: a full re-run cannot separate "the fill changed the policy
at the trigger periods" from "the fill changed what the network learned".  This
script isolates the first effect by re-evaluating a **saved** model with both fill
rules and identical weights.

It first checks itself: re-evaluating the stored model with the *old* (spread) rule
must reproduce the reference ``metrics.json`` of that job bit-for-bit.  Only then is
the comparison against the new (waterfall) rule meaningful.

Usage
-----
    python scripts/05_fill_control.py --run results/<reference_run> \\
        --experiment full_nocost --fold f2_2020_2021
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio import selection as sel_mod  # noqa: E402
from e2e_portfolio.backtest import simulate  # noqa: E402
from e2e_portfolio.config import Config  # noqa: E402
from e2e_portfolio.dataset import build_dataset  # noqa: E402
from e2e_portfolio.experiments import (  # noqa: E402
    experiment_config,
    make_nn_provider,
    n_max_slots,
)
from e2e_portfolio.metrics import (  # noqa: E402
    annualised_return,
    annualised_vol,
    conditional_value_at_risk,
    count_holdings,
    information_ratio,
    max_drawdown,
    sharpe,
)
from e2e_portfolio.train import Trainer  # noqa: E402

_WATERFALL = sel_mod.selection_caps


def spread_caps(pi, y_max, valid=None, score=None):
    """The buggy fill: spread the deficit over every name's headroom.

    Kept here (not in the library) as the *reference implementation of the bug*, so
    the regression control stays runnable without re-introducing the defect.
    """
    caps = sel_mod.raw_caps(pi, y_max, valid)
    free = torch.ones_like(pi) if valid is None else valid.to(pi.dtype)
    n_valid = free.sum(dim=-1, keepdim=True).clamp(min=1.0)
    y_eff = torch.clamp(n_valid.reciprocal(), min=float(y_max))
    room = (y_eff - caps).clamp(min=0.0) * free
    deficit = (1.0 - caps.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    if not bool((deficit > 0).any()):
        return caps
    share = room / room.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return caps + deficit * share


def evaluate(cfg, fold, ds, state_dict, n_max, fill):
    sel_mod.selection_caps = fill
    trainer = Trainer(cfg, ds, n_max=n_max, n_features=len(ds.feature_names), verbose=False)
    trainer.model.load_state_dict(state_dict)
    test = ds.periods_between(fold.test_start, fold.test_end, pad_horizon=True)
    t0 = time.time()
    bt = simulate(
        ds, test, make_nn_provider(trainer), cost_bps=cfg.backtest.cost_bps,
        y_max=cfg.opt.y_max, keep_records=True,
    )
    w = np.asarray([np.asarray(x, dtype=np.float64) for x in bt.target_weights])
    bench = np.asarray(bt.bench_ret, dtype=np.float64)
    ret = np.asarray(bt.ret_net, dtype=np.float64)
    hold = count_holdings(w)
    out = {
        "ann_return": annualised_return(ret),
        "ann_vol": annualised_vol(ret),
        "sharpe": sharpe(ret),
        "max_drawdown": max_drawdown(ret),
        "cvar_95": conditional_value_at_risk(ret),
        "holdings_mean": float(hold.mean()),
        "eff_holdings_mean": float(
            np.mean([1.0 / float((x[x > 0] ** 2).sum()) if (x > 0).any() else 0.0 for x in w])
        ),
        "turnover_ann": float(
            np.mean(bt.turnover) * (len(bt.turnover) / max(ret.size / 252.0, 1 / 252.0))
        ),
        "info_ratio": information_ratio(ret, bench),
    }
    out["_seconds"] = time.time() - t0
    return out, w, ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="reference run directory")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", default=None, help="where to write the comparison json")
    args = ap.parse_args()

    ref_dir = ROOT / args.run / args.experiment / args.fold
    ref_metrics = json.loads((ref_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = Config.from_yaml(ref_dir / "config.yaml")
    fold = next(f for f in cfg.folds if f.name == args.fold)
    ds = build_dataset(cfg, verbose=False)

    train = ds.periods_between(fold.train_start, fold.train_end, pad_horizon=False)
    val = ds.periods_between(fold.val_start, fold.val_end, pad_horizon=False)
    test = ds.periods_between(fold.test_start, fold.test_end, pad_horizon=True)
    n_max = n_max_slots(cfg, [*train, *val, *test], dataset=ds)

    state = torch.load(ref_dir / "model.pt", map_location="cpu")
    print(f"{args.experiment} / {args.fold}: n_max={n_max}, reference model loaded")
    print(f"  reference metrics.json: ann={ref_metrics['ann_return']:.6f} "
          f"sharpe={ref_metrics['sharpe']:.6f} holdings={ref_metrics['holdings_mean']:.4f}")

    rows = {}
    for name, fill in (("spread(旧)", spread_caps), ("waterfall(新)", _WATERFALL)):
        m, w, ret = evaluate(cfg, fold, ds, state, n_max, fill)
        rows[name] = m
        print(f"  [{name}] ann={m['ann_return']:.6f} sharpe={m['sharpe']:.6f} "
              f"vol={m['ann_vol']:.6f} mdd={m['max_drawdown']:.6f} "
              f"holdings={m['holdings_mean']:.4f} ({m['_seconds']:.0f}s)")
        if name.startswith("spread"):
            dev = {k: abs(m[k] - ref_metrics[k]) for k in m if k in ref_metrics and not k.startswith("_")}
            worst = max(dev.items(), key=lambda kv: kv[1])
            print(f"  control check: reproduce reference metrics.json -> worst |delta| "
                  f"{worst[1]:.3e} on '{worst[0]}' "
                  f"({'OK' if worst[1] < 1e-9 else 'MISMATCH'})")

    sel_mod.selection_caps = _WATERFALL
    out = {
        "run": args.run, "experiment": args.experiment, "fold": args.fold,
        "reference_metrics_ann": ref_metrics["ann_return"],
        "reference_metrics_sharpe": ref_metrics["sharpe"],
        "reference_metrics_holdings": ref_metrics["holdings_mean"],
        "spread": rows["spread(旧)"], "waterfall": rows["waterfall(新)"],
    }
    dest = Path(args.out) if args.out else ROOT / "results" / "fill_control" / (
        f"{args.experiment}__{args.fold}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  written -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
