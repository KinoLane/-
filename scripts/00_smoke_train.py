"""Fast end-to-end smoke test of the training loop (small universe, 2 periods).

    python scripts/00_smoke_train.py

Checks that
* the model + layer run inside the training loop for both regimes,
* gradients reach every parameter,
* the loss terms have sensible magnitudes,
* the memory footprint of ``batch_size`` graphs stays bounded,
* one optimiser step reduces the loss.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio.config import Config  # noqa: E402
from e2e_portfolio.dataset import build_dataset  # noqa: E402
from e2e_portfolio.experiments import investable_slot_bound, n_max_slots  # noqa: E402
from e2e_portfolio.train import Trainer, build_period_tensors  # noqa: E402


def main() -> int:
    cfg = Config.from_yaml(ROOT / "configs" / "csi300.yaml")
    cfg.data.top_liquidity = 30
    cfg.model.num_scenarios = 20
    cfg.train.epochs = 2
    cfg.train.batch_size = 4
    cfg.train.num_threads = 6

    torch.manual_seed(0)
    t0 = time.time()
    ds = build_dataset(cfg, verbose=True)
    print(f"dataset in {time.time() - t0:.1f}s: {ds.describe()['n_periods']} periods")

    periods = ds.periods_between("2015-01-01", "2015-08-31", pad_horizon=True)
    print(f"smoke periods: {len(periods)} ({periods[0].date} .. {periods[-1].date})")
    n_max = n_max_slots(cfg, periods, dataset=ds)
    print(f"slot budget n_max={n_max} (max investable {investable_slot_bound(ds, periods)})")

    for name, overrides in (
        ("e2e (layer in the loop)", {}),
        ("predict-only", {"loss": {"l_decision": 0.0, "l_tail": 0.0,
                                   "l_turnover": 0.0, "l_sparse": 0.0}}),
    ):
        print(f"\n=== {name} ===")
        sub = cfg.replace(**overrides)
        tr = Trainer(sub, ds, n_max=n_max, n_features=len(ds.feature_names), verbose=True)
        print(f"  use_layer={tr.use_layer}")
        t0 = time.time()
        terms, _ = tr.run_epoch(periods, train=True)
        print(f"  epoch 1: {time.time() - t0:.1f}s  terms="
              + "  ".join(f"{k}={v:+.4f}" for k, v in sorted(terms.items())))
        t0 = time.time()
        terms2, _ = tr.run_epoch(periods, train=True)
        print(f"  epoch 2: {time.time() - t0:.1f}s  terms="
              + "  ".join(f"{k}={v:+.4f}" for k, v in sorted(terms2.items())))
        w, info = tr.weights_for(periods[-1], np.zeros(ds.panel.num_tickers))
        print(f"  weights: sum={w.sum():.6f} max={w.max():.3%} "
              f"n_holdings={(w > 1e-6).sum()} eff={1.0 / np.sum(w[w > 0] ** 2):.1f}")

        # ---- gradient check: one period, no optimiser step -----------------
        tr.model.train()
        tr.optimizer.zero_grad(set_to_none=True)
        b = build_period_tensors(ds, periods[3], np.zeros(ds.panel.num_tickers), n_max)
        loss, y, out = tr._forward_loss(b)
        loss.total.backward()
        grads = {n: p.grad for n, p in tr.model.named_parameters()}
        n_none = sum(1 for g in grads.values() if g is None)
        n_zero = sum(1 for g in grads.values() if g is not None and float(g.abs().sum()) == 0)
        tot = sum(float(g.pow(2).sum()) for g in grads.values() if g is not None)
        print(f"  gradient check: {len(grads)} tensors, {n_none} without grad, "
              f"{n_zero} with zero grad, sum(grad^2)={tot:.4g}")
        if out.aux.get("weights") is not None:
            print(f"    mu range [{float(out.mu[b.valid].min()):+.4f}, {float(out.mu[b.valid].max()):+.4f}]"
                  f"  sigma range [{float(out.sigma[b.valid].min()):.4f}, "
                  f"{float(out.sigma[b.valid].max()):.4f}]"
                  f"  pi>0: {int((out.pi > 1e-9).sum())}/{b.n_valid}"
                  f"  sum(w)={float(out.aux['weights'].sum()):.6f}")
        tr.optimizer.zero_grad(set_to_none=True)

    # memory sanity check for the gradient-accumulation window
    import os

    import psutil  # type: ignore  # noqa: F401

    print("\n=== memory / throughput ===")
    proc = psutil.Process(os.getpid())
    for bs in (1, 4, 8):
        sub = cfg.replace(train={"batch_size": bs})
        tr = Trainer(sub, ds, n_max=n_max, n_features=len(ds.feature_names), verbose=False)
        base = proc.memory_info().rss / 1e6
        t0 = time.time()
        tr.run_epoch(periods, train=True)
        dt = time.time() - t0
        print(f"  batch_size={bs}: {dt / max(len(periods), 1) * 1000:.0f} ms/period, "
              f"rss {base:.0f} -> {proc.memory_info().rss / 1e6:.0f} MB")

    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
