"""Run the experiment grid (one process per (experiment, fold) pair).

    # everything
    python scripts/02_run_experiments.py --config configs/csi300.yaml --workers 4

    # quick sanity check of the whole pipeline
    python scripts/02_run_experiments.py --config configs/csi300.yaml --smoke

    # a subset
    python scripts/02_run_experiments.py --experiments full,ew --folds f1_2018_2019

Each job writes ``results/<run_name>/<experiment>/<fold>/`` with the fitted
configuration, the metrics, the daily returns, the target weights, the training
history and the model state dict, so any single number can be traced back to the
exact artefact that produced it.  ``results/<run_name>/summary.json`` collects
the headline metrics of every job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio.config import Config  # noqa: E402
from e2e_portfolio.experiments import MAIN_GRID, REGISTRY, run_job  # noqa: E402


def _worker_init(threads: int) -> None:
    try:
        import torch

        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(threads))))


def _run_one(payload) -> Dict:
    """Executed inside a worker process."""
    cfg_dict, experiment, fold_dict, out_root, threads = payload
    _worker_init(threads)
    from e2e_portfolio.config import Config as _C
    from e2e_portfolio.config import FoldConfig as _F

    cfg = _C.from_dict(cfg_dict)
    fold = _F(**fold_dict)
    t0 = time.time()
    try:
        metrics = run_job(cfg, experiment, fold, out_root=Path(out_root), verbose=False)
        return {"ok": True, "experiment": experiment, "fold": fold.name,
                "metrics": metrics, "seconds": time.time() - t0}
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "experiment": experiment,
            "fold": fold.name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "seconds": time.time() - t0,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/csi300.yaml")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--experiments", default=None, help="comma separated subset")
    ap.add_argument("--folds", default=None, help="comma separated subset")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads-per-worker", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: fold f1, epochs=2, first train month only")
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    args = ap.parse_args()

    cfg = Config.from_yaml(ROOT / args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
        cfg.train.patience = args.epochs
        cfg.train.min_epochs = min(cfg.train.min_epochs, args.epochs)

    experiments = [e.strip() for e in (args.experiments or "").split(",") if e.strip()]
    experiments = experiments or cfg.experiments or MAIN_GRID
    unknown = [e for e in experiments if e not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")

    folds = cfg.folds
    if args.folds:
        wanted = {f.strip() for f in args.folds.split(",") if f.strip()}
        folds = [f for f in folds if f.name in wanted]
        if not folds:
            raise SystemExit(f"no fold matched {wanted}")

    if args.smoke:
        import copy as _copy

        smoke_fold = _copy.deepcopy(cfg.folds[0])
        smoke_fold.name = "smoke"
        smoke_fold.train_start = "2010-01-01"
        smoke_fold.train_end = "2011-06-30"
        smoke_fold.val_start = "2011-07-01"
        smoke_fold.val_end = "2011-12-31"
        smoke_fold.test_start = "2012-01-01"
        smoke_fold.test_end = "2012-12-31"
        folds = [smoke_fold]
        cfg.train.epochs = 2
        cfg.train.patience = 2
        cfg.train.min_epochs = 1
        cfg.results_dir = args.results_root or "results_smoke"

    run_name = cfg.run_name or f"{cfg.data.index}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = ROOT / (args.results_root or cfg.results_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.run_name = run_name
    cfg.to_yaml(out_root / "config.yaml")

    jobs = [(e, f) for e in experiments for f in folds]
    cfg_dict = cfg.to_dict()
    payloads = [
        (cfg_dict, e, f.__dict__, str(out_root), args.threads_per_worker) for e, f in jobs
    ]
    total = len(payloads)
    print(f"[run ] {total} jobs -> {out_root}", flush=True)
    print(f"[run ] experiments: {', '.join(experiments)}", flush=True)
    print(f"[run ] folds: {', '.join(f.name for f in folds)} | workers={args.workers}", flush=True)

    results: List[Dict] = []
    t_start = time.time()
    if args.workers <= 1:
        for i, pl in enumerate(payloads, 1):
            r = _run_one(pl)
            results.append(r)
            _report_progress(i, total, r, t_start)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                 initargs=(args.threads_per_worker,)) as ex:
            futs = {ex.submit(_run_one, pl): pl for pl in payloads}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                _report_progress(i, total, r, t_start)

    summary = {
        "run_name": run_name,
        "index": cfg.data.index,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": time.time() - t_start,
        "n_jobs": total,
        "n_ok": sum(1 for r in results if r["ok"]),
        "n_failed": sum(1 for r in results if not r["ok"]),
        "results": [r for r in results if r["ok"]],
        "failures": [r for r in results if not r["ok"]],
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {summary['n_ok']}/{total} ok in {summary['seconds'] / 60:.1f} min", flush=True)
    for r in summary["failures"]:
        print(f"[fail] {r['experiment']} / {r['fold']}: {r['error']}", flush=True)
        print(r.get("traceback", ""), flush=True)
    return 0 if summary["n_failed"] == 0 else 1


def _report_progress(i: int, total: int, r: Dict, t_start: float) -> None:
    el = time.time() - t_start
    eta = el / i * (total - i)
    tag = "ok  " if r["ok"] else "FAIL"
    extra = ""
    if r["ok"]:
        m = r["metrics"]
        extra = (f"sharpe {m.get('sharpe', float('nan')):+.2f} "
                 f"ann {m.get('ann_return', float('nan')):+.2%} "
                 f"mdd {m.get('max_drawdown', float('nan')):.2%}")
    else:
        extra = r["error"]
    print(
        f"[{i:>3}/{total}] {tag} {r['experiment']:<18} {r['fold']:<15} "
        f"{r['seconds'] / 60:5.1f}min  eta {eta / 60:5.1f}min  {extra}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
