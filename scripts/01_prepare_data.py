"""Build / validate the feature panel and cache it, then print a summary.

    python scripts/01_prepare_data.py --config configs/csi300.yaml

Caches:
    results/_cache/panel_<index>.npz      raw tensors (OHLCV + masks + index)
    results/_cache/features_<index>_<hash>.npz   standardised feature tensor

The feature cache is keyed by the feature configuration, so changing ``clip`` or
a window invalidates it automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_portfolio.config import Config  # noqa: E402
from e2e_portfolio.data import load_panel, validate_panel  # noqa: E402
from e2e_portfolio.dataset import build_dataset  # noqa: E402
from e2e_portfolio.features import build_features  # noqa: E402


def feature_cache_key(cfg: Config) -> str:
    payload = json.dumps(
        {"features": cfg.features.__dict__, "exclude_st": cfg.data.exclude_st},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/csi300.yaml")
    ap.add_argument("--force", action="store_true", help="ignore the feature cache")
    ap.add_argument("--validate", action="store_true", help="run the data integrity checks")
    args = ap.parse_args()

    cfg = Config.from_yaml(ROOT / args.config)
    cache = ROOT / cfg.results_dir / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    panel = load_panel(cfg.data, verbose=True)
    print(f"[data] panel loaded in {time.time() - t0:.1f}s")
    print(
        f"[data] {panel.num_dates} dates x {panel.num_tickers} tickers, "
        f"{str(panel.dates[0])} .. {str(panel.dates[-1])}"
    )
    n_obs = int(panel.observed.sum())
    print(
        f"[data] observed cells: {n_obs:,} ({100.0 * n_obs / panel.observed.size:.1f}%), "
        f"ST: {int(panel.is_st.sum()):,}, "
        f"limit-up locked: {int(panel.limit_locked_up.sum()):,}, "
        f"limit-down locked: {int(panel.limit_locked_down.sum()):,}"
    )

    if args.validate:
        rep = validate_panel(panel, cfg.data, verbose=True)
        (cache / f"validate_{cfg.data.index}.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    key = feature_cache_key(cfg)
    fpath = cache / f"features_{cfg.data.index}_{key}.npz"
    t0 = time.time()
    if fpath.exists() and not args.force:
        z = np.load(fpath, allow_pickle=True)
        feats, observed, names = z["features"], z["observed"], list(z["names"])
        print(f"[feat] loaded cached features from {fpath.name} ({feats.shape})")
    else:
        feats, observed, names = build_features(
            panel, cfg.features, exclude_st=cfg.data.exclude_st, verbose=True
        )
        np.savez_compressed(fpath, features=feats, observed=observed, names=np.array(names))
        print(f"[feat] built features in {time.time() - t0:.1f}s -> {fpath.name} {feats.shape}")
    print(f"[feat] {len(names)} features: {', '.join(names)}")

    ds = build_dataset(cfg, verbose=False)
    info = ds.describe()
    print("[set ] " + json.dumps(info, ensure_ascii=False))
    info["folds"] = []
    for f in cfg.folds:
        info["folds"].append(
            {
                "name": f.name,
                "train": f"{f.train_start}..{f.train_end}",
                "train_periods": len(ds.periods_between(f.train_start, f.train_end, pad_horizon=False)),
                "val": f"{f.val_start}..{f.val_end}",
                "val_periods": len(ds.periods_between(f.val_start, f.val_end, pad_horizon=False)),
                "test": f"{f.test_start}..{f.test_end}",
                "test_periods": len(ds.periods_between(f.test_start, f.test_end, pad_horizon=True)),
            }
        )
    for f in info["folds"]:
        print(
            f"[fold] {f['name']:>14}  train {f['train_periods']:>3}  "
            f"val {f['val_periods']:>2}  test {f['test_periods']:>3}"
        )
    (cache / f"dataset_info_{cfg.data.index}.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
