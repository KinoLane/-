"""Shared pytest fixtures.

Two groups of tests:

* **synthetic** (default) -- a small fabricated ``findata``-style directory is
  written to a temporary folder and pushed through the *real* loading, feature
  and dataset pipeline (``load_panel`` -> ``build_features`` -> ``build_dataset``).
  This keeps the tests fast (a few seconds) while still exercising production
  code paths rather than mocks.
* **data-backed** -- tests that need the real CSI 300 panel are marked
  ``data`` and skipped automatically when ``D:\\金创\\findata`` is absent.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA_ROOT = Path(os.environ.get("E2E_FINDATA_ROOT", "D:\\金创\\findata"))

FIELDS = ["open", "high", "low", "close", "volume", "vwap"]


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the real findata dataset")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


requires_data = pytest.mark.skipif(
    not (DATA_ROOT / "csi300").exists(), reason=f"dataset not found at {DATA_ROOT}"
)


@pytest.fixture(scope="session")
def data_root() -> Path:
    return DATA_ROOT


def synth_root(base: Path) -> Path:
    """Write a tiny ``findata``-style dataset and return its root.

    The layout mirrors the real bundle exactly:

    ``<root>/<index>/tensor/dates.npy|tickers.npy|X.npy|membership_mask.npy|observed_mask.npy``
    ``<root>/<index>/<index>_daily.parquet``
    ``<root>/<index>/index_daily.csv``
    """
    index = "synth"
    rng = np.random.default_rng(20240101)
    n_t, n_n = 520, 14
    dates = np.array(
        pd.bdate_range("2018-01-01", periods=n_t).strftime("%Y-%m-%d"), dtype="<U10"
    )
    tickers = np.array([f"{600000 + i}.SH" for i in range(n_n)], dtype="<U11")

    # a weak, learnable cross-sectional signal + a common market factor
    loadings = rng.normal(0.0, 1.0, n_n)
    market = rng.normal(0.0003, 0.011, n_t)
    alpha = 0.002 * loadings[None, :] + rng.normal(0.0, 0.004, (n_t, n_n))
    idio = rng.normal(0.0, 0.011, (n_t, n_n))
    ret = np.clip(0.9 * market[:, None] * loadings[None, :] + alpha + idio, -0.095, 0.095)

    close = 20.0 * np.cumprod(1.0 + ret, axis=0)
    price_pct = np.concatenate([np.zeros((1, n_n)), ret[1:]], axis=0)
    volume = rng.lognormal(15.0, 0.5, (n_t, n_n))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, (n_t, n_n)))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    vwap = (open_ + high + low + close) / 4.0

    X = np.stack([open_, high, low, close, volume, vwap], axis=-1).astype(np.float64)
    membership = np.ones((n_t, n_n), dtype=bool)
    observed = np.ones((n_t, n_n), dtype=bool)
    observed[40, :] = False              # a fully missing session
    observed[7:10, 2] = False            # a short trading halt
    X[40, :, :] = np.nan

    tensor = base / index / "tensor"
    tensor.mkdir(parents=True, exist_ok=True)
    np.save(tensor / "dates.npy", dates)
    np.save(tensor / "tickers.npy", tickers)
    np.save(tensor / "X.npy", X)
    np.save(tensor / "membership_mask.npy", membership)
    np.save(tensor / "observed_mask.npy", observed)
    (tensor / "features.json").write_text(json.dumps(FIELDS), encoding="utf-8")

    # the per-ticker daily frame (qfq close and the percentage change)
    rows = []
    for ti, ticker in enumerate(tickers):
        for di, date in enumerate(dates):
            if not observed[di, ti]:
                continue
            rows.append(
                {
                    "trade_date": date,
                    "ticker": ticker,
                    "is_st_current": False,
                    "raw_pct_change": float(price_pct[di, ti] * 100.0),
                    "pct_change": float(price_pct[di, ti] * 100.0),
                    "raw_close": float(close[di, ti]),
                    "close": float(close[di, ti]),
                    "high": float(high[di, ti]),
                    "low": float(low[di, ti]),
                    "volume": float(volume[di, ti]),
                    "amount": float(volume[di, ti] * close[di, ti]),
                }
            )
    pd.DataFrame(rows).to_parquet(base / index / f"{index}_daily.parquet", index=False)

    index_close = 3000.0 * np.cumprod(1.0 + market)
    pd.DataFrame({"trade_date": dates, "close": index_close}).to_csv(
        base / index / "index_daily.csv", index=False
    )
    return base


@pytest.fixture(scope="session")
def synth_cfg(tmp_path_factory):
    """A configuration pointed at the synthetic dataset."""
    from e2e_portfolio.config import Config

    root = synth_root(tmp_path_factory.mktemp("findata"))
    cfg = Config()
    cfg.data.root = str(root)
    cfg.data.index = "synth"
    cfg.data.cache_dir = str(root / "cache")
    cfg.data.top_liquidity = 8
    cfg.data.exclude_st = True
    cfg.data.exclude_limit_locked = True
    cfg.model.num_scenarios = 12
    cfg.model.hidden_size = 8
    cfg.model.head_hidden = 8
    cfg.train.epochs = 2
    cfg.train.num_threads = 1
    return cfg


@pytest.fixture(scope="session")
def synth_dataset(synth_cfg):
    """The synthetic :class:`PanelDataset` (built once per session)."""
    from e2e_portfolio.dataset import build_dataset

    return build_dataset(synth_cfg, verbose=False)
