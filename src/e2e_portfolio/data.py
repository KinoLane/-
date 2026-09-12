"""Loading of the bundled index panels.

The dataset shipped next to this project (``findata/<index>/``) contains

``<index>_daily.parquet``   per (trade_date, constituent) market data, including
                            both raw and forward-adjusted (``qfq``) prices,
                            ST flags, effective membership dates and index
                            weights;
``tensor/X.npy``            ``(T, N, 6)`` forward-adjusted OHLCV + VWAP panel;
``tensor/membership_mask``  ``(T, N)`` bool, points in time index membership;
``tensor/observed_mask``    ``(T, N)`` bool, data availability;
``index_daily.csv``         index-level OHLC for the benchmark.

``load_panel`` returns everything aligned on the same ``(date, ticker)`` grid.
Aligned per-date matrices that are not shipped as tensors (ST flag, price-limit
lock, traded amount) are extracted once from the parquet file and cached in
``cache/`` so that repeated runs do not re-read the 130 MB parquet file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# --- price-limit rules ----------------------------------------------------- #
MAIN_BOARD_LIMIT = 0.10
ST_LIMIT = 0.05
CHINEXT_STAR_LIMIT = 0.20
#: ChiNext (300xxx) moved from 10% to 20% on 2020-08-24
CHINEXT_20PCT_FROM = "2020-08-24"
#: tolerance used when detecting a price-limit close
LIMIT_TOL = 0.003

FIELDS: List[str] = ["open", "high", "low", "close", "volume", "vwap"]


@dataclass
class Panel:
    """Aligned, point-in-time index panel."""

    index: str
    dates: np.ndarray            # (T,) str 'YYYY-MM-DD'
    tickers: np.ndarray          # (N,) str
    X: np.ndarray                # (T, N, 6) forward-adjusted OHLCV+VWAP
    membership: np.ndarray       # (T, N) bool
    observed: np.ndarray         # (T, N) bool
    is_st: np.ndarray            # (T, N) bool
    limit_locked: np.ndarray     # (T, N) bool  high == low at a price limit
    limit_locked_up: np.ndarray  # (T, N) bool
    limit_locked_down: np.ndarray  # (T, N) bool
    amount: np.ndarray           # (T, N) float64, traded amount in CNY
    index_close: np.ndarray      # (T,) float64
    index_ret: np.ndarray        # (T,) float64, close-to-close index return

    # -- convenience ------------------------------------------------------- #
    @property
    def num_dates(self) -> int:
        return self.X.shape[0]

    @property
    def num_tickers(self) -> int:
        return self.X.shape[1]

    def field(self, name: str) -> np.ndarray:
        return self.X[:, :, FIELDS.index(name)]

    def slice_dates(self, start: Optional[str], end: Optional[str]) -> "Panel":
        """Restrict the panel to ``[start, end]`` (inclusive, ISO strings)."""
        lo, hi = 0, self.num_dates
        if start is not None:
            lo = int(np.searchsorted(self.dates, start, side="left"))
        if end is not None:
            hi = int(np.searchsorted(self.dates, end, side="right"))
        return Panel(
            index=self.index,
            dates=self.dates[lo:hi],
            tickers=self.tickers,
            X=self.X[lo:hi],
            membership=self.membership[lo:hi],
            observed=self.observed[lo:hi],
            is_st=self.is_st[lo:hi],
            limit_locked=self.limit_locked[lo:hi],
            limit_locked_up=self.limit_locked_up[lo:hi],
            limit_locked_down=self.limit_locked_down[lo:hi],
            amount=self.amount[lo:hi],
            index_close=self.index_close[lo:hi],
            index_ret=self.index_ret[lo:hi],
        )

    def subset_tickers(self, keep: np.ndarray) -> "Panel":
        """Restrict the panel to ``keep`` (boolean mask over tickers)."""
        keep = np.asarray(keep, dtype=bool)
        return Panel(
            index=self.index,
            dates=self.dates,
            tickers=self.tickers[keep],
            X=self.X[:, keep],
            membership=self.membership[:, keep],
            observed=self.observed[:, keep],
            is_st=self.is_st[:, keep],
            limit_locked=self.limit_locked[:, keep],
            limit_locked_up=self.limit_locked_up[:, keep],
            limit_locked_down=self.limit_locked_down[:, keep],
            amount=self.amount[:, keep],
            index_close=self.index_close,
            index_ret=self.index_ret,
        )


# --------------------------------------------------------------------------- #
# tensor loading
# --------------------------------------------------------------------------- #
def _tensor_dir(root: str | Path, index: str) -> Path:
    return Path(root) / index / "tensor"


def load_tensor(root: str | Path, index: str) -> dict:
    """Load the raw ``.npy`` artefacts of one index."""
    d = _tensor_dir(root, index)
    if not d.exists():
        raise FileNotFoundError(
            f"tensor directory not found: {d}\n"
            f"expected the bundled dataset layout '<root>/<index>/tensor'."
        )
    return {
        "dates": np.load(d / "dates.npy").astype(str),
        "tickers": np.load(d / "tickers.npy").astype(str),
        "X": np.load(d / "X.npy").astype(np.float64),
        "membership": np.load(d / "membership_mask.npy").astype(bool),
        "observed": np.load(d / "observed_mask.npy").astype(bool),
    }


# --------------------------------------------------------------------------- #
# per-date matrices extracted from the parquet file (cached)
# --------------------------------------------------------------------------- #
_PARQUET_COLUMNS = [
    "trade_date",
    "ticker",
    "is_st_current",
    "raw_pct_change",
    "pct_change",
    "raw_close",
    "close",
    "high",
    "low",
    "volume",
    "amount",
]


def _cache_path(cache_dir: str | Path, index: str) -> Path:
    return Path(cache_dir) / f"{index}_panel_extras.npz"


def build_panel_extras(
    root: str | Path,
    index: str,
    dates: np.ndarray,
    tickers: np.ndarray,
    cache_dir: str | Path = "cache",
    force: bool = False,
) -> dict:
    """Extract ST flag, traded amount, price-limit locks and qfq close.

    The result is cached in ``cache/<index>_panel_extras.npz`` because reading
    the parquet file is by far the slowest step of the pipeline.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, index)

    T, N = len(dates), len(tickers)
    if cache_file.exists() and not force:
        npz = np.load(cache_file, allow_pickle=False)
        if npz["is_st"].shape == (T, N):
            return {k: npz[k] for k in npz.files}

    parquet = Path(root) / index / f"{index}_daily.parquet"
    if not parquet.exists():
        csv = Path(root) / index / f"{index}_daily.csv"
        if not csv.exists():
            raise FileNotFoundError(f"neither {parquet} nor {csv} exists")
        parquet = csv

    frame = _read_parquet_columns(parquet)

    date_pos = {d: i for i, d in enumerate(dates)}
    tick_pos = {t: i for i, t in enumerate(tickers)}

    is_st = np.zeros((T, N), dtype=bool)
    amount = np.zeros((T, N), dtype=np.float64)
    pct = np.zeros((T, N), dtype=np.float64)
    close_qfq = np.full((T, N), np.nan, dtype=np.float64)
    high = np.full((T, N), np.nan, dtype=np.float64)
    low = np.full((T, N), np.nan, dtype=np.float64)
    volume = np.zeros((T, N), dtype=np.float64)

    di = frame["trade_date"].map(date_pos).to_numpy()
    ti = frame["ticker"].map(tick_pos).to_numpy()
    valid = pd.notna(di) & pd.notna(ti)
    di = di[valid].astype(np.int64)
    ti = ti[valid].astype(np.int64)
    sub = frame.loc[valid]

    is_st[di, ti] = sub["is_st_current"].fillna(0).to_numpy().astype(bool)
    amount[di, ti] = sub["amount"].fillna(0.0).to_numpy()
    pct[di, ti] = sub["raw_pct_change"].fillna(0.0).to_numpy()
    close_qfq[di, ti] = sub["close"].to_numpy()
    high[di, ti] = sub["high"].to_numpy()
    low[di, ti] = sub["low"].to_numpy()
    volume[di, ti] = sub["volume"].fillna(0.0).to_numpy()

    # --- price-limit lock detection -------------------------------------- #
    limit_rate = _limit_rate_matrix(dates, tickers, is_st)
    price_pct = pct / 100.0  # the parquet stores percentage points
    at_limit_up = price_pct >= (limit_rate - LIMIT_TOL)
    at_limit_down = price_pct <= -(limit_rate - LIMIT_TOL)
    one_price = np.isfinite(high) & np.isfinite(low) & (np.abs(high - low) < 1e-9)
    limit_locked_up = at_limit_up & one_price & (volume > 0)
    limit_locked_down = at_limit_down & one_price & (volume > 0)
    limit_locked = limit_locked_up | limit_locked_down

    np.savez_compressed(
        cache_file,
        is_st=is_st,
        amount=amount,
        pct=price_pct,
        close_qfq=close_qfq,
        limit_locked=limit_locked,
        limit_locked_up=limit_locked_up,
        limit_locked_down=limit_locked_down,
    )
    return {
        "is_st": is_st,
        "amount": amount,
        "pct": price_pct,
        "close_qfq": close_qfq,
        "limit_locked": limit_locked,
        "limit_locked_up": limit_locked_up,
        "limit_locked_down": limit_locked_down,
    }


def _read_parquet_columns(path: Path) -> pd.DataFrame:
    """Read only the columns we need from parquet / csv."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in _PARQUET_COLUMNS if c in available]
        frame = pq.read_table(path, columns=cols).to_pandas()
    else:
        frame = pd.read_csv(
            path, usecols=lambda c: c in set(_PARQUET_COLUMNS), low_memory=False
        )
    if "trade_date" not in frame.columns or "ticker" not in frame.columns:
        raise ValueError(f"{path} is missing 'trade_date'/'ticker' columns")
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    frame["ticker"] = frame["ticker"].astype(str)
    return frame


def _limit_rate_matrix(
    dates: np.ndarray, tickers: np.ndarray, is_st: np.ndarray
) -> np.ndarray:
    """Daily price-limit rate for every (date, ticker) pair."""
    board = np.full(len(tickers), MAIN_BOARD_LIMIT, dtype=np.float64)
    codes = np.array([t.split(".")[0] for t in tickers])
    chinext = np.array([c.startswith("300") or c.startswith("301") for c in codes])
    star = np.array([c.startswith("688") or c.startswith("689") for c in codes])
    board[star] = CHINEXT_STAR_LIMIT
    after = dates >= CHINEXT_20PCT_FROM
    board_2d = np.where(chinext[None, :] & after[:, None], CHINEXT_STAR_LIMIT, board[None, :])
    return np.where(is_st, ST_LIMIT, board_2d)


# --------------------------------------------------------------------------- #
# index benchmark
# --------------------------------------------------------------------------- #
def load_index_daily(root: str | Path, index: str) -> pd.DataFrame:
    path = Path(root) / index / "index_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["trade_date", "close"])
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    frame = frame.drop_duplicates("trade_date").sort_values("trade_date")
    return frame.rename(columns={"trade_date": "date", "close": "close"}).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def load_panel(
    cfg=None,
    index: str = "csi300",
    cache_dir: str | Path = "cache",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_cache: bool = False,
    verbose: bool = False,
) -> Panel:
    """Load the full aligned panel for one index.

    The first argument may be a :class:`~e2e_portfolio.config.DataConfig`, in
    which case ``index``/``cache_dir``/``start_date``/``end_date`` are taken from
    it; otherwise it is the dataset root and the remaining arguments are used.
    """
    import time as _time

    from .config import DataConfig

    if isinstance(cfg, DataConfig):
        force_cache = force_cache or False
        root, index, cache_dir, start_date, end_date = (
            cfg.root,
            cfg.index,
            cfg.cache_dir,
            cfg.start_date,
            cfg.end_date,
        )
    else:
        root = r"D:\金创\findata" if cfg is None else cfg

    t0 = _time.time()
    tensors = load_tensor(root, index)
    data = tensors["X"]
    observed = tensors["observed"]
    # A stock is observable only if the whole OHLC block is present.
    observed = observed & np.isfinite(data[:, :, :5]).all(axis=2)
    observed = observed & (data[:, :, 3] > 0)

    extras = build_panel_extras(
        root,
        index,
        tensors["dates"],
        tensors["tickers"],
        cache_dir=cache_dir,
        force=force_cache,
    )

    bench = load_index_daily(root, index)
    bench = bench.set_index("date")
    aligned = bench.reindex(tensors["dates"])
    index_close = aligned["close"].to_numpy(dtype=np.float64)
    # forward-fill missing benchmark days, then compute close-to-close returns
    index_close = pd.Series(index_close).ffill().bfill().to_numpy()
    index_ret = np.zeros_like(index_close)
    index_ret[1:] = index_close[1:] / index_close[:-1] - 1.0

    panel = Panel(
        index=index,
        dates=tensors["dates"],
        tickers=tensors["tickers"],
        X=data,
        membership=tensors["membership"],
        observed=observed,
        is_st=extras["is_st"],
        limit_locked=extras["limit_locked"],
        limit_locked_up=extras["limit_locked_up"],
        limit_locked_down=extras["limit_locked_down"],
        amount=extras["amount"],
        index_close=index_close,
        index_ret=index_ret,
    )
    start = start_date or None
    end = end_date or None
    if start or end:
        panel = panel.slice_dates(start, end)
    if verbose:
        print(
            f"[data] {index}: {panel.num_dates} dates x {panel.num_tickers} tickers "
            f"({panel.dates[0]} .. {panel.dates[-1]}) in {_time.time() - t0:.1f}s"
        )
    return panel


def validate_panel(
    panel: Panel,
    cfg=None,
    cache_dir: str | Path = "cache",
    n_probe: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Sanity check: the tensor prices must reproduce the parquet qfq close.

    ``cfg`` may be a :class:`~e2e_portfolio.config.DataConfig` (preferred) or the
    dataset root.
    """
    from .config import DataConfig

    if isinstance(cfg, DataConfig):
        root, cache_dir = cfg.root, cfg.cache_dir
    else:
        root = r"D:\金创\findata" if cfg is None else cfg
    extras = build_panel_extras(
        root, panel.index, panel.dates, panel.tickers, cache_dir=cache_dir
    )
    close_tensor = panel.field("close")
    close_ref = extras["close_qfq"]
    mask = panel.observed & np.isfinite(close_ref)
    idx = np.flatnonzero(mask.ravel())
    rng = np.random.default_rng(seed)
    if len(idx) > n_probe:
        idx = rng.choice(idx, size=n_probe, replace=False)
    a = close_tensor.ravel()[idx]
    b = close_ref.ravel()[idx]
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
    return {
        "n_probe": int(len(idx)),
        "max_rel_diff": float(np.max(rel)) if len(idx) else float("nan"),
        "mean_rel_diff": float(np.mean(rel)) if len(idx) else float("nan"),
        "n_mismatch_gt_1pct": int(np.sum(rel > 0.01)),
    }
