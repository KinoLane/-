"""Rebalancing calendar, walk-forward folds and per-period training samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import DataConfig, FeatureConfig, FoldConfig
from .data import Panel
from .features import build_features, daily_returns, forward_return


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #
def monthly_rebalance_dates(dates: np.ndarray) -> np.ndarray:
    """First trading day of every month."""
    months = np.array([d[:7] for d in dates])
    first = np.ones(len(dates), dtype=bool)
    first[1:] = months[1:] != months[:-1]
    idx = np.flatnonzero(first)
    # the very first observation may be mid-month; keep it anyway
    return idx


def every_n_days(dates: np.ndarray, n: int) -> np.ndarray:
    return np.arange(0, len(dates), n)


def rebalance_index(dates: np.ndarray, cfg) -> np.ndarray:
    if cfg.rebalance == "monthly":
        return monthly_rebalance_dates(dates)
    return every_n_days(dates, int(cfg.rebalance_every))


def build_periods(dates: np.ndarray, reb: np.ndarray) -> List[Tuple[int, int]]:
    """``(rebalance_index, horizon)`` pairs; the last period is dropped when the
    holding window would run past the end of the sample."""
    out: List[Tuple[int, int]] = []
    for k, t in enumerate(reb):
        end = reb[k + 1] if k + 1 < len(reb) else None
        if end is None:
            continue
        horizon = int(end - t)
        if horizon <= 0 or end >= len(dates):
            continue
        out.append((int(t), horizon))
    return out


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #
@dataclass
class Period:
    """Everything the trainer / backtester needs for one rebalancing date."""

    t: int                     # index into ``panel.dates``
    date: str
    horizon: int               # trading days until the next rebalancing date
    universe: np.ndarray       # (N,) bool, buyable & member & tradable at ``t``
    sellable: np.ndarray       # (N,) bool, not locked limit-down at ``t``
    fwd_ret: np.ndarray        # (N,) float32 holding-period return
    daily_ret: np.ndarray      # (H, N) float32 daily returns inside the period
    liquidity_rank: np.ndarray  # (N,) int, 0 = most liquid (average amount)
    n_universe: int

    # indices into the compact per-period universe
    def compact(self, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask)


def attach_lookback(features: np.ndarray, observed: np.ndarray, t: int, lookback: int):
    """Feature window ``(L, N, K)`` and validity mask ``(L, N)`` for date ``t``."""
    lo = t - lookback + 1
    if lo >= 0:
        win = features[lo : t + 1]
        msk = observed[lo : t + 1]
    else:
        pad = -lo
        win = np.concatenate([np.zeros((pad,) + features.shape[1:], features.dtype), features[: t + 1]])
        msk = np.concatenate([np.zeros((pad, observed.shape[1]), bool), observed[: t + 1]])
    return win, msk


class PanelDataset:
    """Feature panel + rebalancing periods, with liquidity filtering."""

    def __init__(
        self,
        panel: Panel,
        features: np.ndarray,
        observed: np.ndarray,
        feature_names: Sequence[str],
        data_cfg: DataConfig,
        backtest_cfg,
    ):
        self.panel = panel
        self.features = features
        self.observed = observed
        self.feature_names = list(feature_names)
        self.data_cfg = data_cfg
        self.lookback = None  # set by the caller
        self.daily = daily_returns(panel)

        reb = rebalance_index(panel.dates, backtest_cfg)
        self.periods: List[Period] = []
        for t, horizon in build_periods(panel.dates, reb):
            self.periods.append(self._make_period(t, horizon))

    # ------------------------------------------------------------------ #
    def _make_period(self, t: int, horizon: int) -> Period:
        panel = self.panel
        universe = panel.membership[t] & self.observed[t]
        if self.data_cfg.exclude_st:
            universe = universe & ~panel.is_st[t]
        buyable = universe & ~panel.limit_locked_up[t]
        if self.data_cfg.exclude_limit_locked:
            universe = buyable
        sellable = panel.observed[t] & ~panel.limit_locked_down[t]

        fwd = forward_return(panel, t, horizon, mask=universe).astype(np.float32)
        daily = self.daily[t + 1 : t + 1 + horizon].astype(np.float32)
        # names that leave the universe mid-period keep their position but are
        # marked to market until the last available price (handled in backtest)

        liq = np.full(panel.num_tickers, 10 ** 9, dtype=np.int64)
        if self.data_cfg.top_liquidity:
            amt = panel.amount[max(0, t - 20) : t + 1].mean(axis=0)
            order = np.argsort(-np.nan_to_num(amt), kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            liq = rank
            keep = liq < self.data_cfg.top_liquidity
            universe = universe & keep
        return Period(
            t=t,
            date=str(panel.dates[t]),
            horizon=horizon,
            universe=universe,
            sellable=sellable,
            fwd_ret=fwd,
            daily_ret=daily,
            liquidity_rank=liq,
            n_universe=int(universe.sum()),
        )

    # ------------------------------------------------------------------ #
    def window(self, t: int, mask: np.ndarray):
        """``(L, M, K)`` features and ``(L, M)`` validity for the masked names."""
        win, msk = attach_lookback(self.features, self.observed, t, self.lookback)
        return win[:, mask, :], msk[:, mask]

    def index_of_date(self, date: str) -> int:
        """Position of ``date`` in the calendar (``len(dates)`` if past the end)."""
        return int(np.searchsorted(self.panel.dates, date, side="left"))

    def periods_between(self, start: str, end: str, pad_horizon: bool = True) -> List[Period]:
        """Periods whose *whole* holding window lies inside ``[start, end]``.

        With ``pad_horizon`` the holding window may extend one period past
        ``end`` -- used for the last fold so that the test sample is not
        truncated; training folds always set it to False, which guarantees that
        no future information leaks into the fit.
        """
        end_idx = int(np.searchsorted(self.panel.dates, end, side="right"))
        out = []
        for p in self.periods:
            if p.date < start:
                continue
            if p.date > end:
                break
            # The holding window of ``p`` realises daily returns up to
            # ``dates[p.t + p.horizon]``; ``end_idx`` is the first index *after*
            # ``end``, hence the strict inequality.
            if pad_horizon or (p.t + p.horizon < end_idx):
                out.append(p)
        return out

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, object]:
        counts = np.array([p.n_universe for p in self.periods])
        return {
            "index": self.panel.index,
            "n_dates": int(self.panel.num_dates),
            "n_tickers": int(self.panel.num_tickers),
            "date_range": [str(self.panel.dates[0]), str(self.panel.dates[-1])],
            "n_periods": len(self.periods),
            "universe_mean": float(counts.mean()) if len(counts) else 0.0,
            "universe_min": int(counts.min()) if len(counts) else 0,
            "universe_max": int(counts.max()) if len(counts) else 0,
            "features": self.feature_names,
        }


# --------------------------------------------------------------------------- #
# convenience factory
# --------------------------------------------------------------------------- #
def build_dataset(cfg, verbose: bool = False) -> PanelDataset:
    """Load the panel, build the features and wrap everything in a dataset.

    Kept in one place so that the training scripts, the tests and the batch
    runner all see exactly the same panel.
    """
    from .data import load_panel

    panel = load_panel(cfg.data, verbose=verbose)
    feats, observed, names = build_features(
        panel, cfg.features, exclude_st=cfg.data.exclude_st, verbose=verbose
    )
    ds = PanelDataset(
        panel=panel,
        features=feats,
        observed=observed,
        feature_names=names,
        data_cfg=cfg.data,
        backtest_cfg=cfg.backtest,
    )
    ds.lookback = int(cfg.features.lookback)
    return ds
