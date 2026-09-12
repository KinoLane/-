"""Causal feature engineering.

All features for date ``t`` only use information available up to the close of
``t``.  Cross-sectional standardisation is performed *within each date*, which
keeps the scale comparable through time without leaking future information.

The implementation is fully vectorised over ``(T, N)`` with cumulative sums, so
building features for 4 000 days x 3 000 stocks takes a few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import FeatureConfig
from .data import Panel


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill along axis 0 (last observation carried forward)."""
    mask = np.isfinite(a)
    idx = np.where(mask, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over the trailing window of length ``w`` (x is 0 where missing)."""
    cs = np.cumsum(x, axis=0, dtype=np.float64)
    out = np.empty_like(cs)
    out[:w] = cs[:w]
    out[w:] = cs[w:] - cs[:-w]
    return out


def _rolling_mean_std(
    x: np.ndarray, valid: np.ndarray, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Trailing mean / (population) std over `w` observations, counting only
    entries flagged by ``valid``."""
    xv = np.where(valid, x, 0.0)
    cnt = _rolling_sum(valid.astype(np.float64), w)
    s1 = _rolling_sum(xv, w)
    s2 = _rolling_sum(np.where(valid, x * x, 0.0), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def _cs_zscore(x: np.ndarray, mask: np.ndarray, clip: float) -> np.ndarray:
    """Cross-sectional z-score for every date, over the entries in ``mask``."""
    m = mask & np.isfinite(x)
    cnt = m.sum(axis=1, keepdims=True).astype(np.float64)
    xm = np.where(m, x, 0.0)
    mean = xm.sum(axis=1, keepdims=True) / np.maximum(cnt, 1.0)
    dev = np.where(m, x - mean, 0.0)
    std = np.sqrt((dev ** 2).sum(axis=1, keepdims=True) / np.maximum(cnt, 1.0))
    z = dev / np.maximum(std, 1e-8)
    z = np.where(cnt > 1, z, 0.0)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -clip, clip)


def _cs_rank(x: np.ndarray, mask: np.ndarray, clip: float) -> np.ndarray:
    """Cross-sectional rank mapped to ``[-clip, clip]`` (robust alternative)."""
    m = mask & np.isfinite(x)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        row = m[t]
        n = int(row.sum())
        if n < 2:
            continue
        vals = x[t, row]
        order = np.argsort(np.argsort(vals))
        out[t, row] = (order / (n - 1) * 2.0 - 1.0) * clip
    return out


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_features(
    panel: Panel,
    cfg: FeatureConfig,
    exclude_st: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``(features, valid_mask, feature_names)``.

    ``features`` has shape ``(T, N, K)`` and is standardised cross-sectionally
    on every date over the *investable* cross-section.
    """
    close = panel.field("close")
    open_ = panel.field("open")
    high = panel.field("high")
    low = panel.field("low")
    volume = panel.field("volume")
    vwap = panel.field("vwap")

    observed = panel.observed.copy()
    if exclude_st:
        observed = observed & ~panel.is_st

    close_ff = _ffill(np.where(observed, close, np.nan))
    prev = np.vstack([close_ff[:1], close_ff[:-1]])
    ret1 = np.where(observed, close_ff / prev - 1.0, np.nan)
    valid_ret = observed & np.isfinite(ret1) & (prev > 0)

    mkt = panel.index_ret[:, None].astype(np.float64)
    mkt = np.repeat(mkt, close.shape[1], axis=1)

    feats: Dict[str, np.ndarray] = {}

    sw, mw, lw = cfg.short_window, cfg.mid_window, cfg.long_window
    for w in (sw, mw, lw):
        base = np.vstack([np.full((w, close.shape[1]), np.nan), close_ff[:-w]])
        feats[f"mom_{w}"] = np.where(observed, close_ff / base - 1.0, np.nan)

    # realised volatility, short and long horizon
    mean_, std_ = _rolling_mean_std(np.nan_to_num(ret1), valid_ret, mw)
    feats[f"vol_{mw}"] = np.where(observed, std_, np.nan)
    mean_, std_ = _rolling_mean_std(np.nan_to_num(ret1), valid_ret, lw)
    feats[f"vol_{lw}"] = np.where(observed, std_, np.nan)

    # intraday shape
    feats["range"] = np.where(observed, (high - low) / np.maximum(close, 1e-12), np.nan)
    feats["open_close"] = np.where(observed, close / np.maximum(open_, 1e-12) - 1.0, np.nan)
    feats["vwap_gap"] = np.where(observed, vwap / np.maximum(close, 1e-12) - 1.0, np.nan)

    # liquidity
    log_vol = np.log1p(np.maximum(volume, 0.0))
    mean_v, _ = _rolling_mean_std(log_vol, observed, mw)
    feats["log_volume"] = np.where(observed, log_vol, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["volume_ratio"] = np.where(
            observed, log_vol - mean_v, np.nan
        )
    amount = np.maximum(panel.amount, 1.0)
    illiq = np.abs(np.nan_to_num(ret1)) * 1e8 / amount
    mean_i, _ = _rolling_mean_std(illiq, valid_ret, mw)
    feats["amihud"] = np.where(observed, np.log1p(mean_i * 1e3), np.nan)

    # market-relative features
    feats["mkt_ret_1"] = np.where(observed, np.tile(panel.index_ret[:, None], (1, close.shape[1])), np.nan)
    cs = np.cumsum(np.nan_to_num(panel.index_ret))
    mkt_cum = np.concatenate([np.zeros(mw), cs[:-mw]])
    feats[f"mkt_ret_{mw}"] = np.where(observed, (cs - mkt_cum)[:, None] * np.ones_like(close), np.nan)

    r1 = np.nan_to_num(ret1)
    mm_ = np.where(valid_ret, mkt, 0.0)
    cov = _rolling_sum(r1 * mm_, lw) / np.maximum(_rolling_sum(valid_ret.astype(float), lw), 1.0)
    mean_r = _rolling_sum(r1, lw) / np.maximum(_rolling_sum(valid_ret.astype(float), lw), 1.0)
    mean_m = _rolling_sum(mm_, lw) / np.maximum(_rolling_sum(valid_ret.astype(float), lw), 1.0)
    var_m = _rolling_sum(mm_ * mm_, lw) / np.maximum(
        _rolling_sum(valid_ret.astype(float), lw), 1.0
    ) - mean_m ** 2
    beta = (cov - mean_r * mean_m) / np.maximum(var_m, 1e-10)
    beta = np.clip(beta, -3.0, 3.0)
    feats[f"beta_{lw}"] = np.where(observed, beta, np.nan)
    feats[f"resid_mom_{mw}"] = np.where(
        observed, feats[f"mom_{mw}"] - beta * feats[f"mkt_ret_{mw}"], np.nan
    )
    # downside-risk asymmetry
    neg = np.where(valid_ret & (r1 < 0), r1, 0.0)
    _, dstd = _rolling_mean_std(neg, valid_ret, mw)
    feats[f"downside_vol_{mw}"] = np.where(observed, dstd, np.nan)
    feats[f"ret_skew_{lw}"] = np.where(
        observed,
        _rolling_sum(np.nan_to_num(ret1) ** 3, lw)
        / np.maximum(_rolling_sum(valid_ret.astype(float), lw), 1.0)
        / np.maximum(feats[f"vol_{lw}"] ** 3, 1e-10),
        np.nan,
    )

    if cfg.include_daily_return:
        # the raw daily return is not fed to the encoder by default (the
        # multi-horizon momentum channels already carry the price path); the
        # plan's feature list asks for it explicitly, so it can be switched on
        with np.errstate(invalid="ignore", divide="ignore"):
            feats["ret_1"] = np.where(valid_ret, ret1, np.nan)
            feats["log_ret_1"] = np.where(
                valid_ret, np.log1p(np.clip(ret1, -0.999, None)), np.nan
            )

    names = sorted(feats.keys())
    mask = observed
    out = np.zeros((close.shape[0], close.shape[1], len(names)), dtype=np.float64)
    for k, name in enumerate(names):
        out[:, :, k] = _cs_zscore(feats[name], mask, cfg.clip)

    # names whose raw value is undefined carry a neutral 0 in the standardised
    # space; remember that with an explicit availability mask
    for k, name in enumerate(names):
        ok = np.isfinite(feats[name]) & mask
        out[:, :, k] = np.where(ok, out[:, :, k], 0.0)

    if verbose:
        print(f"[features] built {len(names)} features: {names}")
    return out.astype(np.float32), mask, names


def forward_return(
    panel: Panel, t: int, horizon: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Total return of every stock from ``t`` to ``t + horizon`` (qff close)."""
    close = _ffill(np.where(panel.observed, panel.field("close"), np.nan))
    end = t + horizon
    if end >= close.shape[0]:
        raise IndexError("horizon runs past the end of the panel")
    r = close[end] / close[t] - 1.0
    r = np.where(np.isfinite(r) & (close[t] > 0), r, 0.0)
    if mask is not None:
        r = np.where(mask, r, 0.0)
    return r


def daily_returns(panel: Panel) -> np.ndarray:
    """Per-stock close-to-close returns, ``0`` where the stock is not observed.

    A suspended (or already delisted) name simply contributes a zero return for
    that day, which is the standard treatment of a frozen position.
    """
    close = _ffill(np.where(panel.observed, panel.field("close"), np.nan))
    prev = np.vstack([close[:1], close[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = close / prev - 1.0
    ret = np.where(np.isfinite(ret), ret, 0.0)
    ret = np.where(panel.observed, ret, 0.0)
    return ret
