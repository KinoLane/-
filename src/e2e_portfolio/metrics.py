"""Performance and risk metrics used to compare the experiments.

All functions take a daily simple-return series (``np.ndarray``) and an
annualisation factor (252 trading days).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

TRADING_DAYS = 252
#: number of largest positions reported by the concentration diagnostics
TOP_K = 5
#: target weight above which a name counts as an economic holding.  The conic
#: solver leaves 1e-8..1e-6 residuals on the names the optimiser rejected, so a
#: plain ``w > 0`` count would report the whole investable cross-section.
HOLDING_EPS = 1e-3


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def equity_curve(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    return initial * np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))


def annualised_return(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    total = float(np.prod(1.0 + r))
    years = r.size / periods_per_year
    if total <= 0 or years <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annualised_vol(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    vol = annualised_vol(r, periods_per_year)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return (annualised_return(r, periods_per_year) - rf) / vol


def sortino(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    downside = r[r < rf / periods_per_year]
    dd = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year) if downside.size else 0.0
    if dd == 0:
        return float("nan")
    return (annualised_return(r, periods_per_year) - rf) / dd


def max_drawdown(returns: np.ndarray) -> float:
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if eq.size else float("nan")


def calmar(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return annualised_return(returns, periods_per_year) / abs(mdd)


def value_at_risk(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Historical VaR: the ``alpha`` quantile of the loss distribution."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    return float(-np.quantile(r, 1.0 - alpha))


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Historical CVaR (expected shortfall) of the loss distribution.

    ``alpha = 0.95`` averages the worst 5% of the daily returns.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    if tail.size == 0:
        return float("nan")
    return float(-tail.mean())


def downside_deviation(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(periods_per_year))


def skewness(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 3:
        return float("nan")
    s = r.std(ddof=0)
    return float(np.mean(((r - r.mean()) / s) ** 3)) if s > 0 else float("nan")


def kurtosis(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 4:
        return float("nan")
    s = r.std(ddof=0)
    return float(np.mean(((r - r.mean()) / s) ** 4)) if s > 0 else float("nan")


def omega(returns: np.ndarray, threshold: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    gains = np.maximum(r - threshold, 0.0).sum()
    losses = np.maximum(threshold - r, 0.0).sum()
    return float(gains / losses) if losses > 0 else float("nan")


def hit_ratio(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    return float((r > 0).mean()) if r.size else float("nan")


# --------------------------------------------------------------------------- #
# portfolio-level metrics
# --------------------------------------------------------------------------- #
def turnover_series(weights: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(weights - prev_weights).sum(axis=-1)


def effective_n(weights: np.ndarray) -> np.ndarray:
    """Participation ratio ``1 / sum(w^2)`` of a weight vector (or matrix)."""
    w = np.atleast_2d(weights)
    return 1.0 / np.maximum((w ** 2).sum(axis=-1), 1e-12)


def concentration(weights: np.ndarray) -> np.ndarray:
    """Herfindahl index ``sum(w^2)``."""
    w = np.atleast_2d(weights)
    return (w ** 2).sum(axis=-1)


def count_holdings(weights: np.ndarray, eps: float = HOLDING_EPS) -> np.ndarray:
    """Number of economic positions, i.e. names with ``w > eps`` (see ``HOLDING_EPS``)."""
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    return (w > eps).sum(axis=-1).astype(np.float64)


def concentration_series(weights: np.ndarray, top_k: int = TOP_K) -> Dict[str, np.ndarray]:
    """Per-rebalance concentration diagnostics of a weight matrix.

    One row per rebalance, one column per asset::

        hhi          Herfindahl index ``sum(w^2)`` (1.0 = a single asset)
        top5_weight  share of the ``top_k`` largest names (``top_k = 5``)
        max_weight   share of the single largest name

    The number of holdings alone cannot separate a 25-name book that is nearly
    equal-weighted from one that parks half the capital in three names, which is
    why the plan asks for holdings *and* concentration.
    """
    w = np.clip(np.atleast_2d(np.asarray(weights, dtype=np.float64)), 0.0, None)
    n_rows, n_cols = w.shape
    empty = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows == 0 or n_cols == 0:
        return {"hhi": empty, "top5_weight": empty.copy(), "max_weight": empty.copy()}
    k = max(1, min(int(top_k), n_cols))
    top = -np.partition(-w, k - 1, axis=-1)[:, :k]
    return {
        "hhi": concentration(w),
        "top5_weight": top.sum(axis=-1),
        "max_weight": w.max(axis=-1),
    }


def concentration_metrics(weights: np.ndarray, top_k: int = TOP_K) -> Dict[str, float]:
    """Mean of :func:`concentration_series` -- the block stored in ``metrics.json``."""
    out: Dict[str, float] = {}
    for name, values in concentration_series(weights, top_k=top_k).items():
        finite = values[np.isfinite(values)]
        out[f"{name}_mean"] = float(finite.mean()) if finite.size else float("nan")
    return out


def alpha_beta(returns: np.ndarray, benchmark: np.ndarray, periods_per_year: int = TRADING_DAYS) -> Dict[str, float]:
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    n = min(r.size, b.size)
    r, b = r[-n:], b[-n:]
    if n < 3 or b.std() == 0:
        return {"alpha": float("nan"), "beta": float("nan"), "r2": float("nan")}
    beta = float(np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    alpha_daily = float(r.mean() - beta * b.mean())
    corr = float(np.corrcoef(r, b)[0, 1])
    return {
        "alpha": alpha_daily * periods_per_year,
        "beta": beta,
        "r2": corr ** 2,
    }


def information_ratio(
    returns: np.ndarray, benchmark: np.ndarray, periods_per_year: int = TRADING_DAYS
) -> float:
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    n = min(r.size, b.size)
    if n < 3:
        return float("nan")
    active = r[-n:] - b[-n:]
    sd = active.std(ddof=1)
    if sd == 0:
        # A constant active return carries no tracking-error information; report
        # 0.0 rather than ``nan`` so that the metric block stays JSON-clean.
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def compute_metrics(
    returns: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    turnover: Optional[np.ndarray] = None,
    n_holdings: Optional[np.ndarray] = None,
    effective_holdings: Optional[np.ndarray] = None,
    alpha: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, float]:
    """The full metric block reported for every experiment / fold."""
    r = np.asarray(returns, dtype=np.float64)
    if benchmark is not None and np.asarray(benchmark).size not in (0, r.size):
        raise ValueError(
            "benchmark must have the same length as returns "
            f"(got {np.asarray(benchmark).size} vs {r.size})"
        )
    out: Dict[str, float] = {
        "n_days": int(r.size),
        "total_return": float(np.prod(1.0 + r) - 1.0) if r.size else float("nan"),
        "ann_return": annualised_return(r, periods_per_year),
        "ann_vol": annualised_vol(r, periods_per_year),
        "sharpe": sharpe(r, periods_per_year=periods_per_year),
        "sortino": sortino(r, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, periods_per_year=periods_per_year),
        "downside_dev": downside_deviation(r, periods_per_year),
        "var_95": value_at_risk(r, alpha),
        "cvar_95": conditional_value_at_risk(r, alpha),
        "skew": skewness(r),
        "kurtosis": kurtosis(r),
        "omega": omega(r),
        "hit_ratio": hit_ratio(r),
    }
    if benchmark is not None and len(benchmark) >= 3:
        b = np.asarray(benchmark, dtype=np.float64)
        out.update(alpha_beta(r, b, periods_per_year))
        out["info_ratio"] = information_ratio(r, b, periods_per_year)
        out["bench_ann_return"] = annualised_return(b, periods_per_year)
        out["bench_sharpe"] = sharpe(b, periods_per_year=periods_per_year)
        out["bench_max_drawdown"] = max_drawdown(b)
        out["excess_ann_return"] = out["ann_return"] - out["bench_ann_return"]
    if turnover is not None and len(turnover):
        t = np.asarray(turnover, dtype=np.float64)
        out["turnover_mean"] = float(t.mean())
        # ``turnover`` holds one entry *per rebalance*, not per day, so it must be
        # annualised by the realised rebalance frequency (≈12 for monthly) rather
        # than by ``periods_per_year``.
        years = max(r.size / periods_per_year, 1.0 / periods_per_year)
        rebalances_per_year = t.size / years
        out["rebalances_per_year"] = float(rebalances_per_year)
        out["turnover_ann"] = float(t.mean() * rebalances_per_year)
    if n_holdings is not None and len(n_holdings):
        out["holdings_mean"] = float(np.mean(n_holdings))
    if effective_holdings is not None and len(effective_holdings):
        out["eff_holdings_mean"] = float(np.mean(effective_holdings))
    return out


def market_state_breakdown(
    returns: np.ndarray,
    benchmark: np.ndarray,
) -> Dict[str, float]:
    """Performance split by the sign of the benchmark return on the same day."""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    n = min(r.size, b.size)
    r, b = r[-n:], b[-n:]
    up, down = b > 0, b <= 0
    out: Dict[str, float] = {}
    for tag, mask in (("up", up), ("down", down)):
        if mask.sum() > 1:
            out[f"ann_return_mkt_{tag}"] = annualised_return(r[mask])
            out[f"hit_mkt_{tag}"] = hit_ratio(r[mask])
        else:
            out[f"ann_return_mkt_{tag}"] = float("nan")
            out[f"hit_mkt_{tag}"] = float("nan")
        out[f"n_mkt_{tag}"] = float(mask.sum())
    return out
