"""Walk-forward backtester.

The simulation is deliberately simple and explicit, so that the results can be
audited line by line:

* at rebalancing date ``t`` (the close of the first trading day of the month) the
  portfolio is moved to the target weights ``y`` computed from information
  available up to and including ``t``;
* the new portfolio is held for ``horizon`` trading days, earning the official
  close-to-close returns ``r_{t+1 .. t+horizon}``;
* transaction cost is ``cost_bps`` per unit of *traded notional*, which is the
  full L1 distance ``sum_i |y_i - w_prev,i|`` (buying 1.0 of notional costs
  ``cost_bps``); the reported ``turnover`` is the conventional one-way turnover
  ``0.5 * sum_i |y_i - w_prev,i|``, so that ``cost = 2 * cost_rate * turnover``.
  The loss term in :mod:`e2e_portfolio.losses` uses the same traded notional, so
  the differentiable layer internalises exactly the cost charged here;
* a name that is suspended (or has left the index) keeps its position, earns a
  zero return for the suspended days and is marked to market with its last
  available price -- exactly what ``daily_returns`` does.

Weights are never allowed to exceed ``y_max`` at construction time; the drift
inside the holding month is reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import PanelDataset, Period
from .metrics import HOLDING_EPS, count_holdings, effective_n, market_state_breakdown, compute_metrics
from .selection import feasible_cap


@dataclass
class BacktestResult:
    dates: List[str] = field(default_factory=list)
    ret_net: List[float] = field(default_factory=list)
    ret_gross: List[float] = field(default_factory=list)
    bench_ret: List[float] = field(default_factory=list)
    weights: List[np.ndarray] = field(default_factory=list)       # per rebalancing date (N,)
    rebalance_dates: List[str] = field(default_factory=list)
    turnover: List[float] = field(default_factory=list)          # 0.5 * L1(y - prev)
    traded_notional: List[float] = field(default_factory=list)    # L1(y - prev)
    cost: List[float] = field(default_factory=list)
    n_holdings: List[float] = field(default_factory=list)
    target_weights: List[np.ndarray] = field(default_factory=list)
    diag: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def arrays(self):
        return (
            np.asarray(self.ret_net, dtype=np.float64),
            np.asarray(self.ret_gross, dtype=np.float64),
            np.asarray(self.bench_ret, dtype=np.float64),
            np.asarray(self.turnover, dtype=np.float64),
            np.asarray(self.n_holdings, dtype=np.float64),
        )

    @property
    def traded_notional_array(self) -> np.ndarray:
        return np.asarray(self.traded_notional, dtype=np.float64)

    def weight_frames(self, dataset: PanelDataset):
        """``(dates, tickers, weights)`` sparse-ish long format for saving."""
        rows = []
        for date, w in zip(self.rebalance_dates, self.target_weights):
            idx = np.flatnonzero(w > HOLDING_EPS)
            for i in idx:
                rows.append((date, str(dataset.panel.tickers[i]), float(w[i])))
        return rows

    def summary(self, alpha: float = 0.95) -> Dict[str, float]:
        net, gross, bench, turn, nhold = self.arrays()
        eff = np.asarray([effective_n(w)[0] for w in self.target_weights]) if self.target_weights else np.array([])
        out = compute_metrics(net, benchmark=bench, turnover=turn, n_holdings=nhold,
                              effective_holdings=eff, alpha=alpha)
        gross_m = compute_metrics(gross, benchmark=None, alpha=alpha)
        out["ann_return_gross"] = gross_m["ann_return"]
        out["sharpe_gross"] = gross_m["sharpe"]
        out["cvar_95_gross"] = gross_m["cvar_95"]
        out["cost_drag_ann"] = gross_m["ann_return"] - out["ann_return"]
        out.update(market_state_breakdown(net, bench))
        if self.traded_notional:
            # ``cost_drag_ann`` above is the exact realised drag from the equity
            # curves; this only exposes the underlying traded notional per rebalance.
            out["traded_notional_mean"] = float(self.traded_notional_array.mean())
        return out


# --------------------------------------------------------------------------- #
def simulate(
    dataset: PanelDataset,
    periods: Sequence[Period],
    target_fn,
    cost_bps: float = 15.0,
    y_max: float = 0.10,
    keep_records: bool = True,
    progress: bool = False,
) -> BacktestResult:
    """Run one backtest.

    ``target_fn(period, prev_weights) -> (weights, info)`` must return a
    full-length ``(N,)`` weight vector (zeros outside the universe) that sums to
    one, computed from information available at ``period.t``.
    """
    res = BacktestResult()
    n = dataset.panel.num_tickers
    prev = np.zeros(n, dtype=np.float64)
    cost_rate = cost_bps / 1e4

    for k, p in enumerate(periods):
        w_target, info = target_fn(p, prev)
        w_target = np.asarray(w_target, dtype=np.float64)
        if w_target.shape != (n,):
            raise ValueError(f"target weights must have shape {(n,)}, got {w_target.shape}")
        s = w_target.sum()
        if not np.isfinite(s) or s <= 0:
            w_target = np.zeros(n)
        elif abs(s - 1.0) > 1e-8:
            w_target = w_target / s

        traded = float(np.abs(w_target - prev).sum())
        turnover = 0.5 * traded  # conventional one-way turnover
        cost = cost_rate * traded
        rets = p.daily_ret  # (H, N) close-to-close returns inside the holding month
        portfolio_daily = rets @ w_target
        portfolio_daily = portfolio_daily.copy()
        if len(portfolio_daily):
            portfolio_daily[0] -= cost

        # drift the weights with the realised returns so that the next turnover
        # is measured against the portfolio actually held
        w_held = w_target.copy()
        for h in range(rets.shape[0]):
            w_held = w_held * (1.0 + rets[h])
            tot = w_held.sum()
            if tot > 0:
                w_held = w_held / tot

        res.dates.extend(str(d) for d in dataset.panel.dates[p.t + 1 : p.t + 1 + p.horizon])
        res.ret_net.extend(portfolio_daily.tolist())
        res.ret_gross.extend((rets @ w_target).tolist())
        res.bench_ret.extend(dataset.panel.index_ret[p.t + 1 : p.t + 1 + p.horizon].tolist())
        res.rebalance_dates.append(p.date)
        res.turnover.append(turnover)
        res.traded_notional.append(traded)
        res.cost.append(cost)
        res.n_holdings.append(float(count_holdings(w_target)[0]))
        if keep_records:
            res.target_weights.append(w_target.copy())
            res.weights.append(w_held.copy())
        prev = w_held
        if progress and k % 20 == 0:
            print(f"    backtest period {k+1}/{len(periods)} {p.date}", flush=True)

    res.diag["n_periods"] = len(periods)
    res.diag["cost_bps"] = cost_bps
    res.diag["y_max"] = y_max
    # audit against the *feasible* cap: a universe with fewer than 1/y_max names
    # cannot satisfy sum(w)=1 with w <= y_max, so the target generator relaxes it.
    # 1e-3 of slack absorbs the conic solver's own residual (SCS defaults are
    # ~1e-4 relative, which on a 10% cap shows up as a ~0.02% overshoot).
    res.diag["over_cap_periods"] = int(
        sum(
            1
            for w, p in zip(res.target_weights, periods)
            if (w - feasible_cap(y_max, p.n_universe) - 1e-3).max() > 0
        )
    ) if res.target_weights else 0
    return res
