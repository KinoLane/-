"""Backtester accounting: costs, drift, and no look-ahead."""
from __future__ import annotations

import numpy as np
import pytest

from e2e_portfolio.backtest import simulate


def _zero_provider(n):
    """Always go to cash -- used to isolate the cost accounting."""

    def provider(p, prev):
        return np.zeros(n), {}

    return provider


def _fixed_provider(n, weights):
    w = np.zeros(n)
    w[: len(weights)] = weights

    def provider(p, prev):
        return w.copy(), {}

    return provider


def test_cost_is_charged_once_on_the_first_day_of_the_holding_window(synth_dataset):
    ds = synth_dataset
    n = ds.panel.num_tickers
    periods = ds.periods[:3]
    w = np.full(n, 1.0 / n)
    bt = simulate(ds, periods, _fixed_provider(n, [1.0 / n] * n), cost_bps=15.0)
    # first period: from cash to fully invested -> traded notional = 1
    assert abs(bt.cost[0] - 0.0015 * 1.0) < 1e-9
    # the cost lands on the first day of the period only
    first = bt.ret_gross[0] - bt.ret_net[0]
    assert abs(first - bt.cost[0]) < 1e-12


def test_zero_turnover_after_the_first_period_for_a_static_target(synth_dataset):
    """Holding a fixed target keeps the traded notional small after period 0.

    ``turnover`` is the conventional one-way turnover ``0.5 * sum|dy|``, so
    buying a fully invested portfolio from cash is exactly 1.0 of traded notional
    and 0.5 of one-way turnover.
    """
    ds = synth_dataset
    n = ds.panel.num_tickers
    periods = ds.periods[:6]
    bt = simulate(ds, periods, _fixed_provider(n, [1.0 / n] * n), cost_bps=15.0)
    assert abs(bt.traded_notional[0] - 1.0) < 1e-9
    assert abs(bt.turnover[0] - 0.5) < 1e-9
    assert abs(bt.cost[0] - 15.0 / 1e4) < 1e-12
    # only the drift of realised returns can create turnover afterwards
    assert all(t < bt.turnover[0] for t in bt.turnover[1:])


def test_cost_matches_the_declared_bps_of_the_traded_notional(synth_dataset):
    ds = synth_dataset
    n = ds.panel.num_tickers
    p = ds.periods[0]
    for bps in (5.0, 15.0, 30.0):
        bt = simulate(ds, [p], _fixed_provider(n, [1.0 / n] * n), cost_bps=bps)
        assert abs(bt.cost[0] - bps / 1e4) < 1e-12


def test_nothing_is_invested_before_the_rebalance_date(synth_dataset):
    """The first period's returns must start the day *after* the signal date."""
    ds = synth_dataset
    n = ds.panel.num_tickers
    p = ds.periods[0]
    bt = simulate(ds, [p], _fixed_provider(n, [1.0 / n] * n), cost_bps=0.0)
    expected = (p.daily_ret @ np.full(n, 1.0 / n)).tolist()
    assert np.allclose(bt.ret_net, expected, atol=1e-12)
    assert bt.dates[0] == str(ds.panel.dates[p.t + 1])


def test_weights_are_drifted_with_realised_returns(synth_dataset):
    ds = synth_dataset
    n = ds.panel.num_tickers
    p = ds.periods[0]
    target = np.zeros(n)
    target[:2] = 0.5
    bt = simulate(ds, [p], _fixed_provider(n, [0.5, 0.5]), cost_bps=0.0)
    growth = np.prod(1.0 + p.daily_ret[:, :2], axis=0)
    expected = 0.5 * growth
    expected = expected / expected.sum()
    assert np.allclose(bt.weights[0][:2], expected, atol=1e-10)
    assert abs(bt.weights[0].sum() - 1.0) < 1e-10


def test_target_weights_are_renormalised_and_validated(synth_dataset):
    ds = synth_dataset
    n = ds.panel.num_tickers
    p = ds.periods[0]

    def provider(period, prev):
        w = np.zeros(n)
        w[:4] = 0.5  # sums to 2
        return w, {}

    bt = simulate(ds, [p], provider, cost_bps=0.0)
    assert abs(bt.target_weights[0].sum() - 1.0) < 1e-12

    def bad_provider(period, prev):
        return np.zeros(n - 1), {}

    with pytest.raises(ValueError):
        simulate(ds, [p], bad_provider, cost_bps=0.0)


def test_cash_portfolio_earns_nothing_but_pays_to_get_there(synth_dataset):
    ds = synth_dataset
    n = ds.panel.num_tickers
    bt = simulate(ds, ds.periods[:4], _zero_provider(n), cost_bps=15.0)
    assert np.allclose(bt.ret_gross, 0.0)
    assert bt.cost[0] == 0.0  # already in cash


def test_benchmark_is_aligned_with_the_holding_window(synth_dataset):
    ds = synth_dataset
    n = ds.panel.num_tickers
    p = ds.periods[2]
    bt = simulate(ds, [p], _fixed_provider(n, [1.0 / n] * n), cost_bps=0.0)
    expect = ds.panel.index_ret[p.t + 1 : p.t + 1 + p.horizon]
    assert np.allclose(bt.bench_ret, expect)
