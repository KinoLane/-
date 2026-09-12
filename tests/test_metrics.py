"""Metrics must be correct on hand-checked inputs."""
from __future__ import annotations

import numpy as np
import pytest

from e2e_portfolio.metrics import (
    annualised_return,
    annualised_vol,
    concentration_metrics,
    concentration_series,
    conditional_value_at_risk,
    count_holdings,
    effective_n,
    equity_curve,
    hit_ratio,
    information_ratio,
    max_drawdown,
    omega,
    sharpe,
    sortino,
    value_at_risk,
)
from e2e_portfolio.metrics import compute_metrics


def test_count_holdings_ignores_solver_dust():
    """A book of 2 real names plus 50 solver residuals holds 2 names, not 52."""
    w = np.zeros(100)
    w[0], w[1] = 0.6, 0.4
    w[2:52] = 1e-7  # the conic solver's leftovers on the rejected names
    assert count_holdings(w)[0] == 2.0
    # a name exactly at the reporting floor does not count, one just above it does
    w2 = np.zeros(100)
    w2[0] = 0.9
    w2[1], w2[2] = 1e-3, 1e-3 + 1e-9
    assert count_holdings(w2)[0] == 2.0
    # works row-wise on a weight matrix as well
    assert np.array_equal(count_holdings(np.vstack([w, w2])), np.array([2.0, 2.0]))


def test_equity_curve_compounds():
    r = np.array([0.1, -0.1])
    assert np.allclose(equity_curve(r), [1.1, 1.1 * 0.9])


def test_max_drawdown_of_a_known_path():
    r = np.array([0.0, 0.25, -0.20, 0.0])
    # equity: 1, 1.25, 1.0 -> trough 20% below the peak
    assert abs(max_drawdown(r) + 0.20) < 1e-12


def test_annualised_return_is_a_cagr():
    years = 2
    r = np.full(252 * years, 0.0)
    r[0] = np.exp(np.log(1.21) * 1.0) - 1.0  # +21% once
    assert abs(annualised_return(r) - (1.21 ** (1 / years) - 1)) < 1e-9


def test_annualised_vol_scales_with_the_square_root_of_time():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 100_000)
    assert abs(annualised_vol(r) - 0.01 * np.sqrt(252)) < 1e-3


def test_value_at_risk_and_cvar_on_a_uniform_grid():
    r = np.linspace(-0.05, 0.05, 1001)
    # 95% VaR -> the 5th percentile
    assert abs(value_at_risk(r, 0.95) - 0.045) < 1e-6
    # CVaR is the *mean* of the worst 5%, i.e. slightly worse than the quantile
    assert conditional_value_at_risk(r, 0.95) > value_at_risk(r, 0.95)


def test_sharpe_of_a_constant_return_series_is_undefined_not_infinite():
    r = np.full(50, 0.01)
    assert np.isnan(sharpe(r)) or np.isinf(sharpe(r)) or abs(sharpe(r)) > 100


def test_sortino_penalises_downside_only():
    rng = np.random.default_rng(1)
    up = rng.normal(0.002, 0.01, 500)
    down = rng.normal(-0.002, 0.01, 500)
    assert sortino(up) > sortino(down)


def test_omega_is_the_gain_over_loss_ratio():
    r = np.array([0.02, 0.02, -0.01, -0.01])
    assert abs(omega(r) - (0.04 / 0.02)) < 1e-12


def test_hit_ratio_counts_positive_days():
    assert abs(hit_ratio(np.array([1e-3, -1e-3, 0.0, 2e-3])) - 0.5) < 1e-12


def test_effective_n_of_a_concentrated_and_a_uniform_portfolio():
    assert abs(effective_n(np.array([1.0, 0.0, 0.0]))[0] - 1.0) < 1e-9
    assert abs(effective_n(np.full(10, 0.1))[0] - 10.0) < 1e-9


def test_information_ratio_is_zero_for_an_identical_series():
    r = np.linspace(-0.02, 0.02, 200)
    assert abs(information_ratio(r, r)) < 1e-9


def test_compute_metrics_handles_a_short_series():
    out = compute_metrics(np.array([0.01, -0.02, 0.03]))
    assert np.isfinite(out["ann_return"])
    assert out["n_days"] == 3


def test_compute_metrics_requires_matching_shapes():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0.01, 0.02]), benchmark=np.array([0.01]))


def test_turnover_is_annualised_by_the_rebalance_frequency():
    """``turnover`` is per rebalance, so a monthly series must scale by ~12."""
    rng = np.random.default_rng(7)
    n_rebal = 24  # two years of monthly rebalances
    days_per_rebal = 21
    daily = rng.normal(0.0, 0.01, n_rebal * days_per_rebal)
    turnover = np.full(n_rebal, 0.30)
    out = compute_metrics(daily, turnover=turnover)
    assert abs(out["turnover_mean"] - 0.30) < 1e-12
    assert abs(out["rebalances_per_year"] - 12.0) < 0.1
    assert abs(out["turnover_ann"] - 0.30 * out["rebalances_per_year"]) < 1e-12
    # the old (buggy) behaviour annualised by 252 trading days
    assert out["turnover_ann"] < 4.0


# --------------------------------------------------------------------------- #
# concentration (the plan's "持仓数量及集中度" metric)
# --------------------------------------------------------------------------- #
def test_concentration_series_on_hand_checked_books():
    w = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # single asset
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # two equal names
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],  # five equal names
        ]
    )
    out = concentration_series(w)
    assert np.allclose(out["hhi"], [1.0, 0.5, 0.2])
    assert np.allclose(out["max_weight"], [1.0, 0.5, 0.2])
    # the top-5 share always sums the five largest names: 1.0, 1.0, 1.0
    assert np.allclose(out["top5_weight"], [1.0, 1.0, 1.0])


def test_top5_share_uses_the_five_largest_names_not_the_first_five_columns():
    w = np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.2, 0.1]])
    # 0.5 + 0.5 + 0.2 + 0.1 + 0.0 (the sixth *column* holds 0.2 but is not a top-5 name)
    assert abs(concentration_series(w)["top5_weight"][0] - 1.3) < 1e-12
    w2 = np.array([[0.60, 0.20, 0.10, 0.04, 0.03, 0.03]])
    assert abs(concentration_series(w2)["top5_weight"][0] - 0.97) < 1e-12


def test_concentration_series_ignores_short_positions():
    """Negative (short) weights must not reduce the index."""
    out = concentration_series(np.array([[0.6, 0.4, -0.2]]))
    assert abs(out["hhi"][0] - 0.52) < 1e-12


def test_concentration_metrics_averages_the_rebalances():
    w = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]])
    out = concentration_metrics(w)
    assert abs(out["hhi_mean"] - 0.6) < 1e-12
    assert abs(out["max_weight_mean"] - 0.6) < 1e-12
    assert abs(out["top5_weight_mean"] - 1.0) < 1e-12


def test_concentration_handles_empty_and_single_column_inputs():
    out = concentration_metrics(np.zeros((0, 4)))
    assert all(np.isnan(v) for v in out.values())
    out = concentration_metrics(np.array([[0.5, 0.5]]))
    assert abs(out["hhi_mean"] - 0.5) < 1e-12


def test_a_equal_weight_book_of_25_names_has_the_expected_hhi():
    """The headline holdings count cannot separate these two 25-name books."""
    ew = np.full((1, 25), 1.0 / 25.0)
    lumpy = np.zeros((1, 25))
    lumpy[0, :3] = [0.34, 0.33, 0.33]
    assert concentration_metrics(ew)["top5_weight_mean"] < 0.25
    assert concentration_metrics(lumpy)["top5_weight_mean"] == pytest.approx(1.0)
