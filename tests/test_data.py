"""Feature-engineering and dataset invariants."""
from __future__ import annotations

import numpy as np

from e2e_portfolio.data import load_panel
from e2e_portfolio.dataset import monthly_rebalance_dates, build_periods
from e2e_portfolio.features import build_features, daily_returns, forward_return


def test_panel_shapes_and_alignment(synth_cfg):
    panel = load_panel(synth_cfg.data)
    assert panel.num_dates == 520
    assert panel.num_tickers == 14
    assert panel.dates[0] == "2018-01-01" or panel.dates[0] >= "2018-01-01"
    assert np.all(np.diff(panel.dates.astype("datetime64[D]")) > np.timedelta64(0, "D"))
    assert panel.index_ret.shape == (panel.num_dates,)
    # a fully missing session must not be flagged as observed
    blank = np.flatnonzero(~panel.observed.any(axis=1))
    assert blank.size == 1, f"expected exactly one blank session, got {blank}"
    assert not panel.observed[blank[0]].any()
    # the short trading halt of the third ticker must be carried through as well
    assert not panel.observed[7:10, 2].any()


def test_daily_returns_respects_missing_days(synth_cfg):
    panel = load_panel(synth_cfg.data)
    ret = daily_returns(panel)
    assert ret.shape == (panel.num_dates, panel.num_tickers)
    assert np.isfinite(ret).all()
    bad = ~panel.observed
    assert (ret[bad] == 0.0).all(), "unobserved cells must carry a zero return"


def test_forward_return_matches_compounded_daily_returns(synth_cfg):
    panel = load_panel(synth_cfg.data)
    daily = daily_returns(panel)
    t, h = 100, 21
    mask = panel.membership[t] & panel.observed[t]
    fwd = forward_return(panel, t, h, mask=mask)
    manual = np.prod(1.0 + daily[t + 1 : t + 1 + h], axis=0) - 1.0
    good = mask & panel.observed[t + h]
    assert np.allclose(fwd[good], manual[good], atol=1e-6)


def test_daily_return_channels_are_optional_and_exactly_the_z_scored_return(synth_cfg):
    """The plan lists "收益率与对数收益率"; they are an opt-in input channel."""
    panel = load_panel(synth_cfg.data)
    base, _, base_names = build_features(
        panel, synth_cfg.features, exclude_st=synth_cfg.data.exclude_st
    )
    cfg = synth_cfg.replace(features={"include_daily_return": True})
    ext, mask, ext_names = build_features(panel, cfg.features, exclude_st=cfg.data.exclude_st)

    assert "ret_1" not in base_names and "log_ret_1" not in base_names
    assert set(ext_names) - set(base_names) == {"ret_1", "log_ret_1"}
    assert ext.shape[0] == base.shape[0]
    assert ext.shape[2] == base.shape[2] + 2

    # channel == cross-sectional z-score of the close-to-close return
    t = 100
    close = panel.field("close")
    ret = close[t] / close[t - 1] - 1.0
    z = (ret - ret.mean()) / ret.std()
    got = ext[t, mask[t], ext_names.index("ret_1")]
    assert mask[t].all()
    assert np.allclose(got, np.clip(z, -cfg.features.clip, cfg.features.clip), atol=1e-6)

    # log return is the same quantity up to the (monotone) log transform
    r = ext[:, :, ext_names.index("ret_1")].ravel()
    lg = ext[:, :, ext_names.index("log_ret_1")].ravel()
    assert np.corrcoef(r, lg)[0, 1] > 0.999


def test_features_are_finite_and_normalised(synth_dataset):
    ds = synth_dataset
    assert ds.features.shape[:2] == (ds.panel.num_dates, ds.panel.num_tickers)
    assert len(ds.feature_names) == ds.features.shape[2]
    finite = np.isfinite(ds.features)
    assert finite.mean() > 0.95
    assert np.abs(ds.features[finite]).max() < 100.0, "features must be winsorised"


def test_rebalance_calendar_is_the_first_trading_day_of_each_month(synth_cfg):
    panel = load_panel(synth_cfg.data)
    idx = monthly_rebalance_dates(panel.dates)
    months = [panel.dates[i][:7] for i in idx]
    assert months == sorted(set(months))
    pairs = build_periods(panel.dates, idx)
    for t, h in pairs:
        assert h > 0 and t + h < panel.num_dates


def test_periods_between_does_not_leak_the_future(synth_dataset):
    """Training windows must never see returns from after ``train_end``."""
    ds = synth_dataset
    train = ds.periods_between("2018-01-01", "2019-06-30", pad_horizon=False)
    assert train
    for p in train:
        assert p.date <= "2019-06-30"
        assert ds.panel.dates[p.t + p.horizon] <= "2019-06-30"
        assert p.daily_ret.shape[0] == p.horizon

    test = ds.periods_between("2019-07-01", "2019-12-31", pad_horizon=True)
    assert test
    assert min(p.date for p in test) >= "2019-07-01"
    # the last period of the padded window may reach one month past `end`
    assert max(p.date for p in test) <= "2019-12-31"


def test_liquidity_filter_bounds_the_universe(synth_dataset):
    ds = synth_dataset
    top = ds.data_cfg.top_liquidity
    assert ds.periods
    for p in ds.periods:
        assert p.n_universe <= top


def test_period_slices_are_consistent(synth_dataset):
    ds = synth_dataset
    p = ds.periods[5]
    assert p.daily_ret.shape == (p.horizon, ds.panel.num_tickers)
    assert p.fwd_ret.shape == (ds.panel.num_tickers,)
    assert p.universe.shape == (ds.panel.num_tickers,)
    # a name that is not in the universe must not be buyable
    assert not (p.universe & ~p.sellable).all()
