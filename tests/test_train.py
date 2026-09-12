"""End-to-end training plumbing on the synthetic dataset."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from e2e_portfolio.experiments import (
    REGISTRY,
    experiment_config,
    investable_slot_bound,
    n_max_slots,
    run_job,
)
from e2e_portfolio.metrics import HOLDING_EPS, concentration_metrics
from e2e_portfolio.selection import feasible_cap
from e2e_portfolio.train import Trainer, build_period_tensors, drift_weights


def _trainer(cfg, ds):
    return Trainer(cfg, ds, n_max=n_max_slots(cfg), n_features=len(ds.feature_names), verbose=False)


def test_period_tensors_are_padded_to_the_fixed_slot_count(synth_cfg, synth_dataset):
    ds = synth_dataset
    n_max = n_max_slots(synth_cfg)
    p = ds.periods[3]
    b = build_period_tensors(ds, p, np.zeros(ds.panel.num_tickers), n_max)
    n_used = len(b.compact)
    assert b.x.shape[0] == n_max
    assert b.prev.shape == (n_max,)
    assert b.floor.shape == (n_max,)
    assert int(b.valid.sum()) == n_used == p.n_universe
    assert b.daily.shape == (p.horizon, n_max)
    # padded slots must be inert
    assert not b.valid[n_used:].any()
    assert float(b.daily[:, n_used:].abs().sum()) == 0.0
    assert float(b.x[n_used:].abs().sum()) == 0.0
    assert (b.realised[~b.valid] == 0).all()


def test_padding_slots_are_inert_for_the_deterministic_forward_pass(synth_cfg, synth_dataset):
    """Extra slots exist only to fix the layer size; they must not move anything.

    Checked in ``eval`` mode, where the dropout mask no longer depends on the
    padded tensor size (during *training* the mask draws do consume RNG slots,
    which is why ``n_max`` must stay constant within an experiment grid).
    """
    ds = synth_dataset
    p = ds.periods[7]
    zeros = np.zeros(ds.panel.num_tickers)
    n_used = int(p.n_universe)
    small = build_period_tensors(ds, p, zeros, n_used)
    big = build_period_tensors(ds, p, zeros, n_used + 25)

    tr = _trainer(experiment_config(synth_cfg, "lstm_topk"), ds)
    tr.model.eval()
    with torch.no_grad():
        a = tr._solve(small, run_layer=False)
        b = tr._solve(big, run_layer=False)
    for key in ("mu", "sigma", "score", "pi", "cap"):
        x, y = getattr(a, key), getattr(b, key)
        assert x.shape[0] == n_used
        assert torch.allclose(x, y[:n_used], atol=1e-6), key
    assert torch.allclose(a.scenarios, b.scenarios[:, :n_used], atol=1e-6)


def test_stuck_holdings_get_a_floor_and_an_equal_cap(synth_dataset):
    ds = synth_dataset
    p = copy.deepcopy(ds.periods[5])
    prev = np.zeros(ds.panel.num_tickers)
    held = np.flatnonzero(p.universe)[:2]
    prev[held] = 0.5
    p.sellable[held[0]] = False  # cannot be sold -> must be kept
    # the sliding name is added *on top of* the universe, which is why the slot
    # count has to be derived instead of guessed
    n_max = int((p.universe | (prev > 0)).sum())
    b = build_period_tensors(ds, p, prev, n_max)
    j = int(np.flatnonzero(b.compact == held[0])[0])
    assert abs(float(b.floor[j]) - 0.5) < 1e-6
    assert abs(float(b.prev[j]) - 0.5) < 1e-6
    assert float(b.floor.sum()) <= 1.0 + 1e-9


def test_build_period_tensors_rejects_a_universe_larger_than_n_max(synth_dataset):
    ds = synth_dataset
    p = ds.periods[4]
    try:
        build_period_tensors(ds, p, np.zeros(ds.panel.num_tickers), 3)
    except ValueError as exc:
        assert "n_max" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError for an undersized slot count")


def test_weight_drift_matches_the_backtester(synth_dataset):
    ds = synth_dataset
    p = ds.periods[6]
    n = ds.panel.num_tickers
    w = np.zeros(n)
    w[:3] = 1.0 / 3
    got = drift_weights(w, p)
    step = w.copy()
    for h in range(p.daily_ret.shape[0]):
        step = step * (1.0 + p.daily_ret[h])
        step = step / step.sum()
    assert np.allclose(got, step, atol=1e-12)
    assert abs(got.sum() - 1.0) < 1e-12


def test_trainer_produces_a_valid_portfolio(synth_cfg, synth_dataset):
    ds = synth_dataset
    cfg = experiment_config(synth_cfg, "full")
    cfg.train.epochs = 1
    tr = _trainer(cfg, ds)
    periods = ds.periods_between("2018-06-01", "2018-09-30", pad_horizon=False)
    assert periods
    terms, prev = tr.run_epoch(periods, train=False)
    assert "L_total" in terms and np.isfinite(terms["L_total"])
    w, info = tr.weights_for(periods[-1], np.zeros(ds.panel.num_tickers))
    assert w.shape == (ds.panel.num_tickers,)
    assert w.min() >= -1e-9
    assert abs(w.sum() - 1.0) < 5e-3
    # the synthetic universe is small, so the hard cap is relaxed to the smallest
    # value that still allows sum(y) = 1 (see ``selection.feasible_cap``)
    limit = feasible_cap(cfg.opt.y_max, periods[-1].n_universe)
    assert w.max() <= limit + 5e-3
    assert tr.layer_failures == 0


def test_gradients_flow_through_the_layer_into_the_encoder(synth_cfg, synth_dataset):
    ds = synth_dataset
    cfg = experiment_config(synth_cfg, "full")
    tr = _trainer(cfg, ds)
    b = build_period_tensors(ds, ds.periods[7], np.zeros(ds.panel.num_tickers), n_max_slots(cfg))
    tr.model.train()
    tr.optimizer.zero_grad(set_to_none=True)
    loss, y, _ = tr._forward_loss(b)
    if y is None:
        import pytest

        pytest.skip("layer fell back on the synthetic problem")
    loss.total.backward()
    encoder_grads = [
        p.grad for n, p in tr.model.named_parameters() if n.startswith(("encoder", "head"))
    ]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in encoder_grads), (
        "no gradient reached the network -- the layer is not differentiable"
    )


def test_predict_only_variant_does_not_run_the_layer_in_training(synth_cfg, synth_dataset):
    ds = synth_dataset
    cfg = experiment_config(synth_cfg, "lstm_sparsemax")
    tr = _trainer(cfg, ds)
    assert tr.use_layer is False
    terms, _ = tr.run_epoch(ds.periods_between("2018-03-01", "2018-06-30"), train=True)
    assert set(terms) <= {"L_pred", "L_sparse", "L_total"}


def test_early_stopping_restores_the_best_epoch(synth_cfg, synth_dataset):
    ds = synth_dataset
    cfg = experiment_config(synth_cfg, "lstm_softmax")
    cfg.train.epochs = 4
    cfg.train.patience = 1
    cfg.train.min_epochs = 1
    tr = _trainer(cfg, ds)
    hist = tr.fit(
        ds.periods_between("2018-01-01", "2018-06-30", pad_horizon=False),
        ds.periods_between("2018-07-01", "2018-08-31", pad_horizon=False),
    )
    assert hist.best_epoch >= 0
    assert len(hist.epochs) <= cfg.train.epochs
    assert np.isfinite(hist.best_val)


def test_every_registered_experiment_runs_on_one_fold(synth_cfg, synth_dataset, tmp_path):
    """Smoke-run the whole grid on the synthetic data (short windows)."""
    import copy

    from e2e_portfolio.config import FoldConfig

    cfg = copy.deepcopy(synth_cfg)
    cfg.train.epochs = 1
    cfg.train.patience = 1
    cfg.train.min_epochs = 1
    fold = FoldConfig(
        name="synth_fold",
        train_start="2018-01-01",
        train_end="2018-09-30",
        val_start="2018-10-01",
        val_end="2018-11-30",
        test_start="2018-12-01",
        test_end="2019-06-30",
    ).validate()

    ds = synth_dataset
    assert ds.periods, "synthetic dataset produced no rebalancing periods"
    for name in REGISTRY:
        metrics = run_job(cfg, name, fold, dataset=ds, out_root=tmp_path, verbose=False)
        assert metrics["experiment"] == name
        assert np.isfinite(metrics["ann_return"])
        assert (tmp_path / name / fold.name / "metrics.json").exists()
        assert (tmp_path / name / fold.name / "returns.csv").exists()
        # every headline number must be the *net-of-cost* one; the gross pass is
        # reported separately with a ``_gross`` suffix
        assert metrics["ann_return"] <= metrics["ann_return_gross"] + 1e-12
        assert abs(
            metrics["cost_drag_ann"]
            - (metrics["ann_return_gross"] - metrics["ann_return"])
        ) < 1e-12
        assert "info_ratio" in metrics, "the net pass must be the one given the benchmark"
        # the layer's slot budget must cover the largest investable set actually
        # simulated -- halted holdings are added on top of the liquid universe,
        # so a fixed ``top_liquidity + margin`` budget is not enough
        assert metrics["n_max"] >= metrics["max_investable"]
        seen = (
            ds.periods_between(fold.train_start, fold.train_end, pad_horizon=False)
            + ds.periods_between(fold.val_start, fold.val_end, pad_horizon=False)
            + ds.periods_between(fold.test_start, fold.test_end, pad_horizon=True)
        )
        assert metrics["n_max"] == n_max_slots(cfg, seen, dataset=ds)
        assert metrics["max_investable"] == investable_slot_bound(ds, seen)
        # the plan asks for holdings *and* concentration; both must be present and
        # must agree with the weight matrix the job wrote to disk
        for key in ("hhi_mean", "top5_weight_mean", "max_weight_mean"):
            assert key in metrics and np.isfinite(metrics[key])
        weights = pd.read_csv(tmp_path / name / fold.name / "weights.csv", index_col=0)
        expected = concentration_metrics(weights.to_numpy())
        for key, value in expected.items():
            assert abs(metrics[key] - value) < 1e-12
        reb = pd.read_csv(tmp_path / name / fold.name / "rebalance.csv")
        for col in ("hhi", "top5_weight", "max_weight"):
            assert col in reb.columns and len(reb[col]) == len(reb)
        # ``hhi >= 1 / n_holdings`` per period, but both sides are averages here,
        # so allow the ~1e-9 slack the conic solver leaves on ``sum(y) == 1``
        # (measured: the tightest real run is +9.2e-06).
        assert metrics["hhi_mean"] >= 1.0 / max(metrics["holdings_mean"], 1.0) - 1e-8, (
            f"{name}: hhi_mean={metrics['hhi_mean']!r} holdings_mean={metrics['holdings_mean']!r}"
        )
        # only economic positions count: the 1e-8..1e-6 residuals the conic solver
        # leaves on the rejected names must not be reported as holdings
        assert abs(
            metrics["holdings_mean"]
            - float((weights.to_numpy() > HOLDING_EPS).sum(axis=1).mean())
        ) < 1e-12
        assert metrics["holdings_mean"] <= metrics["max_universe"]
