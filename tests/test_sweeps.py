"""Registries and helpers behind the reference-figure sweeps.

The sweeps that reproduce ``D:\\figures`` are declared once in
:mod:`e2e_portfolio.experiments` and then sliced by the YAML configs, so the
README's job counts (105 target-return jobs, 75 entropy jobs, 12 tree jobs) only
stay true if the registry and the config files agree.  These tests pin that
agreement down, and cover the tree-baseline feature/score helpers that no other
test touches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from e2e_portfolio.config import Config
from e2e_portfolio.experiments import (
    ENTROPY_LAMBDAS,
    ENTROPY_SWEEP,
    MU_SWEEP,
    MU_TARGETS,
    SWEEP_ARCHS,
    TREE_BASELINES,
    compact_tree_features,
    entropy_experiment_name,
    experiment_config,
    mu_experiment_name,
    prev_slots,
    spec_for,
)
from e2e_portfolio.models import score_to_pi_cap
from e2e_portfolio.selection import build_selection

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #
def test_target_return_sweep_is_the_full_architecture_by_target_grid():
    assert len(MU_TARGETS) == 7 and MU_TARGETS[0] == 0.016 and MU_TARGETS[-1] == 0.028
    assert set(MU_SWEEP) == {
        mu_experiment_name(mu, arch) for mu in MU_TARGETS for arch in SWEEP_ARCHS
    }
    for name in MU_SWEEP:
        spec = spec_for(name)
        assert spec.kind == "nn" and spec.e2e is True and spec.group == "mu_sweep"
        tag, arch = name.split("_")
        assert spec.overrides == {
            "model": {"arch": arch},
            "opt": {"mu_target": round(int(tag[2:]) / 1000, 3)},
        }


def test_target_return_sweep_reaches_the_configuration():
    base = Config()
    for name, mu in (("mu016_lstm", 0.016), ("mu022_gru", 0.022), ("mu028_rbfn", 0.028)):
        cfg = experiment_config(base, name)
        assert cfg.opt.mu_target == pytest.approx(mu)
        assert cfg.model.arch == name.split("_")[1]
    # the default model is unchanged when no sweep override is applied
    assert Config().opt.mu_target is None


def test_entropy_sweep_covers_every_regularisation_level_and_architecture():
    assert ENTROPY_LAMBDAS == (0.0, 0.001, 0.002, 0.005, 0.010)
    assert set(ENTROPY_SWEEP) == {
        entropy_experiment_name(lam, arch)
        for lam in ENTROPY_LAMBDAS
        for arch in SWEEP_ARCHS
    }
    for name in ENTROPY_SWEEP:
        spec = spec_for(name)
        assert spec.kind == "nn" and spec.e2e is True and spec.group == "entropy_sweep"
        tag, arch = name.split("_")
        lam = round(int(tag[3:]) / 10_000, 4)
        assert spec.overrides == {"model": {"arch": arch}, "opt": {"lam_div": lam}}
        assert experiment_config(Config(), name).opt.lam_div == pytest.approx(lam)


def test_tree_baselines_keep_the_layer_and_only_swap_the_forecaster():
    assert len(TREE_BASELINES) == 4
    estimators = {spec_for(n).estimator for n in TREE_BASELINES}
    assert estimators == {"adaboost", "xgboost"}
    targets = sorted({spec_for(n).overrides["opt"]["mu_target"] for n in TREE_BASELINES})
    assert targets == [0.020, 0.022]
    for name in TREE_BASELINES:
        spec = spec_for(name)
        assert spec.kind == "tree" and spec.group == "tree_baseline"
        assert spec.estimator in {"adaboost", "xgboost"}
        assert set(spec.overrides) == {"opt"}  # only tau changes
        # trees are fitted outside the graph, the layer itself stays non-differentiable
        assert spec.e2e is False
        cfg = experiment_config(Config(), name)
        assert cfg.opt.mu_target in (0.020, 0.022)


SWEEP_CONFIGS = {
    "csi300_mu_sweep.yaml": 21,
    "csi300_mu_sweep_ext.yaml": 14,
    "csi300_entropy_sweep.yaml": 15,
    "csi300_entropy_sweep_ext.yaml": 10,
    "csi300_tree_baselines.yaml": 4,
    "sse50_reference.yaml": 4,
    "csi500_reference.yaml": 4,
}


@pytest.mark.parametrize("config_file,expected_jobs", sorted(SWEEP_CONFIGS.items()))
def test_shipped_configs_only_use_registered_experiments(config_file, expected_jobs):
    cfg = Config.from_yaml(ROOT / "configs" / config_file)
    assert len(cfg.experiments) == expected_jobs  # README job counts x 3 folds
    assert len(set(cfg.experiments)) == expected_jobs
    for name in cfg.experiments:
        spec_for(name)  # raises for a typo or a stale name


def test_the_two_halves_of_each_sweep_are_disjoint_and_cover_the_registry():
    """The recurrent half plus the non-recurrent half = the whole sweep."""
    mu_main = Config.from_yaml(ROOT / "configs" / "csi300_mu_sweep.yaml").experiments
    mu_ext = Config.from_yaml(ROOT / "configs" / "csi300_mu_sweep_ext.yaml").experiments
    assert not set(mu_main) & set(mu_ext)
    assert set(mu_main) | set(mu_ext) == set(MU_SWEEP)
    assert len(mu_main) == 21 and len(mu_ext) == 14

    ent_main = Config.from_yaml(ROOT / "configs" / "csi300_entropy_sweep.yaml").experiments
    ent_ext = Config.from_yaml(
        ROOT / "configs" / "csi300_entropy_sweep_ext.yaml"
    ).experiments
    assert not set(ent_main) & set(ent_ext)
    assert set(ent_main) | set(ent_ext) == set(ENTROPY_SWEEP)
    assert len(ent_main) == 15 and len(ent_ext) == 10


@pytest.mark.parametrize(
    "config_file,index,size",
    [("sse50_reference.yaml", "sse50", 50), ("csi500_reference.yaml", "csi500", 120)],
)
def test_cross_market_configs_only_change_the_universe(config_file, index, size):
    cfg = Config.from_yaml(ROOT / "configs" / config_file)
    base = Config()
    assert cfg.data.root == base.data.root       # same local dataset
    assert cfg.data.index == index               # only the panel is swapped
    assert cfg.data.top_liquidity == size
    assert cfg.data.exclude_st == base.data.exclude_st
    assert cfg.model == base.model and cfg.opt == base.opt and cfg.loss == base.loss
    assert cfg.run_name.startswith(index)        # runs never collide
    assert set(cfg.experiments) == {"ew", "meancvar_hist", "mu020_lstm", "full"}


# --------------------------------------------------------------------------- #
# tree feature / score helpers
# --------------------------------------------------------------------------- #
def test_compact_tree_features_summarises_last_day_mean_and_observed_fraction():
    g = np.random.default_rng(0)
    n_features = 3
    win = g.normal(size=(5, 4, n_features))
    msk = np.ones((5, 4), dtype=bool)
    X = compact_tree_features(win, msk)
    assert X.shape == (4, 2 * n_features + 1)
    np.testing.assert_allclose(X[:, :n_features], win[-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        X[:, n_features:2 * n_features], win.mean(axis=0), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(X[:, 2 * n_features], 1.0)


def test_compact_tree_features_ignore_padded_days_and_survive_empty_names():
    win = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)
    msk = np.zeros((2, 3), dtype=bool)
    msk[0, 0] = True          # only day 0 observed
    msk[:, 1] = True          # name 1 fully observed
    # name 2 never observed -> must come out as an all-zero row, not NaN
    X = compact_tree_features(win, msk)
    assert X.shape == (3, 5)
    assert np.isfinite(X).all()
    np.testing.assert_allclose(X[0, :2], 0.0)                  # last day is padding
    np.testing.assert_allclose(X[0, 2:4], win[0, 0])            # mean over observed
    assert X[0, 4] == pytest.approx(0.5)
    np.testing.assert_allclose(X[2], 0.0)

    extreme = np.full((1, 1, 1), 1e6)
    assert compact_tree_features(extreme, np.ones((1, 1), dtype=bool))[0, 0] == 10.0


def test_prev_slots_pack_the_investable_names_into_the_first_slots():
    prev = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    out = prev_slots(prev, np.array([1, 3, 4]), 5)
    np.testing.assert_allclose(out, [0.2, 0.4, 0.5, 0.0, 0.0])


def test_score_to_pi_cap_matches_the_layer_convention_for_flat_scores():
    """One-dimensional scores in, one-dimensional ``(pi, cap)`` out."""
    cfg = Config()
    cfg.model.selection = "sparsemax"
    selection = build_selection(cfg.model.selection, cfg.model)
    score = torch.tensor([3.0, 2.0, 1.0, -5.0, 4.0, 4.0], dtype=torch.float64)
    valid = torch.tensor([True, True, True, True, False, False])
    floor = torch.zeros(6, dtype=torch.float64)
    floor[0] = 0.4  # a name that is suspended and therefore cannot be sold

    pi, cap = score_to_pi_cap(score, valid, selection, cfg.opt, floor)
    assert pi.shape == (6,) and cap.shape == (6,)
    assert float(pi[~valid].detach().abs().sum()) == 0.0
    assert float(pi.detach().sum()) == pytest.approx(1.0)

    # the cap of a blocked name is lifted to its floor, the others are unchanged
    _, cap_free = score_to_pi_cap(score, valid, selection, cfg.opt, None)
    assert float(cap[0]) == pytest.approx(max(float(cap_free[0]), 0.4))
    assert float(cap[1:].sum()) == pytest.approx(float(cap_free[1:].sum()))
    assert float((cap_free >= 0).all())


def test_score_to_pi_cap_falls_back_to_equal_weights_without_valid_names():
    cfg = Config()
    cfg.model.selection = "softmax"
    selection = build_selection(cfg.model.selection, cfg.model)
    score = torch.zeros(4, dtype=torch.float64)
    valid = torch.tensor([True, True, False, False])
    pi, cap = score_to_pi_cap(score, valid, selection, cfg.opt, None)
    np.testing.assert_allclose(pi.detach().numpy(), [0.5, 0.5, 0.0, 0.0])
    assert float(cap[2:].abs().sum()) == 0.0
    assert float(pi.detach().sum()) == pytest.approx(1.0)
