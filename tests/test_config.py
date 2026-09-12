"""Configuration: YAML round-tripping and the experiment override mechanism."""
from __future__ import annotations

from pathlib import Path

import pytest

from e2e_portfolio.config import Config, FoldConfig
from e2e_portfolio.experiments import REGISTRY, experiment_config, investable_slot_bound, n_max_slots


def test_default_config_is_internally_consistent():
    cfg = Config()
    assert cfg.opt.alpha < 1.0
    assert 0.0 < cfg.opt.y_max <= 1.0
    assert cfg.train.batch_size >= 1
    # 0 is the documented "keep every index member" default of the liquidity filter
    assert cfg.data.top_liquidity >= 0


def test_yaml_round_trip_preserves_every_field(tmp_path):
    cfg = Config()
    cfg.data.index = "csi500"
    cfg.opt.gamma = 0.37
    cfg.model.num_scenarios = 42
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    back = Config.from_yaml(path)
    assert isinstance(back.data, type(cfg.data)), "nested sections must be dataclasses"
    assert isinstance(back.opt, type(cfg.opt))
    assert back.data.index == "csi500"
    assert abs(back.opt.gamma - 0.37) < 1e-12
    assert back.model.num_scenarios == 42


def test_replace_applies_nested_overrides_without_mutating_the_base():
    base = Config()
    base.opt.gamma = 0.5
    derived = base.replace(opt={"gamma": 0.9}, loss={"l_tail": 0.0})
    assert abs(base.opt.gamma - 0.5) < 1e-12, "the base config was mutated"
    assert abs(derived.opt.gamma - 0.9) < 1e-12
    assert derived.loss.l_tail == 0.0
    # everything else is carried over
    assert derived.data.index == base.data.index
    assert derived.opt.alpha == base.opt.alpha


def test_shipped_config_loads_and_has_the_expected_folds():
    path = Path(__file__).resolve().parents[1] / "configs" / "csi300.yaml"
    cfg = Config.from_yaml(path)
    assert cfg.data.index == "csi300"
    assert len(cfg.folds) == 3
    for f in cfg.folds:
        assert f.train_end < f.val_start <= f.val_end < f.test_start
    assert cfg.experiments, "the shipped config must declare an experiment list"
    for name in cfg.experiments:
        assert name in REGISTRY, f"{name} is not a registered experiment"
    # the frozen reference grid feeds the 17 price/volume channels only
    assert cfg.features.include_daily_return is False


def test_full_universe_config_keeps_every_index_member():
    """Plan §5's second stage: no liquidity truncation at all."""
    path = Path(__file__).resolve().parents[1] / "configs" / "csi300_full_universe.yaml"
    cfg = Config.from_yaml(path)
    assert cfg.data.index == "csi300"
    assert cfg.data.top_liquidity == 0, "the full-universe run must not filter on liquidity"
    assert cfg.run_name
    for name in cfg.experiments:
        assert name in REGISTRY


def test_extra_return_config_switches_the_plan_feature_list_on():
    path = Path(__file__).resolve().parents[1] / "configs" / "csi300_extra_returns.yaml"
    cfg = Config.from_yaml(path)
    assert cfg.features.include_daily_return is True
    assert cfg.features.lookback == 60
    for name in cfg.experiments:
        assert name in REGISTRY


def test_experiment_overrides_only_touch_their_own_knobs():
    base = Config()
    full = experiment_config(base, "full")
    nocost = experiment_config(base, "full_nocost")
    assert full.opt.lam_turnover == base.opt.lam_turnover
    assert nocost.opt.lam_turnover == 0.0
    assert nocost.loss.l_turnover == 0.0
    assert nocost.loss.l_tail == base.loss.l_tail

    pred = experiment_config(base, "lstm_topk")
    assert pred.model.selection == "topk"
    assert pred.loss.l_pred == 1.0
    assert pred.loss.l_decision == 0.0

    e2e = REGISTRY["e2e_noselect"]
    assert e2e.e2e and e2e.overrides["model"]["selection"] == "identity"


def test_n_max_slots_has_room_for_the_liquidity_buffer():
    cfg = Config()
    cfg.data.top_liquidity = 120
    assert n_max_slots(cfg) > cfg.data.top_liquidity


def test_n_max_slots_sizes_itself_from_the_periods_it_is_given():
    """A fixed +N margin can overflow once halted holdings are added on top."""
    from types import SimpleNamespace

    cfg = Config()
    cfg.data.top_liquidity = 120
    periods = [SimpleNamespace(n_universe=n) for n in (110, 141, 120)]
    assert n_max_slots(cfg, periods) == 141
    # never smaller than ``top_liquidity``, even for an easy fold
    assert n_max_slots(cfg, [SimpleNamespace(n_universe=5)]) == 120


def test_investable_slot_bound_counts_stuck_holdings_outside_the_universe():
    """Names held while unsellable stay investable even when they leave the universe."""
    from types import SimpleNamespace

    import numpy as np

    dataset = SimpleNamespace(panel=SimpleNamespace(num_tickers=4))
    p0 = SimpleNamespace(
        universe=np.array([True, True, False, False]),
        sellable=np.array([True, True, True, True]),
    )
    # name 0 is held from ``p0``, is unobserved (hence unsellable) in ``p1`` and
    # therefore must still get a slot next to the single name ``p1`` selects
    p1 = SimpleNamespace(
        universe=np.array([False, False, False, True]),
        sellable=np.array([False, True, True, True]),
    )
    assert investable_slot_bound(dataset, [p0, p1]) == 2
    assert investable_slot_bound(dataset, [p1]) == 1


def test_the_full_model_runs_the_layer_end_to_end():
    assert REGISTRY["full"].e2e is True
    assert REGISTRY["full"].overrides == {}
    assert REGISTRY["ew"].e2e is False
    assert REGISTRY["meancvar_hist"].e2e is False


def test_ablation_experiments_differ_from_the_full_model_in_exactly_one_knob():
    base = Config()
    full = experiment_config(base, "full")
    for name in ("full_nocvar_loss", "full_nocost", "full_nosparse"):
        alt = experiment_config(base, name)
        diff = []
        for section in ("data", "features", "model", "opt", "loss", "train", "backtest"):
            a, b = getattr(full, section), getattr(alt, section)
            for field in vars(a):
                if getattr(a, field) != getattr(b, field):
                    diff.append(f"{section}.{field}")
        assert 0 < len(diff) <= 2, f"{name} changes {diff}"


def test_fold_config_ordering_is_validated():
    with pytest.raises(ValueError):
        FoldConfig(
            name="bad",
            train_start="2018-01-01",
            train_end="2019-12-31",
            val_start="2019-01-01",  # overlaps the training window
            val_end="2019-06-30",
            test_start="2020-01-01",
            test_end="2020-12-31",
        ).validate()
