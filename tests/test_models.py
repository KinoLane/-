"""Encoder heads, the scenario grid and the two selection-score modes.

The plan's module 3 fixes the selection score as a risk-adjusted function of the
predictions, ``s_i = mu_i - l_sigma * sigma_i - l_cvar * CVaR_i``, while the main
grid learns it with a dedicated head.  Both paths must produce exactly what they
claim to, and ``mu_sigma`` must not leave an unused head in the state dict.
"""
from __future__ import annotations

import pytest
import torch

from e2e_portfolio.config import Config
from e2e_portfolio.models import (
    E2EPortfolioModel,
    StockEncoder,
    gaussian_quantile_grid,
    inverse_softplus,
    scenario_cvar,
)


def _model_cfg(score_mode: str, n_scen: int = 60):
    cfg = Config().model
    cfg.hidden_size = 8
    cfg.head_hidden = 8
    cfg.dropout = 0.0
    cfg.num_scenarios = n_scen
    cfg.selection = "sparsemax"
    cfg.score_mode = score_mode
    return cfg


def _model(score_mode: str, n_features: int = 5, n_scen: int = 60):
    return E2EPortfolioModel(
        n_features, _model_cfg(score_mode, n_scen), Config().opt, opt_layer=None
    )


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def test_gaussian_grid_is_symmetric_and_monotone():
    z = gaussian_quantile_grid(60)
    assert z.shape == (60,)
    assert torch.all(z.diff() > 0)
    assert abs(float(z.mean())) < 1e-6
    assert float(z[29]) < 0 < float(z[30])  # the median straddles zero
    assert float(z[-1]) == pytest.approx(2.394, abs=1e-3)


def test_scenario_cvar_is_the_mean_of_the_worst_tail():
    scen = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    # (1 - 0.9) * 20 = 2 worst scenarios (0 and 1); sign flipped -> -0.5
    assert float(scenario_cvar(scen, 0.9)) == pytest.approx(-0.5)

    narrow = torch.linspace(-1.0, 1.0, 20).unsqueeze(1)
    wide = torch.linspace(-3.0, 3.0, 20).unsqueeze(1)
    assert float(scenario_cvar(wide, 0.9)) > float(scenario_cvar(narrow, 0.9))


def test_scenario_cvar_tail_count_is_ceiled_without_float_noise():
    scen = torch.zeros(20, 1)
    scen[:3, 0] = -1.0
    # (1 - 0.85) * 20 = 3 exactly, so the mean of the 3 worst scenarios is -1
    assert float(scenario_cvar(scen, 0.85)) == pytest.approx(1.0)

    s60 = torch.zeros(60, 1)
    s60[:4, 0] = -1.0
    # 0.05 * 60 = 3.0000000000000004 in binary floats: it must still be 3 days
    assert float(scenario_cvar(s60, 0.95)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# score_mode = "head" (the main grid)
# --------------------------------------------------------------------------- #
def test_head_mode_owns_a_learned_score_head():
    torch.manual_seed(0)
    model = _model("head")
    assert model.encoder.score_layer is not None
    assert any(k.startswith("score_layer") for k in model.encoder.state_dict())

    x = torch.randn(7, 4, 5)
    mask = torch.ones(7, 4, dtype=torch.bool)
    valid = torch.ones(7, dtype=torch.bool)
    valid[-1] = False
    with torch.no_grad():
        model.encoder.score_layer.weight.zero_()
        model.encoder.score_layer.bias.zero_()
    _, _, score, _ = model.predict(x, mask, valid)
    # a zeroed head really owns the score: it does *not* fall back to mu/sigma
    assert torch.allclose(score[valid], torch.zeros(7 - 1), atol=1e-7)


# --------------------------------------------------------------------------- #
# score_mode = "mu_sigma" (the plan's fixed rule)
# --------------------------------------------------------------------------- #
def test_mu_sigma_mode_implements_the_fixed_rule_exactly():
    torch.manual_seed(0)
    model = _model("mu_sigma")
    assert model.encoder.score_layer is None
    assert not any(k.startswith("score_layer") for k in model.encoder.state_dict())
    model.cfg.score_lambda_sigma = 0.3
    model.cfg.score_lambda_cvar = 0.7

    x = torch.randn(9, 4, 5)
    mask = torch.ones(9, 4, dtype=torch.bool)
    valid = torch.ones(9, dtype=torch.bool)
    valid[3:] = False
    mu, sigma, score, scen = model.predict(x, mask, valid)

    expected = (
        mu
        - 0.3 * sigma
        - 0.7 * scenario_cvar(scen, model.opt_cfg.alpha)
    )
    assert torch.allclose(score[valid], expected[valid], atol=1e-6)
    assert torch.all(score[~valid] <= -1e4 + 1.0)


def test_mu_sigma_score_penalises_volatility():
    torch.manual_seed(0)
    model = _model("mu_sigma")
    with torch.no_grad():
        model.encoder.mu_layer.weight.zero_()
        model.encoder.mu_layer.bias.zero_()
        model.encoder.sigma_layer.weight.zero_()
        model.encoder.sigma_layer.bias.fill_(inverse_softplus(0.05))

    # identical feature windows for every name -> identical mu, sigma and score
    x = torch.zeros(4, 3, 5)
    mask = torch.ones(4, 3, dtype=torch.bool)
    valid = torch.ones(4, dtype=torch.bool)
    mu, sigma, score, _ = model.predict(x, mask, valid)
    assert float(mu.std()) == pytest.approx(0.0, abs=1e-6)
    assert float(sigma.std()) == pytest.approx(0.0, abs=1e-6)
    assert float(sigma.mean()) == pytest.approx(0.05, abs=1e-5)
    assert float(score.std()) == pytest.approx(0.0, abs=1e-6)

    # the volatility penalty is applied with the configured weight
    model.cfg.score_lambda_sigma = float(model.cfg.score_lambda_sigma) + 1.0
    _, _, score_hi, _ = model.predict(x, mask, valid)
    assert float(score_hi.mean()) == pytest.approx(
        float(score.mean()) - 0.05, abs=1e-5
    )


def test_selection_weights_are_the_selector_applied_to_the_formula_score():
    torch.manual_seed(0)
    model = _model("mu_sigma")
    x = torch.randn(6, 4, 5)
    mask = torch.ones(6, 4, dtype=torch.bool)
    valid = torch.ones(6, dtype=torch.bool)
    valid[5] = False
    out = model(x, mask, valid, run_layer=False)

    formula = (
        out.mu
        - float(model.cfg.score_lambda_sigma) * out.sigma
        - float(model.cfg.score_lambda_cvar) * scenario_cvar(out.scenarios, model.opt_cfg.alpha)
    )
    # the same masking ``predict`` applies to invalid slots, then the selector
    score_ref = formula - (~valid).to(formula.dtype) * 1e4
    reference = model.selection(score_ref.unsqueeze(0)).squeeze(0)
    reference = reference * valid.to(reference.dtype)
    reference = reference / reference.sum()

    assert torch.allclose(out.pi, reference, atol=1e-7)
    assert float(out.pi[5]) == 0.0
    assert float(out.pi[valid].sum()) == pytest.approx(1.0, abs=1e-6)


def test_caps_are_feasible_in_both_modes():
    torch.manual_seed(0)
    x = torch.randn(6, 4, 5)
    mask = torch.ones(6, 4, dtype=torch.bool)
    valid = torch.ones(6, dtype=torch.bool)
    for mode in ("head", "mu_sigma"):
        out = _model(mode)(x, mask, valid, run_layer=False)
        assert float(out.cap.min()) >= 0.0
        assert float(out.cap.sum()) >= 1.0 - 1e-6


def test_unknown_score_mode_is_rejected():
    with pytest.raises(ValueError, match="score_mode"):
        StockEncoder(4, _model_cfg("head_and_musigma"))


# --------------------------------------------------------------------------- #
# registry wiring
# --------------------------------------------------------------------------- #
def test_registry_wires_both_score_modes():
    from e2e_portfolio.experiments import experiment_config, spec_for

    base = Config()
    assert base.model.score_mode == "head"  # the main grid is untouched
    assert experiment_config(base, "full").model.score_mode == "head"
    assert experiment_config(base, "lstm_musigma").model.score_mode == "mu_sigma"
    assert experiment_config(base, "lstm_musigma").model.selection == "sparsemax"
    assert experiment_config(base, "full_musigma").model.score_mode == "mu_sigma"
    assert spec_for("full_musigma").e2e is True
    assert spec_for("lstm_musigma").e2e is False
    # the plan-faithful baseline learns nothing but the prediction
    assert experiment_config(base, "lstm_musigma").loss.l_decision == 0.0
    assert experiment_config(base, "lstm_musigma").loss.l_tail == 0.0

    # ``full_musigma`` differs from ``full`` *only* in the score rule
    full = experiment_config(base, "full").to_dict()["model"]
    musigma = experiment_config(base, "full_musigma").to_dict()["model"]
    assert {k for k in full if full[k] != musigma[k]} == {"score_mode"}

# --------------------------------------------------------------------------- #
# encoder architectures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["lstm", "gru", "rnn", "mlp", "rbfn"])
def test_every_architecture_shares_one_encoder_and_produces_finite_predictions(arch):
    """Architecture variants are drop-in replacements for the LSTM encoder."""
    cfg = _model_cfg("head")
    cfg.arch = arch
    model = E2EPortfolioModel(3, cfg, Config().opt, opt_layer=None).double()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 4, 3, generator=g, dtype=torch.float64)
    mask = torch.ones(7, 4, dtype=torch.float64)
    mask[3, 2:] = 0.0  # one short history
    valid = torch.ones(7, dtype=torch.float64)
    mu, sigma, score, scen = model.predict(x, mask, valid)
    for t in (mu, sigma, score):
        assert t.shape == (7,)
        assert torch.isfinite(t).all()
    assert float(sigma.min()) > 0
    assert scen.shape == (cfg.num_scenarios, 7)


def test_non_recurrent_architectures_pool_only_over_observed_days():
    """Flipping the padded days of a name must not move its embedding."""
    for arch in ("mlp", "rbfn"):
        cfg = _model_cfg("head")
        cfg.arch = arch
        enc = StockEncoder(3, cfg).double().eval()
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1, 5, 3, generator=g, dtype=torch.float64)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        with torch.no_grad():
            a = enc(x, mask)[0]
            x2 = x.clone()
            x2[0, 3:] = 99.0  # padding content is irrelevant
            b = enc(x2, mask)[0]
        assert torch.allclose(a, b, atol=1e-12)


def test_unknown_architecture_is_rejected():
    cfg = _model_cfg("head")
    cfg.arch = "transformer"
    with pytest.raises(ValueError):
        StockEncoder(3, cfg)
