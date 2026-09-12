"""The composite training objective."""
from __future__ import annotations

import numpy as np
import torch

from e2e_portfolio.config import LossConfig
from e2e_portfolio.losses import (
    composite_loss,
    portfolio_return,
    prediction_loss,
    sparsity_loss,
    tail_loss,
    turnover_loss,
)


def _batch(n=6, h=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, generator=g, requires_grad=False) * 0.02
    sigma = torch.full((n,), 0.06) + 0.01 * torch.rand(n, generator=g)
    realised = torch.randn(n, generator=g) * 0.07
    daily = torch.randn(h, n, generator=g) * 0.02
    valid = torch.ones(n, dtype=torch.bool)
    pi = torch.full((n,), 1.0 / n)
    y = torch.full((n,), 1.0 / n, requires_grad=False)
    prev = torch.zeros(n)
    return mu, sigma, realised, daily, valid, pi, y, prev


def test_prediction_loss_ignores_padded_slots():
    """Zero-padded slots must not shift the NLL (their sigma is zero)."""
    mu, sigma, realised, _, valid, *_ = _batch(n=6)
    mu_ext = torch.cat([mu, torch.zeros(4)])
    sigma_ext = torch.cat([sigma, torch.zeros(4)])
    realised_ext = torch.cat([realised, torch.zeros(4)])
    valid_ext = torch.cat([valid, torch.zeros(4, dtype=torch.bool)])
    a = prediction_loss(mu, sigma, realised, valid)
    b = prediction_loss(mu_ext, sigma_ext, realised_ext, valid_ext)
    assert torch.allclose(a, b, atol=1e-6)


def test_prediction_loss_is_minimised_at_the_realised_value():
    mu, sigma, realised, _, valid, *_ = _batch()
    good = prediction_loss(realised.clone().requires_grad_(True), sigma, realised, valid)
    bad = prediction_loss(realised + 0.05, sigma, realised, valid)
    assert float(good) < float(bad)


def test_prediction_loss_mse_variant_is_normalised():
    mu, sigma, realised, _, valid, *_ = _batch()
    zero = prediction_loss(torch.zeros_like(mu), sigma, torch.zeros_like(realised), valid, "mse")
    assert float(zero) == 0.0


def test_portfolio_return_is_linear_in_the_weights():
    _, _, realised, *_ = _batch()
    w = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert abs(float(portfolio_return(w, realised)) - 0.5 * float(realised[:2].sum())) < 1e-6


def test_tail_loss_is_the_mean_of_the_worst_days():
    daily = torch.zeros(100, 2)
    daily[:5, 0] = -0.10  # five bad days for asset 0
    y = torch.tensor([1.0, 0.0])
    # with alpha = 0.95 the worst 5 of 100 days are averaged
    assert abs(float(tail_loss(y, daily, 0.95)) - 0.10) < 1e-6


def test_tail_loss_is_convex_in_the_weights():
    """CVaR of a linear function of y is convex -- the layer relies on this."""
    g = torch.Generator().manual_seed(2)
    daily = torch.randn(60, 4, generator=g) * 0.03
    y1 = torch.tensor([0.7, 0.1, 0.1, 0.1])
    y2 = torch.tensor([0.1, 0.1, 0.1, 0.7])
    mid = 0.5 * (y1 + y2)
    lhs = float(tail_loss(mid, daily, 0.9))
    rhs = 0.5 * (float(tail_loss(y1, daily, 0.9)) + float(tail_loss(y2, daily, 0.9)))
    assert lhs <= rhs + 1e-9


def test_turnover_loss_counts_the_full_switch():
    prev = torch.tensor([1.0, 0.0])
    new = torch.tensor([0.0, 1.0])
    assert abs(float(turnover_loss(new, prev)) - 2.0) < 1e-6
    assert float(turnover_loss(prev, prev)) == 0.0


def test_sparsity_loss_rewards_a_concentrated_selection():
    """``L_sparse = PR / N``: 1 for a uniform selection, ``1/N`` for a single name.

    Minimising it therefore concentrates the selection -- the "sparse" pressure
    the study needs -- while the value stays scale-free in ``[1/N, 1]``.
    """
    n = 25
    pi = torch.full((n,), 1.0 / n)
    assert abs(float(sparsity_loss(pi, n)) - 1.0) < 1e-6
    concentrated = torch.zeros(n)
    concentrated[0] = 1.0
    assert abs(float(sparsity_loss(concentrated, n)) - 1.0 / n) < 1e-6
    half = torch.zeros(n)
    half[: n // 2] = 2.0 / n
    assert float(sparsity_loss(half, n)) < float(sparsity_loss(pi, n))


def test_composite_loss_equals_the_weighted_sum_of_its_terms():
    mu, sigma, realised, daily, valid, pi, y, prev = _batch()
    cfg = LossConfig()
    out = composite_loss(
        cfg, mu=mu, sigma=sigma, realised=realised, daily=daily, valid=valid,
        pi=pi, y=y, y_prev=prev, alpha=0.95, n_candidates=6,
    )
    expect = (
        cfg.l_pred * out.terms["L_pred"]
        + cfg.l_decision * out.terms["L_decision"]
        + cfg.l_tail * out.terms["L_tail"]
        + cfg.l_turnover * out.terms["L_turnover"]
        + cfg.l_sparse * out.terms["L_sparse"]
    )
    assert abs(float(out.total) - expect) < 1e-5


def test_composite_loss_drops_the_terms_of_the_predict_only_variant():
    mu, sigma, realised, daily, valid, pi, y, prev = _batch()
    cfg = LossConfig(l_pred=1.0, l_decision=0.0, l_tail=0.0, l_turnover=0.0, l_sparse=0.0)
    out = composite_loss(
        cfg, mu=mu, sigma=sigma, realised=realised, daily=daily, valid=valid,
        pi=pi, y=y, y_prev=prev, alpha=0.95, n_candidates=6,
    )
    assert set(out.terms) == {"L_pred", "L_total"}
    assert abs(float(out.total) - float(out.terms["L_pred"])) < 1e-6


def test_composite_loss_is_differentiable_in_mu():
    mu, sigma, realised, daily, valid, pi, y, prev = _batch()
    mu = mu.clone().requires_grad_(True)
    cfg = LossConfig()
    out = composite_loss(
        cfg, mu=mu, sigma=sigma, realised=realised, daily=daily, valid=valid,
        pi=pi, y=y, y_prev=prev, alpha=0.95, n_candidates=6,
    )
    out.total.backward()
    assert mu.grad is not None and float(mu.grad.abs().sum()) > 0
