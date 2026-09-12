"""The differentiable Mean-CVaR layer: feasibility, optimality and gradients.

The most important test here is :func:`test_layer_gradient_matches_finite_differences`:
it is the evidence that ``cvxpylayers`` really back-propagates through the convex
program on this machine (the whole point of the end-to-end study).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from e2e_portfolio.optlayer import MeanCVaRLayer

TOL = 5e-3


def _problem(n=12, s=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = (torch.randn(1, n, generator=g) * 0.02).double()
    scen = (torch.randn(1, s, n, generator=g) * 0.06).double()
    cap = torch.full((1, n), 0.20, dtype=torch.float64)
    prev = torch.zeros(1, n, dtype=torch.float64)
    floor = torch.zeros(1, n, dtype=torch.float64)
    return mu, scen, cap, prev, floor


def _layer(**kw):
    kw.setdefault("n_assets", 12)
    kw.setdefault("n_scenarios", 20)
    kw.setdefault("gamma", 0.1)
    kw.setdefault("lam_turnover", 0.0015)
    kw.setdefault("lam_div", 0.002)
    kw.setdefault("use_entropy", True)
    return MeanCVaRLayer(**kw)


def test_weights_are_a_probability_vector_respecting_the_caps():
    mu, scen, cap, prev, floor = _problem()
    out = _layer()(mu, scen, cap, prev, floor)
    y = out["weights"]
    assert y.shape == (1, 12)
    assert abs(float(y.sum()) - 1.0) < TOL
    assert float(y.min()) > -TOL
    assert float(y.max()) <= 0.20 + TOL


def test_floor_is_honoured_even_when_it_exceeds_the_selection_cap():
    """A stuck holding (cap raised to the floor) must remain feasible."""
    mu, scen, cap, prev, floor = _problem()
    floor[0, 0] = 0.45
    cap[0, 0] = 0.45
    out = _layer()(mu, scen, cap, prev, floor)
    assert float(out["weights"][0, 0]) >= 0.45 - TOL


def test_turnover_penalty_reduces_the_distance_to_the_previous_weights():
    mu, scen, cap, prev, floor = _problem()
    prev[0, :4] = 0.25
    free = _layer(lam_turnover=0.0)(mu, scen, cap, prev, floor)["weights"]
    sticky = _layer(lam_turnover=0.5)(mu, scen, cap, prev, floor)["weights"]
    d_free = float((free - prev).abs().sum())
    d_sticky = float((sticky - prev).abs().sum())
    assert d_sticky < d_free, "a larger turnover penalty must trade less"


def test_gamma_controls_the_risk_aversion():
    """A larger gamma must not increase the realised CVaR of the solution."""
    mu, scen, cap, prev, floor = _problem(seed=3)
    alpha = 0.95

    def cvar(y):
        r = (scen[0] @ y[0]).detach().numpy()
        k = max(1, int(np.ceil((1 - alpha) * len(r))))
        return float(-np.sort(r)[:k].mean())

    low = _layer(gamma=0.01)(mu, scen, cap, prev, floor)["weights"]
    high = _layer(gamma=1.0)(mu, scen, cap, prev, floor)["weights"]
    assert cvar(high) <= cvar(low) + TOL


def test_entropy_variant_is_at_least_as_diversified_as_the_quadratic_one():
    mu, scen, cap, prev, floor = _problem(seed=5)
    ent = _layer(lam_div=0.01, use_entropy=True)(mu, scen, cap, prev, floor)["weights"]
    quad = _layer(lam_div=0.01, use_entropy=False)(mu, scen, cap, prev, floor)["weights"]
    pr_ent = 1.0 / float((ent**2).sum())
    pr_quad = 1.0 / float((quad**2).sum())
    assert pr_ent >= pr_quad - 0.5


def test_solver_failure_is_surfaced_not_silently_swallowed():
    mu, scen, cap, prev, floor = _problem()
    cap[0] = 0.0  # sum(cap) = 0 < 1 -> provably infeasible
    with pytest.raises(Exception):
        _layer()(mu, scen, cap, prev, floor)


def test_layer_gradient_matches_finite_differences():
    """d(loss)/d(mu) from cvxpylayers vs. a central finite difference.

    This is the end-to-end differentiability guarantee: the loss is a smooth
    functional of the optimal weights, so the analytic Jacobian of the convex
    program must match a numerical derivative.

    The check is only meaningful for a *strictly* convex program.  With the plain
    piecewise-linear Mean-CVaR objective the argmin sits on a vertex of the
    feasible polytope, the true Jacobian is zero almost everywhere and any
    "difference" only reflects solver inexactness.  The test therefore uses the
    entropy-regularised variant (which is also the production configuration) and
    a tightly converged solver.
    """
    n, s = 8, 30
    g = torch.Generator().manual_seed(11)
    mu0 = (torch.randn(n, generator=g, dtype=torch.float64) * 0.02)
    scen0 = (torch.randn(s, n, generator=g, dtype=torch.float64) * 0.05)
    cap0 = torch.full((n,), 0.30, dtype=torch.float64)
    prev0 = torch.zeros(n, dtype=torch.float64)
    floor0 = torch.zeros(n, dtype=torch.float64)

    layer = _layer(
        n_assets=n,
        n_scenarios=s,
        lam_div=0.05,
        use_entropy=True,
        solver="SCS",
        solver_kwargs={"eps": 1e-11, "max_iters": 100_000},
    )

    def run(mu_vec):
        out = layer(
            mu_vec.unsqueeze(0),
            scen0.unsqueeze(0),
            cap0.unsqueeze(0),
            prev0.unsqueeze(0),
            floor0.unsqueeze(0),
        )
        y = out["weights"][0]
        target = torch.arange(n, dtype=torch.float64) / n
        return (y * target).sum()

    mu = mu0.clone().requires_grad_(True)
    loss = run(mu)
    loss.backward()
    analytic = mu.grad.detach().numpy()

    eps = 1e-5
    numeric = np.zeros(n)
    for i in range(n):
        plus = mu0.clone()
        plus[i] += eps
        minus = mu0.clone()
        minus[i] -= eps
        numeric[i] = (float(run(plus)) - float(run(minus))) / (2 * eps)

    denom = max(np.abs(numeric).max(), 1e-9)
    err = np.abs(analytic - numeric).max() / denom
    assert np.abs(numeric).max() > 1e-4, "the entropy variant must be genuinely smooth"
    assert err < 0.05, f"gradient mismatch: analytic={analytic}, numeric={numeric}"


def test_layer_converges_to_the_analytic_two_asset_solution():
    """For 2 assets with no risk term the optimum is a corner solution."""
    layer = MeanCVaRLayer(
        n_assets=2,
        n_scenarios=4,
        gamma=0.0,
        lam_turnover=0.0,
        lam_div=0.0,
        lam_herf=0.0,
        use_entropy=False,
        solver_kwargs={"eps": 1e-10, "max_iters": 20000},
    )
    mu = torch.tensor([[0.03, -0.01]], dtype=torch.float64)
    scen = torch.zeros(1, 4, 2, dtype=torch.float64)
    cap = torch.full((1, 2), 1.0, dtype=torch.float64)
    z = torch.zeros(1, 2, dtype=torch.float64)
    y = layer(mu, scen, cap, z, z)["weights"]
    assert float(y[0, 0]) > 0.95

def test_mu_target_is_reached_when_it_is_affordable():
    """The hard constraint ``mu @ y >= tau`` binds the achieved return up."""
    mu, scen, cap, prev, floor = _problem()
    mu = mu.abs() * 0.5  # all-positive predictions, so the target is reachable
    tau = 0.0025
    out = _layer(mu_target=tau)(mu, scen, cap, prev, floor)
    y = out["weights"]
    assert float(mu @ y.t().squeeze(-1)) >= tau - TOL


def test_mu_target_is_clamped_to_what_the_caps_allow():
    """An unreachable target must not make the program infeasible."""
    mu, scen, cap, prev, floor = _problem()
    mu = mu.abs() * 0.5
    layer = _layer(mu_target=5.0)  # absurd target: 500% per period
    out = layer(mu, scen, cap, prev, floor)
    y = out["weights"]
    assert abs(float(y.sum()) - 1.0) < TOL
    reach = layer.feasible_mu(mu, cap, floor)
    assert float(mu @ y.t().squeeze(-1)) <= float(reach) + TOL
    assert layer.timing["failures"] == 0


def test_feasible_mu_matches_the_linear_program():
    """The greedy two-stage fill equals the LP optimum (checked with scipy)."""
    from scipy.optimize import linprog

    n = 6
    mu = torch.tensor([[0.03, 0.01, 0.02, -0.005, 0.015, 0.0]], dtype=torch.float64)
    cap = torch.tensor([[0.4, 0.3, 0.5, 0.6, 0.2, 0.25]], dtype=torch.float64)
    floor = torch.tensor([[0.1, 0.0, 0.0, 0.2, 0.05, 0.0]], dtype=torch.float64)
    reach = float(MeanCVaRLayer.feasible_mu(mu, cap, floor))
    res = linprog(
        -mu[0].numpy(),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(float(floor[0, i]), float(cap[0, i])) for i in range(n)],
        method="highs",
    )
    assert res.status == 0
    assert abs(reach - float(-res.fun)) < 1e-8
