"""The selection layers and the induced per-name caps."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from e2e_portfolio.selection import (
    IdentitySelection,
    SoftmaxSelection,
    SparsemaxSelection,
    TopKSelection,
    participation_ratio,
    raw_caps,
    selection_caps,
    sparsemax,
)


def _score(n=20, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, generator=g) * scale


def test_dense_selection_layers_sum_to_one():
    z = _score()
    for layer in (SoftmaxSelection(), SparsemaxSelection(), IdentitySelection()):
        pi = layer(z)
        assert pi.shape == z.shape
        assert torch.all(pi >= -1e-9), f"{type(layer).__name__} produced a negative weight"
        assert abs(float(pi.sum()) - 1.0) < 1e-5


def test_topk_selects_exactly_k_names_with_a_straight_through_gradient():
    k = 5
    z = _score(20).requires_grad_(True)
    pi = TopKSelection(k=k)(z)
    assert int((pi > 1e-8).sum()) == k
    assert torch.allclose(pi[pi > 1e-8], torch.full((k,), 1.0 / k))
    # ``pi.sum()`` is constant by construction (the straight-through term adds
    # ``soft - soft.detach()``), so the gradient must be probed with a functional
    # that actually depends on *which* names were selected.
    (pi * torch.arange(20.0)).sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def test_sparsemax_produces_exact_zeros_and_a_valid_gradient():
    z = _score(40, scale=3.0).requires_grad_(True)
    pi = sparsemax(z, dim=-1)
    assert float(pi.min()) >= 0.0
    assert abs(float(pi.sum()) - 1.0) < 1e-5
    assert int((pi == 0).sum()) >= 1, "sparsemax should zero out some scores"
    (pi * torch.arange(40.0)).sum().backward()
    assert z.grad is not None


def test_sparsemax_matches_the_analytic_projection_on_a_known_case():
    # projection of (3, 0, -1) onto the simplex is (1, 0, 0)
    z = torch.tensor([[3.0, 0.0, -1.0]])
    pi = sparsemax(z, dim=-1)
    assert torch.allclose(pi, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_participation_ratio_equals_one_over_sum_of_squares():
    pi = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    assert abs(float(participation_ratio(pi)) - 2.0) < 1e-6
    pi = torch.full((1, 8), 1.0 / 8)
    assert abs(float(participation_ratio(pi)) - 8.0) < 1e-6


def test_selection_caps_always_leave_the_program_feasible():
    """``sum(cap) >= 1`` and ``cap > 0`` wherever ``pi > 0``."""
    for layer in (SoftmaxSelection(), SparsemaxSelection(), IdentitySelection()):
        for seed in range(4):
            pi = layer(_score(40, seed=seed, scale=2.0))
            valid = torch.ones_like(pi, dtype=torch.bool)
            caps = selection_caps(pi, 0.10, valid)
            assert float(caps.sum()) >= 1.0 - 1e-6
            assert float(caps.min()) >= 0.0
            assert float(caps.max()) <= 0.10 + 1e-6 or float(pi.max()) > 0.0


def test_caps_respect_the_hard_position_limit_for_topk():
    k, y_max = 5, 0.10
    pi = TopKSelection(k=k)(_score(20))
    caps = selection_caps(pi, y_max, torch.ones_like(pi, dtype=torch.bool))
    assert float(caps.max()) <= y_max + 1e-6
    assert float(caps.sum()) >= 1.0 - 1e-6


def test_padding_slots_never_obtain_a_cap():
    pi = SoftmaxSelection()(_score(10))
    valid = torch.ones_like(pi, dtype=torch.bool)
    valid[0, 6:] = False
    caps = selection_caps(pi, 0.10, valid)
    assert float(caps[0, 6:].abs().sum()) == 0.0
    assert float(caps[0, :6].sum()) >= 1.0 - 1e-6


def test_caps_only_need_the_deficit_fill_when_the_selection_is_too_concentrated():
    """The fill may spill into ``pi = 0`` names, but only if ``sum(cap) < 1``.

    The sparse selectors of the main grid support enough names for
    ``sum(y_eff * min(1, k_eff * pi_i)) >= 1`` (the runner records ``cap_sum``
    and, over the two reference runs, it never fell below 1.1608), so no spill
    ever happens there.  The third block pins the behaviour for a selection that
    is too concentrated to fund the budget on its own.
    """
    valid = torch.ones(1, 20, dtype=torch.bool)

    # 10 selected names, y_max = 0.10: every selected name may hold the full
    # budget, sum(cap) = 1 exactly, so the zero-score names get nothing
    pi = torch.zeros(1, 20)
    pi[0, :10] = 0.1
    caps = selection_caps(pi, 0.10, valid)
    assert float(caps[0, 10:].sum()) == 0.0
    assert float(caps[0, :10].sum()) == pytest.approx(1.0, abs=1e-6)

    # a uniform selection: the effective support is the whole cross-section, so
    # ``k_eff * pi_i >= 1`` everywhere and ``y_max`` binds on every name
    pi = torch.full((1, 20), 1.0 / 20)
    caps = selection_caps(pi, 0.10, valid)
    assert float(caps.min()) == pytest.approx(0.10, abs=1e-6)
    assert float(caps.sum()) == pytest.approx(2.0, abs=1e-6)

    # a smooth softmax selection: still no cap above y_max and no zero-π slot
    wide = SoftmaxSelection()(_score(20, scale=3.0))
    caps = selection_caps(wide, 0.10, valid)
    assert float(caps.sum()) >= 1.0 - 1e-6
    assert float(caps.max()) <= 0.10 + 1e-6

    # one single name cannot hold the whole budget at y_max = 0.10, so the fill
    # widens the *selection*: the remaining 0.90 goes to the next 9 names at the
    # full headroom 0.10 each, and the last ten names stay at zero
    pi = torch.zeros(1, 20)
    pi[0, 0] = 1.0
    caps = selection_caps(pi, 0.10, valid)
    assert float(caps.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(caps[0, 0]) == pytest.approx(0.10, abs=1e-6)
    assert float(caps[0, 1:10].sum()) == pytest.approx(0.90, abs=1e-6)
    assert float(caps[0, 1:10].min()) == pytest.approx(0.10, abs=1e-6)
    assert float(caps[0, 10:].abs().sum()) == 0.0


def test_raw_caps_are_the_budget_before_the_widening():
    """``raw_caps`` is exactly what ``selection_caps`` widens, and nothing more."""
    valid = torch.ones(1, 30, dtype=torch.bool)
    score = _score(30, seed=7, scale=3.0)
    pi = SparsemaxSelection()(score)

    raw = raw_caps(pi, 0.10, valid)
    assert torch.allclose(raw, raw_caps(pi, 0.10, valid))
    assert float(raw.sum()) < 1.0, "the fixture must need the widening"

    caps = selection_caps(pi, 0.10, valid, score)
    assert float(caps.sum()) == pytest.approx(1.0, abs=1e-5)
    # widening only ever *adds* room, never takes it away, and stays inside y_max
    assert bool((caps >= raw - 1e-9).all())
    assert float(caps.max()) <= 0.10 + 1e-6

    # a selection that can fund the budget by itself is returned untouched
    uniform = torch.full((1, 30), 1.0 / 30)
    assert float(raw_caps(uniform, 0.10, valid).sum()) == pytest.approx(3.0, abs=1e-6)
    assert torch.equal(selection_caps(uniform, 0.10, valid), raw_caps(uniform, 0.10, valid))
    # ... while a peaked softmax on the same universe cannot and is widened
    wide = SoftmaxSelection()(_score(30, seed=3, scale=3.0))
    assert float(raw_caps(wide, 0.10, valid).sum()) < 1.0
    assert float(selection_caps(wide, 0.10, valid).sum()) == pytest.approx(1.0, abs=1e-5)

    # padding slots are zero in both
    valid = torch.ones(1, 30, dtype=torch.bool)
    valid[0, 20:] = False
    assert float(raw_caps(pi, 0.10, valid)[0, 20:].abs().sum()) == 0.0


def test_deficit_fill_widens_by_score_and_never_exceeds_y_max():
    """The widening follows the score order and the hard cap still binds."""
    n, y_max = 30, 0.10
    valid = torch.ones(1, n, dtype=torch.bool)
    # Sparsemax on a peaked score: 3 names selected, 0.70 of the budget left over
    score = torch.zeros(1, n)
    score[0, [5, 12, 25]] = 4.0
    score[0, [1, 2, 3]] = 1.0  # the next best names, in a scrambled order
    pi = sparsemax(score, dim=-1)
    caps = selection_caps(pi, y_max, valid, score)
    assert float(caps.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(caps.max()) <= y_max + 1e-6
    assert float(caps.min()) >= 0.0

    # 3 selected + 7 widened names at headroom 0.10 = the budget; float32 rounding
    # of ``pi`` leaves one extra name with a ~1e-7 cap
    funded = (caps[0] > 1e-8).nonzero().flatten().tolist()
    assert 10 <= len(funded) <= 12, funded
    # the funded names are exactly the top of the score ranking: a better-scored
    # name is never left out while a worse-scored one is filled
    order = torch.argsort(score[0], descending=True, stable=True).tolist()
    assert sorted(order.index(i) for i in funded) == list(range(len(funded))), funded
    # the names just below the selection get the full headroom, the far tail zero
    assert float(caps[0, 1]) == pytest.approx(y_max, abs=1e-6)
    assert float(caps[0, 2]) == pytest.approx(y_max, abs=1e-6)
    assert float(caps[0, 9]) == pytest.approx(0.0)
    assert float(caps[0, 10]) == pytest.approx(0.0)
    assert float(caps[0, 29]) == pytest.approx(0.0)


