"""Differentiable stock-selection layers.

All three mechanisms map a per-stock score to *selection weights* ``pi`` that
sum to one, so that the same portfolio cap rule can be applied to all of them:

    cap_i = y_max * min(1, k_eff * pi_i)

``k_eff`` is the participation ratio ``1 / sum(pi^2)`` -- the effective number
of selected names.  For Top-k this gives exactly ``cap_i = y_max`` on the
selected names, for Sparsemax the "average" selected name gets ``y_max`` and
names with a larger score get proportionally more room, and for Softmax (which
spreads mass over the whole cross-section) every name gets ``y_max``.

A small uniform slack is added whenever ``sum(cap) < 1`` so that the portfolio
program is always feasible.
"""
from __future__ import annotations

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# sparsemax / entmax
# --------------------------------------------------------------------------- #
def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016): Euclidean projection on the simplex.

    Produces exactly zero weights for low scores while remaining differentiable
    almost everywhere.
    """
    z = z - z.max(dim=dim, keepdim=True).values.detach()
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    k = k.view(*([1] * (z.dim() - 1)), -1)
    support = (z_sorted - (cumsum - 1.0) / k) > 0
    k_support = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, k_support - 1) - 1.0) / k_support
    return torch.clamp(z - tau, min=0.0)


def entmax15(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax: alternative sparsifying transform (kept for completeness)."""
    z = z / 2.0
    z = z - z.max(dim=dim, keepdim=True).values.detach()
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    k = k.view(*([1] * (z.dim() - 1)), -1)
    support = (z_sorted - (cumsum - 1.0) / k) > 0
    k_support = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, k_support - 1) - 1.0) / k_support
    return torch.clamp(z - tau, min=0.0) ** 2


# --------------------------------------------------------------------------- #
# selection modules
# --------------------------------------------------------------------------- #
class TopKSelection(nn.Module):
    """Hard Top-k selection with a straight-through gradient."""

    def __init__(self, k: int = 25, temperature: float = 1.0):
        super().__init__()
        self.k = k
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-2, 1e2)

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        n = score.shape[-1]
        k = min(self.k, n)
        soft = torch.softmax(score / self.temperature, dim=-1)
        _, idx = torch.topk(score, k, dim=-1)
        hard = torch.zeros_like(score).scatter_(-1, idx, 1.0) / k
        # straight-through: forward value is the hard 0/1 mask, backward uses softmax
        return hard - soft.detach() + soft


class SoftmaxSelection(nn.Module):
    """Dense selection: every name stays eligible."""

    def __init__(self, temperature: float = 1.0, learn: bool = True):
        super().__init__()
        if learn:
            self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        else:
            self.register_buffer("log_temperature", torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-2, 1e2)

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        return torch.softmax(score / self.temperature, dim=-1)


class SparsemaxSelection(nn.Module):
    """Sparse selection: exact zeros, gradients preserved."""

    def __init__(self, temperature: float = 1.0, learn: bool = True):
        super().__init__()
        if learn:
            self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        else:
            self.register_buffer("log_temperature", torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-2, 1e2)

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        return sparsemax(score / self.temperature, dim=-1)


class IdentitySelection(nn.Module):
    """No selection layer: every tradable name may be held up to ``y_max``."""

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        return torch.full_like(score, 1.0 / score.shape[-1])


def build_selection(kind: str, cfg) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "topk":
        return TopKSelection(k=cfg.topk, temperature=cfg.temperature_init)
    if kind == "softmax":
        return SoftmaxSelection(cfg.temperature_init, learn=cfg.learn_temperature)
    if kind == "sparsemax":
        return SparsemaxSelection(cfg.temperature_init, learn=cfg.learn_temperature)
    if kind in ("none", "identity"):
        return IdentitySelection()
    raise ValueError(f"unknown selection mechanism: {kind}")


# --------------------------------------------------------------------------- #
# caps
# --------------------------------------------------------------------------- #
def participation_ratio(pi: torch.Tensor) -> torch.Tensor:
    """``1 / sum(pi^2)`` -- a differentiable proxy for the number of holdings."""
    return 1.0 / (pi.pow(2).sum(dim=-1) + 1e-12)


def feasible_cap(y_max: float, n_valid: int) -> float:
    """The tightest per-name cap that still allows ``sum(y) = 1``.

    With fewer than ``1 / y_max`` investable names the hard limit would make the
    budget constraint infeasible, so the cap is relaxed to the equal-weight
    portfolio weight of a tiny universe.  For a realistic universe
    (``n_valid >= 1 / y_max``) this is exactly ``y_max``.
    """
    return max(float(y_max), 1.0 / max(int(n_valid), 1))


def raw_caps(
    pi: torch.Tensor, y_max: float, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-name caps *before* the deficit widening (see :func:`selection_caps`).

    ``cap_i = y_eff * min(1, k_eff * pi_i)`` with ``k_eff = 1 / sum(pi^2)``.  When
    ``sum_i cap_i >= 1`` the selection can fund the whole budget on its own and
    :func:`selection_caps` returns exactly this tensor.
    """
    free = torch.ones_like(pi) if valid is None else valid.to(pi.dtype)
    n_valid = free.sum(dim=-1, keepdim=True).clamp(min=1.0)
    y_eff = torch.clamp(n_valid.reciprocal(), min=float(y_max))
    k_eff = participation_ratio(pi).unsqueeze(-1).detach()
    return y_eff * torch.clamp(k_eff * pi, max=1.0) * free


def selection_caps(
    pi: torch.Tensor,
    y_max: float,
    valid: torch.Tensor | None = None,
    score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-name weight caps induced by the selection weights.

    ``cap_i = y_eff * min(1, k_eff * pi_i)`` with the effective support
    ``k_eff = 1 / sum(pi^2)``, so a concentrated selection earns the full budget
    while a diffuse one is capped proportionally.  ``y_eff`` is
    :func:`feasible_cap` of the universe size.

    A selection that is *more* concentrated than the position limit allows
    (``sum(cap) < 1``, i.e. the support is smaller than ``1 / y_max``) cannot hold
    the whole budget, because the plan's constraint set ``sum(y) = 1`` with
    ``y_i <= y_max`` is then empty.  The selection is widened instead of the cap:
    the deficit is handed, in waterfall order, to the *next best names by score*,
    each taking at most its headroom ``y_eff - cap_i``.  Widening keeps every hard
    constraint of the plan intact (``0 <= y <= y_max``) and keeps the portfolio as
    sparse as the budget allows -- exactly ``ceil(deficit / y_eff)`` extra names.
    Spreading the deficit over the headroom of the *whole* cross-section would be
    feasible too, but it silently turns a concentrated selection into a dense book
    (hundreds of ~0.3% positions), which defeats the point of sparse selection.

    ``score`` only sets the widening order; ``pi`` is used when it is omitted,
    which is equivalent for Sparsemax (``pi_i = (z_i - tau)^+`` is monotone in
    the score) and deterministic (index order) for the tied Top-k weights.
    """
    caps = raw_caps(pi, y_max, valid)
    free = torch.ones_like(pi) if valid is None else valid.to(pi.dtype)
    n_valid = free.sum(dim=-1, keepdim=True).clamp(min=1.0)
    y_eff = torch.clamp(n_valid.reciprocal(), min=float(y_max))
    # ``sum(headroom) >= deficit`` because ``n_valid * y_eff >= 1``, so a single
    # pass fills the budget exactly without exceeding ``y_eff`` anywhere.
    deficit = (1.0 - caps.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    if not bool((deficit > 0).any()):
        return caps
    room = (y_eff - caps).clamp(min=0.0) * free
    order = torch.argsort(
        pi if score is None else score, dim=-1, descending=True, stable=True
    )
    room_sorted = room.gather(-1, order)
    # headroom of the strictly better-ranked names, i.e. the waterfall level
    taken = room_sorted.cumsum(dim=-1) - room_sorted
    fill_sorted = (deficit - taken).clamp(min=0.0).clamp(max=room_sorted)
    return caps + torch.zeros_like(caps).scatter(-1, order, fill_sorted)
