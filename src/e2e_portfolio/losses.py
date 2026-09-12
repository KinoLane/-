"""Training objective.

The composite loss follows the plan exactly:

    L = l_pred     * L_pred        predicted return / risk accuracy (Gaussian NLL)
      + l_decision * L_decision    realised portfolio return of the solved weights
      + l_tail     * L_tail        realised tail (CVaR) loss inside the holding month
      + l_turnover * L_turnover    realised trading cost
      + l_sparse   * L_sparse      sparsity of the selection weights

Every term is differentiable w.r.t. the network parameters, including the ones
that flow through the convex program (the cvxpylayers backward pass supplies the
sensitivity of the optimal weights to ``mu``, to the scenario matrix and to the
caps).

Scaling
-------
The terms are put on a comparable scale so that the weights are meaningful and
the ablation study is interpretable:

* return-based terms are annualised (``annual_scale`` = 12 for monthly
  rebalancing), so ``L_decision`` is roughly ``0.1`` for a 10% annual return;
* ``L_turnover`` is expressed in the *same return units*, using the very cost
  rate charged by the backtester (``cost_per_unit_turnover`` = 15 bp one-way), so
  the trainer optimises the same objective the backtest measures;
* ``L_sparse`` (the participation ratio, i.e. the effective number of holdings)
  is divided by the number of candidates, which maps it to ``[0, 1]``.

The raw value of every term is returned and logged, so the balance between the
terms can be checked in the training log.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class LossOutput:
    total: torch.Tensor
    terms: Dict[str, float]


# --------------------------------------------------------------------------- #
def prediction_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    realised: torch.Tensor,
    valid: torch.Tensor,
    kind: str = "nll",
) -> torch.Tensor:
    """Gaussian negative log-likelihood (or MSE) of the realised period return."""
    v = valid.to(mu.dtype)
    n = v.sum().clamp(min=1.0)
    resid = (realised - mu) * v
    if kind == "mse":
        # x100 keeps the MSE gradient comparable to the NLL gradient at sigma = 8%
        return (resid.pow(2).sum() / n) * 100.0
    var = (sigma * v).pow(2) + 1e-6
    nll = 0.5 * (torch.log(2.0 * torch.pi * var) + resid.pow(2) / var)
    # padded slots must not contribute (their sigma is zero, which would make the
    # log-density unboundedly small and bias the mean)
    return (nll * v).sum() / n


def portfolio_return(y: torch.Tensor, realised: torch.Tensor) -> torch.Tensor:
    return (y * realised).sum()


def tail_loss(y: torch.Tensor, daily: torch.Tensor, alpha: float) -> torch.Tensor:
    """Empirical CVaR of the daily portfolio returns inside the holding month.

    ``daily``: ``(H, M)`` realised daily returns, ``y``: ``(M,)`` weights held
    over the month.  Returns the *negative* mean of the worst ``(1 - alpha)``
    fraction of days -- what a mean-CVaR investor minimises.
    """
    r = daily @ y  # (H,)
    h = r.numel()
    # ``(1 - alpha) * h`` suffers binary-float noise (0.05 * 100 = 5.000000000000004),
    # which would round the tail up to 6 days; the tolerance pins the exact case.
    k = max(1, int(np.ceil((1.0 - alpha) * h - 1e-9)))
    worst = torch.topk(r, k, largest=False).values
    return -worst.mean()


def turnover_loss(y: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
    """One-way L1 turnover ``sum |y - y_prev|`` (0 = unchanged, 2 = full switch)."""
    return (y - y_prev).abs().sum()


def sparsity_loss(pi: torch.Tensor, n_candidates: int) -> torch.Tensor:
    """Participation ratio ``1 / sum(pi^2)`` (effective number of holdings) / N."""
    pr = 1.0 / (pi.pow(2).sum() + 1e-12)
    return pr / max(n_candidates, 1)


# --------------------------------------------------------------------------- #
def composite_loss(
    cfg,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    realised: torch.Tensor,
    daily: torch.Tensor,
    valid: torch.Tensor,
    pi: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    y_prev: Optional[torch.Tensor] = None,
    alpha: float = 0.95,
    n_candidates: Optional[int] = None,
) -> LossOutput:
    """Assemble the five-term loss for a single rebalancing period."""
    terms: Dict[str, float] = {}
    total = torch.zeros((), dtype=mu.dtype)
    n_cand = int(n_candidates if n_candidates is not None else valid.shape[0])

    if cfg.l_pred:
        l_pred = prediction_loss(mu, sigma, realised, valid, cfg.pred_kind)
        terms["L_pred"] = float(l_pred.detach())
        total = total + cfg.l_pred * l_pred

    if cfg.l_sparse:
        l_sparse = sparsity_loss(pi, n_cand)
        terms["L_sparse"] = float(l_sparse.detach())
        total = total + cfg.l_sparse * l_sparse

    if y is not None:
        if cfg.l_decision:
            l_dec = -portfolio_return(y, realised) * cfg.annual_scale
            terms["L_decision"] = float(l_dec.detach())
            total = total + cfg.l_decision * l_dec
        if cfg.l_tail:
            l_tail = tail_loss(y, daily, alpha) * cfg.annual_scale
            terms["L_tail"] = float(l_tail.detach())
            total = total + cfg.l_tail * l_tail
        if cfg.l_turnover:
            l_turn = turnover_loss(y, y_prev if y_prev is not None else torch.zeros_like(y))
            l_turn = l_turn * cfg.annual_scale * cfg.cost_per_unit_turnover
            terms["L_turnover"] = float(l_turn.detach())
            total = total + cfg.l_turnover * l_turn

    terms["L_total"] = float(total.detach())
    return LossOutput(total=total, terms=terms)
