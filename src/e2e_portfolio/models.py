"""Neural return / risk predictor and the end-to-end pipeline.

Architecture (per rebalancing date, applied to every stock in the investable
cross-section):

    x_{i, 1..L}  --(shared encoder)-->  h_i  --(MLP head)-->
        mu_i         expected return over the next holding period
        sigma_i      volatility of that return
        s_i          selection score

The selection score is either produced by its own linear head (``score_mode =
head``, the default: the score is learned end to end) or built from the
predictions with the fixed rule of the research plan (``score_mode =
mu_sigma``),

    s_i = mu_i - score_lambda_sigma * sigma_i
                - score_lambda_cvar * CVaR_i(scenarios)

Two ways of turning ``(mu, sigma)`` into the ``S`` return scenarios required by
the CVaR layer:

``gaussian``  deterministic Gaussian quantile grid,
              ``r_{i,s} = mu_i + sigma_i * z_s`` with ``z_s = Phi^{-1}((s-.5)/S)``.
              Every stock then has a scenario distribution with exactly the
              predicted mean and standard deviation, and the empirical CVaR of
              those scenarios approximates the Gaussian CVaR
              ``mu - sigma * phi(z_alpha) / (1 - alpha)``.
``direct``    the head emits the ``S`` scenario returns directly, which lifts the
              Gaussian assumption at the cost of more parameters.

The LSTM is *shared* across stocks (one model, not one per stock).  Because the
features are standardised cross-sectionally, the network learns a ranking rule
that is invariant to the level of the market -- the usual design for
cross-sectional stock selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from . import selection as sel_mod


def gaussian_quantile_grid(n_scenarios: int) -> torch.Tensor:
    """``z_s = Phi^{-1}((s - 0.5) / S)``: the symmetric quantile grid."""
    probs = (torch.arange(n_scenarios, dtype=torch.float64) + 0.5) / n_scenarios
    z = math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)  # Phi^{-1}
    return z.float()


def inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def scenario_cvar(scenarios: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-name CVaR of the *loss* implied by the scenario distribution.

    ``scenarios``: ``(S, M)``; returns ``(M,)``, the mean of the worst
    ``ceil((1 - alpha) * S)`` scenarios of every name with the sign flipped, so
    a positive value means a large tail loss.  A single name with a wide
    scenario spread therefore scores lower than a narrow one with the same mean.
    """
    n_scen = scenarios.shape[0]
    # same binary-float tolerance as losses.tail_loss: 0.05 * 60 = 3.0000000000000004
    k = max(1, int(math.ceil((1.0 - alpha) * n_scen - 1e-9)))
    worst = torch.topk(scenarios, k, dim=0, largest=False).values
    return -worst.mean(dim=0)


@dataclass
class ModelOutput:
    mu: torch.Tensor            # (M,) predicted return
    sigma: torch.Tensor         # (M,) predicted volatility
    score: torch.Tensor         # (M,) selection score
    pi: torch.Tensor            # (M,) selection weights, sum = 1 over valid
    cap: torch.Tensor           # (M,) per-name cap induced by pi
    scenarios: torch.Tensor     # (S, M) scenario returns
    aux: Dict[str, torch.Tensor]


class StockEncoder(nn.Module):
    """Shared encoder (``model.arch``) followed by a per-stock prediction head."""

    def __init__(self, n_features: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.score_mode = str(getattr(cfg, "score_mode", "head"))
        if self.score_mode not in ("head", "mu_sigma"):
            raise ValueError(
                f"unknown model.score_mode: {self.score_mode!r} "
                "(expected 'head' or 'mu_sigma')"
            )
        self.input_norm = nn.LayerNorm(n_features)
        self.arch = str(getattr(cfg, "arch", "lstm")).lower()
        if self.arch not in ("lstm", "gru", "rnn", "mlp", "rbfn"):
            raise ValueError(
                f"unknown model.arch: {self.arch!r} "
                "(expected 'lstm', 'gru', 'rnn', 'mlp' or 'rbfn')"
            )
        self.lstm = None
        self.td_mlp = None
        self.rbf_centres = None
        if self.arch in ("lstm", "gru", "rnn"):
            recur = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[self.arch]
            kwargs = dict(
                input_size=n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            if self.arch == "rnn":
                kwargs["nonlinearity"] = "tanh"
            self.lstm = recur(**kwargs)
        elif self.arch == "mlp":
            # no state: a time-distributed MLP whose per-day embeddings are
            # pooled over the observed window (a "stateless" encoder variant)
            self.td_mlp = nn.Sequential(
                nn.Linear(n_features, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.GELU(),
            )
        else:
            # RBF network: Gaussian kernels centred at learnable points, one
            # feature dimension per kernel, again pooled over the window
            self.rbf_centres = nn.Parameter(torch.randn(cfg.hidden_size, n_features) * 0.5)
            self.rbf_log_sigma = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(cfg.dropout)
        self.body = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.GELU(),
        )
        self.mu_layer = nn.Linear(cfg.head_hidden, 1)
        self.sigma_layer = nn.Linear(cfg.head_hidden, 1)
        self.score_layer = (
            nn.Linear(cfg.head_hidden, 1) if self.score_mode == "head" else None
        )
        self.direct_layer = (
            nn.Linear(cfg.head_hidden, cfg.num_scenarios) if cfg.scenario_mode == "direct" else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # the model starts from an almost flat forecast: small head weights and a
        # bias that reproduces the typical monthly volatility
        for layer in (self.mu_layer, self.score_layer):
            if layer is None:
                continue
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.sigma_layer.weight, std=0.01)
        nn.init.constant_(self.sigma_layer.bias, inverse_softplus(self.cfg.sigma_init))
        if self.direct_layer is not None:
            nn.init.normal_(self.direct_layer.weight, std=0.01)
            nn.init.zeros_(self.direct_layer.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """``x``: ``(M, L, K)``; ``mask``: ``(M, L)`` (1 = observed)."""
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        x = self.input_norm(x)
        if self.lstm is not None:
            out, _ = self.lstm(x)
            if mask is not None:
                lengths = mask.sum(dim=1).clamp(min=1).long() - 1
                h = out[torch.arange(out.shape[0], device=out.device), lengths]
            else:
                h = out[:, -1]
        else:
            if self.arch == "mlp":
                feats = self.td_mlp(x)
            else:
                sigma = torch.nn.functional.softplus(self.rbf_log_sigma)
                d = torch.cdist(x, self.rbf_centres)  # (M, L, H)
                feats = torch.exp(-(d * d) / (2.0 * sigma * sigma))
            if mask is None:
                h = feats.mean(dim=1)
            else:
                w = mask.unsqueeze(-1).to(feats.dtype)
                h = (feats * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        z = self.body(h)
        mu = self.mu_layer(z).squeeze(-1)
        raw_sigma = self.sigma_layer(z).squeeze(-1)
        score = None if self.score_layer is None else self.score_layer(z).squeeze(-1)
        sigma = torch.nn.functional.softplus(raw_sigma).clamp(
            min=self.cfg.sigma_min, max=self.cfg.sigma_max
        )
        direct = self.direct_layer(z) if self.direct_layer is not None else None
        return mu, sigma, score, direct


def score_to_pi_cap(
    score: torch.Tensor,
    valid: torch.Tensor,
    selection,
    opt_cfg,
    y_floor: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a selection *score* into ``(pi, cap)``.

    Shared by the learned encoder and by the tree-ensemble baselines, so both
    feed the very same selection + cap construction into the Mean-CVaR layer.
    ``score`` and ``valid`` are ``(M,)``, the returned tensors as well.
    """
    pi = selection(score.unsqueeze(0)).squeeze(0)
    pi = pi * valid.to(pi.dtype)
    total = pi.sum()
    uniform = valid.to(pi.dtype) / valid.sum().clamp(min=1).to(pi.dtype)
    pi = torch.where(total > 1e-8, pi / total.clamp(min=1e-12), uniform)
    cap = sel_mod.selection_caps(
        pi.unsqueeze(0), opt_cfg.y_max, valid.unsqueeze(0), score.unsqueeze(0)
    ).squeeze(0)
    if y_floor is not None:
        # a name that cannot be sold must be allowed to keep its current
        # weight, otherwise y >= y_floor and y <= cap are contradictory and the
        # convex program is reported as infeasible
        cap = torch.maximum(cap, y_floor)
    return pi, cap


class E2EPortfolioModel(nn.Module):
    """Encoder + selection + differentiable Mean-CVaR optimisation layer."""

    def __init__(self, n_features: int, model_cfg, opt_cfg, opt_layer=None):
        super().__init__()
        self.cfg = model_cfg
        self.opt_cfg = opt_cfg
        self.encoder = StockEncoder(n_features, model_cfg)
        self.selection = sel_mod.build_selection(model_cfg.selection, model_cfg)
        self.register_buffer("z_grid", gaussian_quantile_grid(model_cfg.num_scenarios))
        self.opt_layer = opt_layer

    # ------------------------------------------------------------------ #
    def predict(self, x: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor):
        """Return ``(mu, sigma, score, scenarios)`` restricted to the investable set."""
        mu, sigma, score, direct = self.encoder(x, mask)
        v = valid.to(mu.dtype)
        mu = mu * v
        sigma = sigma * v
        if self.cfg.scenario_mode == "direct" and direct is not None:
            scen = direct.transpose(0, 1)  # (S, M)
            sigma = scen.std(dim=0) * v
        else:
            scen = mu.unsqueeze(0) + sigma.unsqueeze(0) * self.z_grid.unsqueeze(1)
        scen = scen * v.unsqueeze(0)
        if score is None:
            # fixed rule of the research plan: risk-adjusted prediction
            score = (
                mu
                - float(self.cfg.score_lambda_sigma) * sigma
                - float(self.cfg.score_lambda_cvar) * scenario_cvar(scen, self.opt_cfg.alpha)
            )
        score = score - (1.0 - v) * 1e4  # invalid names get a very small score
        return mu, sigma, score, scen

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
        y_prev: Optional[torch.Tensor] = None,
        y_floor: Optional[torch.Tensor] = None,
        run_layer: bool = True,
    ) -> ModelOutput:
        """``x``: ``(M, L, K)``, ``mask``: ``(M, L)``, ``valid``: ``(M,)`` buyable."""
        mu, sigma, score, scen = self.predict(x, mask, valid)
        pi, cap = score_to_pi_cap(score, valid, self.selection, self.opt_cfg, y_floor)
        out = ModelOutput(mu=mu, sigma=sigma, score=score, pi=pi, cap=cap,
                          scenarios=scen, aux={})
        if not run_layer:
            return out
        res = self.opt_layer(
            mu.unsqueeze(0),
            scen.unsqueeze(0),
            cap.unsqueeze(0),
            None if y_prev is None else y_prev.unsqueeze(0),
            None if y_floor is None else y_floor.unsqueeze(0),
        )
        out.aux["weights"] = res["weights"].squeeze(0)
        out.aux["var"] = res["var"].reshape(-1)
        return out
