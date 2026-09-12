"""Differentiable Mean-CVaR portfolio optimisation layer.

Solves, for every rebalancing date, the convex program

    minimize_y, zeta, u   -mu^T y + gamma * zeta
                          + gamma / ((1-alpha) * S) * sum_s u_s
                          + lam_turn * || y - y_prev ||_1
                          + lam_div  * sum_i y_i log y_i          (entropy)
                          [ + lam_div_2 * sum_i y_i^2 ]           (herfindahl)
    subject to            u_s >= -r_s^T y - zeta        s = 1..S
                          u_s >= 0
                          sum_i y_i = 1
                          y_floor <= y <= cap
                          y >= 0

``y`` is the portfolio weight vector, ``zeta`` the VaR level and ``u`` the CVaR
slack variables; ``sum(u)/S`` is the empirical CVaR in excess of ``zeta``.

Notes on the formulation
------------------------
* The plan's entropy term was written as ``-lam_H * H(y)``.  Since ``H`` is
  concave, that would make the program non-convex.  Entropy is therefore
  implemented in its convex form ``lam_div * sum_i y_i log y_i`` (rewarding
  diversification) -- the *same* economic effect, but DPP-compliant.
* Every product of two *values* is written so that at most one factor is a
  parameter, and quantities such as ``gamma / ((1-alpha) S)`` are folded into
  constants.  The transmission-cost cap ``y_max * k_eff * pi`` is pre-computed
  in torch and enters as a *single* parameter vector ``cap``.
* The number of names ``N`` is fixed when the layer is built (the maximum
  universe size of the experiment).  Periods with a smaller universe simply set
  ``cap = 0`` on the unused slots, which pins those weights to zero.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - import guard
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    CVXPYLAYERS_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CvxpyLayer = None
    CVXPYLAYERS_AVAILABLE = False


# Keeps the ``y_prev`` parameter alive even when the turnover penalty is switched
# off, so that the layer's parameter list never changes between experiments.
_TURNOVER_EPS = 1e-9


class MeanCVaRLayer(nn.Module):
    """Entropy-regularised Mean-CVaR program with turnover and box constraints."""
    def __init__(
        self,
        n_assets: int,
        n_scenarios: int,
        alpha: float = 0.95,
        gamma: float = 1.0,
        lam_turnover: float = 0.0,
        lam_div: float = 0.0,
        lam_herf: float = 0.0,
        use_entropy: bool = True,
        solver: str = "SCS",
        solver_kwargs: dict | None = None,
        differentiable: bool = True,
        mu_target: float | None = None,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_scenarios = int(n_scenarios)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lam_turnover = float(lam_turnover)
        self.lam_div = float(lam_div)
        self.lam_herf = float(lam_herf)
        self.mu_target = None if mu_target is None else float(mu_target)
        self.use_mu_target = mu_target is not None
        self.use_entropy = bool(use_entropy) and lam_div != 0.0
        self.solver = solver
        self.solver_kwargs = dict(solver_kwargs or {})
        self.differentiable = bool(differentiable) and CVXPYLAYERS_AVAILABLE
        self._layer = None
        self._problem = None
        self._y = self._zeta = self._u = None
        self.timing = {"forward": 0.0, "backward": 0.0, "calls": 0, "failures": 0}

    # ------------------------------------------------------------------ #
    def _build(self):
        n, s = self.n_assets, self.n_scenarios
        y = cp.Variable(n, name="y")
        zeta = cp.Variable(1, name="zeta")
        u = cp.Variable(s, name="u")

        mu_p = cp.Parameter(n, name="mu")
        R_p = cp.Parameter((s, n), name="R")
        cap_p = cp.Parameter(n, name="cap")
        prev_p = cp.Parameter(n, name="y_prev")
        floor_p = cp.Parameter(n, name="y_floor")

        cvar_coef = self.gamma / ((1.0 - self.alpha) * s)
        objective = -mu_p @ y + self.gamma * zeta + cvar_coef * cp.sum(u)
        # ``y_prev`` is only reachable through the turnover term.  cvxpylayers
        # prunes parameters that do not appear in the problem, which would break
        # the (mu, R, cap, y_prev, y_floor) calling convention whenever
        # lam_turnover == 0 (e.g. the ``full_nocost`` ablation).  A vanishing
        # coefficient keeps the parameter alive without changing the solution.
        objective = objective + max(self.lam_turnover, _TURNOVER_EPS) * cp.norm1(y - prev_p)
        if self.use_entropy:
            # convex form of "-H(y)": sum y log y
            objective = objective - self.lam_div * cp.sum(cp.entr(y))
        if self.lam_herf:
            objective = objective + self.lam_herf * cp.sum_squares(y)

        constraints = [
            u >= -R_p @ y - zeta,
            u >= 0,
            cp.sum(y) == 1.0,
            y >= floor_p,
            y <= cap_p,
        ]
        tau_p = None
        if self.use_mu_target:
            # hard target-return constraint ``mu @ y >= tau``; ``tau`` is a
            # parameter (not a constant) because the model clamps it per period
            # to the level the box constraints actually allow.
            tau_p = cp.Parameter(1, name="tau")
            constraints.append(mu_p @ y >= tau_p)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        self._problem = problem
        self._y, self._zeta, self._u = y, zeta, u
        if self.differentiable:
            self._layer = CvxpyLayer(
                problem,
                parameters=[mu_p, R_p, cap_p, prev_p, floor_p]
                + ([tau_p] if tau_p is not None else []),
                variables=[y, zeta, u],
            )
        return problem

    # ------------------------------------------------------------------ #
    @staticmethod
    def feasible_mu(mu: torch.Tensor, cap: torch.Tensor, floor: torch.Tensor | None = None) -> torch.Tensor:
        """Highest reachable ``mu @ y`` under ``sum y = 1`` and the box bounds.

        Because the feasible set is a box intersected with a budget equality,
        the maximum is reached by a two-stage greedy fill: commit the mandatory
        floor first, then spend the remaining budget on the largest predictions.
        Returns ``(B, 1)``.
        """
        out = []
        lo = floor if floor is not None else torch.zeros_like(cap)
        for b in range(mu.shape[0]):
            m = mu[b].detach().to(torch.float64)
            upper = cap[b].detach().to(torch.float64).clamp(min=0.0)
            lower = lo[b].detach().to(torch.float64).clamp(min=0.0)
            upper = torch.maximum(upper, lower)
            if float(upper.sum()) <= 1e-12:
                out.append(0.0)
                continue
            y = lower.clone()
            left = 1.0 - float(y.sum())
            if left <= 0.0:  # the mandatory positions already exhaust the budget
                out.append(float((m * y).sum()))
                continue
            room = upper - y
            order = torch.argsort(m, descending=True)
            for i in order.tolist():
                take = min(float(room[i]), left)
                if take <= 0.0:
                    continue
                y[i] += take
                left -= take
                if left <= 1e-12:
                    break
            out.append(float((m * y).sum()))
        return torch.tensor(out, dtype=mu.dtype, device=mu.device).unsqueeze(1)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        mu: torch.Tensor,
        scenarios: torch.Tensor,
        cap: torch.Tensor,
        y_prev: torch.Tensor | None = None,
        y_floor: torch.Tensor | None = None,
    ) -> dict:
        """Solve the layer.  Shapes follow cvxpylayers' batching convention.

        ``mu``: (B, N); ``scenarios``: (B, S, N) or (S, N); ``cap``/``y_prev``/
        ``y_floor``: (B, N) or (N).
        """
        if self._problem is None:
            self._build()

        mu_t = mu if mu.dim() > 1 else mu.unsqueeze(0)
        batch = mu_t.shape[0]
        sc = scenarios if scenarios.dim() == 3 else scenarios.unsqueeze(0).expand(batch, -1, -1)
        cap_t = (cap if cap.dim() > 1 else cap.unsqueeze(0)).expand(batch, -1).contiguous()
        prev_t = y_prev if y_prev is not None else torch.zeros_like(mu_t)
        prev_t = (prev_t if prev_t.dim() > 1 else prev_t.unsqueeze(0)).expand(batch, -1).contiguous()
        floor_t = y_floor if y_floor is not None else torch.zeros_like(mu_t)
        floor_t = (floor_t if floor_t.dim() > 1 else floor_t.unsqueeze(0)).expand(batch, -1).contiguous()

        cap_t = torch.clamp(cap_t, min=0.0)
        floor_t = torch.clamp(floor_t, min=0.0)

        tau_t = None
        if self.use_mu_target:
            reach = self.feasible_mu(mu_t, cap_t, floor_t)
            # stay a hair below the reachable maximum: a target the box can only
            # just attain makes conic solvers wobble between optimal and
            # infeasible on nearly-identical problems.
            tau_t = torch.minimum(
                torch.full_like(reach, self.mu_target),
                reach - torch.clamp(0.002 * reach.abs(), min=1e-6),
            )

        if self.differentiable:
            t0 = time.time()
            try:
                args = [mu_t, sc, cap_t, prev_t, floor_t] + ([tau_t] if tau_t is not None else [])
                y, zeta, u = self._layer(*args, solver_args=dict(self.solver_kwargs))
            except Exception:
                self.timing["failures"] += 1
                raise
            self.timing["forward"] += time.time() - t0
            self.timing["calls"] += 1
            return {"weights": y, "var": zeta.squeeze(-1), "slack": u}

        # ---- non-differentiable reference solve (numpy in / numpy out) ----
        t0 = time.time()
        weights, zetas = [], []
        prob = self._problem
        # ``Problem.parameters()`` is *not* ordered like the constructor call, so
        # the parameters are looked up by name instead of unpacked positionally.
        p_mu = prob.param_dict["mu"]
        p_R = prob.param_dict["R"]
        p_cap = prob.param_dict["cap"]
        p_prev = prob.param_dict["y_prev"]
        p_floor = prob.param_dict["y_floor"]
        p_tau = prob.param_dict.get("tau")
        for b in range(batch):
            p_mu.value = mu_t[b].detach().cpu().numpy()
            p_R.value = sc[b].detach().cpu().numpy()
            p_cap.value = cap_t[b].detach().cpu().numpy()
            p_prev.value = prev_t[b].detach().cpu().numpy()
            p_floor.value = floor_t[b].detach().cpu().numpy()
            if p_tau is not None and tau_t is not None:
                p_tau.value = tau_t[b].detach().cpu().numpy()
            try:
                prob.solve(solver=self.solver, **self.solver_kwargs)
                status = prob.status
            except Exception:
                status = "solver_error"
            if status not in ("optimal", "optimal_inaccurate"):
                self.timing["failures"] += 1
                np_cap = cap_t[b].detach().cpu().numpy().astype(np.float64)
                np_floor = floor_t[b].detach().cpu().numpy().astype(np.float64)
                weights.append(_feasible_fallback(np_cap, np_floor))
                zetas.append(float(-np_cap @ mu_t[b].detach().cpu().numpy().astype(np.float64)))
            else:
                weights.append(np.asarray(self._y.value, dtype=np.float64).ravel())
                zetas.append(float(np.asarray(self._zeta.value).ravel()[0]))
        self.timing["forward"] += time.time() - t0
        self.timing["calls"] += 1
        y = torch.as_tensor(np.stack(weights), dtype=mu_t.dtype)
        return {"weights": y, "var": torch.as_tensor(zetas, dtype=mu_t.dtype), "slack": None}


# --------------------------------------------------------------------------- #
def _feasible_fallback(cap: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Box-feasible, budget-feasible weight vector used when a solver fails.

    The problem is feasible by construction (``sum(floor) <= 1 <= sum(cap)``),
    but a solver can still terminate without a usable point.  Rather than
    propagating ``NaN`` into the backtest we fall back to a capped
    equal-weight allocation.
    """
    upper = np.maximum(cap, floor)
    n = upper.size
    if n == 0 or float(upper.sum()) <= 0:
        return np.zeros(n)
    w = upper / float(upper.sum())
    for _ in range(3):  # project back onto the box, then renormalise
        w = np.clip(w, floor, upper)
        total = float(w.sum())
        if total <= 0:
            return np.zeros(n)
        w = w / total
    return np.clip(w, floor, upper)


def build_opt_layer(cfg, n_assets: int, n_scenarios: int, differentiable: bool = True) -> MeanCVaRLayer:
    """Factory reading an :class:`~e2e_portfolio.config.OptConfig`."""
    kind = (cfg.diversify or "none").lower()
    return MeanCVaRLayer(
        n_assets=n_assets,
        n_scenarios=n_scenarios,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        lam_turnover=cfg.lam_turnover,
        lam_div=cfg.lam_div if kind == "entropy" else 0.0,
        lam_herf=cfg.lam_div if kind == "quadratic" else 0.0,
        use_entropy=kind == "entropy",
        solver=cfg.solver,
        solver_kwargs=dict(cfg.solver_kwargs),
        differentiable=differentiable and cfg.differentiable,
        mu_target=getattr(cfg, "mu_target", None),
    )
