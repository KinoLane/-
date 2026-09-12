"""Experiment registry and the runner for one (experiment, fold) job.

Every experiment is described declaratively as a set of overrides on top of the
base :class:`~e2e_portfolio.config.Config`, so the whole study is reproducible
from a single YAML file plus the registry below.

The grid follows the plan.

Baselines
    ``ew``              equal weight over the investable universe (no learning,
                        no optimisation)
    ``meancvar_hist``   traditional Mean-CVaR: historical overlapping-horizon
                        scenario returns, non-differentiable convex solve

Predict-then-optimise (network trained on the forecast loss only, the
Mean-CVaR layer is applied when the portfolio is formed)
    ``lstm_topk``       straight-through top-k selection
    ``lstm_softmax``    dense softmax selection
    ``lstm_sparsemax``  differentiable sparsemax selection

End-to-end (the layer is inside the training loop, gradients flow through the
optimal weights)
    ``e2e_noselect``    no selection -- uniform per-name cap
    ``full``            the complete model
    ``full_nocvar_loss``  no realised tail term (L_tail = 0)
    ``full_nocost``     cost-blind: no turnover penalty in the layer or the loss
    ``full_nosparse``   no sparsity incentive (L_sparse = 0)
    ``full_quad``       quadratic (Herfindahl) diversification instead of entropy
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .backtest import simulate
from .config import Config, FoldConfig
from .dataset import PanelDataset, Period, build_dataset
from .metrics import compute_metrics, concentration_metrics, concentration_series, market_state_breakdown
from .optlayer import MeanCVaRLayer
from .selection import feasible_cap
from .train import Trainer


# --------------------------------------------------------------------------- #
@dataclass
class ExperimentSpec:
    name: str
    kind: str  # 'ew' | 'hist_meancvar' | 'nn'
    description: str
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    group: str = "main"
    #: train the network end-to-end through the optimisation layer
    e2e: bool = False
    #: estimator of the ``tree`` kind: ``adaboost`` | ``xgboost``
    estimator: Optional[str] = None


def _predict_only(selection: str, topk: int = 25, **model_overrides: Any) -> Dict[str, Dict[str, Any]]:
    """Overrides shared by the predict-then-optimise variants.

    ``model_overrides`` are merged into the ``model`` section, which is how the
    plan's fixed ``mu_sigma`` score rule is attached to the baseline.
    """
    return {
        "model": {"selection": selection, "topk": topk, **model_overrides},
        "loss": {
            "l_pred": 1.0,
            "l_decision": 0.0,
            "l_tail": 0.0,
            "l_turnover": 0.0,
            "l_sparse": 0.0,
        },
    }


REGISTRY: Dict[str, ExperimentSpec] = {
    s.name: s
    for s in [
        ExperimentSpec(
            "ew",
            "ew",
            "Equal-weight 1/N over the investable universe (no learning).",
            group="baseline",
        ),
        ExperimentSpec(
            "meancvar_hist",
            "hist_meancvar",
            "Traditional Mean-CVaR on historical overlapping-horizon scenarios "
            "(non-differentiable, no learning).",
            group="baseline",
        ),
        ExperimentSpec(
            "lstm_topk",
            "nn",
            "Predict-then-optimise: LSTM + straight-through top-k selection, "
            "Mean-CVaR layer applied at test time only.",
            _predict_only("topk", 25),
        ),
        ExperimentSpec(
            "lstm_softmax",
            "nn",
            "Predict-then-optimise: LSTM + dense softmax selection, Mean-CVaR "
            "layer applied at test time only.",
            _predict_only("softmax", 25),
        ),
        ExperimentSpec(
            "lstm_sparsemax",
            "nn",
            "Predict-then-optimise: LSTM + sparsemax selection, Mean-CVaR layer "
            "applied at test time only.",
            _predict_only("sparsemax", 25),
        ),
        ExperimentSpec(
            "e2e_noselect",
            "nn",
            "End-to-end without selection: uniform per-name cap, the layer is "
            "inside the training loop.",
            {"model": {"selection": "identity"}, "loss": {"l_sparse": 0.0}},
            e2e=True,
        ),
        ExperimentSpec(
            "full",
            "nn",
            "Full model: LSTM + sparsemax selection + differentiable Mean-CVaR "
            "layer, trained with all five loss terms.",
            {},
            e2e=True,
        ),
        ExperimentSpec(
            "full_nocvar_loss",
            "nn",
            "Ablation: no realised tail (CVaR) term in the training loss.",
            {"loss": {"l_tail": 0.0}},
            e2e=True,
        ),
        ExperimentSpec(
            "full_nocost",
            "nn",
            "Ablation: cost-blind training and optimisation (trading cost is "
            "still charged in the backtest).",
            {"opt": {"lam_turnover": 0.0}, "loss": {"l_turnover": 0.0}},
            e2e=True,
        ),
        ExperimentSpec(
            "full_nosparse",
            "nn",
            "Ablation: no sparsity incentive in the loss (the layer still caps "
            "each name through pi).",
            {"loss": {"l_sparse": 0.0}},
            e2e=True,
        ),
        ExperimentSpec(
            "full_quad",
            "nn",
            "Robustness: quadratic (Herfindahl) diversification instead of the "
            "entropy regulariser.",
            {"opt": {"diversify": "quadratic"}},
            e2e=True,
        ),
        ExperimentSpec(
            "lstm_musigma",
            "nn",
            "Plan-faithful baseline: the selection score is the *fixed* rule "
            "s = mu - l_sigma*sigma - l_cvar*CVaR(name), no learned score head.",
            _predict_only("sparsemax", 25, score_mode="mu_sigma"),
        ),
        ExperimentSpec(
            "full_musigma",
            "nn",
            "Full model with the plan's fixed score rule instead of the learned "
            "score head (everything else identical to ``full``).",
            {"model": {"score_mode": "mu_sigma"}},
            e2e=True,
        ),
    ]
}

MAIN_GRID = list(REGISTRY.keys())


# --------------------------------------------------------------------------- #
# target-return / entropy sweeps
#
# The reference study's panels 2-5 and 13-14 sweep a *target return* tau and the
# panels 8-9 sweep the diversification strength; both axes are implemented here
# as ordinary registry entries so every run stays reproducible from the config
# files alone (no command-line magic).
# --------------------------------------------------------------------------- #
#: requested monthly (per rebalancing period) portfolio return of the hard
#: constraint ``mu @ y >= tau`` used by the target-return sweep
MU_TARGETS: tuple[float, ...] = (0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028)
#: ``opt.lam_div`` values of the entropy sweep (0.0 = no diversification term)
ENTROPY_LAMBDAS: tuple[float, ...] = (0.0, 0.001, 0.002, 0.005, 0.010)
#: encoders compared by both sweeps -- the five architectures of the reference
#: study's panels 2-4 / 8-9
SWEEP_ARCHS: tuple[str, ...] = ("lstm", "gru", "rnn", "mlp", "rbfn")

MU_SWEEP: List[str] = []
ENTROPY_SWEEP: List[str] = []


def mu_experiment_name(mu: float, arch: str) -> str:
    """``mu020_lstm`` = target return 0.020 per period on the LSTM encoder."""
    return f"mu{round(mu * 1000):03d}_{arch}"


def entropy_experiment_name(lam: float, arch: str) -> str:
    """``ent20_lstm`` = ``lam_div = 0.002`` on the LSTM encoder."""
    return f"ent{round(lam * 10000):03d}_{arch}"


for _mu in MU_TARGETS:
    for _arch in SWEEP_ARCHS:
        _name = mu_experiment_name(_mu, _arch)
        REGISTRY[_name] = ExperimentSpec(
            _name,
            "nn",
            f"Target-return sweep: full model with the {_arch.upper()} encoder "
            f"and a hard constraint ``mu @ y >= {_mu:.3f}`` "
            f"(per-period portfolio return).",
            {"model": {"arch": _arch}, "opt": {"mu_target": _mu}},
            group="mu_sweep",
            e2e=True,
        )
        MU_SWEEP.append(_name)

for _lam in ENTROPY_LAMBDAS:
    for _arch in SWEEP_ARCHS:
        _name = entropy_experiment_name(_lam, _arch)
        REGISTRY[_name] = ExperimentSpec(
            _name,
            "nn",
            f"Entropy sweep: full model with the {_arch.upper()} encoder and "
            f"``opt.lam_div = {_lam:.4f}`` "
            + ("(diversification term switched off)." if _lam == 0 else f"({_lam:.4f})."),
            {"model": {"arch": _arch}, "opt": {"lam_div": _lam}},
            group="entropy_sweep",
            e2e=True,
        )
        ENTROPY_SWEEP.append(_name)


# --------------------------------------------------------------------------- #
# tree-ensemble baselines (reference panels 13-14)
#
# ``AdaBoost`` and ``XGBoost`` replace the recurrent encoder, but *not* the rest
# of the pipeline: the same fixed score rule, the same sparsemax selection, the
# same per-name caps and the very same Mean-CVaR layer are used, so the panels
# isolate the forecasting model.
# --------------------------------------------------------------------------- #
TREE_BASELINES: List[str] = []
#: target return of the tree baselines, so they are comparable with ``mu020_*``
TREE_MU_TARGET = 0.020
#: the reference study compares AdaBoost/XGBoost at two target returns
TREE_MU_TARGETS: tuple[float, ...] = (0.020, 0.022)

for _key, _label in (("adaboost", "AdaBoost (50 depth-3 trees)"),
                     ("xgboost", "XGBoost (200 depth-3 trees)")):
    for _mu in TREE_MU_TARGETS:
        _name = (
            f"{_key}_mcvar"
            if _mu == TREE_MU_TARGET
            else f"{_key}_mcvar_mu{round(_mu * 1000):03d}"
        )
        REGISTRY[_name] = ExperimentSpec(
            _name,
            "tree",
            f"Tree baseline: {_label} forecasts mu, the 60-day realised volatility "
            "scales the fixed score rule; the selection and the Mean-CVaR layer are "
            f"the same as in ``mu{round(_mu * 1000):03d}_lstm``.",
            {"opt": {"mu_target": _mu}},
            group="tree_baseline",
            estimator=_key,
        )
        TREE_BASELINES.append(_name)


def spec_for(name: str) -> ExperimentSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown experiment '{name}'; available: {', '.join(REGISTRY)}")
    return REGISTRY[name]


# --------------------------------------------------------------------------- #
# job level
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    experiment: str
    fold: FoldConfig

    @property
    def key(self) -> str:
        return f"{self.experiment}__{self.fold.name}"


def experiment_config(base: Config, name: str) -> Config:
    """Apply the registry overrides of ``name`` to ``base``."""
    spec = spec_for(name)
    return base.replace(**spec.overrides)


def investable_slot_bound(dataset: PanelDataset, periods: Sequence[Period]) -> int:
    """Upper bound on the per-period investable set size over ``periods``.

    The investable set is ``universe | (held & ~sellable)``: a suspended or
    limit-locked position cannot be sold, so it is added *on top of* the
    ``top_liquidity`` liquid names.  A name can only be held if an earlier period
    put it in the universe, so replaying ``pool = universe | (~sellable & pool)``
    from an empty book visits every name a portfolio could still be carrying and
    gives the exact worst case for this fold.
    """
    pool = np.zeros(dataset.panel.num_tickers, dtype=bool)
    best = 0
    for p in periods:
        pool = p.universe | (~p.sellable & pool)
        best = max(best, int(pool.sum()))
    return best


def n_max_slots(
    cfg: Config,
    periods: Optional[Sequence[Period]] = None,
    dataset: Optional[PanelDataset] = None,
) -> int:
    """Fixed number of asset slots used by the optimisation layer.

    Passing the periods of the job (and its dataset) sizes the tensors exactly;
    without them we fall back to the loose worst case ``2 * top_liquidity``.
    """
    base = max(int(cfg.data.top_liquidity), 1)
    if periods and dataset is not None:
        return max(base, investable_slot_bound(dataset, periods))
    if periods:
        return max(base, max(int(p.n_universe) for p in periods))
    return 2 * base


# --------------------------------------------------------------------------- #
# scenarios for the traditional (historical) Mean-CVaR baseline
# --------------------------------------------------------------------------- #
def historical_scenarios(dataset: PanelDataset, p: Period, n_scen: int, horizon: int) -> np.ndarray:
    """``(S, N)`` overlapping historical returns of length ``horizon`` ending at ``p.t``.

    Uses only information available at ``p.t`` (returns up to and including day
    ``t``), so it is a fair, leakage-free baseline.
    """
    d = dataset.daily[: p.t + 1]
    h = max(int(horizon), 1)
    if d.shape[0] <= h + 1:
        return np.zeros((n_scen, dataset.panel.num_tickers), dtype=np.float64)
    cum = np.cumprod(1.0 + np.clip(d, -0.9, 10.0), axis=0)
    m = np.zeros_like(cum)
    m[h:] = cum[h:] / np.maximum(cum[:-h], 1e-8) - 1.0
    s = m[-n_scen:]
    out = np.zeros((n_scen, dataset.panel.num_tickers), dtype=np.float64)
    out[-s.shape[0] :] = np.clip(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), -0.9, 3.0)
    return out


# --------------------------------------------------------------------------- #
# weight providers for the backtester
# --------------------------------------------------------------------------- #
def make_ew_provider(dataset: PanelDataset) -> Callable:
    """1/N over the investable names, keeping the weight of any stuck holding."""
    n = dataset.panel.num_tickers

    def provider(p: Period, prev: np.ndarray):
        w = np.zeros(n, dtype=np.float64)
        stuck = (prev > 0) & ~p.sellable
        w[stuck] = prev[stuck]
        free = p.universe & ~stuck
        k = int(free.sum())
        if k:
            w[free] = max(0.0, 1.0 - w.sum()) / k
        return w, {}

    return provider


def make_hist_meancvar_provider(dataset: PanelDataset, cfg: Config, layer: MeanCVaRLayer) -> Callable:
    n = dataset.panel.num_tickers
    horizon = max(int(dataset.periods[0].horizon), 1) if dataset.periods else 21
    n_scen = int(cfg.model.num_scenarios)

    def provider(p: Period, prev: np.ndarray):
        scen = historical_scenarios(dataset, p, n_scen, horizon)
        held = prev > 0
        investable = p.universe | (held & ~p.sellable)
        idx = np.flatnonzero(investable)
        n_slots = layer.n_assets
        mu = np.zeros(n_slots, dtype=np.float64)
        R = np.zeros((n_scen, n_slots), dtype=np.float64)
        cap = np.zeros(n_slots, dtype=np.float64)
        prev_v = np.zeros(n_slots, dtype=np.float64)
        floor = np.zeros(n_slots, dtype=np.float64)
        k = len(idx)
        if k > n_slots:
            raise ValueError(
                f"period {p.date}: {k} investable names exceed the fixed slot count "
                f"n_max={n_slots}; size the job with "
                f"experiments.n_max_slots(cfg, periods, dataset=ds)"
            )
        sel = idx[:k]
        mu[:k] = np.clip(scen[:, sel].mean(axis=0), -0.5, 0.5)
        R[:, :k] = np.clip(scen[:, sel], -0.5, 0.5)
        stuck = ~p.sellable[sel] & (prev[sel] > 0)
        floor[:k] = np.where(stuck, prev[sel], 0.0)
        # A small selection needs a slackened cap, otherwise sum(y) = 1 becomes
        # infeasible; ``cap`` is never allowed to exceed the (effective) limit.
        y_eff = feasible_cap(cfg.opt.y_max, k)
        cap[:k] = np.maximum(y_eff, floor[:k])
        prev_v[:k] = prev[sel]
        # a stuck holding can consume more than its share of the budget, so any
        # remaining deficit is filled from the headroom of the selected names
        deficit = max(0.0, 1.0 - cap.sum())
        headroom = np.maximum(y_eff - cap, 0.0)
        if deficit > 0 and headroom.sum() > 0:
            cap = cap + deficit * headroom / headroom.sum()
        t = torch.as_tensor
        res = layer(
            t(mu[None, :], dtype=torch.float32),
            t(R[None, :, :], dtype=torch.float32),
            t(cap[None, :], dtype=torch.float32),
            t(prev_v[None, :], dtype=torch.float32),
            t(floor[None, :], dtype=torch.float32),
        )
        y = res["weights"].detach().cpu().numpy().astype(np.float64)[0][:k]
        w = np.zeros(n, dtype=np.float64)
        w[sel] = np.maximum(y, 0.0)
        return w, {"scenarios": len(scen)}

    return provider


def make_nn_provider(trainer: Trainer) -> Callable:
    def provider(p: Period, prev: np.ndarray):
        w, info = trainer.weights_for(p, prev)
        return w, info

    return provider


# --------------------------------------------------------------------------- #
# tree-ensemble baselines
# --------------------------------------------------------------------------- #
def compact_tree_features(win: np.ndarray, msk: np.ndarray) -> np.ndarray:
    """``(L, n, K)`` feature window -> ``(n, 2K + 1)`` tabular matrix.

    Trees cannot consume the raw window, so the same information is summarised
    by the last observed day (``K`` channels), the window mean of every channel
    and the observed fraction.  The first ``K`` columns are the last-day values,
    which is also where the volatility estimate is read from.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        masked = np.where(msk[:, :, None], win, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(masked, axis=0)
        last = masked[-1]
        frac = msk.mean(axis=0)[:, None]
        out = np.concatenate([last, mean, frac], axis=1)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)


def _fit_tree(estimator: str, X: np.ndarray, y: np.ndarray, seed: int):
    if estimator == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=1,
            verbosity=0,
        )
    elif estimator == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

        model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=50,
            learning_rate=0.05,
            random_state=seed,
        )
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"unknown tree estimator '{estimator}'")
    model.fit(X, y)
    return model


def make_tree_provider(
    dataset: PanelDataset,
    cfg: Config,
    layer: MeanCVaRLayer,
    estimator: str,
    horizon: Optional[int] = None,
) -> Callable:
    """Walk-forward AdaBoost / XGBoost baseline for the whole pipeline.

    The model is refitted at every rebalancing date on an *expanding* panel of
    past periods whose labels are already realised (``t <= p.t - horizon``), so
    the baseline never sees the future.  ``mu`` comes from the trees, ``sigma``
    from the 60-day realised volatility of the same feature window, the score is
    the plan's fixed rule, and the selection + Mean-CVaR layer are shared with
    the neural model.
    """
    from .models import gaussian_quantile_grid, score_to_pi_cap
    from .selection import build_selection

    horizon = int(horizon or (dataset.periods[0].horizon if dataset.periods else 21))
    n_scen = int(cfg.model.num_scenarios)
    z_grid = np.asarray(gaussian_quantile_grid(n_scen), dtype=np.float64)
    k_tail = max(1, int(np.ceil((1.0 - float(cfg.opt.alpha)) * n_scen)))
    z_tail_mean = float(np.sort(z_grid)[:k_tail].mean())
    vol_name = f"vol_{int(cfg.features.mid_window)}"
    vol_idx = dataset.feature_names.index(vol_name) if vol_name in dataset.feature_names else None
    lam_sigma = float(cfg.model.score_lambda_sigma)
    lam_cvar = float(cfg.model.score_lambda_cvar)
    selection = build_selection(cfg.model.selection, cfg.model)
    seed = int(cfg.train.seed)
    n_slots = int(layer.n_assets)

    # ---- compact features of every period, built once -------------------- #
    panel: List[tuple] = []
    for q in dataset.periods:
        idx = np.flatnonzero(q.universe)
        if idx.size == 0:
            continue
        win, msk = dataset.window(q.t, q.universe)
        yq = np.asarray(q.fwd_ret[idx], dtype=np.float64)
        keep = np.isfinite(yq)
        panel.append((int(q.t), compact_tree_features(win, msk)[keep], np.clip(yq[keep], -0.5, 0.5)))
    print(f"[tree] {estimator}: cached {len(panel)} periods x "
          f"{sum(len(e[1]) for e in panel)} samples", flush=True)

    state: Dict[str, Any] = {}

    def provider(p: Period, prev: np.ndarray):
        cut = int(p.t) - horizon
        rows_x, rows_y = [], []
        for t_q, X_q, y_q in panel:
            if t_q <= cut:
                rows_x.append(X_q)
                rows_y.append(y_q)
        if not rows_x:  # pragma: no cover - only for the earliest folds
            raise ValueError(f"no realised training period before {p.date}")
        X_tr = np.concatenate(rows_x)
        y_tr = np.concatenate(rows_y)
        model = _fit_tree(estimator, X_tr, y_tr, seed)
        state["fit_t"] = int(p.t)
        state["n_train"] = int(X_tr.shape[0])

        held = prev > 0
        investable = p.universe | (held & ~p.sellable)
        idx = np.flatnonzero(investable)
        win, msk = dataset.window(p.t, investable)
        X_p = compact_tree_features(win, msk)
        mu_raw = np.asarray(model.predict(X_p), dtype=np.float64)
        if vol_idx is not None:
            sig = np.abs(X_p[:, vol_idx]) * np.sqrt(horizon)
        else:  # pragma: no cover - both shipped configs carry vol_20
            sig = np.full(idx.size, float(np.std(y_tr - model.predict(X_tr))))
        sig = np.clip(sig, float(cfg.model.sigma_min), float(cfg.model.sigma_max))
        cvar = -(mu_raw + sig * z_tail_mean)  # Gaussian tail under this sigma
        score = mu_raw - lam_sigma * sig - lam_cvar * cvar

        k = idx.size
        if k > n_slots:
            raise ValueError(
                f"period {p.date}: {k} investable names exceed the fixed slot count "
                f"n_max={n_slots}; size the job with "
                f"experiments.n_max_slots(cfg, periods, dataset=ds)"
            )
        t = torch.as_tensor
        floor = np.zeros(n_slots, dtype=np.float64)
        stuck = ~p.sellable[idx] & held[idx]
        floor[:k] = np.where(stuck, prev[idx], 0.0)
        score_slots = np.full(n_slots, -1e4, dtype=np.float64)
        score_slots[:k] = score
        valid = np.zeros(n_slots, dtype=bool)
        valid[:k] = True
        pi_t, cap_t = score_to_pi_cap(
            t(score_slots, dtype=torch.float32),
            t(valid),
            selection,
            cfg.opt,
            t(floor, dtype=torch.float32),
        )
        mu_slots = np.zeros(n_slots, dtype=np.float64)
        mu_slots[:k] = mu_raw
        R = np.zeros((n_scen, n_slots), dtype=np.float64)
        R[:, :k] = np.clip(
            mu_raw[None, :] + sig[None, :] * z_grid[:, None], -0.9, 3.0
        )
        res = layer(
            t(mu_slots[None, :], dtype=torch.float32),
            t(R[None, :, :], dtype=torch.float32),
            t(np.maximum(cap_t.detach().cpu().numpy(), floor)[None, :], dtype=torch.float32),
            t(prev_slots(prev, idx, n_slots)[None, :], dtype=torch.float32),
            t(floor[None, :], dtype=torch.float32),
        )
        y = res["weights"].detach().cpu().numpy().astype(np.float64)[0][:k]
        w = np.zeros(dataset.panel.num_tickers, dtype=np.float64)
        w[idx] = np.maximum(y, 0.0)
        return w, {
            "n_train": int(X_tr.shape[0]),
            "pi_support": int((pi_t.detach().cpu().numpy() > 1e-6).sum()),
            "mu_book": float(mu_raw @ np.maximum(y, 0.0) / max(float(np.maximum(y, 0.0).sum()), 1e-12)),
        }

    return provider


def prev_slots(prev: np.ndarray, idx: np.ndarray, n_slots: int) -> np.ndarray:
    """Pack the previous weights of the investable names into the fixed slots."""
    out = np.zeros(n_slots, dtype=np.float64)
    out[: idx.size] = prev[idx]
    return out


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def run_job(
    base_cfg: Config,
    experiment: str,
    fold: FoldConfig,
    dataset: Optional[PanelDataset] = None,
    out_root: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train (when needed) and backtest one experiment on one fold.

    Writes every artefact under ``out_root/<experiment>/<fold>/`` and returns a
    dictionary with the headline metrics.
    """
    t_start = time.time()
    cfg = experiment_config(base_cfg, experiment)
    spec = spec_for(experiment)
    ds = dataset if dataset is not None else build_dataset(cfg, verbose=verbose)
    out_dir = Path(out_root or cfg.results_dir) / experiment / fold.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    train_periods = ds.periods_between(fold.train_start, fold.train_end, pad_horizon=False)
    val_periods = ds.periods_between(fold.val_start, fold.val_end, pad_horizon=False)
    test_periods = ds.periods_between(fold.test_start, fold.test_end, pad_horizon=True)
    if not test_periods:
        raise ValueError(f"fold {fold.name} has no test periods")

    # one slot budget for the whole job, sized from the periods actually seen
    seen_periods = [*train_periods, *val_periods, *test_periods]
    n_max = n_max_slots(cfg, seen_periods, dataset=ds)

    history = None
    trainer = None
    layer = None
    timings: Dict[str, Any] = {}

    if spec.kind == "nn":
        trainer = Trainer(cfg, ds, n_max=n_max, n_features=len(ds.feature_names), verbose=verbose)
        t0 = time.time()
        history = trainer.fit(train_periods, val_periods)
        timings["train_seconds"] = time.time() - t0
        timings["epochs_run"] = len(history.epochs)
        timings["best_epoch"] = history.best_epoch
        timings["stopped_early"] = bool(history.stopped_early)
        timings["layer_failures"] = int(history.layer_failures)
        timings.update({f"layer_{k}": v for k, v in trainer.opt_layer.timing.items()})
        pd.DataFrame(history.to_rows()).to_csv(out_dir / "history.csv", index=False,
                                               encoding="utf-8")
        torch.save(trainer.model.state_dict(), out_dir / "model.pt")
        provider = make_nn_provider(trainer)
        layer = trainer.opt_layer
    elif spec.kind == "ew":
        provider = make_ew_provider(ds)
        layer = MeanCVaRLayer(n_max, cfg.model.num_scenarios, differentiable=False)
    elif spec.kind == "hist_meancvar":
        layer = MeanCVaRLayer(
            n_max,
            cfg.model.num_scenarios,
            alpha=cfg.opt.alpha,
            gamma=cfg.opt.gamma,
            lam_turnover=cfg.opt.lam_turnover,
            lam_div=cfg.opt.lam_div if cfg.opt.diversify == "entropy" else 0.0,
            lam_herf=cfg.opt.lam_div if cfg.opt.diversify == "quadratic" else 0.0,
            use_entropy=cfg.opt.diversify == "entropy",
            solver=cfg.opt.solver,
            solver_kwargs=dict(cfg.opt.solver_kwargs),
            differentiable=False,
        )
        provider = make_hist_meancvar_provider(ds, cfg, layer)
    elif spec.kind == "tree":
        layer = MeanCVaRLayer(
            n_max,
            cfg.model.num_scenarios,
            alpha=cfg.opt.alpha,
            gamma=cfg.opt.gamma,
            lam_turnover=cfg.opt.lam_turnover,
            lam_div=cfg.opt.lam_div if cfg.opt.diversify == "entropy" else 0.0,
            lam_herf=cfg.opt.lam_div if cfg.opt.diversify == "quadratic" else 0.0,
            use_entropy=cfg.opt.diversify == "entropy",
            solver=cfg.opt.solver,
            solver_kwargs=dict(cfg.opt.solver_kwargs),
            differentiable=False,
            mu_target=cfg.opt.mu_target,
        )
        provider = make_tree_provider(ds, cfg, layer, spec.estimator or "adaboost")
    else:  # pragma: no cover
        raise ValueError(f"unsupported experiment kind '{spec.kind}'")

    t0 = time.time()
    bt = simulate(
        ds,
        test_periods,
        provider,
        cost_bps=cfg.backtest.cost_bps,
        y_max=cfg.opt.y_max,
        keep_records=True,
    )
    timings["backtest_seconds"] = time.time() - t0
    if layer is not None:
        timings.update({f"solve_{k}": v for k, v in layer.timing.items()})

    # ---- metrics ---------------------------------------------------------
    ret_net = np.asarray(bt.ret_net, dtype=np.float64)
    ret_gross = np.asarray(bt.ret_gross, dtype=np.float64)
    bench = np.asarray(bt.bench_ret, dtype=np.float64)
    turnover = np.asarray(bt.turnover, dtype=np.float64)
    eff = np.asarray(
        [
            (1.0 / float(np.sum(w[w > 0] ** 2))) if np.sum(w) > 0 else 0.0
            for w in bt.target_weights
        ],
        dtype=np.float64,
    )
    # the plan asks for "number of holdings *and* concentration" (section 6)
    conc = concentration_series(bt.target_weights)

    # ``compute_metrics`` returns the same key names for both curves, so the gross
    # figures must be *suffixed*: updating in place would overwrite the net numbers
    # and silently report a cost-free portfolio as the main result.
    net_metrics = compute_metrics(
        ret_net,
        benchmark=bench,
        turnover=turnover,
        n_holdings=np.asarray(bt.n_holdings, dtype=np.float64),
        alpha=cfg.opt.alpha,
    )
    gross_metrics = compute_metrics(ret_gross, benchmark=None, alpha=cfg.opt.alpha)
    metrics = dict(net_metrics)
    for key in (
        "total_return", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "downside_dev", "var_95", "cvar_95",
        "skew", "kurtosis", "omega", "hit_ratio",
    ):
        if key in gross_metrics:
            metrics[f"{key}_gross"] = gross_metrics[key]
    metrics["cost_drag_ann"] = gross_metrics["ann_return"] - net_metrics["ann_return"]
    metrics.update(concentration_metrics(bt.target_weights))
    metrics.update({f"mkt_{k}": v for k, v in market_state_breakdown(ret_net, bench).items()})
    metrics.update(
        {
            "experiment": experiment,
            "group": spec.group,
            "fold": fold.name,
            "index": cfg.data.index,
            "test_start": str(test_periods[0].date),
            "test_end": str(test_periods[-1].date),
            "n_rebalances": len(test_periods),
            "n_max": int(n_max),
            "max_universe": int(max(p.n_universe for p in seen_periods)),
            "max_investable": int(investable_slot_bound(ds, seen_periods)),
            "cost_bps": float(cfg.backtest.cost_bps),
            "cost_total": float(np.asarray(bt.cost, dtype=np.float64).sum()),
            "eff_holdings_mean": float(eff.mean()),
            "seconds": time.time() - t_start,
        }
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "timings.json").write_text(
        json.dumps(_jsonable(timings), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pd.DataFrame(
        {
            "date": bt.dates,
            "ret_net": bt.ret_net,
            "ret_gross": bt.ret_gross,
            "bench": bt.bench_ret,
        }
    ).to_csv(out_dir / "returns.csv", index=False, encoding="utf-8")

    tickers = ds.panel.tickers
    pd.DataFrame(
        bt.target_weights, index=[str(d) for d in bt.rebalance_dates], columns=tickers
    ).to_csv(out_dir / "weights.csv", encoding="utf-8")
    pd.DataFrame(
        {
            "date": [str(d) for d in bt.rebalance_dates],
            "turnover": bt.turnover,
            "traded_notional": bt.traded_notional,
            "cost": bt.cost,
            "n_holdings": bt.n_holdings,
            "eff_holdings": eff,
            "hhi": conc["hhi"],
            "top5_weight": conc["top5_weight"],
            "max_weight": conc["max_weight"],
        }
    ).to_csv(out_dir / "rebalance.csv", index=False, encoding="utf-8")

    return metrics
