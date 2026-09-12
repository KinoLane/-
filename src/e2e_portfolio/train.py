"""Training loop for the end-to-end sparse Mean-CVaR model.

Two training regimes are supported, selected by the loss configuration:

``predict-then-optimise``  (``l_decision = l_tail = l_turnover = 0``)
    the network is trained purely on the Gaussian NLL for the cross-section, and
    the selection + Mean-CVaR layer is applied only when the portfolio is formed.
    This is the classic two-stage baseline.

``end-to-end``  (any of ``l_decision/l_tail/l_turnover`` > 0)
    the convex program is solved inside the forward pass and the network is
    optimised through the optimal weights (cvxpylayers / diffcp).

Implementation notes
--------------------
* Samples are whole cross-sections (one per rebalancing date), processed in
  chronological order.  ``y_prev`` -- needed by the turnover term -- is the
  *detached* optimal weight vector of the previous period, so a gradient step
  accumulates ``batch_size`` consecutive periods before the optimiser is called.
* The number of asset slots ``N`` is fixed for a whole run (the largest universe
  plus a buffer) because the cvxpy problem is compiled once.  Periods with a
  smaller universe simply set ``cap = 0`` on the unused slots, which pins those
  weights to zero.
* Names that are held but cannot be traded any more (suspended, or locked at the
  daily price limit) are kept in the optimisation with a lower bound equal to
  their current weight, so the program cannot "sell" them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .dataset import PanelDataset, Period
from .losses import composite_loss
from .models import E2EPortfolioModel
from .optlayer import build_opt_layer


# --------------------------------------------------------------------------- #
@dataclass
class PeriodTensors:
    """Torch view of one rebalancing period, padded to ``n_max`` slots."""

    x: torch.Tensor          # (N_max, L, K)
    mask: torch.Tensor       # (N_max, L) bool
    valid: torch.Tensor      # (N_max,) bool
    floor: torch.Tensor      # (N_max,) float -- minimum weight (stuck holdings)
    prev: torch.Tensor       # (N_max,) float -- current (drifted) weights
    realised: torch.Tensor   # (N_max,) float
    daily: torch.Tensor      # (H, N_max) float
    compact: np.ndarray      # ticker indices of the used slots
    p: Period

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def drift_weights(weights: np.ndarray, p: Period, horizon: Optional[int] = None) -> np.ndarray:
    """Hold the portfolio over the period: weights drift with realised returns.

    Returns the (renormalised) weights actually held at the end of the holding
    window.  An intermediate daily renormalisation would not change the relative
    proportions, so the product form is exact.
    """
    if weights.sum() <= 0:
        return weights.copy()
    h = int(horizon or p.daily_ret.shape[0])
    growth = np.prod(1.0 + p.daily_ret[:h], axis=0)
    out = weights * growth
    s = out.sum()
    return out / s if s > 0 else weights.copy()


def build_period_tensors(
    dataset: PanelDataset,
    p: Period,
    prev_full: np.ndarray,
    n_max: int,
    dtype=torch.float32,
) -> PeriodTensors:
    """Materialise the feature window, labels and constraints of one period."""
    held = prev_full > 0
    investable = p.universe | (held & ~p.sellable)
    idx = np.flatnonzero(investable)
    if len(idx) > n_max:
        raise ValueError(
            f"period {p.date}: {len(idx)} investable names exceed the fixed slot "
            f"count n_max={n_max}; size the job with "
            f"experiments.n_max_slots(cfg, periods, dataset=ds)"
        )
    n = len(idx)
    win, msk = dataset.window(p.t, investable)  # (L, n, K), (L, n)

    x = np.zeros((n_max,) + win.shape[:1] + win.shape[2:], dtype=np.float32)
    mask = np.zeros((n_max, msk.shape[0]), dtype=bool)
    valid = np.zeros(n_max, dtype=bool)
    floor = np.zeros(n_max, dtype=np.float64)
    prev = np.zeros(n_max, dtype=np.float64)
    realised = np.zeros(n_max, dtype=np.float64)
    daily = np.zeros((p.daily_ret.shape[0], n_max), dtype=np.float32)

    x[:n] = np.transpose(win, (1, 0, 2))
    mask[:n] = msk.T
    valid[:n] = True
    stuck = ~p.sellable[idx] & held[idx]
    floor[:n] = np.where(stuck, prev_full[idx], 0.0)
    prev[:n] = prev_full[idx]
    realised[:n] = p.fwd_ret[idx]
    daily[:, :n] = p.daily_ret[:, idx]

    return PeriodTensors(
        x=torch.as_tensor(x, dtype=dtype),
        mask=torch.as_tensor(mask),
        valid=torch.as_tensor(valid),
        floor=torch.as_tensor(floor, dtype=dtype),
        prev=torch.as_tensor(prev, dtype=dtype),
        realised=torch.as_tensor(realised, dtype=dtype),
        daily=torch.as_tensor(daily, dtype=dtype),
        compact=idx,
        p=p,
    )


# --------------------------------------------------------------------------- #
@dataclass
class TrainHistory:
    epochs: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    stopped_early: bool = False
    seconds: float = 0.0
    layer_failures: int = 0

    def to_rows(self) -> List[Dict[str, float]]:
        rows = []
        for e, rec in enumerate(self.epochs):
            row = {"epoch": e}
            row.update(rec)
            rows.append(row)
        return rows


class Trainer:
    """Owns the model, the optimisation layer and the training loop."""

    def __init__(
        self,
        cfg,
        dataset: PanelDataset,
        n_max: int,
        n_features: int,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.n_max = int(n_max)
        self.n_features = int(n_features)
        self.seed = int(cfg.train.seed if seed is None else seed)
        self.verbose = verbose

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if cfg.train.num_threads:
            torch.set_num_threads(int(cfg.train.num_threads))

        self.device = torch.device("cpu")
        self.opt_layer = build_opt_layer(
            cfg.opt, self.n_max, cfg.model.num_scenarios, differentiable=cfg.opt.differentiable
        )
        self.model = E2EPortfolioModel(n_features, cfg.model, cfg.opt, self.opt_layer).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.use_layer = bool(
            cfg.opt.differentiable and (cfg.loss.l_decision or cfg.loss.l_tail or cfg.loss.l_turnover)
        )
        self.layer_failures = 0

    # ------------------------------------------------------------------ #
    def _solve(self, batch: PeriodTensors, run_layer: bool = True):
        return self.model(
            batch.x,
            batch.mask,
            batch.valid,
            y_prev=batch.prev,
            y_floor=batch.floor,
            run_layer=run_layer,
        )

    def _forward_loss(self, batch: PeriodTensors):
        """Forward pass + loss for a single period."""
        y = None
        if self.use_layer:
            try:
                out = self._solve(batch, run_layer=True)
                y = out.aux["weights"]
            except Exception as exc:  # pragma: no cover - solver robustness
                self.layer_failures += 1
                if self.verbose:
                    print(f"      [warn] layer failed on {batch.p.date}: {type(exc).__name__}: {exc}")
                out = self._solve(batch, run_layer=False)
        else:
            out = self._solve(batch, run_layer=False)

        loss = composite_loss(
            self.cfg.loss,
            mu=out.mu,
            sigma=out.sigma,
            realised=batch.realised,
            daily=batch.daily,
            valid=batch.valid,
            pi=out.pi,
            y=y,
            y_prev=batch.prev,
            alpha=self.cfg.opt.alpha,
            n_candidates=batch.n_valid,
        )
        weights = y.detach() if y is not None else None
        return loss, weights, out

    # ------------------------------------------------------------------ #
    def run_epoch(self, periods: Sequence[Period], train: bool) -> Tuple[Dict[str, float], np.ndarray]:
        self.model.train(train)
        totals: Dict[str, float] = {}
        n_terms = 0
        prev_full = np.zeros(self.dataset.panel.num_tickers, dtype=np.float64)
        pending = 0
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for p in periods:
                batch = build_period_tensors(self.dataset, p, prev_full, self.n_max)
                loss, weights, _ = self._forward_loss(batch)
                if train:
                    (loss.total / max(self.cfg.train.batch_size, 1)).backward()
                    pending += 1
                    if pending >= max(self.cfg.train.batch_size, 1):
                        if self.cfg.train.grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.cfg.train.grad_clip
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        pending = 0
                for k, v in loss.terms.items():
                    totals[k] = totals.get(k, 0.0) + v
                n_terms += 1
                # carry the portfolio state into the next period: with the solved
                # weights as the new target and then drifted by the realised
                # returns of the holding window (exactly like the backtester)
                if weights is not None:
                    w = np.maximum(weights.cpu().numpy().astype(np.float64), 0.0)
                    full = np.zeros_like(prev_full)
                    full[batch.compact] = w[: len(batch.compact)]
                    s = full.sum()
                    full = full / s if s > 0 else full
                    prev_full = drift_weights(full, p)
            if train and pending:
                if self.cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        denom = max(n_terms, 1)
        return {k: v / denom for k, v in totals.items()}, prev_full

    # ------------------------------------------------------------------ #
    def fit(self, train_periods: Sequence[Period], val_periods: Sequence[Period]) -> TrainHistory:
        cfg = self.cfg.train
        hist = TrainHistory()
        best_state = None
        t_start = time.time()
        for epoch in range(int(cfg.epochs)):
            t0 = time.time()
            train_terms, _ = self.run_epoch(train_periods, train=True)
            t_train = time.time() - t0
            t0 = time.time()
            val_terms, _ = self.run_epoch(val_periods, train=False) if val_periods else ({}, None)
            t_val = time.time() - t0
            rec = {f"train_{k}": v for k, v in train_terms.items()}
            rec.update({f"val_{k}": v for k, v in val_terms.items()})
            rec["seconds_train"] = t_train
            rec["seconds_val"] = t_val
            hist.epochs.append(rec)

            val_key = "val_L_decision" if (self.use_layer and cfg.val_metric == "decision") else "val_L_total"
            val = rec.get(val_key, rec.get("val_L_total", float("nan")))
            improved = np.isfinite(val) and val < hist.best_val
            if improved:
                hist.best_val = float(val)
                hist.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if self.verbose:
                print(
                    f"    epoch {epoch + 1:3d}/{cfg.epochs} | train {rec.get('train_L_total', float('nan')):+.4f}"
                    f" | val {val:+.4f} | {t_train:.0f}s" + ("" if improved else ""),
                    flush=True,
                )
            if epoch + 1 >= cfg.min_epochs and epoch - hist.best_epoch >= cfg.patience:
                hist.stopped_early = True
                if self.verbose:
                    print(f"    early stop at epoch {epoch + 1} (best {hist.best_epoch + 1})", flush=True)
                break
        hist.seconds = time.time() - t_start
        hist.layer_failures = self.layer_failures
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return hist

    # ------------------------------------------------------------------ #
    def weights_for(
        self, period: Period, prev_full: np.ndarray, n_max: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Portfolio weights for one period, evaluated without gradient tracking.

        ``prev_full`` must be the *currently held* weight vector (i.e. already
        drifted by the backtester), which is what the optimisation layer needs
        for the turnover term and the no-sell constraint.
        """
        self.model.eval()
        batch = build_period_tensors(self.dataset, period, prev_full, n_max or self.n_max)
        with torch.no_grad():
            try:
                out = self.model(
                    batch.x,
                    batch.mask,
                    batch.valid,
                    y_prev=batch.prev,
                    y_floor=batch.floor,
                    run_layer=True,
                )
                w = out.aux["weights"].cpu().numpy().astype(np.float64)[: len(batch.compact)]
            except Exception as exc:  # pragma: no cover
                self.layer_failures += 1
                print(f"      [warn] test solve failed on {period.date}: {exc}")
                out = self.model(
                    batch.x, batch.mask, batch.valid, y_prev=batch.prev, run_layer=False
                )
                w = out.pi.cpu().numpy().astype(np.float64)[: len(batch.compact)]
        full = np.zeros(self.dataset.panel.num_tickers, dtype=np.float64)
        full[batch.compact] = np.maximum(w, 0.0)
        info = {
            "mu": out.mu.cpu().numpy(),
            "sigma": out.sigma.cpu().numpy(),
            "pi": out.pi.cpu().numpy(),
            "cap": out.cap.cpu().numpy(),
            "compact": batch.compact,
        }
        return full, info
