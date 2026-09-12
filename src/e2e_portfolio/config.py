"""Typed configuration objects.

Every experiment is fully described by a :class:`Config`, which is loaded from a
YAML file (see ``configs/``).  Keeping the configuration explicit is what makes
the study reproducible: each run directory stores the exact YAML that produced
it, together with the resolved git/package versions.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    #: root of the bundled dataset (``.../findata``)
    root: str = r"D:\金创\findata"
    #: index name, e.g. ``csi300``
    index: str = "csi300"
    #: keep only the ``top_liquidity`` most liquid members at each rebalance
    #: (``0`` keeps every member of the index)
    top_liquidity: int = 0
    #: drop ST / *ST names from the investable universe
    exclude_st: bool = True
    #: drop names that are locked at the price limit on the rebalance day
    exclude_limit_locked: bool = True
    #: restrict the sample to [start_date, end_date] (``None`` = everything)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    #: directory that caches the parquet-derived extras (ST flags, amount, ...)
    cache_dir: str = "cache"


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    #: number of past trading days fed to the encoder
    lookback: int = 60
    #: cross-sectional winsorisation bound applied to standardised features
    clip: float = 3.0
    #: windows of the momentum / volatility / liquidity features
    short_window: int = 5
    mid_window: int = 20
    long_window: int = 60
    #: add the 1-day simple and log return of every stock as input channels
    #: (the plan's feature list mentions "收益率与对数收益率"; they are off by
    #: default so that the frozen 17-channel reference runs stay comparable)
    include_daily_return: bool = False


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    #: encoder architecture shared by every stock:
    #: ``lstm`` (default) | ``gru`` | ``rnn``   -> recurrent sequence encoder
    #: ``mlp``  -> time-distributed MLP + mask-mean pooling
    #: ``rbfn`` -> time-distributed RBF (Gaussian kernel) layer + mask-mean pooling
    #: Only the recurrent variants carry a state and keep the historical
    #: ``encoder.lstm.*`` parameter names, so ``lstm`` stays bit-compatible with
    #: the frozen runs.
    arch: str = "lstm"
    #: hidden size of the shared encoder
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    #: size of the per-stock MLP head
    head_hidden: int = 64
    #: ``gaussian`` -> scenarios mu + sigma * z_s (deterministic quantile grid)
    #: ``direct``   -> the head emits S scenarios per stock directly
    scenario_mode: str = "gaussian"
    #: number of return scenarios handed to the CVaR layer
    num_scenarios: int = 60
    #: initial value of the predicted monthly volatility
    sigma_init: float = 0.08
    #: bounds applied to the predicted volatility
    sigma_min: float = 0.005
    sigma_max: float = 0.60
    #: selection mechanism: ``topk`` | ``softmax`` | ``sparsemax`` | ``none``
    selection: str = "sparsemax"
    #: number of names kept by Top-k selection
    topk: int = 25
    #: initial temperature of the sparsemax/softmax score -> weight mapping
    temperature_init: float = 1.0
    #: learn the temperature jointly with the network
    learn_temperature: bool = True
    #: how the selection score is produced:
    #: ``head``     -> a free per-stock linear head (default, and what the main
    #:                 grid uses: the score is learned end to end)
    #: ``mu_sigma`` -> the fixed rule of the research plan,
    #:                 ``s_i = mu_i - l_sigma * sigma_i - l_cvar * CVaR_i``
    score_mode: str = "head"
    #: weight of the predicted volatility in the ``mu_sigma`` score
    score_lambda_sigma: float = 0.5
    #: weight of the per-name scenario CVaR in the ``mu_sigma`` score
    score_lambda_cvar: float = 0.5


# --------------------------------------------------------------------------- #
# differentiable Mean-CVaR layer
# --------------------------------------------------------------------------- #
@dataclass
class OptConfig:
    #: CVaR confidence level (alpha): CVaR_alpha of the loss distribution
    alpha: float = 0.95
    #: risk-aversion gamma on the CVaR term.  With Gaussian scenarios the layer
    #: objective reduces to ``-(1+gamma)*mu + gamma*CVaR_alpha(loss)``, so
    #: gamma = 0.1 makes a 1% expected return worth about one unit of 1%
    #: monthly volatility (0.1 * 2.06 * 5% = 1%).
    gamma: float = 0.1
    #: L1 turnover penalty charged *inside* the layer.  Defaults to the same
    #: cost the backtester charges (15 bp one-way), so the optimal weights are
    #: cost-aware; ``0.0`` gives the cost-blind ablation.
    lam_turnover: float = 0.0015
    #: strength of the diversification regulariser (entropy / Herfindahl)
    lam_div: float = 0.002
    #: diversification formulation: ``entropy`` (sum y log y) | ``quadratic``
    #: (sum y^2) | ``none``.  Entropy costs ~5x more in the backward pass, so
    #: ``quadratic`` is offered as a cheaper convex alternative.
    diversify: str = "entropy"
    #: per-name cap y_max, applied through the selection weights pi
    y_max: float = 0.10
    #: hard target-return constraint ``mu @ y >= tau``.  ``None`` (default)
    #: disables it, keeping the frozen runs bit-compatible.  ``tau`` is a
    #: *monthly* portfolio return, e.g. 0.02 = 2% per rebalancing period; the
    #: layer clamps it per period to the highest level the box constraints
    #: actually allow (see ``MeanCVaRLayer.feasible_mu``), so an unreachable
    #: target never turns the convex program infeasible.
    mu_target: Optional[float] = None
    #: conic solver handed to cvxpylayers (only ``SCS`` is supported by diffcp)
    solver: str = "SCS"
    solver_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: use the differentiable (cvxpylayers) layer; when False the same convex
    #: program is solved with plain cvxpy and treated as a non-differentiable
    #: block (used for the "traditional Mean-CVaR" baselines)
    differentiable: bool = True


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    #: weights of L = l_pred*L_pred + l_decision*L_decision + l_tail*L_tail
    #:                 + l_turnover*L_turnover + l_sparse*L_sparse
    l_pred: float = 0.1
    l_decision: float = 1.0
    l_tail: float = 0.5
    l_turnover: float = 1.0
    l_sparse: float = 0.1
    #: scale of the Gaussian NLL prediction loss (``mse`` uses plain MSE)
    pred_kind: str = "nll"
    #: annualisation used to convert the per-period losses to a comparable scale
    annual_scale: float = 12.0
    #: cost of one unit of one-way L1 turnover, in return units; must match
    #: ``BacktestConfig.cost_bps`` so that the trainer optimises the objective
    #: the backtester measures (15 bp one-way -> 0.0015 per unit of sum|dy|)
    cost_per_unit_turnover: float = 0.0015


# --------------------------------------------------------------------------- #
# training / backtest
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    #: early-stopping patience (in epochs) on the validation objective
    patience: int = 8
    #: minimum number of epochs before early stopping can trigger
    min_epochs: int = 5
    #: torch device; ``auto`` picks cuda when available
    device: str = "auto"
    #: base random seed (fold ``i`` uses ``seed + i``)
    seed: int = 20240101
    #: torch intra-op threads (0 = leave the default)
    num_threads: int = 0
    #: cap on the number of training periods per epoch (0 = all); useful to
    #: trade compute for statistics when the sample is long
    max_train_periods_per_epoch: int = 0
    #: validation objective used for early stopping
    #: ``loss`` (full training loss) | ``sharpe`` | ``ann_return``
    val_metric: str = "loss"


@dataclass
class BacktestConfig:
    #: one-way transaction cost in basis points of the traded notional
    cost_bps: float = 15.0
    #: rebalancing frequency: ``monthly`` | ``every_n_days``
    rebalance: str = "monthly"
    #: used when ``rebalance == 'every_n_days'``
    rebalance_every: int = 20
    #: realised return used for the reported equity curve
    #: ``daily`` uses the official close-to-close daily returns, ``period``
    #: uses the compounded holding-period return of each rebalancing period
    return_kind: str = "daily"


@dataclass
class FoldConfig:
    """One walk-forward split.  All dates are inclusive and ISO formatted."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def validate(self) -> "FoldConfig":
        """Reject splits that would overlap (and therefore leak information)."""
        ok = (
            self.train_start <= self.train_end < self.val_start <= self.val_end < self.test_start
            <= self.test_end
        )
        if not ok:
            raise ValueError(
                f"fold '{self.name}' is not a valid walk-forward split: train "
                f"[{self.train_start}, {self.train_end}] < val [{self.val_start}, "
                f"{self.val_end}] < test [{self.test_start}, {self.test_end}]"
            )
        return self


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def _build_dataclass(cls, raw: Dict[str, Any]):
    """Instantiate a config dataclass from a plain dict (unknown keys rejected)."""
    kwargs = {k: v for k, v in raw.items() if k in {f.name for f in fields(cls)}}
    unknown = set(raw) - set(kwargs)
    if unknown:
        raise KeyError(f"unknown fields for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptConfig = field(default_factory=OptConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    folds: List[FoldConfig] = field(default_factory=list)
    #: experiments to run (names of :data:`e2e_portfolio.experiments.REGISTRY`)
    experiments: List[str] = field(default_factory=list)
    #: directory that receives every artefact of a run
    results_dir: str = "results"
    #: run name; defaults to ``<index>_<timestamp>``
    run_name: Optional[str] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        raw = dict(raw)
        folds_raw = raw.pop("folds", [])
        nested = {
            "data": DataConfig,
            "features": FeatureConfig,
            "model": ModelConfig,
            "opt": OptConfig,
            "loss": LossConfig,
            "train": TrainConfig,
            "backtest": BacktestConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            section = nested.get(key)
            kwargs[key] = (
                _build_dataclass(section, value) if section is not None and isinstance(value, dict) else value
            )
        cfg = cls(**kwargs)
        cfg.folds = [
            (f if isinstance(f, FoldConfig) else FoldConfig(**f)).validate() for f in folds_raw
        ]
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)

    def replace(self, **overrides: Dict[str, Any]) -> "Config":
        """Return a deep copy with nested dataclass fields overridden.

        ``cfg.replace(model={"selection": "topk"}, loss={"l_tail": 0.0})``
        """
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            if not hasattr(new, key):
                raise KeyError(f"unknown config section: {key}")
            section = getattr(new, key)
            if dataclasses.is_dataclass(section):
                for k, v in value.items():
                    if not hasattr(section, k):
                        raise KeyError(f"unknown config field: {key}.{k}")
                    setattr(section, k, v)
            else:
                setattr(new, key, value)
        return new
