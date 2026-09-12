"""End-to-end differentiable Mean-CVaR portfolio learning with sparse neural
stock selection.

Package layout
--------------
``data``        : load the bundled index panel tensors (``findata/*/tensor``).
``features``    : causal feature engineering + cross-sectional standardisation.
``dataset``     : rebalancing calendar, walk-forward folds, per-period samples.
``models``      : shared-LSTM encoder + return/risk (scenario) prediction heads.
``selection``   : Top-k / Softmax / Sparsemax differentiable selection.
``optlayer``    : cvxpylayers Mean-CVaR portfolio optimisation layer.
``losses``      : the five joint-training loss terms.
``backtest``    : daily portfolio simulation with transaction costs.
``metrics``     : annualised return, Sharpe/Sortino, VaR/CVaR, MDD, Calmar, ...
``train``       : walk-forward training loop.
``experiments`` : the full comparison + ablation grid required by the protocol.
"""

__version__ = "1.0.0"
