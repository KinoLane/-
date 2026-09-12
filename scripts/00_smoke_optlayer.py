"""Environment / performance smoke test for the differentiable Mean-CVaR layer.

Run this first after installing ``requirements.txt``::

    .venv\\Scripts\\python.exe scripts\\00_smoke_optlayer.py

It verifies that ``cvxpylayers`` can (a) be constructed for the problem shape
used by the study, (b) be differentiated through, and (c) reports the wall-clock
cost of a forward + backward pass.  The timing numbers are what let us size the
number of scenarios ``S``, the universe ``N`` and the number of epochs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e2e_portfolio.optlayer import CVXPYLAYERS_AVAILABLE, MeanCVaRLayer  # noqa: E402


def main() -> int:
    print(f"cvxpylayers available : {CVXPYLAYERS_AVAILABLE}")
    import cvxpy, diffcp, cvxpylayers, torch as _t  # noqa: F401

    print(f"torch                 : {_t.__version__}")
    print(f"cvxpy                 : {cvxpy.__version__}")
    print(f"cvxpylayers           : {cvxpylayers.__version__}")
    print(f"threads               : {_t.get_num_threads()}")

    ok = True
    for n, s, divergence in [(300, 30, "entropy"), (300, 30, "none"), (300, 60, "quadratic"), (100, 20, "entropy")]:
        layer = MeanCVaRLayer(
            n_assets=n,
            n_scenarios=s,
            alpha=0.95,
            gamma=1.0,
            lam_turnover=0.02,
            lam_div=0.01 if divergence == "entropy" else 0.0,
            lam_herf=0.01 if divergence == "quadratic" else 0.0,
            use_entropy=divergence == "entropy",
        )
        torch.manual_seed(0)
        mu = (torch.randn(1, n) * 0.02).requires_grad_(True)
        sc = torch.randn(1, s, n) * 0.05
        pi = torch.softmax(torch.randn(1, n), dim=-1)
        k_eff = 1.0 / pi.pow(2).sum(-1, keepdim=True)
        cap = (0.10 * torch.clamp(k_eff * pi, max=1.0)).clamp(min=1e-4)
        cap = cap / cap.sum(-1, keepdim=True) * max(1.5, 1.0)  # keep feasible
        prev = torch.zeros(1, n)
        prev[0, :20] = 1 / 20

        t0 = time.time()
        out = layer(mu, sc, cap, prev)
        t_fwd = time.time() - t0
        y = out["weights"]
        port = (y * mu).sum()
        t0 = time.time()
        port.backward()
        t_bwd = time.time() - t0
        grad_ok = mu.grad is not None and torch.isfinite(mu.grad).all().item()
        sum_ok = abs(float(y.sum()) - 1.0) < 1e-4
        print(
            f"N={n:4d} S={s:3d} div={divergence:9s} | fwd {t_fwd*1000:7.1f} ms  "
            f"bwd {t_bwd*1000:8.1f} ms | sum(y)={float(y.sum()):.6f}  "
            f"grad_ok={grad_ok}  feasible={sum_ok}"
        )
        ok = ok and grad_ok and sum_ok
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
