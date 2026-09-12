"""The report script must recover missing concentration metrics from disk.

Runs produced before the concentration block existed still carry their
per-rebalance weight matrices, so ``scripts/03_report.py`` recomputes
``hhi`` / top-5 share / largest weight instead of requiring a retrain.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "e2e_report_script", ROOT / "scripts" / "03_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def legacy_run(tmp_path: Path) -> Path:
    """A run directory shaped like the pre-concentration grids."""
    run = tmp_path / "csi300_legacy"
    fold_dir = run / "full" / "f1_2018_2019"
    fold_dir.mkdir(parents=True)

    weights = pd.DataFrame(
        [[0.5, 0.5, 0.0], [0.2, 0.2, 0.0]],
        index=["2020-01-31", "2020-02-28"],
        columns=["a.SH", "b.SH", "c.SH"],
    )
    weights.to_csv(fold_dir / "weights.csv")

    pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-28"],
            "turnover": [0.0, 0.6],
            "traded_notional": [0.0, 0.6],
            "cost": [0.0, 0.0009],
            "n_holdings": [2, 2],
            "eff_holdings": [2.0, 2.0],
        }
    ).to_csv(fold_dir / "rebalance.csv", index=False)

    rng = np.random.default_rng(3)
    daily = rng.normal(0.0005, 0.01, 40)
    pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-31", periods=40).strftime("%Y-%m-%d"),
            "ret_net": daily,
            "ret_gross": daily + 1e-4,
            "bench": daily * 0.8,
        }
    ).to_csv(fold_dir / "returns.csv", index=False)

    # note: no concentration keys -- this is what an old run looks like
    (fold_dir / "metrics.json").write_text(
        json.dumps({"experiment": "full", "fold": "f1_2018_2019", "sharpe": 0.5}),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"folds": [{"name": "f1_2018_2019"}], "n_jobs": 1, "n_ok": 1}),
        encoding="utf-8",
    )
    return run


def test_load_jobs_backfills_concentration_from_weights(legacy_run):
    report = _load_report_module()
    jobs, _ = report.load_jobs(legacy_run)
    metrics = jobs[("full", "f1_2018_2019")]["metrics"]
    # HHI: mean of (0.25+0.25) and (0.04+0.04) = mean(0.5, 0.08)
    assert metrics["hhi_mean"] == pytest.approx(0.29)
    assert metrics["top5_weight_mean"] == pytest.approx(0.7)
    assert metrics["max_weight_mean"] == pytest.approx(0.35)

    # the same numbers must be recomputable straight from the weight matrix
    weights = pd.read_csv(legacy_run / "full" / "f1_2018_2019" / "weights.csv", index_col=0)
    from e2e_portfolio.metrics import concentration_metrics

    assert concentration_metrics(weights.to_numpy()) == pytest.approx(
        {k: metrics[k] for k in ("hhi_mean", "top5_weight_mean", "max_weight_mean")}
    )


def test_backfilled_rebalance_columns_reach_the_pooled_table(legacy_run):
    report = _load_report_module()
    jobs, _ = report.load_jobs(legacy_run)
    reb = jobs[("full", "f1_2018_2019")]["rebalance"]
    for col in ("hhi", "top5_weight", "max_weight"):
        assert col in reb.columns
    pooled = report.pooled_table(jobs, ["full"], ["f1_2018_2019"], 0.95)
    row = pooled.iloc[0]
    assert row["hhi_mean"] == pytest.approx(0.29)
    assert row["top5_weight_mean"] == pytest.approx(0.7)
    assert row["max_weight_mean"] == pytest.approx(0.35)


def test_backfill_is_idempotent_and_never_overwrites(legacy_run):
    report = _load_report_module()
    jobs, _ = report.load_jobs(legacy_run)
    fold_dir = legacy_run / "full" / "f1_2018_2019"
    jobs[("full", "f1_2018_2019")]["metrics"]["hhi_mean"] = 0.42  # a hand-edited value
    assert report.ensure_concentration(jobs[("full", "f1_2018_2019")], fold_dir) is False
    assert jobs[("full", "f1_2018_2019")]["metrics"]["hhi_mean"] == 0.42


def test_holdings_mean_is_recomputed_without_solver_dust(legacy_run):
    """An old run counted solver residuals as positions; the report must not."""
    report = _load_report_module()
    fold_dir = legacy_run / "full" / "f1_2018_2019"
    weights = pd.read_csv(fold_dir / "weights.csv", index_col=0)
    weights["d.SH"] = 1e-7  # dust left behind by the conic solver
    weights.to_csv(fold_dir / "weights.csv")

    metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["holdings_mean"] = float(weights.shape[1])  # the inflated legacy value
    (fold_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    jobs, _ = report.load_jobs(legacy_run)
    job = jobs[("full", "f1_2018_2019")]
    assert job["metrics"]["holdings_mean"] == pytest.approx(2.0)
    assert job["rebalance"]["n_holdings"].tolist() == [2.0, 2.0]
    # the repaired numbers must land back in the artefacts so that a reader of the
    # raw run directory sees the same holdings count as the report
    on_disk = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["holdings_mean"] == pytest.approx(2.0)
    assert on_disk["hhi_mean"] == pytest.approx(0.29)
    reb_on_disk = pd.read_csv(fold_dir / "rebalance.csv")
    assert reb_on_disk["n_holdings"].tolist() == [2.0, 2.0]
    # a second pass must be a no-op now that the value matches the artefacts
    assert report.ensure_holdings(job, fold_dir) is False
