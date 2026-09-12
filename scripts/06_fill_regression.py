"""Regression report for the deficit-fill fix (README section 9.8).

Two independent controls are combined here:

* **retrain** -- the affected jobs were re-run from scratch with the fixed rule, so
  the numbers differ both because the rule changed at the trigger periods *and*
  because the fill sits inside the differentiable layer and therefore changes what
  the network learns;
* **frozen weights** -- ``scripts/05_fill_control.py`` loads the *pre-fix* model and
  re-evaluates the test fold with both rules, which isolates the rule itself.

Usage
-----
    python scripts/06_fill_regression.py          # prints markdown tables
    python scripts/06_fill_regression.py --write  # also writes results/fill_control/

Table **C** is the census of README §9.8: it walks every ``selection_diag.csv`` on
disk and counts the jobs whose diagnostics ever show a positive fill deficit, which
is exactly「哪些作业走进了填充分支」.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONTROL_DIR = RESULTS / "fill_control"

#: label -> (pre-fix run, post-fix run, experiment, fold)
CASES = [
    ("主网格 full_nocost f2", "csi300_20260910_235309", "csi300_fill_fix_check_abl", "full_nocost", "f2_2020_2021"),
    ("主网格 full_quad f2", "csi300_20260910_235309", "csi300_fill_fix_check_abl", "full_quad", "f2_2020_2021"),
    ("补充网格 full_musigma f3", "csi300_score_rule_20260911", "csi300_fill_fix_check", "full_musigma", "f3_2022_2026"),
    ("19 通道 full f2", "_invalidated_extra_returns_prefixfillbug", "csi300_extra_returns", "full", "f2_2020_2021"),
    ("19 通道 full f3", "_invalidated_extra_returns_prefixfillbug", "csi300_extra_returns", "full", "f3_2022_2026"),
    ("完整池 full f2", "csi300_full_universe", "csi300_fill_fix_check_fu", "full", "f2_2020_2021"),
]

#: run dir -> fill rule that produced that dir's per-period diagnostics
#: (``selection_diag.csv`` is a *replay* of the selection layer with the code as of
#: the moment it was written, so a run keeps the old-rule numbers until re-diagnosed)
CENSUS_RUNS = [
    ("_invalidated_extra_returns_prefixfillbug", "旧"),
    ("csi300_20260910_235309", "旧"),
    ("csi300_score_rule_20260911", "旧"),
    ("csi300_full_universe", "旧"),
    ("csi300_extra_returns", "新"),
    ("csi300_fill_fix_check", "新"),
    ("csi300_fill_fix_check_abl", "新"),
    ("csi300_fill_fix_check_fu", "新"),
]

#: control json written by 05_fill_control.py
CONTROLS = {
    "主网格 full_nocost f2": "full_nocost__f2_2020_2021.json",
    "主网格 full_quad f2": "full_quad__f2_2020_2021.json",
    "补充网格 full_musigma f3": "full_musigma__f3_2022_2026.json",
    "19 通道 full f2": "full__f2_2020_2021.json",
    "19 通道 full f3": "full__f3_2022_2026.json",
    "完整池 full f2": "full_universe__full__f2_2020_2021.json",
}


def metrics(run: str, exp: str, fold: str) -> dict | None:
    p = RESULTS / run / exp / fold / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def diag(run: str, exp: str, fold: str) -> pd.DataFrame | None:
    p = RESULTS / run / exp / fold / "selection_diag.csv"
    if p.exists():
        return pd.read_csv(p)
    agg = RESULTS / run / "report" / "selection_diagnostics.csv"
    if agg.exists():
        d = pd.read_csv(agg)
        d = d[(d.experiment == exp) & (d.fold == fold)]
        if len(d):
            return None  # aggregate only: no per-period columns
    return None


def trigger_stats(run: str, exp: str, fold: str) -> tuple[int, int, float, float]:
    """(periods, triggers, min cap_budget, max deficit) from the per-period csv."""
    d = diag(run, exp, fold)
    if d is None or "deficit" not in d:
        return 0, 0, float("nan"), float("nan")
    return (
        len(d),
        int((d["deficit"] > 0).sum()),
        float(d["cap_budget"].min()),
        float(d["deficit"].max()),
    )


def fmt(v, pct=False, dec=2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v * 100:.{dec}f}%" if pct else f"{v:.{dec}f}"


def census_run(run: str) -> dict[str, int]:
    """Count diagnosed jobs of one run dir and how many ever entered the fill branch."""
    jobs = sorted((RESULTS / run).glob("*/*/selection_diag.csv"))
    periods = triggers = hit = 0
    for path in jobs:
        d = pd.read_csv(path)
        periods += len(d)
        n = int((d["deficit"] > 0).sum())
        triggers += n
        hit += n > 0
    return {"jobs": len(jobs), "hit": hit, "triggers": triggers, "periods": periods}


def census() -> tuple[list[str], list[str]]:
    """（分规则汇总表, 分运行目录明细表）.

    The two tables answer「有多少作业真的走进了填充分支」— the question the README
    §9.8 census answers — straight from the per-period diagnostics on disk.
    """
    detail = {}
    for run, rule in CENSUS_RUNS:
        detail[run] = (rule, census_run(run))
    groups: dict[str, dict[str, int]] = {}
    for rule, stat in detail.values():
        acc = groups.setdefault(rule, {"jobs": 0, "hit": 0, "triggers": 0, "periods": 0})
        for k, v in stat.items():
            acc[k] += v
    rows_a = [
        "| 填充规则 | 已诊断的「实验×折」作业 | 至少触发一期 | 触发期次 / 总期次 |",
        "| --- | --- | --- | --- |",
    ]
    for rule, acc in groups.items():
        name = ("旧（把缺口摊到整个横截面）" if rule == "旧"
                else "新（按分数水位线加宽选择）")
        rows_a.append(f"| {name} | {acc['jobs']} | {acc['hit']} | {acc['triggers']} / {acc['periods']} |")
    rows_b = [
        "| 运行目录 | 填充规则 | 已诊断作业 | 至少触发一期 | 触发期次 / 总期次 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run, (rule, stat) in detail.items():
        rows_b.append(f"| `{run}` | {rule} | {stat['jobs']} | {stat['hit']} | "
                      f"{stat['triggers']} / {stat['periods']} |")
    return rows_a, rows_b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    lines_a = [
        "| 受影响作业 | 期次(旧) | 触发期次(旧→新) | cap_budget 最低(旧→新) | 账面持仓(旧→新) | 年化净收益(旧→新) | Sharpe(旧→新) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines_b = [
        "| 受影响作业 | 报告值(旧规则) 年化/Sharpe/持仓 | 同一权重+新规则 年化/Sharpe/持仓 | 仅推理的差值 |",
        "| --- | --- | --- | --- |",
    ]
    for label, pre_run, post_run, exp, fold in CASES:
        n0, t0, cb0, df0 = trigger_stats(pre_run, exp, fold)
        n1, t1, cb1, _ = trigger_stats(post_run, exp, fold)
        m0, m1 = metrics(pre_run, exp, fold), metrics(post_run, exp, fold)
        per = f"{n0}" if n0 else "n/a"
        trig = f"{t0}/{n0} → {t1}/{n1}" if n0 else "n/a"
        caps = f"{fmt(cb0, dec=3)} → {fmt(cb1, dec=3)}" if n0 else "n/a"
        book = (
            f"{fmt(m0['holdings_mean'])} → {fmt(m1['holdings_mean'])}"
            if m0 and m1 else "n/a"
        )
        ann = (
            f"{fmt(m0['ann_return'], pct=True)} → {fmt(m1['ann_return'], pct=True)}"
            if m0 and m1 else "n/a"
        )
        shp = (
            f"{fmt(m0['sharpe'], dec=3)} → {fmt(m1['sharpe'], dec=3)}"
            if m0 and m1 else "n/a"
        )
        lines_a.append(f"| {label} | {per} | {trig} | {caps} | {book} | {ann} | {shp} |")

        cpath = CONTROL_DIR / CONTROLS[label]
        if cpath.exists():
            c = json.loads(cpath.read_text(encoding="utf-8"))
            for side in ("spread", "waterfall"):
                c[side] = {k: v for k, v in c[side].items() if k != "_seconds"}
            reported = (
                f"{fmt(c['reference_metrics_ann'], pct=True)} / "
                f"{fmt(c['reference_metrics_sharpe'], dec=3)} / "
                f"{fmt(c['reference_metrics_holdings'])}"
            )
            frozen = (
                f"{fmt(c['waterfall']['ann_return'], pct=True)} / "
                f"{fmt(c['waterfall']['sharpe'], dec=3)} / "
                f"{fmt(c['waterfall']['holdings_mean'])}"
            )
            delta = (
                f"{fmt(c['waterfall']['ann_return'] - c['reference_metrics_ann'], pct=True)} / "
                f"{fmt(c['waterfall']['sharpe'] - c['reference_metrics_sharpe'], dec=3)} / "
                f"{fmt(c['waterfall']['holdings_mean'] - c['reference_metrics_holdings'])}"
            )
        else:
            reported = frozen = delta = "n/a"
        lines_b.append(f"| {label} | {reported} | {frozen} | {delta} |")

    print("### A. 用修复后的规则原样重跑（重训练 + 规则改动）\n")
    print("\n".join(lines_a))
    print("\n### B. 冻结旧权重、只换填充规则（隔离规则本身）\n")
    print("\n".join(lines_b))
    rows_a, rows_b = census()
    print("\n### C. 触发范围普查（每期诊断重放，数据来自 selection_diag.csv）\n")
    print("\n".join(rows_a))
    print()
    print("\n".join(rows_b))

    if args.write:
        CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        out = CONTROL_DIR / "fill_regression.md"
        out.write_text(
            "# 缺陷填充修复的回归对照\n\n"
            "## A. 用修复后的规则原样重跑（重训练 + 规则改动）\n\n" + "\n".join(lines_a) + "\n\n"
            "## B. 冻结旧权重、只换填充规则（隔离规则本身）\n\n" + "\n".join(lines_b) + "\n\n"
            "## C. 触发范围普查（每期诊断重放，数据来自 `selection_diag.csv`）\n\n"
            + "\n".join(rows_a) + "\n\n" + "\n".join(rows_b) + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
