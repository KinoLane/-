"""Check the README's headline tables against the artefacts they claim to quote.

Every markdown table in ``README.md`` whose header starts with「实验」and contains
「年化净收益」is parsed, the backticked experiment name in its first column is looked
up in the ``report/pooled.csv`` of the section's run, and each numeric cell is
compared at the printed precision.

``§9.9–§9.13`` are transcribed by hand from ``results/logs/summary_tables.md``
(which [10_summary_tables.py](./10_summary_tables.py) regenerates straight from the
runs), so those sections get a second pass: every number printed inside one of their
markdown tables must appear verbatim in that dump.

``§9.2/§9.3/§9.7/§9.8`` are prose sections that quote statistics *derived* from the
artefacts (period-weighted averages, trigger censuses, before/after diffs), which no
table check can catch: a wrong weighting or a stale census still "looks" like a
plausible number.  ``check_claims()`` therefore recomputes those statistics from
``selection_diag.csv`` / ``metrics.json`` / ``pooled.csv`` / ``fill_control/*.json``
and requires the README to contain the recomputed rendering.

This is what backs the「README 表格逐格复核」claim: run it after editing the README.

Usage
-----
    python scripts/07_verify_readme.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FOLDS = ("f1", "f2", "f3")
MAIN_RUN = "results/csi300_20260910_235309"
EXT_RUN = "results/csi300_extra_returns"
OLD_EXT_RUN = "results/_invalidated_extra_returns_prefixfillbug"
FULL_UNIVERSE_RUN = "results/csi300_full_universe"
SCORE_RUN = "results/csi300_score_rule_20260911"

#: §9.8 table 2/3 rows: (row label, old run dir, new run dir, experiment, fold prefix)
FILL_REGRESSION_JOBS = [
    ("主网格 full_nocost f2", MAIN_RUN, "results/csi300_fill_fix_check_abl", "full_nocost", "f2"),
    ("主网格 full_quad f2", MAIN_RUN, "results/csi300_fill_fix_check_abl", "full_quad", "f2"),
    ("补充网格 full_musigma f3", "results/csi300_score_rule_20260911",
     "results/csi300_fill_fix_check", "full_musigma", "f3"),
    ("19 通道 f2", OLD_EXT_RUN, EXT_RUN, "full", "f2"),
    ("19 通道 f3", OLD_EXT_RUN, EXT_RUN, "full", "f3"),
    ("完整池 full f2", FULL_UNIVERSE_RUN, "results/csi300_fill_fix_check_fu", "full", "f2"),
]

#: README column header -> (pooled.csv column, is the README value in percent)
COLUMNS = {
    "年化净收益": ("ann_return", True),
    "年化波动": ("ann_vol", True),
    "Sharpe": ("sharpe", False),
    "最大回撤": ("max_drawdown", True),
    "日 CVaR95": ("cvar_95", True),
    "CVaR95(日)": ("cvar_95", True),
    "年化换手": ("turnover_ann", False),
    "持仓数(>1e-3)": ("holdings_mean", False),
    "持仓数": ("holdings_mean", False),
    "有效持仓": ("eff_holdings_mean", False),
    "成本拖累": ("cost_drag_ann", True),
    "beta": ("beta", False),
    "β": ("beta", False),
    "信息比": ("info_ratio", False),
}

#: README section -> run whose report/pooled.csv the section quotes
RUNS = {
    "9.1": "results/csi300_20260910_235309",
    "9.5": "results/csi300_score_rule_20260911",
    "9.6": "results/csi300_full_universe",
}

_POOLED: dict[str, dict[str, dict[str, str]]] = {}


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_cell(text: str) -> tuple[float, bool, int] | None:
    """Return ``(value, is_percent, decimals)`` for a markdown cell, or ``None``."""
    t = text.replace("*", "").replace("`", "").replace("−", "-").strip()
    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()
    if t.lower() in ("", "n/a"):
        return None
    try:
        value = float(t)
    except ValueError:
        return None
    decimals = len(t.split(".")[1]) if "." in t else 0
    return value, pct, decimals


def load_pooled(run: str) -> dict[str, dict[str, str]]:
    with (ROOT / run / "report" / "pooled.csv").open(encoding="utf-8") as fh:
        return {row["experiment"]: row for row in csv.DictReader(fh)}


def lookup(run: str, name: str) -> dict[str, str] | None:
    """Find ``name`` in the section's run, falling back to the other runs.

    The README quotes a few rows across runs (the ``*_musigma`` rows of the main
    table come from the supplementary grid, and the supplementary table repeats
    main-grid rows for comparison), so the fallback mirrors that.
    """
    for candidate in [run] + [r for r in RUNS.values() if r != run]:
        if candidate not in _POOLED:
            _POOLED[candidate] = load_pooled(candidate)
        if name in _POOLED[candidate]:
            return _POOLED[candidate][name]
    return None


def check(run: str, section: str) -> tuple[int, int, list[str]]:
    checked = failed = 0
    problems: list[str] = []
    header: list[tuple[str, bool] | None] | None = None
    active = False
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^#{2,4}\s+(\d+(?:\.\d+)?)", line)
        if heading:
            active = heading.group(1) == section
        if not line.startswith("|"):
            header = None
            continue
        if not active:
            continue
        cols = cells(line)
        if "年化净收益" in cols:  # a results table header
            header = [COLUMNS.get(c) for c in cols]
            continue
        if header is None or set(cols[0]) <= set("-: "):
            continue
        name = re.search(r"`([A-Za-z_0-9]+)`", cols[0])
        if not name:
            continue
        row = lookup(run, name.group(1))
        if row is None:
            problems.append(f"[{section}] {name.group(1)}: not in pooled.csv")
            failed += 1
            continue
        for cell, spec in zip(cols, header):
            if spec is None:
                continue
            parsed = parse_cell(cell)
            if parsed is None:
                continue
            want, pct, decimals = parsed
            key, _ = spec
            got = float(row[key]) * (100.0 if pct else 1.0)
            checked += 1
            if abs(round(got, decimals) - want) > 0.5 * 10 ** (-decimals) + 1e-9:
                failed += 1
                problems.append(f"[{section}] {name.group(1)}.{key}: README {want} vs csv {got}")
    return checked, failed, problems


def check_against_tables_dump() -> tuple[int, list[str]]:
    """Every number in the §9.9–§9.13 tables must occur in ``summary_tables.md``.

    Those sections are transcribed by hand from the dump, and a single mistyped
    digit in a Sharpe ratio is exactly the kind of error a reader cannot spot.
    """
    dump = (ROOT / "results" / "logs" / "summary_tables.md").read_text(encoding="utf-8")
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("### 9.9"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("## 10."))
    checked = 0
    problems: list[str] = []
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= set("|-: "):
            continue
        for cell in cells(stripped):
            for num in re.findall(r"-?\d+\.\d+", cell):
                checked += 1
                if num not in dump and num.lstrip("-") not in dump:
                    problems.append(f"[9.9-9.13] {num} not in summary_tables.md: {stripped[:90]}")
    return checked, problems


def section(number: str) -> list[str]:
    """Lines of README ``§number``, up to the next heading of the same or higher level."""
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if re.match(rf"^#{{2,4}}\s+{re.escape(number)}(?![\d.])", ln))
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    fence = False
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        head = re.match(r"^(#{1,4})\s", ln)
        if head and len(head.group(1)) <= level:
            return lines[start:i]
    return lines[start:]


def prose(number: str) -> str:
    """Section text with markdown noise removed, so any rendering of a value matches."""
    text = "\n".join(section(number))
    text = text.replace("**", "").replace("`", "")
    return text.replace(" / ", "/").replace("－", "-")


def fold_of(run: str, exp: str, prefix: str) -> str:
    base = ROOT / run / exp
    hits = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix + "_"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{run}/{exp}: {len(hits)} dirs start with {prefix}_")
    return hits[0]


def diag(run: str, exp: str, fold: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / run / exp / fold / "selection_diag.csv")


def metrics(run: str, exp: str, fold: str) -> dict:
    path = ROOT / run / exp / fold / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def report_row(run: str, exp: str, fold: str) -> dict[str, str]:
    """One row of a run's aggregated ``report/selection_diagnostics.csv``.

    Those columns are period means (``04_selection_diagnostics.py``), which is what
    §9.7's table quotes; §9.8 quotes the per-period minimum instead.
    """
    with (ROOT / run / "report" / "selection_diagnostics.csv").open(encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh)
                if row["experiment"] == exp and row["fold"] == fold]
    if len(rows) != 1:
        raise KeyError(f"{run}/{exp}/{fold}: {len(rows)} rows in report CSV")
    return rows[0]


def digest(run: str, exp: str, fold: str) -> str:
    """Encoder fingerprint of one job as recorded in the run's aggregated report."""
    return report_row(run, exp, fold)["encoder_sha256"]


def pooled(run: str) -> dict[str, dict[str, str]]:
    with (ROOT / run / "report" / "pooled.csv").open(encoding="utf-8") as fh:
        return {row["experiment"]: row for row in csv.DictReader(fh)}


def pct1(fraction: float) -> str:
    """``1.0`` -> ``100``, ``0.0417`` -> ``4.2`` (the README never writes ``100.0``)."""
    text = f"{fraction * 100:.1f}"
    return text[:-2] if text.endswith(".0") else text


def wmean(frame: pd.DataFrame, column: str) -> float:
    """Period-weighted mean — the README's §9.3 statistic (weights = ``n_valid``)."""
    return float((frame[column] * frame["n_valid"]).sum() / frame["n_valid"].sum())


def table_body(lines: list[str], marker: str) -> list[str]:
    """Data rows of the first markdown table whose header line contains ``marker``."""
    start = next(i for i, ln in enumerate(lines) if marker in ln)
    body = []
    for ln in lines[start + 1:]:
        if not ln.strip().startswith("|"):
            break
        if set(ln.strip()) <= set("|-: "):
            continue
        body.append(ln)
    return body


class Claims:
    """Collects「recomputed value → must appear in this section」claims."""

    def __init__(self, number: str) -> None:
        self.number = number
        self.text = prose(number)
        self.lines = section(number)
        # Same lines with markdown emphasis removed, so a table row can be located
        # by its plain-text label (``| 主网格 `full_quad` f2 | …``).
        self.plain = [ln.replace("**", "").replace("`", "") for ln in self.lines]
        self.checked = 0
        self.problems: list[str] = []

    def want(self, value: str, label: str, where: str | None = None,
             rows: list[str] | None = None) -> None:
        self.checked += 1
        if rows is None:
            haystack = self.text if where is None else "\n".join(
                ln for ln in self.plain if where in ln)
            if where is not None and not haystack:
                self.problems.append(f"[{self.number}] {label}: no row matching {where!r}")
                return
        else:
            haystack = "\n".join(rows)
            if not rows:
                self.problems.append(f"[{self.number}] {label}: row not found")
                return
        if value not in haystack:
            self.problems.append(f"[{self.number}] {label}: README lacks {value!r}")

    def forbid(self, value: str, label: str) -> None:
        """A value the section must *not* quote (stale statistic, wrong weighting)."""
        self.checked += 1
        if value in self.text:
            self.problems.append(f"[{self.number}] {label}: README still quotes {value!r}")


def claims_93() -> Claims:
    """§9.3: period-weighted logit statistics of the frozen-encoder trio."""
    c = Claims("9.3")
    frozen = pd.concat([diag(MAIN_RUN, exp, fold_of(MAIN_RUN, exp, pref))
                        for exp in ("lstm_softmax", "lstm_sparsemax") for pref in FOLDS])
    spread_w, thresh_w = wmean(frozen, "score_spread"), wmean(frozen, "spar_thresh")
    c.want(f"{spread_w:.2e}", "加权 logit 极差")
    c.want(f"{thresh_w:.2e}", "加权稀疏阈值")
    c.forbid(f"{frozen.score_spread.mean():.2e}", "未加权的 logit 极差（旧错误）")
    for prefix in FOLDS:
        sub = frozen[frozen.fold.str.startswith(prefix)]
        below = int((sub.score_spread < sub.spar_thresh).sum())
        c.want(f"{pct1(below / len(sub))}%", f"{prefix} 低于阈值的期数占比")
        if below == len(sub):
            # the README spells the count out only for the fold that never crosses
            c.want(f"{below}/{len(sub)}", f"{prefix} 低于阈值的期数")
    book = pooled(MAIN_RUN)
    for exp in ("lstm_softmax", "lstm_sparsemax"):
        sub = pd.concat([diag(MAIN_RUN, exp, fold_of(MAIN_RUN, exp, pref))
                         for pref in FOLDS])
        c.want(f"{wmean(sub, 'pi_support'):.1f}", f"{exp} 平均支撑集")
        c.want(f"{float(book[exp]['sharpe']):.2f}", f"{exp} 拼接 Sharpe")
    c.want(f"{float(book['lstm_sparsemax']['sharpe']) - float(book['lstm_softmax']['sharpe']):.2f}",
           "两种选择层的 Sharpe 差")
    full = pd.concat([diag(MAIN_RUN, "full", fold_of(MAIN_RUN, "full", pref))
                      for pref in FOLDS])
    spread_f, thresh_f = wmean(full, "score_spread"), wmean(full, "spar_thresh")
    c.want(f"{spread_f:.2e}", "full 折加权 logit 极差")
    c.want(f"{spread_f / thresh_f:.1f}", "极差 / 阈值倍数")
    below = int((full.score_spread < full.spar_thresh).sum())
    c.want(f"{below}/{len(full)}", "full 折低于阈值的期数")
    c.want(f"{wmean(full, 'pi_support'):.1f}", "full 折平均支撑集")
    c.want(f"{full.book_n.mean():.1f}", "full 折平均账面持仓")
    return c


def claims_census() -> dict[str, dict[str, int]]:
    """Trigger census straight from the per-period diagnostics (shared by §9.2/§9.8)."""
    spec = importlib.util.spec_from_file_location(
        "fill_regression", ROOT / "scripts" / "06_fill_regression.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups: dict[str, dict[str, int]] = {}
    for run, rule in module.CENSUS_RUNS:
        acc = groups.setdefault(rule, {"jobs": 0, "hit": 0, "triggers": 0, "periods": 0})
        for key, value in module.census_run(run).items():
            acc[key] += value
    return groups


def claims_92() -> Claims:
    """§9.2's prose census: how many diagnosed jobs ever entered the fill branch."""
    c = Claims("9.2")
    old = claims_census()["旧"]
    c.want(f"{old['jobs']} 个", "已诊断作业数")
    c.want(f"{old['hit']} 个", "至少触发一期的作业数")
    c.want(f"{old['triggers']}/{old['periods']}", "触发期次 / 总期次")
    c.want(f"{old['jobs'] - old['hit']} 个", "从未触发填充的作业数")
    return c


def claims_97() -> Claims:
    """§9.7: the 19-channel feature switch, and the fill bug it exposed."""
    c = Claims("9.7")
    for run, tag in ((MAIN_RUN, "17 通道"), (EXT_RUN, "19 通道")):
        folds = [metrics(run, "full", fold_of(run, "full", prefix)) for prefix in FOLDS]
        for key, fmt, label in (("ann_return", "{:+.2f}%", "年化"),
                                ("sharpe", "{:+.3f}", "Sharpe"),
                                ("max_drawdown", "{:+.2f}%", "回撤"),
                                ("turnover_ann", "{:.2f}", "换手")):
            scale = 100.0 if fmt.endswith("%") else 1.0
            c.want("/".join(fmt.format(fold[key] * scale) for fold in folds),
                   f"{tag} 逐折{label}")
        row = pooled(run)["full"]
        c.want(f"{float(row['ann_return']) * 100:.2f}%", f"{tag} 拼接年化")
        c.want(f"{float(row['sharpe']):.3f}", f"{tag} 拼接 Sharpe")
        c.want(f"{float(row['max_drawdown']) * 100:.2f}%", f"{tag} 拼接最大回撤")
    row = pooled(EXT_RUN)["full"]
    for key, fmt, label in (("ann_vol", "{:.2f}%", "年化波动"),
                            ("cvar_95", "{:.2f}%", "日 CVaR95"),
                            ("cost_drag_ann", "{:.2f}%", "成本拖累"),
                            ("holdings_mean", "{:.1f}", "持仓数"),
                            ("eff_holdings_mean", "{:.1f}", "有效持仓"),
                            ("beta", "{:.2f}", "β"),
                            ("info_ratio", "{:.3f}", "信息比")):
        value = float(row[key]) * (100.0 if fmt.endswith("%") else 1.0)
        c.want(fmt.format(value), f"19 通道 {label}")
    for run, tag in ((OLD_EXT_RUN, "修复前"), (EXT_RUN, "修复后")):
        for prefix in FOLDS:
            fold = fold_of(run, "full", prefix)
            frame = diag(run, "full", fold)
            c.want(f"{frame.pi_support.mean():.1f}", f"{tag} 支撑集 ({prefix})")
            c.want(f"{frame.book_n.mean():.1f}", f"{tag} 持仓数 ({prefix})")
            c.want(f"{float(report_row(run, 'full', fold)['cap_budget']):.3f}",
                   f"{tag} cap_budget 均值 ({prefix})")
            c.want(f"{pct1((frame.deficit > 0).mean())}%", f"{tag} 填充触发率 ({prefix})")
            if prefix == "f1":
                c.want(f"{frame.cap_budget.min():.3f}", f"{tag} f1 逐期最低 cap_budget")
    old, new = (metrics(run, "full", fold_of(run, "full", "f1"))
                for run in (OLD_EXT_RUN, EXT_RUN))
    for key in ("ann_return", "sharpe", "max_drawdown", "cvar_95", "turnover_ann",
                "holdings_mean"):
        c.checked += 1
        if old[key] != new[key]:
            c.problems.append(f"[9.7] f1 折 {key} 修复前后不一致: {old[key]} vs {new[key]}")
    for key, fmt in (("ann_return", "{:.2f}%"), ("sharpe", "{:.4f}"),
                     ("max_drawdown", "{:.2f}%"), ("cvar_95", "{:.2f}%"),
                     ("turnover_ann", "{:.4f}")):
        scale = 100.0 if fmt.endswith("%") else 1.0
        c.want(fmt.format(new[key] * scale), f"f1 折 {key} 的逐位相同值")
    c.want(digest(EXT_RUN, "full", fold_of(EXT_RUN, "full", "f1"))[:16], "f1 折编码器指纹")
    for run in (OLD_EXT_RUN, EXT_RUN):
        for prefix in ("f2", "f3"):
            tag = "修复前" if run == OLD_EXT_RUN else "修复后"
            c.want(digest(run, "full", fold_of(run, "full", prefix))[:8],
                   f"{tag} {prefix} 编码器指纹前缀")
    return c


def claims_98() -> Claims:
    """§9.8: census tables, the retrain table and the frozen-weight table."""
    c = Claims("9.8")
    groups = claims_census()
    census_table = table_body(c.plain, "已诊断的「实验×折」作业")
    for rule, name in (("旧", "旧（把缺口摊到整个横截面）"), ("新", "新（按分数水位线加宽选择）")):
        acc = groups[rule]
        rows = [ln for ln in census_table if name in ln]
        c.want(f"| {name} | {acc['jobs']} | {acc['hit']} | "
               f"{acc['triggers']} / {acc['periods']} |", f"{rule}规则汇总行", rows=rows)
    spec = importlib.util.spec_from_file_location(
        "fill_regression_rows", ROOT / "scripts" / "06_fill_regression.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for run, rule in module.CENSUS_RUNS:
        stat = module.census_run(run)
        c.want(f"| {stat['jobs']} | {stat['hit']} | {stat['triggers']} / {stat['periods']} |",
               f"{run} 明细行", where=run)
    retrain = table_body(c.plain, "触发期次(旧→新)")
    frozen = table_body(c.plain, "仅规则带来的差")
    for label, old_run, new_run, exp, prefix in FILL_REGRESSION_JOBS:
        for run, pref, side in ((old_run, prefix, "旧"), (new_run, prefix, "新")):
            fold = fold_of(run, exp, pref)
            frame, met = diag(run, exp, fold), metrics(run, exp, fold)
            rows = [ln for ln in retrain if label in ln]
            for value, what in (
                (f"{int((frame.deficit > 0).sum())}/{len(frame)}", "触发期次"),
                (f"{frame.cap_budget.min():.3f}", "最低 cap_budget"),
                (f"{met['holdings_mean']:.2f}", "持仓数"),
                (f"{met['ann_return'] * 100:.2f}%", "年化净收益"),
                (f"{met['sharpe']:.3f}", "Sharpe"),
            ):
                c.want(value, f"表二 {label} {side} {what}", rows=rows)
    old_met = metrics(FULL_UNIVERSE_RUN, "full", fold_of(FULL_UNIVERSE_RUN, "full", "f2"))
    new_fold = fold_of(FILL_REGRESSION_JOBS[-1][2], "full", "f2")
    new_met = metrics(FILL_REGRESSION_JOBS[-1][2], "full", new_fold)
    old_frame = diag(FULL_UNIVERSE_RUN, "full", fold_of(FULL_UNIVERSE_RUN, "full", "f2"))
    new_frame = diag(FILL_REGRESSION_JOBS[-1][2], "full", new_fold)
    c.want(f"均值 {old_frame.cap_budget.mean():.3f} → {new_frame.cap_budget.mean():.3f}",
           "完整池 f2 cap_budget 均值（旧 → 新）", where="完整池")
    c.want(f"缺口最大 {old_frame.deficit.max():.3f}", "完整池 f2 旧规则缺口最大值")
    c.want(f"+{(new_met['ann_return'] - old_met['ann_return']) * 100:.2f}% / "
           f"+{new_met['sharpe'] - old_met['sharpe']:.3f}", "完整池 f2 重训的联合效应",
           rows=[ln for ln in c.plain if "重训后变成" in ln])
    for path in sorted((ROOT / "results" / "fill_control").glob("*.json")):
        control = json.loads(path.read_text(encoding="utf-8"))
        base = control["run"].split("/")[-1]
        label = next((lbl for lbl, old_run, _n, exp, pref in FILL_REGRESSION_JOBS
                      if (old_run.split("/")[-1], exp, fold_of(old_run, exp, pref))
                      == (base, control["experiment"], control["fold"])), None)
        if label is None:
            c.checked += 1
            c.problems.append(f"[9.8] fill_control/{path.name} 没有对应的表三行")
            continue
        rows = [ln for ln in frozen if label in ln]
        water = control["waterfall"]
        for value in (f"{control['reference_metrics_ann'] * 100:.2f}%",
                      f"{control['reference_metrics_sharpe']:.3f}",
                      f"{control['reference_metrics_holdings']:.2f}",
                      f"{water['ann_return'] * 100:.2f}%",
                      f"{water['sharpe']:.3f}",
                      f"{water['holdings_mean']:.2f}",
                      f"{water['ann_return'] * 100 - control['reference_metrics_ann'] * 100:+.2f}%",
                      f"{water['sharpe'] - control['reference_metrics_sharpe']:+.3f}",
                      f"{water['holdings_mean'] - control['reference_metrics_holdings']:.2f}"):
            c.want(value, f"表三 {label}（{path.name}）", rows=rows)
    return c


def claims_117() -> Claims:
    """§11.7's data claims: the trigger census and the four pieces of trigger evidence.

    §11.7 is the section a reviewer reads last, and it quotes the same facts as §9.7/§9.8
    in prose form — exactly where stale numbers survived the last rewrite once already.
    """
    c = Claims("11.7")
    census = claims_census()["旧"]
    c.want(f"{census['jobs']} 个", "已诊断作业数")
    c.want(f"{census['hit']} 个", "至少触发一期的作业数")
    c.want(f"{census['triggers']}/{census['periods']} 期", "触发期次 / 总期次")
    c.forbid("1133", "旧的普查分母（已作废）")
    # (row, run, experiment, fold prefix, trigger periods, does §11.7 quote the
    #  min cap_budget / max deficit of this row?)
    rows = (("主网格 full_nocost", MAIN_RUN, "full_nocost", "f2", 13, True),
            ("主网格 full_quad", MAIN_RUN, "full_quad", "f2", 2, True),
            ("完整池 full", FULL_UNIVERSE_RUN, "full", "f2", 12, False),
            ("补充网格 full_musigma", SCORE_RUN, "full_musigma", "f3", 4, False))
    for label, run, exp, prefix, triggers, quote_bounds in rows:
        frame = diag(run, exp, fold_of(run, exp, prefix))
        c.want(f"{triggers}/{len(frame)} 期", f"{label} 触发期次")
        if quote_bounds:
            c.want(f"{frame.cap_budget.min():.3f}", f"{label} 最低 cap_budget")
            c.want(f"{frame.deficit.max():.3f}", f"{label} 最大缺口")
        if label.startswith("补充网格"):
            fired = frame[frame.deficit > 0].sort_values("date")
            for date in fired.date:
                c.want(str(date), "触发滑期日期")
            for deficit in fired.deficit:
                c.want(f"{deficit * 100:.1f}%", "触发期缺口")
            c.want(f"{fired.pi_support.min():.0f}–{fired.pi_support.max():.0f} 只",
                   "触发期 π 支撑集区间")
    old_f2 = diag(OLD_EXT_RUN, "full", fold_of(OLD_EXT_RUN, "full", "f2"))
    old_f3 = diag(OLD_EXT_RUN, "full", fold_of(OLD_EXT_RUN, "full", "f3"))
    c.want(f"{len(old_f2)} 个调仓期", "19 通道 f2 期数")
    for frame, prefix in ((old_f2, "f2"), (old_f3, "f3")):
        c.want(f"{pct1((frame.deficit > 0).mean())}%", f"19 通道 {prefix} 触发率")
        c.want(f"{frame.book_n.mean():.1f} 只", f"19 通道 {prefix} 账面持仓")
    c.want(f"{old_f2.pi_support.mean():.1f} 只", "19 通道 f2 π 支撑集")
    c.want(f"{float(report_row(OLD_EXT_RUN, 'full', fold_of(OLD_EXT_RUN, 'full', 'f2'))['cap_budget']):.3f}",
           "19 通道 f2 cap_budget 均值")
    return c


def check_claims() -> tuple[int, list[str]]:
    """Recompute the derived statistics quoted by §9.2/§9.3/§9.7/§9.8/§11.7."""
    checked = 0
    problems: list[str] = []
    for claim in (claims_92(), claims_93(), claims_97(), claims_98(), claims_117()):
        checked += claim.checked
        problems += claim.problems
        print(f"section {claim.number} (derived statistics): "
              f"{claim.checked - len(claim.problems)}/{claim.checked} claims match")
    return checked, problems


def main() -> int:
    total_checked = total_failed = 0
    for section, run in RUNS.items():
        checked, failed, problems = check(run, section)
        total_checked += checked
        total_failed += failed
        print(f"section {section} ({run}): {checked - failed}/{checked} cells match")
        for p in problems:
            print("   " + p)
    dump_checked, dump_problems = check_against_tables_dump()
    total_checked += dump_checked
    total_failed += len(dump_problems)
    print(f"section 9.9-9.13 (results/logs/summary_tables.md): "
          f"{dump_checked - len(dump_problems)}/{dump_checked} numbers present")
    for p in dump_problems:
        print("   " + p)
    claim_checked, claim_problems = check_claims()
    total_checked += claim_checked
    total_failed += len(claim_problems)
    for p in claim_problems:
        print("   " + p)
    print(f"TOTAL {total_checked - total_failed}/{total_checked} cells match")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
