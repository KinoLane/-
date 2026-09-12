# 交付物清单（DELIVERABLES）

本文件是**交付索引**：这个仓库里到底有哪些东西、各自在哪、是什么、怎么验证。
内容与仓库当前状态一一对应（`main` 分支），不涉及任何未入库的外部文件。

```
e2e_meancvar/              ← 交付根目录（Git 仓库）
├── README.md              ← 主文档（§1 环境 … §11 逐条对应）114 KB
├── DELIVERABLES.md        ← 本文件
├── requirements.txt       ← 依赖（产出 §9 全部结果的那套版本）
├── .gitignore
├── configs/               ← 16 个 YAML 实验配置
├── src/e2e_portfolio/     ← 13 个源码模块
├── scripts/               ← 12 个流水线脚本（00–10）
├── tests/                 ← 12 个测试文件 / 134 个用例
└── results/               ← 实验结果、图表、日志（2506 个已跟踪文件）
```

---

## 1. 文档

| 交付物 | 路径 | 说明 |
| --- | --- | --- |
| 主文档 | [README.md](./README.md) | §1 环境 / §2 数据 / §3 模块与源文件对应 / §4 复现步骤 / §5 实验设计 / §6 实现要点与踩坑 / §7 成本与口径 / §8 测试 / §9 结果与结论（§9.1–§9.13）/ §10 局限 / §11 与原研究方案逐条对应 |
| 本清单 | [DELIVERABLES.md](./DELIVERABLES.md) | 交付索引与验证入口 |
| 图集说明 | [results/figures_reference/README.md](./results/figures_reference/README.md) | 11 张交付图 → 数据来源对照、配色/版式实测值、与范例的真实差异逐条说明 |
| 机器可读图清单 | [results/figures_reference/panels.json](./results/figures_reference/panels.json) | 每张图对应的实验名、指标定义、来源 run 目录、对照范例文件名（`reference_file`） |
| 数值权威来源 | [results/logs/summary_tables.md](./results/logs/summary_tables.md) | README §9.9–§9.13 表格的全部数值即从此处抄录（由 `10_summary_tables.py` 从各 run 重算） |

---

## 2. 源码（src/e2e_portfolio/）

13 个模块，职责与研究方案的模块对应关系见 README §3；优化层的数学形式与联合损失的定义同样在 README §3。

| 模块 | 内容 |
| --- | --- |
| [data.py](./src/e2e_portfolio/data.py) | 读取行情张量、npz 缓存、涨跌停/停牌掩码、指数序列 |
| [features.py](./src/e2e_portfolio/features.py) | 17 个因果特征 + 截面标准化 |
| [dataset.py](./src/e2e_portfolio/dataset.py) | 调仓日历、walk-forward 折、逐期样本（期权池/流动性/未来收益） |
| [models.py](./src/e2e_portfolio/models.py) | 共享 LSTM 编码器 + 收益头 μ + 波动头 σ + 选择分数；高斯参数化场景生成 |
| [selection.py](./src/e2e_portfolio/selection.py) | `TopKSelection`（直通估计）、`SoftmaxSelection`、`SparsemaxSelection`、`IdentitySelection` |
| [optlayer.py](./src/e2e_portfolio/optlayer.py) | `cvxpylayers` 包装的 Mean-CVaR 锥规划 + 解析 KKT 反传 |
| [losses.py](./src/e2e_portfolio/losses.py) | 预测损失、决策损失、尾部风险、换手成本、稀疏选择 5 项 |
| [train.py](./src/e2e_portfolio/train.py) | 逐期张量打包、早停、最佳权重回滚、梯度裁剪 |
| [backtest.py](./src/e2e_portfolio/backtest.py) | 日频模拟、权重漂移、线性交易成本、cap 审计 |
| [metrics.py](./src/e2e_portfolio/metrics.py) | 年化收益/波动、Sharpe/Sortino、VaR/CVaR、MDD、Calmar、信息比率、α/β、涨跌市分解 |
| [experiments.py](./src/e2e_portfolio/experiments.py) | 实验注册表、历史情景 Mean-CVaR 基线、单作业 runner、结果落盘 |
| [config.py](./src/e2e_portfolio/config.py) | 强类型 YAML 配置（`from_yaml` / `to_yaml` / `replace`） |
| [__init__.py](./src/e2e_portfolio/__init__.py) | 包导出 |

---

## 3. 配置（configs/，16 个）

| 配置 | 对应实验 / README 章节 |
| --- | --- |
| [csi300.yaml](./configs/csi300.yaml) | 主参考网格（11 实验 × 3 折）= §9.1 |
| [csi300_mu_sweep.yaml](./configs/csi300_mu_sweep.yaml) · [csi300_mu_sweep_ext.yaml](./configs/csi300_mu_sweep_ext.yaml) | μ 扫描（Fig2–5、Fig13–14 数据源）= §9.9 |
| [csi300_entropy_sweep.yaml](./configs/csi300_entropy_sweep.yaml) · [csi300_entropy_sweep_ext.yaml](./configs/csi300_entropy_sweep_ext.yaml) | 熵扫描（Fig8–9 数据源）= §9.10 |
| [csi300_tree_baselines.yaml](./configs/csi300_tree_baselines.yaml) | AdaBoost / XGBoost 基线 = §9.11 |
| [csi300_score_rule.yaml](./configs/csi300_score_rule.yaml) | 选择分数改用方案给定公式 = §9.5 |
| [csi300_full_universe.yaml](./configs/csi300_full_universe.yaml) | 不截断流动性的完整沪深 300 = §9.6 |
| [csi300_extra_returns.yaml](./configs/csi300_extra_returns.yaml) | 特征通道 17 → 19 敏感性 = §9.7 |
| [csi300_fill_fix_check.yaml](./configs/csi300_fill_fix_check.yaml) · [..._abl.yaml](./configs/csi300_fill_fix_check_abl.yaml) · [..._fu.yaml](./configs/csi300_fill_fix_check_fu.yaml) | 缺陷填充修复的 3 组回归 = §9.8 |
| [csi500_reference.yaml](./configs/csi500_reference.yaml) · [csi500_arch_compare.yaml](./configs/csi500_arch_compare.yaml) | 中证 500 = §9.12（Fig11 曲线来源） |
| [sse50_reference.yaml](./configs/sse50_reference.yaml) · [sse50_arch_compare.yaml](./configs/sse50_arch_compare.yaml) | 上证 50 = §9.12（Fig12 曲线来源） |

---

## 4. 脚本（scripts/，12 个）

| 脚本 | 作用 |
| --- | --- |
| [00_smoke_optlayer.py](./scripts/00_smoke_optlayer.py) | 冒烟：cvxpylayers 反传是否可用 |
| [00_smoke_train.py](./scripts/00_smoke_train.py) | 冒烟：训练回路是否跑通 |
| [01_prepare_data.py](./scripts/01_prepare_data.py) | 原始行情 → 特征缓存（`results/_cache/*.npz`） |
| [02_run_experiments.py](./scripts/02_run_experiments.py) | 按配置批量跑实验（并行作业、断点续跑） |
| [03_report.py](./scripts/03_report.py) | 生成 `report/report.md` + 8 张诊断图 + 汇总 CSV |
| [04_selection_diagnostics.py](./scripts/04_selection_diagnostics.py) | 选择层稀疏性来源诊断（softmax / sparsemax / top-k） |
| [05_fill_control.py](./scripts/05_fill_control.py) · [06_fill_regression.py](./scripts/06_fill_regression.py) | 填充缺陷修复的对照实验与回归汇总 |
| [07_verify_readme.py](./scripts/07_verify_readme.py) | README 数字审计：每个数字回查结果文件 |
| [08_paper_style_figures.py](./scripts/08_paper_style_figures.py) | ⚠️ **早期版式尝试，已被 09 取代**（见 §7） |
| [09_reference_figures.py](./scripts/09_reference_figures.py) | ✅ **最终 11 张交付图的生成器** |
| [10_summary_tables.py](./scripts/10_summary_tables.py) | 从各 run 重算 §9.9–§9.12 表格 → `results/logs/summary_tables.md` |

---

## 5. 测试（tests/，134 个用例）

[tests/](./tests) 覆盖可微优化层的梯度正确性（含有限差分对照）、模型前向、训练/回测口径、指标实现、
选择层、损失项、数据与特征、扫描配置、报告生成与配置解析。

---

## 6. 结果（results/）

### 6.1 实验 run

每个 run 目录 = 一个实验网格；每个实验下三折 walk-forward（`f1_2018_2019` / `f2_2020_2021` / `f3_2022_2026`），
每折包含：

```
config.yaml      实际生效的完整配置
metrics.json     折内全部指标
returns.csv      日频净收益（图与汇总的原始来源）
weights.csv      每期持仓权重
rebalance.csv    每期调仓明细 + 实现的组合熵
history.csv      逐 epoch 训练曲线
model.pt         训练好的权重
timings.json     耗时
```

| 目录 | README 章节 | 内容 |
| --- | --- | --- |
| [csi300_20260910_235309/](./results/csi300_20260910_235309) | §9.1 | 参考运行：11 实验 × 3 折；含 `report/`（report.md、pooled.csv、per_fold.csv、selection_diagnostics.csv、summary.json + 8 张诊断图） |
| [csi300_score_rule_20260911/](./results/csi300_score_rule_20260911) | §9.5 | 选择分数改用方案给定公式 |
| [csi300_full_universe/](./results/csi300_full_universe) | §9.6 | 完整沪深 300（不截断流动性） |
| [csi300_extra_returns/](./results/csi300_extra_returns) | §9.7 | 特征通道 17 → 19 敏感性 |
| [csi300_mu_sweep_20260912/](./results/csi300_mu_sweep_20260912) | §9.9 | μ 扫描主网格（21 实验） |
| [csi300_mu_sweep_ext_20260912/](./results/csi300_mu_sweep_ext_20260912) | §9.9 | μ 扫描补充（MLP / RBFN） |
| [csi300_entropy_sweep_20260912/](./results/csi300_entropy_sweep_20260912) | §9.10 | 熵扫描 |
| [csi300_tree_baselines_20260912/](./results/csi300_tree_baselines_20260912) | §9.11 | AdaBoost / XGBoost 基线 |
| [csi500_arch_compare_20260913/](./results/csi500_arch_compare_20260913) · [sse50_arch_compare_20260913/](./results/sse50_arch_compare_20260913) | §9.12.1 | 两个池的 5 结构对照（Fig11–12 曲线来源） |
| [csi500_reference_20260912/](./results/csi500_reference_20260912) · [sse50_reference_20260912/](./results/sse50_reference_20260912) | §9.12 | 两个池的基准/对照实验 |
| [csi300_fill_fix_check/](./results/csi300_fill_fix_check) · [..._abl/](./results/csi300_fill_fix_check_abl) · [..._fu/](./results/csi300_fill_fix_check_fu) | §9.8 | 填充修复的 3 组回归 |
| [\_invalidated_extra_returns_prefixfillbug/](./results/_invalidated_extra_returns_prefixfillbug) | §9.8 | ⚠️ **已知无效**的旧结果，仅作缺陷复现证据，**请勿引用** |

### 6.2 汇总、日志与其他

| 交付物 | 路径 | 说明 |
| --- | --- | --- |
| 汇总表格 | [results/logs/summary_tables.md](./results/logs/summary_tables.md) | §9.9–§9.12 全部表格的权威数值 |
| 运行日志 | [results/logs/](./results/logs) | 各网格 stdout/stderr、pytest 全量日志、早期作业重跑修复日志 |
| 填充修复回归 | [results/fill_control/fill_regression.md](./results/fill_control/fill_regression.md) | §9.8 两张表的生成物（+ 6 个对照 json） |
| 批处理日志 | [results/_logs/](./results/_logs) | 各批次运行的控制台日志 |
| 特征缓存 | `results/_cache/` | 69 MB npz，**未入库**（`.gitignore` 排除），可由 `01_prepare_data.py` 重建 |

---

## 7. 最终交付图（11 张）

目录：[results/figures_reference/](./results/figures_reference)（11 PNG + README.md + panels.json）

| 图 | 对照范例 | 内容 |
| --- | --- | --- |
| [fig02_std_vs_target.png](./results/figures_reference/fig02_std_vs_target.png) | Fig2 | 7 档目标收益 × 5 结构的年化波动 |
| [fig03_sharpe_vs_target.png](./results/figures_reference/fig03_sharpe_vs_target.png) | Fig3 | 夏普比率 |
| [fig04_tracking_error_vs_target.png](./results/figures_reference/fig04_tracking_error_vs_target.png) | Fig4 | 跟踪误差 |
| [fig05_cumulative_return_mu020.png](./results/figures_reference/fig05_cumulative_return_mu020.png) | Fig5 | τ=0.020 的 5 条累计收益曲线 |
| [fig08_entropy_sharpe_ratio.png](./results/figures_reference/fig08_entropy_sharpe_ratio.png) | Fig8 | 实现熵 × 3 结构的夏普（图例间距、跨零基线说明见 §9.13 第 10 条） |
| [fig09_entropy_cumulative_return.png](./results/figures_reference/fig09_entropy_cumulative_return.png) | Fig9 | 熵正则开/关累计收益（上下双面板） |
| [fig10_out_of_sample_return.png](./results/figures_reference/fig10_out_of_sample_return.png) | Fig10 | 样本外月度已实现收益（左右双面板） |
| [fig11_csi500_cumulative_return.png](./results/figures_reference/fig11_csi500_cumulative_return.png) | Fig11 | 中证 500 累计收益 |
| [fig12_sse50_cumulative_return.png](./results/figures_reference/fig12_sse50_cumulative_return.png) | Fig12 | 上证 50 累计收益 |
| [fig13_adaboost_xgboost_mu020.png](./results/figures_reference/fig13_adaboost_xgboost_mu020.png) | Fig13 | τ=0.020：AdaBoost / XGBoost / LSTM |
| [fig14_adaboost_xgboost_mu022.png](./results/figures_reference/fig14_adaboost_xgboost_mu022.png) | Fig14 | τ=0.022：同上 |

**重新生成：**

```bash
.venv\Scripts\python.exe scripts\09_reference_figures.py --out results\figures_reference
```

（脚本默认还会去 `D:\figures` 找 11 张范例图做像素级对照；找不到范例也能出图，只是跳过对照步骤。）

### 容易拿错的图（重要）

- ✅ 正式交付图：`results/figures_reference/*.png`（由 `scripts/09_reference_figures.py` 生成）。
- ⚠️ 仓库外的 `figures_ours\_preliminary_deprecated\` 是**早期版式草稿**（目录名从 `_preliminary` 改来，
  README 顶部已加废弃警示），与范例对照会得出完全错误的结论，请勿使用。
- ⚠️ `scripts/08_paper_style_figures.py` 默认把图写到仓库外的 `figures_ours\`，产物同样**不是**交付图。
- 📌 各 run 内 `report/figures/` 下的 8 张 PNG 是**诊断图**（训练曲线、权重分布、换手、选择诊断等），
  与上面 11 张交付图是两类东西。

---

## 8. 未随仓库交付的内容

| 缺什么 | 原因 | 怎么补 |
| --- | --- | --- |
| 原始行情数据 | 本地共约 6.7 GB（10 个指数），超出 GitHub 单仓库体积限制 | 本项目实际使用 `csi300`（约 799 MB）+ `csi500`（约 1.37 GB）+ `sse50`（约 139 MB）≈ 2.3 GB；目录与文件约定见 README §2，具备同款数据即可复现 |
| `.venv`（约 1.2 GB） | 可重建 | 按 README §1，用 uv 或 pip 安装 `requirements.txt` |
| `cache/`、`results/_cache/` | 可重建 | `scripts/01_prepare_data.py` |
| 仓库外的草稿目录 | 非交付物 | 无需补；正式产物见 §7 |

---

## 9. 如何验证交付物完好

在仓库根目录执行：

```bash
.venv\Scripts\python.exe -m pytest -q                    # 134 passed
.venv\Scripts\python.exe scripts\07_verify_readme.py     # 802/802 README 数字全部回查通过
.venv\Scripts\python.exe scripts\09_reference_figures.py --out results\figures_reference   # 11/11 张图重新生成
```

重跑 `09` 后再执行 `07`，若 `07` 仍为 802/802，说明图与 README 里的数值描述一致。

---

## 10. 规模

| 项目 | 数值 |
| --- | --- |
| 工作区（不含 `.venv`、`.git`） | 约 243 MB / 2568 个文件 |
| `.git` | 约 48 MB |
| Git 已跟踪文件 | 2561（results 2506 / configs 16 / scripts 12 / src 13 / tests 12 + 3 个根文件） |
| 结果文件构成 | CSV 1190、JSON 611、YAML 302、模型权重 244、PNG 105、日志 31 |
| 测试 | 134 个用例全部通过 |
| README 数字审计 | 802/802 |
