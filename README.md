# 端到端可微 Mean-CVaR 投资组合学习（Sparse Neural Stock Selection）

本项目是对「端到端可微投资组合学习」研究方案的完整实现 + 可复现实验：

**LSTM 收益/风险预测 → 稀疏选择层（Top-k / Softmax / Sparsemax）→ 可微 Mean-CVaR 优化层 → 联合损失端到端训练 → 月度调仓回测**

核心思想是把传统「先预测、再优化」两段式流水线里的**优化层也变成可微的**，
从而让投资组合的**决策损失（含 CVaR、换手成本）直接反传到预测网络**，
而不是只让网络去最小化一个与最终组合无关的预测误差。

全部源代码、配置、实验结果与图表都保存在仓库内，任何人按下面 4 条命令即可复现。

---

## 1. 环境

| 项目 | 要求 |
| --- | --- |
| Python | **3.11**（必须；`diffcp` 这个 `cvxpylayers` 的微分后端在 Windows 上只有 ≤3.11 的 wheel） |
| 硬件 | 纯 CPU 即可（本仓库结果由 AMD Ryzen 5 7535H / 16 GB 内存 / 无 GPU 产出） |
| 关键依赖 | torch 2.14.0 (CPU)、cvxpy 1.4.2、cvxpylayers 0.1.6、diffcp 1.0.23、xgboost 3.2.0、numpy 1.26.4、pandas 2.2.3、matplotlib 3.11.1 |

`requirements.txt` 里钉的**就是产出 §9 全部结果的这套版本**（含 `xgboost`——只有树基线用得到，
以及 `pytest`、`psutil` 这两个跑测试/自检要用的包）。逐条核对本机装了什么：

```bash
.venv\Scripts\python.exe -c "import importlib.metadata as m; [print(f'{p}=={m.version(p)}') for p in ['numpy','pandas','torch','cvxpy','cvxpylayers','diffcp','xgboost','matplotlib','scikit-learn','pytest']]"
```

```bash
# Windows
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# （可选）torch 用 CPU wheel 更小更快：
#   uv pip install --python .venv\Scripts\python.exe torch==2.14.0 \
#       --index-url https://download.pytorch.org/whl/cpu
```

> `uv venv` 生成的虚拟环境没有 `pip` 模块，请用 `uv pip install --python .venv\Scripts\python.exe <pkg>` 安装新包。

---

## 2. 数据

数据来自本地 `findata` 数据集（默认 `D:\金创\findata`，可在 `configs/csi300.yaml` 的 `data.root` 修改）：

```
<root>/<index>/
    tensor/
        dates.npy            交易日，字符串数组 (T,)
        tickers.npy          全部出现过的股票代码 (N_union,)
        X.npy                原始行情张量 (T, N_union, C)
        membership_mask.npy  当日是否为该指数成分股 (T, N_union)
        observed_mask.npy    当日是否有有效行情（停牌/未上市为 False） (T, N_union)
        features.json        原始通道名
    <index>_daily.parquet    日频行情
    index_daily.csv          指数本身
```

`csi300`：T = 4046（2010-01-04 → 2026-08-31），成分股并集 810 只，有效行情单元 36.3%。

**数据处理约定**

* 每个调仓日在「当日成分股 ∩ 有有效行情」中按**近 60 日均成交额**取流动性最高的 `top_liquidity=120` 只构成当期权池（成分股是动态的，不存在幸存者偏差：某只股票只有在其真的属于成分股的日期才会进入期权池）。
* ST、涨跌停锁死的股票在候选集中剔除（`exclude_st`、`exclude_limit_locked`）。
* 停牌股票在持有期内**不可卖出**：优化层对这类持仓施加 `y_floor`（下界），模型层再把 `cap` 抬到不低于 `y_floor` 以满足可行性。
* 原始行情里的涨跌幅是**百分点**，代码里统一除以 100 转成小数收益。
* 特征工程严格因果：任何特征在 `t` 时刻只使用 `≤ t` 的信息；截面标准化按当日横截面做。共 17 个特征：

  `mom_5, mom_20, mom_60, resid_mom_20, vol_20, vol_60, downside_vol_20, ret_skew_60, amihud, log_volume, volume_ratio, vwap_gap, open_close, range, mkt_ret_1, mkt_ret_20, beta_60`

  这 17 个通道已经覆盖了方案 §2 模块一列出的输入分组：**动量与反转**（`mom_5/20/60`、
  `resid_mom_20`）、**波动率**（`vol_20/60`、`downside_vol_20`、`ret_skew_60`）、
  **成交量/成交额/换手率**（`log_volume`、`volume_ratio`、`amihud`、`vwap_gap`、`open_close`、`range`）、
  **市场指数收益**（`mkt_ret_1`、`mkt_ret_20`，以及相对市场的 `beta_60`、`resid_mom_20`）。
  其中「收益率与对数收益率」没有单独作为通道：`mom_*` 本身就是收盘价的收益率序列，
  再加一列 1 日收益是冗余的。方案原文对这个清单用的是「其中**可以**包括」，属于开放式建议，
  所以默认保持 17 通道不变（已冻结的结果表全部基于它）；需要严格对齐时把
  `features.include_daily_return` 置为 `true` 即可追加 `ret_1` / `log_ret_1` 两列（⇒ 19 通道），
  见 [configs/csi300_extra_returns.yaml](./configs/csi300_extra_returns.yaml) 与 §9.7 的敏感性对照。
  行业信息不在 `findata` 的行情张量里（没有行业分类字段），实现上以 `beta_60` 与
  `resid_mom_20` 这两个「相对市场」的暴露度作为代理，这一点在 §10 局限里如实记录。

* 标签是未来持有期收益 `forward_return`，持有期 = 下一个调仓日之前的交易日数（月度 ≈ 20 天）。
  测试期最后一个调仓日不足一个完整持有期时用 `pad_horizon=True` 截断补齐，训练/验证期一律不补齐。

---

## 3. 模块与源文件对应关系

| 研究方案模块 | 源文件 | 内容 |
| --- | --- | --- |
| 数据加载 | [data.py](./src/e2e_portfolio/data.py) | 读取 tensor 张量、npz 缓存、涨跌停/停牌掩码、指数序列 |
| 特征工程 | [features.py](./src/e2e_portfolio/features.py) | 17 个因果特征 + 截面标准化 |
| 样本构造 | [dataset.py](./src/e2e_portfolio/dataset.py) | 调仓日历、walk-forward 折、逐期样本（期权池/流动性/未来收益） |
| 预测网络 | [models.py](./src/e2e_portfolio/models.py) | 共享 LSTM 编码器 + 收益头 μ + 波动头 σ + 选择分数（学习头，或按方案给定公式直接算） |
| 稀疏选择 | [selection.py](./src/e2e_portfolio/selection.py) | `TopKSelection`（直通估计）、`SoftmaxSelection`、`SparsemaxSelection`、`IdentitySelection` |
| 场景生成 | [models.py](./src/e2e_portfolio/models.py) | 高斯参数化场景 `R = μ·h + σ·√h ⊙ ε`（对 μ、σ 都能反传） |
| 可微优化层 | [optlayer.py](./src/e2e_portfolio/optlayer.py) | `cvxpylayers` 包装的 Mean-CVaR 锥规划 + 解析 KKT 反传 |
| 损失函数 | [losses.py](./src/e2e_portfolio/losses.py) | 预测损失、决策损失、尾部风险、换手成本、稀疏选择 5 项 |
| 训练循环 | [train.py](./src/e2e_portfolio/train.py) | 逐期张量打包、早停、最佳权重回滚、梯度裁剪 |
| 回测引擎 | [backtest.py](./src/e2e_portfolio/backtest.py) | 日频模拟、权重漂移、线性交易成本、cap 审计 |
| 评价指标 | [metrics.py](./src/e2e_portfolio/metrics.py) | 年化收益/波动、Sharpe/Sortino、VaR/CVaR、MDD、Calmar、信息比率、α/β、上涨/下跌市分解 |
| 实验编排 | [experiments.py](./src/e2e_portfolio/experiments.py) | 实验注册表、历史情景 Mean-CVaR 基线、单作业 runner、结果落盘 |
| 配置 | [config.py](./src/e2e_portfolio/config.py) | 强类型 YAML 配置（`from_yaml` / `to_yaml` / `replace`） |

### 优化层的数学形式

在每个调仓期 $t$，给定预测收益 $\mu$、场景收益矩阵 $R \in \mathbb{R}^{S\times N}$、上一期持仓 $y_{prev}$，
选择头给出参与度 $\pi$，求解

```text
min_y   -μᵀy + γ·ζ + γ/((1-α)S)·Σ_s u_s
        + λ_turn·‖y - y_prev‖₁ - λ_div·Σ_i entr(y_i) + λ_herf·Σ_i y_i²

s.t.    u_s ≥ -R_sᵀ y - ζ,   u ≥ 0,      Σ_i y_i = 1,   y_floor ≤ y ≤ cap
```

其中 `ζ` 是 VaR 辅助变量、`u` 是尾部超额损失，`entr(y) = y·log y` 使目标**严格凸**，
从而 $y^\star(\mu)$ 对 $\mu$ 是光滑可导的（这也是本文档第 6 节里有限差分梯度检验有意义的前提）。
由于 `ζ` 与 `u` 都在目标里出现，整条 `y` 对参数的依赖由 cvxpylayers 求锥规划的最优解并做隐式微分。

默认超参：`γ=0.1`、`α=0.95`、`λ_turn=0.0015`（= 回测的单边 15 bp）、`λ_div=0.002`、`y_max=0.10`。

### 联合损失

```text
L = l_pred·L_pred + l_decision·L_decision + l_tail·L_tail
  + l_turnover·L_turnover + l_sparse·L_sparse
```

* `L_pred`：收益的负对数似然（`nll`），或 MSE。
* `L_decision`：可微优化层输出的组合在未来真实收益上的负收益（决策质量）。
* `L_tail`：真实收益下最差 `(1-α)` 比例日的平均损失（实现 CVaR）。
* `L_turnover`：`annual_scale · cost_per_unit_turnover · Σ|Δw|`，与回测成本口径完全一致。
* `L_sparse`：`PR(π)/N = (Σπ²)⁻¹/N ∈ [1/N, 1]`，**最小化它会让选择更集中**（这是有意的：
  稀疏选择的"稀疏"由 Top-k / Sparsemax 的硬约束保证，损失项只负责让参与度分布更自信，从而释放 cap 额度）。

默认权重：`l_pred=0.1, l_decision=1.0, l_tail=0.5, l_turnover=1.0, l_sparse=0.1`。

### 单票权重上限（可行性规则）

硬约束 `y ≤ y_max` 与 `Σy = 1` 只有在 `n_valid ≥ 1/y_max` 时才可行。
代码分两层保证可行性：

1. `selection.feasible_cap(y_max, n_valid) = max(y_max, 1/n_valid)`：期权池足够大时就是 `y_max`，
   期权池过小时自动放宽到刚好可行的最小值；
2. `selection.selection_caps(π, y_max, valid, score)` 先给每只票 `cap_i = y_eff · min(1, k_eff·π_i)`
   （`k_eff = 1/Σπ²`），再检查 `Σcap` 是否够买满预算。**选择集比 `1/y_max` 还集中时**
   （例如 Sparsemax 只选出 3 只而 `y_max = 0.10`，`Σcap ≈ 0.3`），缺口按**分数从高到低**依次
   交给下一名，每名最多吃满自己的剩余额度 `y_eff - cap_i`，直到预算填满。

第 2 步为什么是「按分数放宽选择集」而不是「把缺口摊到全体横截面」：后者会让 `cap` 在
上百只票上各留下一小块（实测 `π` 支持集只有 8–17 只时账面却变成 105–123 只、每只约 0.3%），
与「稀疏神经选股」直接矛盾；按分数放宽等价于**把选择集扩大**到刚好能承载预算的大小，
账面始终是「分数最高的那一段」，且硬约束 `0 ≤ y ≤ y_max` 原封不动成立
（可行性由 `n_valid · y_eff ≥ 1` 保证）。
两次参考运行（21 个作业、721 个调仓期）的 `cap_sum` 最小 1.1608、中位数 3.39、最大 12.00，
即缺口恒为 0，这段填充逻辑在参考网格里只在 `full_musigma` f3 的第 2023-01-03 期被触发过
一次、量级 4.8e-8（`selection_caps` 里 `deficit > 0` 才会进入填充分支）。
窄幅但真实的反例见 §9.7 的作废运行与对照运行。

### 两种选择分数（`model.score_mode`）

研究方案把选择分数写成一个**固定公式**，而不是让网络自由学：

```text
s_i = μ̂_i − λ_σ·σ̂_i − λ_c·CVaR_i
```

代码同时支持这两条路，由 `model.score_mode` 切换（见 [models.py](./src/e2e_portfolio/models.py) 与
[config.py](./src/e2e_portfolio/config.py)）：

| `score_mode` | 分数来源 | 谁在用 |
| --- | --- | --- |
| `head`（默认） | 独立的线性头 `s_i = wᵀh_i + b`，end-to-end 学习 | 主网格 11 个实验全部 |
| `mu_sigma` | 上述固定公式，`λ_σ`/`λ_c` 由 `score_lambda_sigma` / `score_lambda_cvar` 给出（默认 0.5/0.5） | `full_musigma`、`lstm_musigma`（见 §5 的补充网格） |

`mu_sigma` 模式下**不会构造 `score_layer`**，因此 `state_dict` 里没有该层的参数
（`tests/test_models.py` 会断言这一点，避免留下永不更新的死参数）。其中 `CVaR_i` 取该股票
场景分布里最差 `ceil((1-α)S)` 个场景的平均损失（`models.scenario_cvar`，与 `losses.tail_loss`
的取整容差一致），所以「预测越差、波动越大、尾部越厚」的票分数越低。
`μ`/`σ` 仍然是由 LSTM 头预测、并且通过 `l_pred` 训练的——两种模式的区别只在于
「如何把预测变成选择依据」。

### 选择层为什么会「看起来不稀疏」

三种选择层（`TopKSelection` / `SoftmaxSelection` / `SparsemaxSelection`）作用在同一个
logit `z` 上（`π = Selector(z)`），但稀疏性条件完全不同：

* Top-k：硬性保留 `k=25` 个非零权重（`π_i = 1/25` 或 0），**一定稀疏**。
* Sparsemax：`π = argmin_{p∈Δ} ½‖p − z‖²`，只有 `max(z) − z_i > (1 + Σ_{j≠i} z_j)/n`
  的坐标才会被压到 0。等价地，**当 `mean(z) − min(z) < 1/n_valid` 时输出满支撑**
  （与 softmax 几乎相同），只有当 logit 的离散程度超过 `1/n` 时才开始真正稀疏。
* 因为选择层不参与 `L_pred`（`_predict_only` 把 `l_decision/l_tail/l_turnover/l_sparse` 全部置零），
  三种「先预测后优化」实验训练出的是**逐位相同**的编码器网络
  （`04_selection_diagnostics.py` 用 `encoder*` 参数的 SHA-256 校验），
  它们的样本外差异 100% 来自选择层的支撑集与由此产生的 `cap`。

`scripts/04_selection_diagnostics.py` 就是把这套判断做成可复现证据：逐期记录
logit 的标准差/极差、稀疏阈值 `1/n_valid`、`π` 的支持集大小、`π` 的有效参与数（PR）、
`cap` 的占空比，以及**实际持仓**的数量与 PR（从 `weights.csv` 读回，阈值 `1e-3`，
因为 SCS 会留下 ~1e-7 的数值残差，用 `1e-8` 会把所有票都数成持仓）。
这个阈值由 `metrics.py` 的 `HOLDING_EPS` 统一提供，`metrics.json` 的 `holdings_mean`、
`rebalance.csv` 的 `n_holdings` 与本脚本的 `book_n` 用的是同一个常量——
早期运行把求解器残差也数成了持仓（例如 `full` 折一 75.5 只 vs 真实的 15.0 只），
`03_report.py` 会按 `HOLDING_EPS` 从 `weights.csv` 重算并把修正后的
`metrics.json` / `rebalance.csv` 写回，见 §4。

---

## 4. 复现步骤

```bash
# 0) 单元测试（134 个用例，本机约 45 秒）
.venv\Scripts\python.exe -m pytest tests -q

# 0b) 两个端到端 smoke（验证 cvxpylayers 反传与训练回路）
.venv\Scripts\python.exe scripts\00_smoke_optlayer.py     # RESULT: OK
.venv\Scripts\python.exe scripts\00_smoke_train.py        # RESULT: OK

# 1) 数据检查（加载 + 特征 + 折划分）
.venv\Scripts\python.exe scripts\01_prepare_data.py --validate

# 2) 跑完整实验网格（11 个实验 × 3 折 = 33 个作业，4 进程并行，约 1 小时）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300.yaml \
    --workers 4 --threads-per-worker 2

# 2b) 只想快速验证流程（2 epoch、单折、1 年）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300.yaml --smoke

# 2c)（可选）补充网格：方案里那个「固定分数公式」的版本，
#     5 个实验（2 个基线 + full + 2 个公式版）× 3 折 = 15 个作业，约 25 分钟；
#     `full` 会被原样重跑一遍，因此这一次运行同时是主网格的复现指纹检查
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_score_rule.yaml \
    --workers 4 --threads-per-worker 2

# 2d) 为补充网格单独生成报告（3 和 4 用 --latest 取最新一次运行；要指定运行目录就换成 --run）
.venv\Scripts\python.exe scripts\04_selection_diagnostics.py --run results\csi300_score_rule_20260911 \
    --experiments full,full_musigma,lstm_musigma
.venv\Scripts\python.exe scripts\03_report.py --run results\csi300_score_rule_20260911

# 2e)（可选）规模扩展：不截断流动性，用完整沪深 300 期权池重跑方案 §6 的核心对比，
#     4 个实验（2 基线 + lstm_topk + full）× 3 折 = 12 个作业，约 30 分钟
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_full_universe.yaml \
    --workers 4 --threads-per-worker 2

# 2f)（可选）特征通道敏感性：把方案 §2 特征清单里的「收益率与对数收益率」两条通道
#     打开（17 → 19 通道），只重跑 full × 3 折 = 3 个作业；结果见 §9.7
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_extra_returns.yaml \
    --workers 4 --threads-per-worker 2

# 2g)（可选）目标收益扫描：给 Mean-CVaR 层加上「收益硬约束」（mu @ y >= tau），
#     5 种网络结构（LSTM/GRU/RNN/MLP/RBFN）× 7 档 tau = 0.016/0.018/…/0.028 × 3 折，
#     21 + 14 个实验 = 105 个作业；结果见 §9.9（对应参考图的 Fig2–Fig5）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_mu_sweep.yaml \
    --workers 4 --threads-per-worker 2
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_mu_sweep_ext.yaml \
    --workers 4 --threads-per-worker 2

# 2h)（可选）熵正则扫描：3 种结构（LSTM/GRU/RNN）× 5 档 lambda = 0 / 0.001 / 0.002 / 0.005 / 0.01
#     × 3 折 = 15 个实验 = 45 个作业；结果见 §9.10（对应参考图的 Fig8–Fig9）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_entropy_sweep.yaml \
    --workers 4 --threads-per-worker 2
#     csi300_entropy_sweep_ext.yaml 把 MLP / RBFN 也补齐（10 个实验），本轮按参考图的口径
#     （Fig8/Fig9 只有三条线）**没有跑**，需要时直接执行下面这行即可
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_entropy_sweep_ext.yaml \
    --workers 4 --threads-per-worker 2

# 2i)（可选）树模型基线：AdaBoost / XGBoost 打分 + 同一个 Mean-CVaR 层，tau = 0.020 与 0.022，
#     4 个实验 × 3 折 = 12 个作业；结果见 §9.11（对应参考图的 Fig13–Fig14）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_tree_baselines.yaml \
    --workers 2 --threads-per-worker 2

# 2j)（可选）换股票池：同一套流程搬到上证 50 与中证 500 上，各 4 个实验 × 3 折；
#     结果见 §9.12（对应参考图的 Fig12 与 Fig11）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\sse50_reference.yaml \
    --workers 2 --threads-per-worker 2
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi500_reference.yaml \
    --workers 2 --threads-per-worker 2

# 2k)（可选）两个池里各补 5 种结构（tau = 0.020，各 5 个实验 × 3 折 = 15 个作业），
#     这是 Fig11/Fig12 那 5 条曲线的数据源；结果见 §9.12.1
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\sse50_arch_compare.yaml \
    --workers 4 --threads-per-worker 2
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi500_arch_compare.yaml \
    --workers 4 --threads-per-worker 2

# 3) 选择层诊断（softmax / sparsemax / top-k 的稀疏性来源，先跑这个）
.venv\Scripts\python.exe scripts\04_selection_diagnostics.py --latest

# 4) 汇总表格 + 图 + 结论（会读取上一步的 report\selection_diagnostics.csv）
.venv\Scripts\python.exe scripts\03_report.py --latest

# 5)（可选）逐格复核：README 引用数字 vs report\pooled.csv（§9.1/§9.5/§9.6），
#    并把 §9.9–§9.13 表格里的每个数字回查 results\logs\summary_tables.md
.venv\Scripts\python.exe scripts\07_verify_readme.py

# 6)（可选）缺陷填充修复的回归对照（复现 §9.8 的两张表）
#    6a) 用修复后的规则重训受影响的 6 个作业（3 个配置，共 7 个作业，约 2 小时）
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_fill_check.yaml --workers 1
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_fill_check_ablations.yaml --workers 2
.venv\Scripts\python.exe scripts\02_run_experiments.py --config configs\csi300_fill_check_full_universe.yaml --workers 1
#    6b) 冻结旧权重、只换填充规则（隔离规则本身，每个作业约 1 分钟）
.venv\Scripts\python.exe scripts\05_fill_control.py --run results\_invalidated_extra_returns_prefixfillbug `
    --experiment full --fold f2_2020_2021
#    6c) 汇总成 results\fill_control\fill_regression.md
.venv\Scripts\python.exe scripts\06_fill_regression.py --write

# 7)（可选）把「参考图」用我们自己的数据重画一遍：11 张图只写进一个独立文件夹，
#    并同时输出 README.md（图 → 数据来源对照）与 panels.json（每张图用到的实验名）
.venv\Scripts\python.exe scripts\09_reference_figures.py --out results\figures_reference

# 8)（可选）把 §9.9–§9.12 的表格从各个 run 里重新算一遍并落盘（README 里的数字
#    就是从这份 results\logs\summary_tables.md 抄过去的；--check 只打印进度表）
.venv\Scripts\python.exe scripts\10_summary_tables.py --out results\logs\summary_tables.md
```

> 第 2c 步的补充网格结果见 §9.5；第 2d 步只是把 3/4 两步指到那个运行目录上；
> 第 2e 步的规模扩展结果见 §9.6；第 2f 步的特征通道敏感性结果见 §9.7；
> 第 6 步的缺陷填充修复回归对照见 §9.8；
> 第 2g/2h/2i/2j 步与第 7 步对应参考研究的 11 张图，结果见 §9.9–§9.13。

> `03_report.py` 的「选择层诊断」一节依赖第 3 步产物；若该文件不存在，报告会打印
> 生成它的命令并跳过本节，其余内容不受影响。

### 每个作业落盘的产物

```
results/<run_name>/<experiment>/<fold>/
    config.yaml     该实验完整解析后的配置（唯一复现入口）
    history.csv     逐 epoch 的 5 个损失项 + 验证损失
    model.pt        模型 state_dict
    metrics.json    全部评价指标（净/毛、分市况、成本拖累……）
    timings.json    训练/回测/求解耗时、层失败次数
    returns.csv     样本外日频 净收益/毛收益/基准
    weights.csv     每次调仓的目标权重（行=调仓日，列=全部 810 只并集股票）
    rebalance.csv   每次调仓的换手、成交名义额、成本、持仓数、有效持仓数
    selection_diag.csv  选择层逐期诊断（仅 nn 类实验，由 04 脚本生成）
                        encoder_hash/score_std/score_spread/spar_thresh/
                        pi_support/pi_pr/cap_sum/cap_at_max/book_n/book_pr
                        cap_budget/deficit/fill_periods（填充修复的判据列，
                        见 §11.7）
results/<run_name>/summary.json        33 个作业的指标汇总
results/<run_name>/report/
    report.md       完整中文报告（表格 + 图 + 结论）
    pooled.csv      所有折拼接后的主结果表
    per_fold.csv    每个 (实验, 折) 一行的明细
    selection_diagnostics.csv  选择层诊断汇总（04 脚本生成，03 脚本消费）
    summary.json    机器可读版
    figures/*.png   净值、回撤、指标柱状、风险收益散点、Sharpe 热力图、
                    训练曲线、成本拖累、组合画像
```

重新跑 `03_report.py` 只会读这些文件，表格是确定性的（不依赖随机数），
所以任何一个数字都能顺着 `<experiment>/<fold>/metrics.json` 与 `returns.csv` 追到源头。

> 老运行的**派生指标**（集中度 `hhi/top5/max`、以及早期用 `w > 1e-8` 数出来的
> `holdings_mean`）如果与 `weights.csv` 不一致，`03_report.py` 会按
> `HOLDING_EPS = 1e-3` 重算并**把修好的 `metrics.json` / `rebalance.csv` 写回**，
> 使运行目录、报告与 README 三处数字一致（幂等，重跑不会反复改写）。

### 缓存（可以删，删了只是变慢）

```
cache/<index>_panel_extras.npz          行情面板附加字段（ST、成交额、涨跌停锁死、qfq 收盘）
                                        读 parquet 是整条流水线最慢的一步，所以单独缓存
                                        （路径由 configs/csi300.yaml 的 data.cache_dir 控制）
results/_cache/panel_<index>.npz        原始张量
results/_cache/features_<index>_<hash>.npz  标准化后的特征张量（hash 由特征配置决定）
results/_cache/dataset_info_<index>.json    数据检查结果（01 脚本产物）
results/_cache/validate_<index>.json        数据校验结果
```

这两个缓存目录**不含实验结论**，删掉后重跑会自动重建（`--force-cache` 可强制重建），
因此不影响他人复现结果；`results/_cache` 名字以下划线开头，`03_report.py` 不会把它当成一次运行。

---

## 5. 实验设计

### 折划分（扩张窗口 walk-forward，验证集 1 年）

| 折 | 训练 | 验证 | 测试 |
| --- | --- | --- | --- |
| F1 | 2010-01 → 2016-12 | 2017 | 2018–2019 |
| F2 | 2010-01 → 2018-12 | 2019 | 2020–2021 |
| F3 | 2010-01 → 2020-12 | 2021 | 2022–2026 |

折之间的测试窗口首尾相接，因此报告里的「主结果」是把 3 折的样本外日收益**拼接后重算**的，
等价于 2018-01 → 2026 的一段连续样本外回测。

### 11 个实验

| 实验 | 类别 | 说明 |
| --- | --- | --- |
| `ew` | 基线 | 期权池内等权 1/N，不学习 |
| `meancvar_hist` | 基线 | 传统 Mean-CVaR：用重叠历史窗口当场景，不学习 |
| `lstm_topk` | 预测-再优化 | LSTM + 直通 top-k，Mean-CVaR 层**只在测试期**使用 |
| `lstm_softmax` | 预测-再优化 | 同上，选择层换成 dense softmax |
| `lstm_sparsemax` | 预测-再优化 | 同上，选择层换成 sparsemax |
| `e2e_noselect` | 端到端 | 没有选择层（均匀 cap），优化层在训练回路内 |
| `full` | 端到端 | **完整模型**：sparsemax + 可微 Mean-CVaR + 5 项损失 |
| `full_nocvar_loss` | 消融 | 去掉损失里的尾部（CVaR）项 |
| `full_nocost` | 消融 | 训练与优化都不看交易成本（回测仍收费） |
| `full_nosparse` | 消融 | 去掉稀疏选择损失 |
| `full_quad` | 消融 | 用二次（Herfindahl）分散项替代熵正则 |

三组对照回答三个问题：
**（1）** 学习型选择相比 1/N 与历史 Mean-CVaR 有没有真实增量；
**（2）** 端到端可微相比「先预测再优化」有没有价值；
**（3）** CVaR 项、成本项、稀疏项、凸正则各自贡献多少。

### 补充网格：把选择分数换回方案给定的公式

主网格默认用「学习出来的分数头」（`score_mode = head`）。为了严格对齐研究方案的
模块三（`s_i = μ̂_i − λ_σ σ̂_i − λ_c CVaR_i`），另有一个独立配置
[configs/csi300_score_rule.yaml](./configs/csi300_score_rule.yaml)，跑 5 个实验 × 3 折 = 15 个作业：

| 实验 | 类别 | 说明 |
| --- | --- | --- |
| `ew` | 基线 | 等权 1/N，与主网格同一折划分（只做回测，秒级） |
| `meancvar_hist` | 基线 | 历史情景 Mean-CVaR，与主网格同参数（让补充报告自带参照） |
| `full` | 端到端 | 主网格的完整模型，原样重跑（同时用作复现指纹检查） |
| `full_musigma` | 端到端 | 与 `full` 唯一的差别是 `score_mode = mu_sigma` |
| `lstm_musigma` | 预测-再优化 | `lstm_sparsemax` + 固定分数公式（编码器仍只由 `l_pred` 训练） |

主网格（`configs/csi300.yaml`）的 `experiments` 列表**没有**包含这两个新实验，
所以已有的 33 个作业结果与本次补充运行互不干扰；两次运行的产物分别落在
`results/csi300_20260910_235309/` 与 `results/csi300_score_rule_20260911/`。
两次运行的结果对比与读法见 §9.5。

### 规模扩展：不截断流动性的完整沪深 300（方案 §5 的第二阶段）

方案 §5 写的是「开始调试时可以先选流动性最高的 50 只或 100 只股票，模型稳定后再扩展到完整沪深 300」。
主网格取的是 `top_liquidity: 120`（调试期的截断），因此另有一个配置
[configs/csi300_full_universe.yaml](./configs/csi300_full_universe.yaml) 把 `top_liquidity` 设为 `0`
（不做流动性过滤，池子 = 当期全部沪深 300 成分股），并把方案 §6 的核心对比线
（两个基线 `ew` / `meancvar_hist`、先预测再优化 `lstm_topk`、完整模型 `full`）在完整沪深 300 上重跑一遍。

不截断会显著放大槽位数：`investable_slot_bound` 给出 f1/f2/f3 = **613 / 682 / 778** 个资产槽位
（截断到前 120 名时是 154 / 155 / 165），因为「停牌且不可卖出」的持仓会逐期累积在必须保留的集合里，
再加上每期约 287 只成分股。实测一次真实 `run_epoch`（f3 折）在 N=778 下约 342 秒，
而 N=165 下约 82 秒（≈4.2 倍），所以这一次只跑核心 4 个实验，消融仍留在 §9.1 的截断网格里。
结果与读法见 §9.6。

### 目标收益扫描（方案模块一/二的硬约束版本）

参考研究的 Fig2–Fig5 是「**要求组合每期达到某个目标收益** τ，然后看波动/夏普/跟踪误差/净值」。
为此在可微 Mean-CVaR 层里加了一条硬约束 `mu @ y >= tau`（`opt.mu_target`，默认 `null` = 不加），
τ 是 cvxpy 的**参数**而不是常量，因此同一个层可以在每个调仓期用当期不同的 τ 求解（τ 逐期重算见 §6）。
`model.arch` 同时支持 5 种编码器（`lstm` / `gru` / `rnn` / `mlp` / `rbfn`），
于是「5 种结构 × 7 档 τ」就是参考图 Fig2–Fig5 的完整网格：

| 配置 | 实验 | 作业数 | τ |
| --- | --- | --- | --- |
| [configs/csi300_mu_sweep.yaml](./configs/csi300_mu_sweep.yaml) | `mu016_lstm` … `mu028_rbfn` 中的 LSTM/GRU/RNN 共 21 个 | 63 | 0.016 / 0.018 / 0.020 / 0.022 / 0.024 / 0.026 / 0.028 |
| [configs/csi300_mu_sweep_ext.yaml](./configs/csi300_mu_sweep_ext.yaml) | 同名的 MLP/RBFN 共 14 个 | 42 | 同上 |

每个实验都是「完整模型」（sparsemax 选择 + 可微 Mean-CVaR + 5 项损失），与 `full` 的唯一差别
就是编码器结构与 `opt.mu_target`。结果与读法见 §9.9。

### 熵正则扫描（方案模块二的 λ 敏感性）

参考研究的 Fig8–Fig9 横轴不是 λ 而是**实际实现的组合熵**。这里同样跑 5 档
`opt.lam_div`（熵正则权重，`0` 表示关掉分散项），再用每期权重算出真实熵：

| 配置 | 实验 | 作业数 | λ（`opt.lam_div`） |
| --- | --- | --- | --- |
| [configs/csi300_entropy_sweep.yaml](./configs/csi300_entropy_sweep.yaml) | `ent000_lstm` … `ent100_rbfn` 中的 LSTM/GRU/RNN 共 15 个 | 45 | 0 / 0.001 / 0.002 / 0.005 / 0.01 |
| [configs/csi300_entropy_sweep_ext.yaml](./configs/csi300_entropy_sweep_ext.yaml) | 同名的 MLP/RBFN 共 10 个 | 30 | 同上 |

结果与读法见 §9.10。`_ext` 这一份（MLP/RBFN）照参考图 Fig8–Fig9 只需要三条曲线，
配置文件已写好但**没有跑**，真要补一次直接按 §9.10 的命令加 `--config` 即可（见 §11.7（E）③）。

### 树模型基线（参考图 Fig13–Fig14）

AdaBoost / XGBoost 在这里**只替换预测器**，其余流水线完全不动：同一套固定分数公式
（`s = μ̂ − λ_σ σ̂ − λ_c CVaR`，σ 用 60 日已实现波动率）、同一个 sparsemax 选择、
同一个单票上限、以及**同一个不可微的 Mean-CVaR 层**（带 `mu_target`）。
特征是把窗口压平成「最后一天 + 窗口均值 + 观测比例」共 2K+1 通道（`compact_tree_features`），
每个调仓期在 `t <= p.t - horizon` 的样本上重新拟合，不做未来函数。
配置 [configs/csi300_tree_baselines.yaml](./configs/csi300_tree_baselines.yaml) 跑 4 个实验 × 3 折：

| 实验 | 预测器 | τ |
| --- | --- | --- |
| `adaboost_mcvar` | AdaBoost（50 棵深度 3 的树） | 0.020 |
| `xgboost_mcvar` | XGBoost（200 棵深度 3 的树） | 0.020 |
| `adaboost_mcvar_mu022` | 同上 | 0.022 |
| `xgboost_mcvar_mu022` | 同上 | 0.022 |

结果与读法见 §9.11。

### 换股票池复现（参考图 Fig11–Fig12）

同一套代码、同一套折划分，只改 `data.index` 与 `data.top_liquidity`：

| 配置 | 股票池 | 期权池上限 | 实验 |
| --- | --- | --- | --- |
| [configs/sse50_reference.yaml](./configs/sse50_reference.yaml) | 上证 50（`sse50`） | 50 | `ew`、`meancvar_hist`、`mu020_lstm`、`full` |
| [configs/csi500_reference.yaml](./configs/csi500_reference.yaml) | 中证 500（`csi500`） | 120 | 同上 |
| [configs/sse50_arch_compare.yaml](./configs/sse50_arch_compare.yaml) | 上证 50（`sse50`） | 50 | `mu020_{rnn,lstm,gru,mlp,rbfn}`（Fig12 的 5 条曲线） |
| [configs/csi500_arch_compare.yaml](./configs/csi500_arch_compare.yaml) | 中证 500（`csi500`） | 120 | 同上（Fig11 的 5 条曲线） |

参考图 Fig12 用的就是上证 50，可以一一对应；Fig11 用的是巴西 IBrX50，本地数据里没有这个指数，
用**中证 500** 替代（同样起「换一个股票池看结论是否还成立」的作用，差别已在 §9.12 与图 README 里写明）。
参考图 Fig11/Fig12 每条线是同一策略下的不同网络结构，所以两个池各补了一套
**5 结构 × 3 折**的网格（`*_arch_compare.yaml`），结果见 §9.12.1。

---

## 6. 实现要点与踩坑记录

这些是复现时最容易踩的坑，都已修好并留了注释：

1. **熵项必须是凸的**：写成 `-λ·Σ entr(y)`（即 `+λ Σ y·log y`）。
   写成 `-λH(y) = +λ Σ y log y` 才对；若误写成 `-λ Σ y log y` 会让目标非凸，SCS 会给出不稳定解。
2. **`cvxpy` 的 `Problem.parameters()` 不保证是构造顺序**（实测是 `[μ, y_prev, R, y_floor, cap]`），
   所以代码一律用 `prob.param_dict[name]` 按名字取参数，否则非微分路径会把场景矩阵塞到错误的参数上。
3. **`cvxpylayers` 会剪掉问题里没出现的参数**：当 `λ_turn=0` 时 `y_prev` 会消失，5 参数的调用直接报错。
   代码给换手项始终保留一个 `_TURNOVER_EPS = 1e-9` 的下限，保证参数个数恒定。
4. **`CvxpyLayer.forward(*params, solver_args={})` 没有 `solver` 参数**；
   传参顺序按声明的 `parameters=[...]` 列表，而不是创建顺序。
5. **SCS 默认精度 ~1e-4**，对分段线性的规划做有限差分梯度检验没有意义（几乎处处雅可比为 0）。
   梯度检验用严格凸的熵变体 + `eps=1e-11`，详见 `tests/test_optlayer.py`。
6. **净收益与毛收益不能同名**：`compute_metrics` 对两条曲线返回同样的键名，
   早期版本 `metrics.update(compute_metrics(ret_gross, ...))` 会把净收益指标**全部覆盖成毛收益**，
   于是「成本拖累」看起来是 0。现在毛收益一律带 `_gross` 后缀（见 `experiments.py`）。
7. **换手率的年化口径**：`turnover` 是**每次调仓**的（月度），
   早期版本乘 252（交易日）得到 42 这种没有意义的值；现在按实际调仓频率年化（月度 ≈ ×12）。
8. **折边界不能泄漏**：`periods_between` 用严格 `< end_idx`，避免持有期跨到下一折的样本里。
9. **尾部风险的 `k`**：`k = ceil((1-α)·h - 1e-9)`，浮点噪声会让 5 天变成 6 天。
10. **求解失败要有兜底**：任何非最优状态都不返回 NaN，而是回落到「带上限的等权投影」，
    并把失败次数记在 `timings.json` 的 `layer_failures` 里。
11. **资产槽位数 `n_max` 必须由数据算出来，不能拍脑袋**：期权池每期是
    「成交额前 120 名」**并上**「已持有但不可卖出（停牌 / 观测缺失 / 跌停锁死）的票」，
    后者是**叠加**在 120 之上的，所以 `top_liquidity + 20` 这种固定余量会被击穿
    （实测 f2 折 2017-03-01 出现 141 个可投资标的，直接抛错）。
    现在 `experiments.n_max_slots(cfg, periods, dataset=ds)` 用
    `investable_slot_bound()` 从空仓开始重放 `pool = universe | (~sellable & pool)`，
    得到这一折里**确切**的最大槽位数（实测 f1/f2/f3 = 154/155/165），
    并把 `n_max` 与 `max_universe`、`max_investable` 一起写进 `metrics.json` 便于核对。
    槽位数偏小会抛错（不再静默截断），偏大只是让 SCS 多算几列。

---

## 7. 成本、换手与评价口径

* 交易成本：`cost = 15bp × Σ|Δw|`（**单边 15 bp，双向成交名义额**），在持有期第一天从净值里扣。
* 换手率：`turnover = ½·Σ|Δw|`（**单边口径**，与业界惯例一致）；另有 `traded_notional = Σ|Δw|` 供对账。
* 权重在持有期内随真实收益**漂移**，下一次调仓的换手是对「实际持有权重」而不是「上一期目标权重」计算的。
* `turnover_ann = turnover_mean × 实际年化调仓频率`（月度 ≈ 12 次/年）。
* 主结果一律是**净收益**；`ann_return_gross` / `sharpe_gross` / `cost_drag_ann` 单独给出，用来量化成本的影响。
* 基准是同期沪深 300 指数本身的日收益。

---

## 8. 测试

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

134 个用例，覆盖：指标的手算校验（MDD、VaR/CVaR、CAGR、信息比率、换手年化）、
持仓数量与集中度（HHI / 前 5 大权重 / 最大权重的手算值、取的是**最大的** 5 个而非前 5 列、
空头权重被截掉、空输入不报错、等权 25 只与「3 只票独占」能被区分开、
`holdings_mean` 只数经济持仓、求解器的 1e-7 残差不算持仓）、
报告脚本的集中度回填（老运行缺 `hhi_mean` 时能从 `weights.csv` 精确恢复，且不会覆盖已有值）
与 `holdings_mean` 迁移（老运行把求解器残差数成持仓，报告会按 `HOLDING_EPS` 重算并把修好的
`metrics.json` / `rebalance.csv` 写回），
选择层的性质（top-k 恰好 k 个非零、sparsemax 稀疏且和为 1、softmax 稠密、
梯度可通、`cap` 的补平只在不满足 `Σcap ≥ 1` 时触发、触发时严格按分数从高到低放宽选择集、
被放款的票一定是分数排序的前缀、每票额度不超过 `y_max`、`raw_caps` 就是补平前的额度
且 `Σraw ≥ 1` 时选择层原样返回）、
优化层（与两资产解析解对比、有限差分梯度一致性、不可行兜底、`mu_target` 可达到时达标、
不可行时被钳制到上限能支撑的最大值、以及 `feasible_mu` 与 scipy 参考线性规划的一致解）、
网络与分数（高斯分位点网格、`scenario_cvar` 与手算逐位一致且取整容差正确、
两种 `score_mode` 各自的语义与参数表、非法 `score_mode` 直接报错、
`model.arch` 五种结构共用同一个编码器且输出有限、非循环结构只在观测到的日期上池化、
未注册的结构直接报错）、
损失项（单调性、方向性）、数据集（折边界不泄漏、停牌/空行处理、可选的两条日收益通道与手算 z-score 逐位一致）、
配置（三个随仓库发布的 `configs/*.yaml` 都能加载、折边界递增、实验名都在注册表里、`include_daily_return` 默认关闭）、
扫描网格（τ × 5 种结构正好铺满 `csi300_mu_sweep(*_ext)`、λ × 结构铺满 `csi300_entropy_sweep(*_ext)`、
树基线只换预测器而不动 Mean-CVaR 层、压缩特征的 2K+1 通道与手算一致、
padding 行与空股票名都不会污染特征、`prev_slots` 把可投资标的压进前几个槽位）、
回测（静态目标零换手、成本口径）、训练回路（梯度能传回编码器、早停回滚、
权重漂移与回测一致、padding 行不影响任何输出），以及 `tests/test_train.py::test_every_registered_experiment_runs_on_one_fold`
——它在合成数据上把注册表里的实验（含 2 个 `mu_sigma` 变体）全部跑通一次，并校验净/毛指标没有被写错。

---

## 9. 结果与结论

### 9.1 参考运行（本仓库已跑完的一次完整结果）

* 运行目录：[results/csi300_20260910_235309/](./results/csi300_20260910_235309/)
* 规模：33/33 作业成功，0 失败，墙钟 52.4 分钟（AMD Ryzen 5 7535H，CPU only，4 进程 × 2 线程）
* 完整报告：[report/report.md](./results/csi300_20260910_235309/report/report.md)（含 8 张图与逐条结论）
* 机器可读结果：[report/pooled.csv](./results/csi300_20260910_235309/report/pooled.csv)、
  [report/per_fold.csv](./results/csi300_20260910_235309/report/per_fold.csv)、
  [report/summary.json](./results/csi300_20260910_235309/report/summary.json)、
  [report/selection_diagnostics.csv](./results/csi300_20260910_235309/report/selection_diagnostics.csv)

主结果（3 折测试窗口拼接后重算，**净收益**）：

| 实验 | 年化净收益 | 年化波动 | Sharpe | 最大回撤 | 日 CVaR95 | 年化换手 | 持仓数(>1e-3) | 有效持仓 | 成本拖累 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `meancvar_hist` | 5.27% | 32.11% | **0.16** | -55.83% | 4.46% | 5.77 | 13.1 | 10.6 | 1.84% |
| `e2e_noselect` | 2.56% | 17.72% | 0.14 | -50.62% | 2.63% | 5.80 | 14.1 | 10.7 | 1.80% |
| `full_nocost` † | 2.70% | 19.77% | 0.14 | -43.43% | 2.94% | 8.14 | 25.3 | 12.0 | 2.54% |
| `ew`（等权基准） | 2.94% | 21.71% | 0.14 | -45.38% | 3.11% | 2.13 | 123.4 | 123.1 | 0.66% |
| **`full`（本文方法）** | 1.10% | 16.80% | 0.07 | -46.82% | 2.38% | 6.16 | 22.4 | 12.2 | 1.88% |
| `full_nosparse` | 0.47% | 18.02% | 0.03 | -49.09% | 2.50% | 6.14 | 19.3 | 11.8 | 1.87% |
| `full_nocvar_loss` | -0.43% | 20.03% | -0.02 | -58.36% | 2.88% | 6.65 | 30.9 | 15.2 | 2.01% |
| `lstm_topk` | -0.71% | 24.55% | -0.03 | -54.47% | 3.35% | 8.53 | 27.6 | 17.0 | 2.58% |
| `lstm_sparsemax` | -0.95% | 20.80% | -0.05 | -44.08% | 2.94% | 3.96 | 91.5 | 46.2 | 1.19% |
| `lstm_softmax` | -1.58% | 19.85% | -0.08 | -46.25% | 2.82% | 3.20 | 93.4 | 47.7 | 0.95% |
| `full_quad` † | -2.73% | 22.34% | -0.12 | -67.20% | 3.31% | 8.92 | 16.1 | 12.7 | 2.64% |

† 该实验有折次在旧版「缺陷填充」规则下触发了填充缺陷（实现 bug，不是模型行为），
本行是**修复前**的数字；修复后的重训结果与冻结权重对照见 §9.8。触发的是
`full_nocost` 的 f2（13/24 期）与 `full_quad` 的 f2（2/24 期），其余 9 个实验、
共 33 个作业里没有一期触发。

### 9.2 结论摘要

1. **端到端没有跑赢最简单的等权基准。** 等权 1/N 的 Sharpe 为 0.14、年化 2.94%；
   完整方法 `full` 的 Sharpe 为 0.07、年化 1.10%。只有传统历史情景 Mean-CVaR
   在 Sharpe 上更高（0.16），代价是年化波动 32%、最大回撤 -55.8%。
2. **端到端确实优于同结构的「先预测再优化」。** `full` 相对 `lstm_sparsemax`
   Sharpe +0.11、日 CVaR 从 2.94% 降到 2.38%、年化换手从 3.96 升到 6.16；
   但 0.11 的差距落在重训噪声量级内（见 §10），只能算方向性证据。
3. **交易成本是这段样本里最大的确定性损耗。** `full` 的成本拖累 1.88%/年，
   与 1.10% 的净收益同量级；`full_nocost`（训练与优化都不看成本）毛收益虽高，
   真实成本拖累升到 2.54%、换手升到 8.14。**在这类高换手策略里，成本口径比模型结构更关键。**
4. **损失项的消融比选择层的消融更有信息量**（相对 `full`）：去掉 CVaR 尾部项最伤
   （Sharpe -0.09、回撤恶化 11.5 个百分点）；去掉稀疏激励 `l_sparse` 小幅变差
   （-0.04）；用二次分散项替代熵正则明显更差（-0.19）。也就是说本框架的收益主要来自
   「尾部风险 + 稀疏」这两项，而不是复杂的正则化设计。
5. **三个预测-再优化实验的训练完全相同，差异只来自选择层。** 三者编码器参数的 SHA-256
   逐位相同（`scripts/04_selection_diagnostics.py` 可复核），Sharpe 极差仅 0.05，
   说明在 logit 尺度很小时，换一个「稀疏」选择层并不会真的变稀疏（详见 9.3）。
6. **没有稳健的赢家。** 3 折里没有任何一个实验在全部 3 折上都为正夏普；
   本文用 3 折结论主要是为了展示流程与量化差异，不足以支撑泛化性判断。
   §9.5 里那个「把分数换回方案给定公式」的补充网格给出的是同一个读数。
7. **一处实现缺陷已定位、修复并做了对照实验。** 可行性规则的「缺口填充」分支最初把缺口
   摊到整个横截面，导致 `π` 极稀疏时账面上百只小权重，与「稀疏神经选股」自相矛盾；
   改动是把它换成「按分数水位线加宽选择集」。45 个已诊断作业里只有 6 个触发过这个分支
   （105/1545 期），其余 39 个一期都没有走进填充分支、修复与它们无关；触发过的 6 个作业有重训对照与冻结权重对照两套结果，
   修复明显变好/变差都有（详见 §9.7、§9.8 与 §11.7(B)）——**它是一个 bug 修复，不是绩效改进**。

### 9.3 一个可复核的负面发现：sparsemax 在此设置下退化为 softmax

冻结编码器（预测-再优化范式）输出的 logit 极差按**期数加权**平均 **9.93e-03**，与 sparsemax 的稀疏阈值
`1/n`（约 **8.45e-03**，同样按期数加权）同量级：f1 折 **100%**（48/48 期）的测试期都低于阈值，f2 折 **4.2%**、f3 折 **1.8%**。
于是 sparsemax 的平均支撑集为 112.8 只、softmax 为 118.4 只（可投资域约 118 只），
两者的可行集与净值曲线几乎重合（Sharpe -0.05 vs -0.08，差 0.03）。

同一个 sparsemax 层在 `full`（损失里带 `l_sparse`）里 logit 极差被训练到 **4.37e-02**，
是阈值的 5.2 倍、**没有任何一期退化**（0/103 期），π 的支撑集收缩到 48.6 只、账本收缩到 22.4 只。

> **结论：稀疏性是损失函数学出来的，不是选择层的代数性质送的。**
> 如果要用 sparsemax 做稀疏选择，必须让选择层参与训练并把 logit 尺度抬到 `1/n` 以上；
> 若只想拿到固定数量的持仓，直接用 top-k（本仓库 `lstm_topk` 支撑集恒为 25 只）。

### 9.4 报告包含的内容

`report/report.md` 由 `scripts/03_report.py` 自动生成，包含：

1. 折划分与运行信息
2. 主结果表（所有折拼接后重算的年化净收益、波动、Sharpe、MDD、CVaR、换手、有效持仓、成本拖累）
3. 分折明细表
4. 选择层诊断（三种选择层的稀疏性对比，数据来自 `scripts/04_selection_diagnostics.py`）
5. 消融结论（相对 `full` 的逐项差值）
6. 图表：净值曲线、回撤、指标柱状图、风险收益散点、Sharpe 热力图、训练曲线、成本拖累、组合画像
7. 复现命令

### 9.5 补充网格：把选择分数换回方案给定的固定公式

方案模块三给的是一个**固定**分数 `s_i = μ̂_i − λ_σ σ̂_i − λ_c CVaR_i`，
主网格默认用的是可学习分数头（§3「两种选择分数」）。为了不让这一条只停留在代码层面，
另跑了一个补充网格：5 实验 × 3 折 = 15 个作业，15/15 成功，墙钟 20.9 分钟
（配置里同时带上 1/N 与历史 Mean-CVaR 两个基线，让这份报告能独立阅读）。

* 运行目录：[results/csi300_score_rule_20260911/](./results/csi300_score_rule_20260911/)
* 配置：[configs/csi300_score_rule.yaml](./configs/csi300_score_rule.yaml)（`λ_σ = λ_c = 0.5`，其余超参与主网格完全一致）
* 完整报告：[report/report.md](./results/csi300_score_rule_20260911/report/report.md)
* 选择层诊断只覆盖 3 个神经网络实验：`04_selection_diagnostics.py --run <run> --experiments full,full_musigma,lstm_musigma`

| 实验 | 年化净收益 | 年化波动 | Sharpe | 最大回撤 | 日 CVaR95 | 年化换手 | 持仓数(>1e-3) | 有效持仓 | 成本拖累 | β | 信息比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `meancvar_hist`（历史情景 Mean-CVaR） | 5.27% | 32.11% | 0.164 | -55.83% | 4.46% | 5.77 | 13.1 | 10.6 | 1.84% | 1.11 | 0.297 |
| `full_musigma`（方案公式 + 端到端） † | 4.39% | 28.30% | 0.155 | -46.34% | 3.83% | 9.23 | 28.6 | 13.8 | 2.94% | 1.19 | 0.304 |
| `ew`（等权 1/N） | 2.94% | 21.71% | 0.135 | -45.38% | 3.11% | 2.13 | 123.4 | 123.1 | 0.66% | 1.08 | **0.362** |
| **`full`（分数头，主网格默认）** | 1.10% | 16.80% | 0.065 | -46.82% | 2.38% | 6.16 | 22.4 | 12.2 | 1.88% | 0.65 | -0.052 |
| `lstm_musigma`（方案公式 + 先预测再优化） | -1.29% | 18.09% | -0.071 | -46.70% | 2.63% | 3.60 | 62.0 | 43.5 | 1.07% | 0.79 | -0.273 |

逐折 Sharpe（f1 / f2 / f3）：`ew` = -0.04 / +0.69 / -0.04，
`meancvar_hist` = +0.27 / +0.83 / -0.20，`full` = +0.30 / -0.20 / +0.09，
`full_musigma` = -0.08 / +0.77 / +0.01，`lstm_musigma` = -0.04 / -0.28 / +0.03。

四点读法：

1. **公式版的名次更高，但它买到的是 beta，不是选股能力。** `full_musigma` 的画像与
   `meancvar_hist` 几乎重合（β 1.19 vs 1.11、年化波动 28.3% vs 32.1%、
   Sharpe 0.155 vs 0.164、逐折节奏 -0.08/+0.77/+0.01 vs +0.27/+0.83/-0.20），
   收益也几乎全部来自 f2 一折（+22.6%，Sharpe 0.77）——而同等权在那一折也有 0.69。
   换成学习出来的分数头后，`full` 把 β 压到 0.65、波动压到 16.8%，代价是净收益降到 1.10%。
   **两者的差别主要在「承担多少市场风险」，不在「谁挑的股票更好」**：
   按信息比排，`ew`（0.362）还高于 `full_musigma`（0.304）与 `meancvar_hist`（0.297）。
2. **换分数来源救不回「先预测再优化」。** `lstm_musigma` 反而更差（年化 -1.29%、Sharpe -0.071）：
   编码器只由 `l_pred` 训练时，学出的 μ̂/σ̂ 与组合目标不对齐，CVaR 项在很多期把 π 压成全支撑
   （f1 折 100% 的期数 logit 极差低于稀疏阈值，π 支撑集 = 全部 116.6 只、有效持仓 43.5 只），
   净值随之走平。这与 §9.3 是同一条结论：**收益来自端到端的决策损失，不是分数公式。**
3. **成本口径不变地更贵。** 公式版年化换手 9.23、成本拖累 2.94%/年，都高于 `full` 的
   6.16 与 1.88%（与 §7 的约定一致），也高于两个基线。
4. **没有一折出现「稳健赢家」。** f1 折只有 `meancvar_hist`（+0.27）与 `full`（+0.30）为正，
   f2 折只有 `full`（-0.20）与 `lstm_musigma`（-0.28）为负，f3 折五个全在 ±0.1 以内；
   没有任何一个实验在三折上全为正。f2 折是指数上行期，
   `meancvar_hist`、`full_musigma`、`ew` 的高分主要来自吃到了那一折的 beta。
   3 折结论只能用于展示流程与量级，不能当作显著性证据。

结论：默认保持 `score_mode: "head"`（可学习分数头）——在本样本上，公式版的优势
无法与「换成更高 beta 的历史 Mean-CVaR 画像」区分开，而分数头版是唯一一个把市场暴露
实质压下来的变体；方案给定的公式以 `score_mode: "mu_sigma"` 完整保留在代码里，
`λ_σ`/`λ_c` 在配置里可直接调，想严格复刻原方案只需把配置改成 `mu_sigma`。

† `full_musigma` 的 f3 折在旧版「缺陷填充」规则下有 4/55 期触发填充缺陷
（实现 bug，不是模型行为），本行是**修复前**的数字；修复后的重训结果与冻结权重对照见 §9.8。
本表其余 4 行为无状态基线或分数头版本，未触发填充。

**可复现性核对（这一步本身也是流水线的指纹检查）：** 补充网格里的 `full` 与主网格的 `full`
在三个折上**逐位相同**——183 个指标字段全部相等（除 `seconds` 这类计时字段），
`weights.csv` 与 `model.pt` 的 SHA-256 也一致。这说明新增的 `score_mode` 选项
没有扰动默认路径，也说明整条流水线（含 cvxpylayers 求解、dropout 随机序列）是确定性可复现的。

### 9.6 规模扩展：不截断流动性的完整沪深 300（方案 §5 第二阶段）

方案 §5 要求「先在缩小后的股票池上验证，再扩展到完整沪深 300」。主网格用的是
流动性截断后的 120 名（`top_liquidity: 120`），这里给出扩展结果：
[configs/csi300_full_universe.yaml](./configs/csi300_full_universe.yaml) 把 `top_liquidity` 设为 0，
每个调仓期的可选股票数从 116.6–118.9 只涨到 **286.2 / 293.8 / 298.4** 只
（即当期全部可交易成分股），`n_max`（`y_max` 与现金可行性决定的持仓槽位）随之变成 **613 / 682 / 778**。
方案 §6 的 4 个核心对比（1/N、历史 Mean-CVaR、`lstm_topk`、`full`）在这套池子上重跑：

* 运行目录：[results/csi300_full_universe/](./results/csi300_full_universe/)（12/12 成功，墙钟 92.3 分钟）
* 完整报告：[report/report.md](./results/csi300_full_universe/report/report.md)
* 选择层诊断：[report/selection_diagnostics.csv](./results/csi300_full_universe/report/selection_diagnostics.csv)

| 实验 | 年化净收益 | 年化波动 | Sharpe | 最大回撤 | 日 CVaR95 | 年化换手 | 持仓数 | 有效持仓 | 成本拖累 | β | 信息比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `meancvar_hist` | **5.16%** | 28.05% | **0.184** | -60.08% | 4.10% | 5.00 | 12.7 | 10.7 | 1.59% | 0.98 | 0.282 |
| `ew`（1/N，约 290 只） | 2.16% | 17.11% | 0.127 | -32.42% | 2.49% | 0.60 | 342.6 | 340.4 | 0.18% | 0.84 | 0.075 |
| `full`（端到端） ‡ | 0.31% | **14.15%** | 0.022 | **-29.74%** | **2.11%** | 5.11 | 48.6 | 27.4 | 1.55% | **0.55** | -0.147 |
| `lstm_topk`（先选择后优化） | -2.14% | 16.02% | -0.133 | -47.76% | 2.52% | 5.19 | 36.5 | 23.2 | 1.54% | 0.64 | -0.327 |

逐折年化（f1 / f2 / f3）：`ew` = -2.70% / +13.16% / -0.20%，
`meancvar_hist` = +15.19% / +21.77% / -5.27%，`full` = -3.40% / +0.42% / +1.93%，
`lstm_topk` = -0.78% / +15.97% / -9.71%；逐折 Sharpe 依次是
`ew` -0.14 / +0.71 / -0.01、`meancvar_hist` +0.57 / +0.63 / -0.21、
`full` -0.23 / +0.02 / +0.16、`lstm_topk` -0.04 / +0.83 / -0.74。

三点读法：

1. **规模扩展不改变主结论，只把差异放大。** 与 120 名池子（§9.1）对照，同样的四个实验里
   只有 `lstm_topk` 明显变差（年化 -0.71% → -2.14%，f3 折 -4.70% → -9.71%）：
   可选名字从 118 涨到 ~290 之后，「直接取 top-25」在更大的截面里挑错的代价更高；
   `full` 的年化（1.10% → 0.31%）与 IR（-0.052 → -0.147）都掉到 1/N 之下，
   仍然是「把市场暴露压下去、但不产生超额」的画像（β 0.65 → 0.55，
   波动 16.80% → 14.15%，最大回撤 -46.82% → -29.74%，是四个实验里最防御的一组）。
   三折里依然**没有任何实验全为正**，f2 折（指数上行）仍是唯一普遍为正的一折。
2. **1/N 在完整池子上几乎是另一个策略。** 因为可选名字变成近 300 只，
   `ew` 的每期换手掉到 0.60（120 名池子是 2.13），成本拖累 0.18%，回撤从 -45.38% 收窄到 -32.42%，
   但信息比也从 0.362 掉到 0.075——**原本是「低成本分散」，现在更像「持有一整个市场」**。
   这说明 §9.2 里「1/N 的信息比最高」这条结论**与股票池大小强相关**，不能外推。
3. **代价是可观的计算量。** 同一份 `full` 的三折训练时间从 465 / 382 / 499 秒涨到
   **2483 / 5314 / 4273 秒**（约 5× / 14× / 9×；`n_max` 变大后每个 diffcp 图的规模随之变大），
   12 个作业 4 worker 并行共用 92.3 分钟。方案 §5「先缩小池子验证、再扩展」的顺序在本仓库被验证为必要的：
   若一开始就上完整池子，主网格的 33 个作业大约需要 5 小时以上。

如实说明一处口径（表内 `full` 行的 ‡）：`full` f2 折的 24 个调仓期里有 12 期触发了选择层的缺陷填充
（旧规则），其余 11 个作业都不触发，因此 §11.7(B) 的修复对这张表里其它行的默认路径没有影响；
该折用修复后规则重跑的对照结果见 §9.8（完整池 `full` f2：年化 +0.42% → +3.92%、
Sharpe 0.024 → 0.268、持仓 26.1 → 29.1，且修复后该折不再触发填充，`cap_budget` 2.047）。

### 9.7 特征通道扩展：把「收益率与对数收益率」也喂进网络（方案模块一）

方案模块一列基础特征时明确写了「**收益率与对数收益率**」。为了让「逐条对应」不留模糊地带，
新增开关 `include_daily_return: true`：输入从 17 通道变成 19 通道（追加当日收益 `ret_1`
与当日对数收益 `log_ret_1`，两者都只用 t 日收盘及之前的信息），并用 19 通道重跑 `full` 三折。

* 配置：[configs/csi300_extra_returns.yaml](./configs/csi300_extra_returns.yaml)（只改 `include_daily_return`，其余超参与主网格逐字相同）
* 运行目录：[results/csi300_extra_returns/](./results/csi300_extra_returns/)（3/3 成功，墙钟 16.9 分钟）
* 完整报告：[report/report.md](./results/csi300_extra_returns/report/report.md)

| 特征通道 | 年化净收益(f1/f2/f3) | Sharpe(f1/f2/f3) | 最大回撤(f1/f2/f3) | 年化换手(f1/f2/f3) | 拼接后年化 | 拼接后 Sharpe | 拼接后最大回撤 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 通道（主网格 `full`） | +5.33% / -3.69% / +1.43% | +0.299 / -0.195 / +0.093 | -27.44% / -26.11% / -31.65% | 7.62 / 5.39 / 5.86 | 1.10% | 0.065 | -46.82% |
| **19 通道（+ `ret_1`/`log_ret_1`）** | -3.75% / +3.04% / +11.82% | -0.226 / +0.148 / +0.707 | -29.01% / -20.24% / -35.49% | 5.99 / 6.82 / 7.92 | 5.92% | 0.335 | -46.39% |

拼接后的其它口径（19 通道）：年化波动 17.66%、日 CVaR95 2.54%、成本拖累 2.32%/年、
持仓数 14.7（有效 11.3）、β 0.67、信息比 0.305。

读法：**加这两个通道确实改变了训练出来的解，但没有改变「结论是分裂的」这一点。**
19 通道在 f3 折明显更好（+11.82% / Sharpe 0.707 vs +1.43% / 0.093），f1 折明显更差
（-3.75% / -0.226 vs +5.33% / +0.299），拼接后更好（年化 5.92% vs 1.10%，Sharpe 0.335 vs 0.065）
——这仍然只是 3 折上的量级差异（§10），不能当成「加通道带来 alpha」的证据。
**方案要求的这两类特征已经以开关形式完整实现，并且有可复现的结果**，
默认配置关闭只是为了让主网格与 §9.1/§9.5 的结果保持可比。

#### 顺带修掉的一个稀疏性 bug（缺陷填充把组合摊稠了）

19 通道第一次跑完时，诊断立刻暴露出反常：f2/f3 的 π 支撑集只有 9.5 / 13.5 只，
但 `weights.csv` 里 `>1e-3` 的持仓数是 **119.0 / 101.4 只**——「稀疏神经选股」
被解成了一个 300 只里的一百多只的小权重组合。根因在可行性规则（§3）：

* 旧实现：当 `Σᵢ min(1, k_eff·πᵢ) < 1`（π 高度集中、`k_eff ≈ 6.7`、`y_eff = 0.10`，上限和凑不满 1）时，
  把缺口**按每只票的剩余空间摊到整个横截面**，于是一直被选中的票各分到 0.3% 左右，组合被摊稠。
* 新实现 `selection.raw_caps` + **按分数降序的水位线（waterfall）填充**：缺口只补给分数排名靠前的票，
  每只票最多拿到自己的剩余空间 `y_eff − cap_i`，**「加宽选择」而不是「摊薄选择」**。
  当 `Σᵢ raw_cap ≥ 1` 时函数提前返回，逐个 `y` 与旧实现逐位相同。

修复前后（同一配置、同一数据，仅选择层填充规则不同）：

| 19 通道 `full` | π 支撑集(f1/f2/f3) | 持仓数(f1/f2/f3) | cap_budget 均值(f1/f2/f3) | 填充触发率(f1/f2/f3) | 年化净收益(f1/f2/f3) | 拼接后 Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| 修复前（旧填充，[证据目录](./results/_invalidated_extra_returns_prefixfillbug/)） | 32.2 / 9.5 / 13.5 | 17.0 / **119.0** / **101.4** | 2.063 / 0.589 / 0.741 | 0% / **100%** / **90.9%** | -3.75% / +9.62% / +6.43% | 0.267 |
| 修复后（水位线填充） | 32.2 / 8.9 / 15.9 | 17.0 / **10.4** / **15.6** | 2.063 / 0.544 / 0.974 | 0% / 100% / 56.4% | -3.75% / +3.04% / +11.82% | 0.335 |

三个可复核的点：

1. **修复在不触发填充时是严格的空操作。** f1 折逐期 `cap_budget` 全部 ≥ 1
   （均值 2.063、逐期最低 1.628），填充从未触发
   （触发率 0%），修复前后 f1 的**编码器 SHA-256（`495d7f55842e6560`）、年化 -3.75%、
   Sharpe -0.2257、最大回撤 -29.01%、CVaR95 2.60%、年化换手 5.9924 全部逐位相同**——
   证明改动没有扰动默认路径。
2. **f2/f3 的差异不是「填充规则 A/B」，而是两个不同的训练解。** 填充发生在前向传播里，
   会改变梯度，修复前后 f2/f3 的编码器指纹不同（`c670931c…`/`0c0fbe30…` → `5f82a782…`/`16b34fe4…`）。
   所以上表只能读成「修复后的解更稀疏、且拼接后更好」，不能读成「填充规则本身带来 6.6 个百分点」。
3. **稀疏性回到正确语义。** 修复后持仓数（17.0 / 10.4 / 15.6）与 π 支撑集（32.2 / 8.9 / 15.9）
   同一量级，f2 折的 10.4 甚至是「支撑集 8.9 + 水位线多补 1~2 只」的结果；
   修复前的 119.0 / 101.4 与支撑集相差一个数量级，属于实现缺陷而非模型行为。

回归证据：旧运行目录保留在
[results/_invalidated_extra_returns_prefixfillbug/](./results/_invalidated_extra_returns_prefixfillbug/)，
两个目录各有一份 `report/selection_diagnostics.csv`，`cap_budget`/`fill_periods`/`book_n`
三列就是上面这张表的数据来源（`04_selection_diagnostics.py` 会重放每期的选择层，
所以诊断口径与训练时逐位一致）。

---

### 9.8 缺陷填充修复的回归对照

上一节末尾那个 bug 的完整对照。旧运行目录**原样保留**（未被覆盖、未被删除），
所有结论都能从下面这些落盘文件重算出来。

**一、触发范围普查。** `04_selection_diagnostics.py` 会重放每期的选择层，把
`cap_budget = Σᵢ raw_cap_i` 与缺口 `deficit = max(0, 1 − cap_budget)` 逐期写盘，
所以「哪些作业真的走进了填充分支」是可数的：

| 填充规则 | 已诊断的「实验×折」作业 | 至少触发一期 | 触发期次 / 总期次 |
| --- | --- | --- | --- |
| 旧（把缺口摊到整个横截面） | 45 | 6 | 105 / 1545 |
| 新（按分数水位线加宽选择） | 7 | 4 | 79 / 230 |

按运行目录拆开（`results/fill_control/fill_regression.md` 的 C 表由脚本自动生成）：

| 运行目录 | 填充规则 | 已诊断作业 | 至少触发一期 | 触发期次 / 总期次 |
| --- | --- | --- | --- | --- |
| `csi300_20260910_235309`（主网格） | 旧 | 27 | 2 | 15 / 927 |
| `csi300_score_rule_20260911`（补充网格） | 旧 | 9 | 1 | 4 / 309 |
| `csi300_full_universe`（完整池） | 旧 | 6 | 1 | 12 / 206 |
| `_invalidated_extra_returns_prefixfillbug`（19 通道旧） | 旧 | 3 | 2 | 74 / 103 |
| `csi300_extra_returns`（19 通道新） | 新 | 3 | 2 | 55 / 103 |
| `csi300_fill_fix_check_abl` | 新 | 2 | 2 | 24 / 48 |
| `csi300_fill_fix_check`（`full_musigma` f3） | 新 | 1 | 0 | 0 / 55 |
| `csi300_fill_fix_check_fu`（完整池 `full` f2） | 新 | 1 | 0 | 0 / 24 |

旧规则下触发的 6 个作业是：主网格 `full_nocost` f2（13/24 期）、`full_quad` f2（2/24）、
补充网格 `full_musigma` f3（4/55）、19 通道 f2（24/24）与 f3（50/55）、完整池 `full` f2（12/24）。
**其余 39 个作业一期都没触发**（`deficit = 0`）：例如 19 通道 f1 的 `cap_budget = 2.063 > 1`，
修复前后编码器 SHA-256（`495d7f55842e6560`）、年化 -3.75%、Sharpe -0.2257、
最大回撤 -29.01%、日 CVaR95 2.60%、年化换手 5.9924 **逐位相同**
（`report/selection_diagnostics.csv` 的 `encoder_sha256` 列可复核）。审计时还把主网格
三种选择层的 9 个作业（`lstm_topk`/`lstm_softmax`/`lstm_sparsemax` × 3 折）在修复后
重新诊断了一遍，9 份 `selection_diag.csv` 与旧文件的**字节完全一致**。

**二、重训对照（改了规则 + 重新训练）。** 受影响作业全部用修复后的代码重跑：

| 作业 | 期次 | 触发期次(旧→新) | 最低 cap_budget(旧→新) | 持仓数(旧→新) | 年化净收益(旧→新) | Sharpe(旧→新) |
| --- | --- | --- | --- | --- | --- | --- |
| 主网格 `full_nocost` f2 | 24 | 13/24 → 1/24 | 0.589 → 0.916 | 50.21 → 14.46 | 10.89% → 9.64% | 0.409 → 0.416 |
| 主网格 `full_quad` f2 | 24 | 2/24 → 23/24 | 0.886 → 0.398 | 17.88 → 12.83 | 5.26% → 9.97% | 0.215 → 0.381 |
| 补充网格 `full_musigma` f3 | 55 | 4/55 → 0/55 | 0.927 → 1.216 | 15.62 → 14.29 | 0.16% → -2.80% | 0.006 → -0.110 |
| 19 通道 f2 | 24 | 24/24 → 24/24 | 0.427 → 0.410 | 119.00 → 10.38 | 9.62% → 3.04% | 0.460 → 0.148 |
| 19 通道 f3 | 55 | 50/55 → 31/55 | 0.470 → 0.580 | 101.44 → 15.64 | 6.43% → 11.82% | 0.396 → 0.707 |
| 完整池 `full` f2 | 24 | 12/24 → 0/24 | 0.729 → 1.680（均值 1.043 → 2.047） | 26.08 → 29.08 | 0.42% → 3.92% | 0.024 → 0.268 |

**三、冻结权重对照（只换规则、不重训）。** 填充发生在前向传播里，重训会连带改变学到的解，
所以上表回答不了「规则本身带来什么」。`scripts/05_fill_control.py` 载入每个作业的
**旧权重**，对同一个测试折用两套规则各评一次（`spread_caps` 是脚本里按旧语义重写的
参考实现，`selection_caps` 直接用库里的新实现）：

| 作业 | 报告值（旧规则，权重冻结）年化/Sharpe/持仓 | 同一权重 + 新规则 年化/Sharpe/持仓 | 仅规则带来的差 |
| --- | --- | --- | --- |
| 主网格 `full_nocost` f2 | 10.89% / 0.409 / 50.21 | 10.35% / 0.382 / **16.92** | -0.54% / -0.027 / -33.29 |
| 主网格 `full_quad` f2 | 5.26% / 0.215 / 17.88 | **8.58%** / **0.345** / 13.21 | +3.32% / +0.130 / -4.67 |
| 补充网格 `full_musigma` f3 | 0.16% / 0.006 / 15.62 | -1.29% / -0.047 / 15.42 | -1.45% / -0.053 / -0.20 |
| 19 通道 f2 | 9.62% / 0.460 / 119.00 | 5.77% / 0.261 / **10.67** | -3.85% / -0.199 / **-108.33** |
| 19 通道 f3 | 6.43% / 0.396 / 101.44 | **8.23%** / **0.494** / **13.25** | +1.80% / +0.098 / **-88.18** |
| 完整池 `full` f2 | 0.42% / 0.024 / 26.08 | -0.24% / -0.012 / 23.79 | -0.66% / -0.036 / -2.29 |

**harness 自校验**：把**修复后**运行的 `model.pt` 交给脚本、两套规则都用新的，
输出与该运行自己的 `metrics.json` 在 9 个键（`n_days`/`total_return`/`ann_return`/`ann_vol`/
`sharpe`/`max_drawdown`/`cvar_95`/`turnover_ann`/`holdings_mean`）上**逐位相同（Δ = 0.0）**，
说明这条评测路径是确定性的、不受脚本影响。表里的「报告值」列取各运行报告的原值，
不取重建值；重建的旧规则与报告值的绝对误差在 6.6e-6 ~ 1.3e-2 之间（最大一项是完整池 f2 的
有效持仓），结构、符号与量级一致，足够当基准。

**四、怎么读这三张表。**

1. **没有触发填充时，修复是严格的空操作**（19 通道 f1、完整池 f1 等 39 个未触发的作业
   一期都没走进填充分支，编码器指纹与全部指标逐位相同；审计时重跑过诊断的 9 个作业字节一致）。
2. **规则本身的效果是「把持仓数拉回 π 支撑集的量级」**：`full_nocost` f2 50.2 → 16.9、
   19 通道 f2 119.0 → 10.7、f3 101.4 → 13.3；另外两处本来就没被摊稠，变化很小
   （完整池 f2 26.1 → 23.8，缺口最大只有 0.271；`full_musigma` f3 15.6 → 15.4，
   那一折缺口很小、只有 4 期、最大缺口 7.3 个百分点）。
3. **收益方向不稳定，必须如实报告。** 冻结权重下 `full_quad` f2 +3.32%、19 通道 f3 +1.80% 变好，
   19 通道 f2 −3.85%、`full_nocost` f2 −0.54%、`full_musigma` f3 −1.45% 变差。
   修复消除的是一个**明确的实现错误**（「稀疏选股」被执行成一百多只小权重），
   它**不是绩效改进**，不能拿它当卖点。
4. **表二的差值是「规则 + 重新求解」的联合效应**，比表三的纯规则差值大得多
   （如 `full_quad` f2：+4.71% vs +3.32%；完整池 f2：+3.50% vs −0.66%），
   且 6 个作业的编码器指纹与逐折权重全部改变。完整池 f2 是最极端的一例：
   冻结旧权重只换规则是 **−0.66% / −0.036**（几乎不动，那一折实际只有 12/24 期触发、
   缺口最大 0.271），重训后变成 **+3.50% / +0.245**，说明这点改善来自「重新训练出一个
   不再需要填补的解」（最低 `cap_budget` 从 0.729 抬到 1.680、均值 2.047，触发率 0），不是填充本身。
5. **修复后触发方向会反转**：`full_nocost` f2 从 13/24 期降到 1/24 期、完整池 f2 从 12/24 期降到 0/24 期，
   `full_quad` f2 则从 2/24 升到 23/24——「要不要加宽选择」是训练解的函数，
   修复只保证**一旦加宽就不再摊薄**（`book_n` 与 `pi_support` 同量级）。

产物与复现入口：`results/fill_control/*.json`（每个作业一份，含两套规则的完整指标）、
`results/fill_control/fill_regression.md`（本节三张表的自动生成版，
由 `python scripts/06_fill_regression.py --write` 写出；不带 `--write` 则只打印到终端）、
三个对照运行目录
[csi300_fill_fix_check](./results/csi300_fill_fix_check/)（1 作业）、
[csi300_fill_fix_check_abl](./results/csi300_fill_fix_check_abl/)（2 作业）、
[csi300_fill_fix_check_fu](./results/csi300_fill_fix_check_fu/)（1 作业），
以及 19 通道的旧/新两个目录
（[旧（证据）](./results/_invalidated_extra_returns_prefixfillbug/) /
[新](./results/csi300_extra_returns/)）。

诚实说明两点：旧填充分支的源码在修复时被直接覆盖（本仓库没有版本控制），
脚本里的 `spread_caps` 是按 §9.7 描述的语义重写的，所以「旧」列一律以报告原值为准；
另外，本节涉及的 6 个作业在 §9.1 / §9.5 / §9.6 的表里引用的是**修复前**的数字，
那些行已用 `†` / `‡` 标出并回指本节。

---

### 9.9 目标收益扫描：7 档 μ × 5 种网络结构（复刻图 Fig2–Fig5）

方案模块一/二要求"给定目标收益 μ，学习稀疏组合、并在此约束下优化 CVaR 意义上的稳健性"。
参考研究把 μ 从 0.016 扫到 0.028；我们把 `opt.mu_target` 实现成**硬约束**（`μᵀy ≥ τ`，
逐期可行性钳制，见 [optlayer.py](./src/e2e_portfolio/optlayer.py) 的 `MeanCVaRLayer.feasible_mu`）
并让 5 种编码器跑同一套 7 档目标。这是与 §9.1 主网格的关键差别：主网格里的 μ 是**软惩罚**，
这里是可以逐期核验是否满足的硬约束。

| 实验族 | run 目录 | 实验 × 折 = 作业 | 编码器 |
|---|---|---|---|
| μ 扫描主体 | [csi300_mu_sweep_20260912](./results/csi300_mu_sweep_20260912/) | 21 × 3 = 63 | LSTM / RNN / GRU |
| μ 扫描扩展 | [csi300_mu_sweep_ext_20260912](./results/csi300_mu_sweep_ext_20260912/) | 14 × 3 = 42 | MLP / RBFN |

三个折的测试窗口首尾相接（2018–2019 / 2020–2021 / 2022–2026），下表把拼接后的日度净收益
当成一段样本外序列重算指标（口径见 §7）。**每个配置只跑了 1 个种子**，所以表里小于
0.2 的夏普差异、小于 2 个百分点的年化差异不应被解释为真实效应（见 §10）。

**年化波动率（%）**

| 编码器 | τ=0.016 | τ=0.018 | τ=0.020 | τ=0.022 | τ=0.024 | τ=0.026 | τ=0.028 |
|---|---|---|---|---|---|---|---|
| LSTM+MCVaR | 16.79 | 18.09 | 16.94 | 18.16 | 17.24 | 18.54 | 17.78 |
| RNN+MCVaR | 22.75 | 23.35 | 22.06 | 21.82 | 17.50 | 20.19 | 22.34 |
| GRU+MCVaR | 22.37 | 19.86 | 19.30 | 22.18 | 23.64 | 28.08 | 20.63 |
| MLP+MCVaR | 18.33 | 24.17 | 18.53 | 17.33 | 23.48 | 24.27 | 21.05 |
| RBFN+MCVaR | 22.29 | 23.18 | 22.28 | 22.28 | 21.50 | 22.30 | 22.33 |

**夏普比率**

| 编码器 | τ=0.016 | τ=0.018 | τ=0.020 | τ=0.022 | τ=0.024 | τ=0.026 | τ=0.028 |
|---|---|---|---|---|---|---|---|
| LSTM+MCVaR | 0.04 | 0.01 | 0.10 | -0.01 | 0.24 | 0.02 | 0.10 |
| RNN+MCVaR | -0.05 | 0.40 | -0.41 | -0.01 | 0.10 | 0.22 | 0.38 |
| GRU+MCVaR | 0.21 | 0.16 | 0.48 | **0.55** | -0.19 | 0.20 | -0.15 |
| MLP+MCVaR | **0.60** | -0.16 | 0.30 | -0.08 | -0.16 | 0.09 | 0.55 |
| RBFN+MCVaR | 0.14 | 0.11 | 0.14 | 0.14 | 0.08 | 0.14 | 0.14 |

**相对沪深 300 的跟踪误差（%）**

| 编码器 | τ=0.016 | τ=0.018 | τ=0.020 | τ=0.022 | τ=0.024 | τ=0.026 | τ=0.028 |
|---|---|---|---|---|---|---|---|
| LSTM+MCVaR | 12.76 | 13.06 | 13.74 | 12.10 | 14.05 | 14.37 | 13.43 |
| RNN+MCVaR | 12.71 | 15.52 | 12.83 | 14.99 | 15.10 | 12.45 | 12.79 |
| GRU+MCVaR | 15.36 | 14.13 | 15.41 | 16.57 | 15.27 | 17.81 | 12.07 |
| MLP+MCVaR | 14.98 | 11.45 | 12.31 | 10.39 | 14.15 | 13.86 | 16.29 |
| RBFN+MCVaR | 6.20 | 7.74 | 6.21 | 6.21 | 5.14 | 6.23 | 6.31 |

**怎么读这张表**

1. **波动率的排序基本由结构决定，不受 τ 支配**：LSTM 全程 16.8–18.5%（最低），
   MLP 18.3–24.3%，RNN/GRU/RBFN 在 20–23% 之间。RBFN 几乎是一条水平线
   （22.29/23.18/22.28/22.28/21.50/22.30/22.33）——它的 RBF 基函数把输出压成了一个
   与 μ 目标近似无关的稳定组合，这既是优点（跟踪误差 5.1–7.7%，全场最低）也是缺点
   （夏普 0.08–0.14，收益端没有响应）。
2. **GRU 是收益端最有效的编码器**：τ=0.020 年化 +9.30%（夏普 0.48）、τ=0.022 年化 +12.15%
   （夏普 0.55、最大回撤 −45.9%），是三折合计最好的两条曲线。
3. **τ 与业绩不是单调关系**。GRU 在 0.020–0.022 有明确甜点区、τ≥0.024 之后转负
   （0.024 夏普 −0.19），RNN 则是 0.020 最差（−0.41）而 0.018/0.028 最好（0.40 / 0.38）。
   用硬约束把目标收益往上抬，并不会让预测质量跟着变好；它只是把可行集切掉一部分，
   在预测有信号时能兑现成收益，在预测没信号时只能放大换手和跟踪误差。
4. **MLP 的 0.60（τ=0.016）看似最高，但同一结构的 τ=0.018 是 −0.16**，
   这种相邻两档之间的剧烈跳变是典型的单种子噪声，不能当作结论；RBFN 的
   "稳定但平庸" 与 GRU 的 "高波动高收益" 才是可复现的结构差异。
5. 复刻图 [fig02](./results/figures_reference/fig02_std_vs_target.png) /
   [fig03](./results/figures_reference/fig03_sharpe_vs_target.png) /
   [fig04](./results/figures_reference/fig04_tracking_error_vs_target.png) 画的就是这三张表
   （5 种结构 × 7 档 τ，分组柱状图），[fig05](./results/figures_reference/fig05_cumulative_return_mu020.png)
   画 τ=0.020 时 5 条累计净值。

产物：每个作业的 `metrics.json` / `backtest/*.csv` 都在上面两个 run 目录里；
跑完后的汇总表由 [scripts/10_summary_tables.py](./scripts/10_summary_tables.py) 自动生成
（`python scripts/10_summary_tables.py --out results/logs/summary_tables.md`）。

---

### 9.10 熵正则扫描：λ 与实际组合熵（复刻图 Fig8–Fig9）

方案模块二要求"用熵正则提高组合分散度"。我们把 `opt.lam_div` 从 0 扫到 0.010
（5 档 × LSTM/GRU/RNN，run 目录 [csi300_entropy_sweep_20260912](./results/csi300_entropy_sweep_20260912/)，
15 × 3 = 45 个作业），同时记录**每一期的实际实现熵**
（`-Σ w log w`，由 `scripts/09_reference_figures.py` 的 `portfolio_entropy` 在 `rebalance.csv` 上算出）。

注意 λ 命名的读法：实验名 `entKXX_arch` 的 KXX = λ×10000，
即 `ent010`=0.001、`ent020`=0.002、`ent050`=0.005、`ent100`=0.010。

**夏普比率（按请求的分散度 λ）**

| 编码器 | λ=0.000 | λ=0.001 | λ=0.002 | λ=0.005 | λ=0.010 |
|---|---|---|---|---|---|
| LSTM+MCVaR | 0.13 | -0.19 | 0.17 | 0.20 | 0.11 |
| GRU+MCVaR | 0.00 | 0.08 | -0.06 | -0.13 | 0.10 |
| RNN+MCVaR | 0.08 | 0.30 | -0.04 | 0.15 | -0.32 |

**实际实现的组合熵（nats，折内均值）**

| 编码器 | λ=0.000 | λ=0.001 | λ=0.002 | λ=0.005 | λ=0.010 |
|---|---|---|---|---|---|
| LSTM+MCVaR | 2.39 | 2.46 | 2.50 | 2.60 | 2.85 |
| GRU+MCVaR | 3.08 | 2.64 | 2.69 | 2.40 | 2.70 |
| RNN+MCVaR | 2.58 | 2.45 | 2.45 | 2.70 | 3.14 |

**年化净收益（%）**

| 编码器 | λ=0.000 | λ=0.001 | λ=0.002 | λ=0.005 | λ=0.010 |
|---|---|---|---|---|---|
| LSTM+MCVaR | 3.27 | -3.80 | 2.90 | 3.45 | 1.87 |
| GRU+MCVaR | 0.03 | 1.55 | -1.13 | -2.24 | 1.75 |
| RNN+MCVaR | 2.07 | 7.42 | -1.05 | 3.19 | -5.93 |

**结论**

1. **熵项确实按预期起作用，但只在 LSTM/RNN 上**：λ 从 0 升到 0.010，
   LSTM 的实际熵单调上升 2.39 → 2.85，RNN 2.58 → 3.14。
   GRU 相反：λ=0 时它的实际熵已经是 3.08（三种结构里最高），加熵后反而降到 2.4–2.7——
   GRU 学到的解本来就分散，此时熵梯度主要在扰动组合而不是在增加分散度。
   这正是参考图 Fig8 把横轴画成"实现熵"而不是 λ 的原因：同一条 λ 曲线在不同结构上
   落到完全不同的分散度。
2. **分散度换不来收益**。除了 LSTM 在 λ=0.005（夏普 0.13 → 0.20）和 RNN 在 λ=0.001
   （0.08 → 0.30）这两处，夏普没有系统性改善；RNN 在 λ=0.010 反而恶化到 −0.32
   （年化 −5.93%）。在我们的样本上，"加熵 → 更分散 → 更稳" 的链条只有前半段成立。
3. 综合 §9.9，**这个设置下最有效的稳健化手段是 GRU 编码器 + 中档目标收益（τ≈0.020–0.022）**，
   而不是加大熵正则。

复刻图：[fig08](./results/figures_reference/fig08_entropy_sharpe_ratio.png)（横轴 = 实现熵，
3 条曲线）、[fig09](./results/figures_reference/fig09_entropy_cumulative_return.png)
（λ=0 与 λ=0.002 的累计净值上下两面板）。

补充说明：参考图上熵扫描只有 RNN/LSTM/GRU 三条线，所以 MLP/RBFN 的熵扫描
（`configs/csi300_entropy_sweep_ext.yaml`，已写好但未跑）没有纳入本次结果，
如需补齐可直接执行该配置。

---

### 9.11 树模型基线：AdaBoost / XGBoost（复刻图 Fig13–Fig14）

参考研究用 AdaBoost 与 XGBoost 作为"传统机器学习选股"基线。我们在**完全相同的数据、
折划分、CVaR 优化层、成本与回测口径**下补了这一族：树模型给出每票的收益分数，
交给同一个 `MeanCVaRLayer` 求组合（run 目录
[csi300_tree_baselines_20260912](./results/csi300_tree_baselines_20260912/)，
4 × 3 = 12 个作业；特征为 60 日窗口的压缩统计量，见 `compact_tree_features`）。

| 目标 τ | 预测器 | 年化净收益 % | 年化波动 % | 夏普 | 最大回撤 % | 跟踪误差 % |
|---|---|---|---|---|---|---|
| 0.020 | AdaBoost+MCVaR | -2.57 | 19.96 | -0.13 | -46.67 | 15.19 |
| 0.020 | XGBoost+MCVaR | -2.99 | 22.24 | -0.13 | -47.52 | 14.79 |
| 0.020 | LSTM+MCVaR | 1.62 | 16.94 | 0.10 | -51.15 | 13.74 |
| 0.020 | **GRU+MCVaR** | **9.30** | 19.30 | **0.48** | -50.99 | 15.41 |
| 0.020 | RNN+MCVaR | -8.94 | 22.06 | -0.41 | -67.63 | 12.83 |
| 0.022 | AdaBoost+MCVaR | -0.12 | 20.76 | -0.01 | -45.78 | 14.23 |
| 0.022 | XGBoost+MCVaR | -4.22 | 24.50 | -0.17 | -49.87 | 14.53 |
| 0.022 | LSTM+MCVaR | -0.12 | 18.16 | -0.01 | -54.08 | 12.10 |
| 0.022 | **GRU+MCVaR** | **12.15** | 22.18 | **0.55** | -45.87 | 16.57 |
| 0.022 | RNN+MCVaR | -0.12 | 21.82 | -0.01 | -46.28 | 14.99 |

**结论**

1. **两档目标下，两种树模型都没能打败最好的递归结构**：τ=0.020 时 GRU 的年化是
   AdaBoost 的 3.6 倍（+9.30% vs −2.57%）；τ=0.022 时是 +12.15% vs −0.12%。
   在同样的约束层与成本下，这个差距只能归给预测器本身。
2. **树模型的波动率与跟踪误差并不占优**：AdaBoost/XGBoost 波动 19.96–24.50%、
   跟踪误差 14.2–15.2%，与 GRU（19.3–22.2%、15.4–16.6%）同量级，
   但收益端为负 —— 它们把目标收益约束"顶"到了很高的换手，却没有相应的信号。
3. **方向与参考研究一致**：Fig13/Fig14 显示 Ada/XGB 低于 LSTM/GRU。
   我们的数据支持这个定性结论，但**不支持"神经网络稳定胜出"**——RNN 在 τ=0.020 时
   比树模型还差（−8.94% / 夏普 −0.41），说明优势来自具体结构（门控 + 时序记忆），
   不是"用了深度学习"这件事本身。

复刻图：[fig13](./results/figures_reference/fig13_adaboost_xgboost_mu020.png)（τ=0.020）、
[fig14](./results/figures_reference/fig14_adaboost_xgboost_mu022.png)（τ=0.022），
每张 5 条曲线：AdaBoost / XGBoost / LSTM / GRU / RNN。

---

### 9.12 换股票池：中证 500 与上证 50（复刻图 Fig11–Fig12）

参考研究在图 11/12 里把同一套流程搬到另一个股票池，并且**在同一个股票池里换网络结构**
看曲线怎么散开。本地数据没有巴西 IBrX50，我们用**两个性质不同的 A 股池**替代
（都是在同一套数据管线、同样 3 折、同样的
`ew` / `meancvar_hist` / `mu020_lstm` / `full` 四种策略下跑出来的）：

* 上证 50（`sse50`，50 只，大盘蓝筹，config `sse50_reference.yaml`，`top_liquidity: 50`）
* 中证 500（`csi500`，500 只，中盘，config `csi500_reference.yaml`，`top_liquidity: 120`）

| 股票池 | 策略 | 年化净收益 % | 年化波动 % | 夏普 | 最大回撤 % | 跟踪误差 % |
|---|---|---|---|---|---|---|
| 沪深 300 | ew（1/N） | 2.94 | 21.71 | 0.14 | -45.38 | 5.75 |
| 沪深 300 | meancvar_hist | 5.27 | 32.11 | 0.16 | -55.83 | 23.80 |
| 沪深 300 | mu020_lstm※ | 1.62 | 16.94 | 0.10 | -51.15 | 13.74 |
| 沪深 300 | full | 1.10 | 16.80 | 0.07 | -46.82 | 12.93 |
| 中证 500 | ew | -2.71 | 21.84 | -0.12 | -46.53 | 6.87 |
| 中证 500 | meancvar_hist | -11.00 | 29.62 | -0.37 | -65.63 | 20.38 |
| 中证 500 | mu020_lstm | -7.35 | 19.67 | -0.37 | -53.72 | 15.53 |
| 中证 500 | full | -0.88 | 16.82 | -0.05 | -41.79 | 17.39 |
| 上证 50 | ew | 2.14 | 15.99 | 0.13 | -29.63 | 7.63 |
| 上证 50 | meancvar_hist | 2.06 | 28.57 | 0.07 | -47.54 | 24.43 |
| 上证 50 | **mu020_lstm** | **5.71** | 13.96 | **0.41** | -23.41 | 12.40 |
| 上证 50 | full | 2.32 | 13.17 | 0.18 | -27.04 | 11.84 |

※ 沪深 300 的 `mu020_lstm` 取自 §9.9/§9.11 的 LSTM+MCVaR（τ=0.020），
口径与其他两池同源，列在这里只是为了三池可比。

#### 9.12.1 同一池里换编码器：五种结构的 τ=0.020 横截面

参考图 11/12 的每条曲线是同一策略下的不同网络结构。为此我们为两个池各补了一套
**五结构 × 3 折**的网格（config `csi500_arch_compare.yaml` / `sse50_arch_compare.yaml`，
`run_name` 为 `csi500_arch_compare_20260913` / `sse50_arch_compare_20260913`，
15 个作业各池），除 `model.arch` 外与上面的 `mu020_lstm` 完全同配置：

| 股票池 | 编码器 | 年化净收益 % | 年化波动 % | 夏普 | 最大回撤 % | 跟踪误差 % |
|---|---|---|---|---|---|---|
| 中证 500 | RNN+MCVaR | -0.80 | 24.20 | -0.03 | -60.45 | 15.25 |
| 中证 500 | LSTM+MCVaR | -7.35 | 19.67 | -0.37 | -53.72 | 15.53 |
| 中证 500 | GRU+MCVaR | -5.01 | 19.32 | -0.26 | -49.68 | 14.08 |
| 中证 500 | MLP+MCVaR | -3.85 | 20.55 | -0.19 | -46.39 | 15.81 |
| 中证 500 | RBFN+MCVaR | -2.89 | 21.41 | -0.13 | -46.25 | 8.05 |
| 上证 50 | RNN+MCVaR | 4.89 | 16.23 | 0.30 | -29.49 | 11.90 |
| 上证 50 | LSTM+MCVaR | 5.71 | 13.96 | 0.41 | -23.41 | 12.40 |
| 上证 50 | GRU+MCVaR | -1.93 | 15.36 | -0.13 | -39.88 | 13.25 |
| 上证 50 | MLP+MCVaR | -4.02 | 16.25 | -0.25 | -45.68 | 10.46 |
| 上证 50 | RBFN+MCVaR | 2.22 | 16.07 | 0.14 | -29.71 | 6.68 |

两个池里 `mu020_lstm` 与第一张表的同一行数值完全一致，说明两套 run 的可比性成立。

**结论**

1. **模型优势不是普适的，它取决于股票池的性质。** 上证 50 上 `mu020_lstm` 全面领先：
   年化 +5.71%（vs 等权 +2.14%）、夏普 0.41（vs 0.13）、最大回撤 −23.4%（vs −29.6%）、
   波动 13.96%（vs 15.99%）。中证 500 上则**全线为负**：
   等权 −2.71%、历史 Mean-CVaR −11.00%、`mu020_lstm` −7.35%、`full` −0.88%，
   相对最好的是"完整流程"那一档（−0.88%、夏普 −0.05）。
2. **大盘池的结果与参考研究方向一致**：IBrX50 是巴西的大盘蓝筹指数，我们的上证 50
   替代跑出了正收益和正夏普，说明这套"神经选股 + Mean-CVaR"在大盘蓝筹上确实有效；
   而中证 500 的反例（含等权在内全部亏损、策略间排序被打乱）提醒：
   在中盘股上，滚动 60 日特征 + 高斯场景的 CVaR 层没有提供可用的信号，
   这时任何"最优化"都在放大噪声。
3. 三池合看还有一条一致的规律：**`meancvar_hist` 历史上从没好过**——它波动率最高
   （28.6–32.1%）、跟踪误差最大（20.4–24.4%）、回撤最深，只是在沪深 300 上靠高波动
   换到了一点收益（+5.27%）。这与 §9.3/§9.5 的观察一致：用历史情景直接做 Mean-CVaR
   在 A 股上是一个高风险的选择。
4. **换编码器的效果同样取决于股票池，而且没有"哪个结构一定更好"的规律。**
   上证 50 上只有两个循环结构站住了：LSTM 夏普 0.41、RNN 0.30（两者差 0.11，
   低于 §10 的 0.2 噪声阈值，只能算并列），GRU 反之为 −0.13；
   RBFN（0.14）基本等于等权（0.13），MLP（−0.25）是唯一明显跑输等权的。
   中证 500 上五者全负，排序还与上证 50 相反：RNN（−0.03）反而最好、LSTM（−0.37）最差。
   唯一跨池一致的是：**MLP 和 GRU 在两个池上都拿不到正夏普**，
   说明"用纯前馈网络（MLP）替掉循环编码器"在本设置下不是升级。
5. **跟踪误差维度给出另一条线索**：上证 50 上 RBFN 的年化收益（2.22%）与等权相当，
   但跟踪误差只有 6.68%，是五种结构里最低的；中证 500 上 RBFN 的跟踪误差 8.05% 同样
   最低。RBFN 的持仓更贴近基准，代价是几乎没有超额收益——它更接近"低跟踪误差的
   基准增强"，而不是 α 来源。

复刻图：[fig11](./results/figures_reference/fig11_csi500_cumulative_return.png)（中证 500）、
[fig12](./results/figures_reference/fig12_sse50_cumulative_return.png)（上证 50）。

---

### 9.13 11 张复刻图对照总表

`D:\figures` 里的 11 张参考图，全部用本仓库 `results/` 下自己跑出来的数据重画，
统一放在**独立文件夹** [results/figures_reference](./results/figures_reference/)，
不动参考图目录。画布尺寸与原图逐张一致（除 fig14 差 3 px，见下），
字号 / 图例版面 / 坐标盒位置均按参考图逐像素量取后照抄（复核实测见下表的第 8–10 条）。

| 本仓库的图 | 参考图 | 内容 | 数据来源（run 目录） |
|---|---|---|---|
| [fig02_std_vs_target.png](./results/figures_reference/fig02_std_vs_target.png) | Fig2 | τ × 5 结构的年化波动 | `csi300_mu_sweep*` |
| [fig03_sharpe_vs_target.png](./results/figures_reference/fig03_sharpe_vs_target.png) | Fig3 | τ × 5 结构的夏普 | `csi300_mu_sweep*` |
| [fig04_tracking_error_vs_target.png](./results/figures_reference/fig04_tracking_error_vs_target.png) | Fig4 | τ × 5 结构的跟踪误差 | `csi300_mu_sweep*` |
| [fig05_cumulative_return_mu020.png](./results/figures_reference/fig05_cumulative_return_mu020.png) | Fig5 | τ=0.020 的 5 条累计净值 | `csi300_mu_sweep*` |
| [fig08_entropy_sharpe_ratio.png](./results/figures_reference/fig08_entropy_sharpe_ratio.png) | Fig8 | 实现熵 × 结构 的夏普 | `csi300_entropy_sweep*` |
| [fig09_entropy_cumulative_return.png](./results/figures_reference/fig09_entropy_cumulative_return.png) | Fig9 | 熵正则开/关的累计净值（上下双面板） | `csi300_entropy_sweep*` |
| [fig10_out_of_sample_return.png](./results/figures_reference/fig10_out_of_sample_return.png) | Fig10 | 样本外月度已实现收益（左右双面板） | `csi300_mu_sweep*` |
| [fig11_csi500_cumulative_return.png](./results/figures_reference/fig11_csi500_cumulative_return.png) | Fig11 | 中证 500 上的累计净值（5 种结构） | `csi500_reference*`（ew / hist / full）+ `csi500_arch_compare*`（5 结构） |
| [fig12_sse50_cumulative_return.png](./results/figures_reference/fig12_sse50_cumulative_return.png) | Fig12 | 上证 50 上的累计净值（5 种结构） | `sse50_reference*`（ew / hist / full）+ `sse50_arch_compare*`（5 结构） |
| [fig13_adaboost_xgboost_mu020.png](./results/figures_reference/fig13_adaboost_xgboost_mu020.png) | Fig13 | τ=0.020：Ada/XGB/LSTM/GRU/RNN | `csi300_mu_sweep*` + `csi300_tree_baselines*` |
| [fig14_adaboost_xgboost_mu022.png](./results/figures_reference/fig14_adaboost_xgboost_mu022.png) | Fig14 | τ=0.022：同上 | `csi300_mu_sweep*` + `csi300_tree_baselines*` |

**与参考图的差别（诚实说明，逐条对照）**

| # | 差别 | 说明 |
|---|---|---|
| 1 | **样本区间** | 参考图横轴 2016–2023；我们的折是 2018–2019 / 2020–2021 / 2022–2026（首尾相接，横轴 2018–2026）。数字不可与参考图逐点比对。 |
| 2 | **Fig11 的市场** | 参考图用巴西 IBrX50，本地数据没有该指数，改用中证 500（"换一个股票池"的同等作用），文件名与图内标注都写明是 CSI 500。 |
| 3 | **Fig8 的横轴** | 两图一样是"实际实现的组合熵"（nats），不是 λ；横轴刻度是每一档 λ 在三条结构上实现的平均熵。 |
| 4 | **Fig8/Fig9 的结构数** | 参考图只有 RNN/LSTM/GRU 三条线，所以我们的熵扫描也只跑这三种结构（MLP/RBFN 的熵配置已备好未跑，见 §9.10）。 |
| 5 | **Fig10 的面板** | 参考图里左右两块到底差在哪个维度（窗口？标的？参数？）无法从图上看出，我们改成**同一个 11 个月窗口（2023-08 ~ 2024-06）**、目标收益最低档 τ=0.016 与最高档 τ=0.028 的对比，每块面板画全 5 种结构的月度已实现净收益；这个假设写在 `panels.json` 的 `assumption` 字段里。 |
| 6 | **指标口径** | 波动/夏普/跟踪误差在"三折首尾相接的日度净收益"上重算，与 `report/pooled.csv` 同口径（年化 252；TE = std(组合−基准)×√252）。 |
| 7 | **fig14 尺寸** | 1324×603 等 10 张与参考图逐像素一致；`fig14` 比参考图窄 3 px（1360 vs 1363），原因是 matplotlib 的自适应 `bbox_inches` 对图例文字宽度的取整，不影响内容。 |
| 8 | **折线图右侧留白：我们 44 px、参考图 17 px（已查明是数据端点造成的，不是版式差异）** | 我们的样本止于 2026-08-20，matplotlib 默认的 5% 横轴边距把视窗推到 2027-01-01，最后一格「2027」年刻度正好压在坐标区右缘、标签一半悬在轴外，`tight_layout` 必须为它留宽：16.5 px pad + ≈27 px 悬出 = 44 px，与实测逐像素闭合。参考图数据止于 2023 年中，末刻度「2023」深处轴内，所以只要 17 px。除留白外的刻度要素两边实测一致：年度刻度定位、末刻度墨迹宽（58 vs 60 px）、图例版面（左上角、单列 5 行、行距同为 ≈37 px）、无网格线。 |
| 9 | **折线图左侧留白差 15 px（参考图坐标区左缘 x≈119，我们 x≈134）** | 两边纵轴标题都是 `"Cumulative return"`（OCR 复核一致）、左缘都在画布 x≈15；差值来自纵轴刻度标签块的宽度（103 vs 117 px）经 `tight_layout` 分配，同样属数据驱动的派生量。 |
| 10 | **柱状图的图例与基线（已修）** | 图例 17 pt，并让图例不参与 `tight_layout`（`set_in_layout(False)`）——居中的 5 列宽图例若参与布局，会把绘图区压窄到 1032 px；修正后绘图区宽 1193 px，参考图实测 1192 px。Fig8 的图例内部间距按参考图另行反解（色块→文字 23 px、条目间距 51 px，复测 22 / 52 px）。另经程序化逐柱核对：Fig8 的 15 根柱子下沿的数据坐标**全部为 0**（同一根 y=0 基线），只因我方夏普跨零（−0.32 ~ +0.30）而 0 基线落在坐标区中部约 277 px、没有贴住下边界——这是数据差、不是画法差。 |

**复现命令**（详细版见 [results/figures_reference/README.md](./results/figures_reference/README.md)）

```powershell
# 1) 五结构 × 七档目标收益（105 个作业）
python scripts/02_run_experiments.py --config configs/csi300_mu_sweep.yaml --workers 3 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/csi300_mu_sweep_ext.yaml --workers 3 --threads-per-worker 2
# 2) 熵扫描（45 个作业）
python scripts/02_run_experiments.py --config configs/csi300_entropy_sweep.yaml --workers 3 --threads-per-worker 2
# 3) 树模型基线（12 个作业）
python scripts/02_run_experiments.py --config configs/csi300_tree_baselines.yaml --workers 3 --threads-per-worker 2
# 4) 另外两个股票池（各 12 个作业）
python scripts/02_run_experiments.py --config configs/csi500_reference.yaml --workers 3 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/sse50_reference.yaml --workers 3 --threads-per-worker 2
# 4b) 两个池里的五结构对照（各 15 个作业，fig11/fig12 的曲线来源）
python scripts/02_run_experiments.py --config configs/csi500_arch_compare.yaml --workers 3 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/sse50_arch_compare.yaml --workers 3 --threads-per-worker 2
# 5) 汇总报告与画图
python scripts/03_report.py --run results/csi300_mu_sweep_20260912
python scripts/10_summary_tables.py --out results/logs/summary_tables.md
python scripts/09_reference_figures.py --out results/figures_reference
```

**这一轮新增的算力规模**：216 个 (实验, 折) 作业（μ 扫描 105 + 熵扫描 45 + 树模型 12 +
两池 24 + 两池五结构 30），全部单机 CPU 完成；每个作业 5–23 分钟。`panels.json` 记录了每张图对应的
实验名、指标定义与数据来源目录，可逐条核对。

---

## 10. 局限

* 单一指数（沪深 300 动态成分股），单一线性成本假设；未建模冲击成本、涨跌停延迟成交、融券约束。
* 3 折、约 8 年样本外，统计功效有限；不同年份区间结论差异很大。
* CVaR 的"稳健"只相对于高斯参数化场景模型成立。
* 深度网络随机性带来的同配置重复训练差异可达数个百分点，
  报告中小于该量级的差异不应被解释为真实效应。
* `n_max` 会通过 dropout 的随机数消耗量影响训练轨迹：槽位数不同，同一实验在同一折上
  的结果不能直接横向比较（本仓库同一折内所有实验共用同一个 `n_max`，折间则不同）。
* 稀疏性只在选择层的输出上成立，不代表最终持仓稀疏：单票 10% 上限会把权重摊到很多票上，
  有效持仓数应由 `rebalance.csv` 的 `eff_holdings` 判断，而不是 `π` 的支持集大小。
* 编码器只用了方案建议的「第一版」共享 LSTM（`StockEncoder`）。方案里提到的后续替换
  （MASTER、StockMixer 之类的横截面模型）不在本次范围内；接口上它们是可插拔的
  ——只要保持 `(N, L, F) → 每票嵌入` 的形状，`models.py` 里的 `StockEncoder` 可以直接替换。

---

## 11. 与原研究方案的逐条对应

下表把原方案里每一条可执行的要求映射到本仓库的实现位置与可核对的产物。
第 0 节给出实验数据；模块、损失、实验、指标四张表逐条对账；最后列出五处**有意偏离**。

### 11.1 数据协议

| 方案要求 | 实现 | 证据 |
| --- | --- | --- |
| 沪深 300 **动态成分股**，无幸存者偏差 | `data.py` 读 `membership_mask.npy`，每期只用当期成分股 | `01_prepare_data.py --validate`；`dataset_info_csi300.json` 记录逐期成分股数 |
| 日频数据 | tensor 面板 T=4046 天（2010-01-04 → 2026-08-31） | `metrics.json` 的 `n_days` |
| 60 日回看窗口 | `features.lookback: 60` | 每个作业的 `config.yaml` |
| 20 日持有/预测期 | `backtest.rebalance_every: 20`、`dataset` 的持有期 `h=20` | `config.yaml`、`dataset_info_*.json` |
| 月度调仓 | `backtest.rebalance: "monthly"`，实测年化调仓频率 ≈ 12.47 次/年 | `rebalance.csv` |
| 前复权（qfq）价格 | `data.py` 用 qfq 收盘价构造收益与特征 | `data.py` 顶部注释、`features.json` |
| 停牌 / 涨跌停处理 | `observed_mask`（停牌）+ `exclude_limit_locked`（涨跌停锁死不可成交），不可卖出的持仓进入「必须保留」约束 | `dataset.py`、`backtest.py`、`metrics.json` 的 `max_investable` |
| 严格 walk-forward，按时间切分 | `folds` 三段（F1 2018–19 / F2 2020–21 / F3 2022–26），训练窗口扩张，验证 1 年 | §5 折划分表；`03_report.py` 的折边界检查 |
| 调试阶段可先用流动性前 50–100 名，模型稳定后再扩展到完整沪深 300 | 调试网格 `data.top_liquidity: 120`；规模扩展网格 `data.top_liquidity: 0`（完整沪深 300，槽位 613/682/778） | `config.yaml`；[configs/csi300_full_universe.yaml](./configs/csi300_full_universe.yaml)；结果见 §9.6 |

### 11.2 模块

| 方案模块 | 实现 | 备注 |
| --- | --- | --- |
| ① 共享 LSTM 编码器，输入 `(N, L, F)` | [models.py](./src/e2e_portfolio/models.py) `StockEncoder`：LayerNorm → 单层 LSTM(64) → MLP(64) → 三个头 | 编码器**跨股票共享**，不是一只票一个网络；`model.arch` 还可切换 `lstm`/`gru`/`rnn`/`mlp`/`rbfn` 五种结构，用于复刻图 Fig2–Fig5 的结构对比，见 §9.9 |
| ② 收益与风险预测（μ̂、σ̂ 或 S 个场景） | `mu_layer`、`sigma_layer`（softplus + 上下限）；场景由 `scenario_mode` 决定：`gaussian` 用分位点网格 `R = μ + σ·z`，`direct` 直接输出 S 个场景 | `num_scenarios: 60` |
| ③ 可微稀疏选择 `π = Sparsemax(s)`，第一版不用 Gumbel | [selection.py](./src/e2e_portfolio/selection.py) `SparsemaxSelection`（Martins & Astudillo 2016，`π = argmin_{p∈Δ} ½‖p−z‖²`）；另有 `TopKSelection`、`SoftmaxSelection` 供第四步对比 | 分数 `s` 的两种来源见 §3「两种选择分数」；`mu_sigma` 就是方案给的公式 |
| ④ 可微 Mean-CVaR 层（cvxpylayers） | [optlayer.py](./src/e2e_portfolio/optlayer.py)：目标 `−μᵀy + γ[ζ + 1/((1−α)S)·Σu_s] + λ_turn‖y−y_prev‖₁ − λ_H H(y)`，约束 `u_s ≥ −R_sᵀy − ζ, u ≥ 0, Σy = 1, 0 ≤ y_i ≤ cap_i`，另可选**目标收益硬约束** `μᵀy ≥ τ`（`opt.mu_target`，逐期可行性钳制 `MeanCVaRLayer.feasible_mu`，见 §9.9） | 数学形式见 §3；有限差分梯度一致性见 `tests/test_optlayer.py`；硬约束与 scipy 参考 LP 的一致性有单测 |

### 11.3 损失

| 方案要求 | 实现 |
| --- | --- |
| 五项联合损失 `L = λ1L_pred + λ2L_decision + λ3L_tail + λ4L_turnover + λ5L_sparse` | [losses.py](./src/e2e_portfolio/losses.py) `composite_loss`，权重 `l_pred=0.1, l_decision=1.0, l_tail=0.5, l_turnover=1.0, l_sparse=0.1` |
| 逐 epoch 可核查 | 每个作业的 `history.csv` 记录 5 个损失项与验证损失；`03_report.py` 画训练曲线 |

### 11.4 四步递进

| 方案步骤 | 实现 | 实验名 |
| --- | --- | --- |
| 第 1 步：复现基线（LSTM 预测 + **固定 Top-k 选股** + 传统 Mean-CVaR 优化） | LSTM 只由 `l_pred` 训练，固定 Top-k 选股，Mean-CVaR 优化层**只在测试期**使用（即「先预测再优化」）；另有不学习的 1/N 与历史情景 Mean-CVaR 作参照 | `lstm_topk`、`lstm_softmax`（选择层对照）、`ew`、`meancvar_hist` |
| 第 2 步：加入可微 Mean-CVaR 层，**暂时不做神经选股**，先验证梯度能否稳定通过优化层传回预测网络 | 优化层进入训练回路，选择层退化为均匀 cap（`selection: identity`），损失保留预测/决策/尾部/换手四项（`l_sparse = 0`）；「梯度能否稳定回传」有两条独立单测 | `e2e_noselect`；`tests/test_optlayer.py::test_layer_gradient_matches_finite_differences`（与有限差分对比）、`tests/test_train.py::test_gradients_flow_through_the_layer_into_the_encoder`（编码器参数确实拿到非零梯度） |
| 第 3 步：把固定 Top-k 换成可微稀疏选择，并比较 Top-k / Softmax / Sparsemax | 三个实验的编码器参数**逐位相同**（SHA-256 校验），差异只来自选择层；稀疏性来源另有专门诊断 | `lstm_topk`、`lstm_softmax`、`lstm_sparsemax`；`04_selection_diagnostics.py` → `report/selection_diagnostics.csv` |
| 第 4 步：端到端联合训练（实际组合收益、CVaR、换手率、交易成本损失） | 优化层在训练回路内，5 项损失联合训练 | `full` |

### 11.5 方案要求的 9 个对比实验

| 方案要求 | 实验名 | 是否跑完 |
| --- | --- | --- |
| Equal Weight | `ew` | ✅ 3 折 |
| Historical Mean-CVaR | `meancvar_hist` | ✅ 3 折 |
| LSTM + Top-k + Mean-CVaR | `lstm_topk` | ✅ 3 折 |
| LSTM + Sparsemax + Mean-CVaR | `lstm_sparsemax` | ✅ 3 折 |
| End-to-End CVaR（无选择层） | `e2e_noselect` | ✅ 3 折 |
| 完整模型 | `full` | ✅ 3 折 |
| 完整模型 − CVaR 损失 | `full_nocvar_loss` | ✅ 3 折 |
| 完整模型 − 交易成本 | `full_nocost` | ✅ 3 折 |
| 完整模型 − 稀疏选择 | `full_nosparse` | ✅ 3 折 |
| （额外）LSTM + Softmax + Mean-CVaR | `lstm_softmax` | ✅ 3 折 |
| （额外）二次分散项替代熵正则 | `full_quad` | ✅ 3 折 |
| （额外）方案给定的固定分数公式 | `full_musigma`、`lstm_musigma` | ✅ 补充网格 3 折 |
| （额外）5 种编码器 × 7 档目标收益 τ=0.016…0.028（复刻图 Fig2–Fig5） | `mu016_lstm` … `mu028_rbfn`（35 个实验） | ✅ 105 个作业，见 §9.9 |
| （额外）熵正则 λ ∈ {0, 0.001, 0.002, 0.005, 0.010} × LSTM/GRU/RNN（复刻图 Fig8–Fig9） | `ent000_*` … `ent100_*`（15 个实验） | ✅ 45 个作业，见 §9.10 |
| （额外）AdaBoost / XGBoost + 同一个 Mean-CVaR 层（复刻图 Fig13–Fig14） | `adaboost_mcvar`、`xgboost_mcvar` 及 τ=0.022 版（4 个实验） | ✅ 12 个作业，见 §9.11 |
| （额外）换股票池：中证 500 / 上证 50（复刻图 Fig11–Fig12） | `ew`/`meancvar_hist`/`mu020_lstm`/`full` × 2 池 | ✅ 24 个作业，见 §9.12 |
| （额外）两个池里的 5 种编码器对照 τ=0.020（复刻图 Fig11–Fig12 的曲线） | `mu020_rnn` … `mu020_rbfn` × 2 池（10 个实验） | ✅ 30 个作业，见 §9.12.1 |

### 11.6 方案要求的 10 类指标

| 指标族 | `metrics.json` / `pooled.csv` 中的键 |
| --- | --- |
| 年化收益 | `ann_return`（净）、`ann_return_gross`（毛）、`total_return` |
| Sharpe / Sortino | `sharpe`、`sortino`（+`_gross`，以及 `bench_sharpe`） |
| VaR / CVaR | `var_95`、`cvar_95`（+`_gross`） |
| 最大回撤 | `max_drawdown`（另有 `max_drawdown_gross`、`bench_max_drawdown`） |
| Calmar | `calmar` |
| 换手率 | `turnover_mean`、`turnover_ann`、`rebalances_per_year`、`cost_total`（成交名义额在 `rebalance.csv`） |
| 扣费后收益 | 主结果一律为净收益，成本影响另有 `cost_drag_ann = ann_return_gross − ann_return` |
| 持仓数与集中度 | `holdings_mean`（≥1e-3 的持仓数）、`eff_holdings_mean`（1/Σw²，有效持仓）、`hhi_mean`（Σw²）、`top5_weight_mean`（前 5 大权重之和）、`max_weight_mean`（最大单一权重）；逐期值另见 `rebalance.csv` 的 `n_holdings` / `eff_holdings` / `hhi` / `top5_weight` / `max_weight`，可用 `weights.csv` 直接复核 |
| 不同市场状态下的表现 | `mkt_ann_return_mkt_up` / `mkt_hit_mkt_up` / `mkt_n_mkt_up` 与对应的 `_down` 三件套（按指数当期涨跌划分） |
| 基准对比 | `alpha`、`beta`、`r2`、`info_ratio`、`excess_ann_return`、`bench_ann_return` |

### 11.7 五处有意偏离（已尽量用代码补齐，如实记录）

**（A）选择分数的来源。** 方案写的是一个固定公式 `s_i = μ̂_i − λ_σ σ̂_i − λ_c CVaR_i`，
主网格默认用的是可学习的分数头（`score_mode: "head"`）。现在两条路都在代码里：
`score_mode: "mu_sigma"` 就是方案的公式（`score_lambda_sigma` / `score_lambda_cvar` 可调），
并配了 `full_musigma` / `lstm_musigma` 两个实验与补充网格，样本外实测见 §9.5：
公式版的名义 Sharpe 更高（0.155 vs 0.065），但它的画像与历史 Mean-CVaR 几乎重合
（β 1.19 vs 1.11、波动 28.3% vs 32.1%、收益集中在 f2 单折），故默认仍保留 `head`；
要做严格复刻只需把配置改成 `mu_sigma`。

**（B）单票权重上限的规则。** 方案写的是 `y_i ≤ y_max·π_i`。这条规则在
`Σ_i y_max·π_i < 1` 时**与 `Σy = 1` 直接冲突**（例如 π 的支撑集只有 5 只票、`y_max = 0.1`，
则上界之和只有 0.5），凸规划无可行解。代码改用
`cap_i = y_eff·min(1, k_eff·π_i)`（`k_eff = 1/Σπ²`，`y_eff = max(y_max, 1/n_valid)`），
并把不足 1 的部分**按分数从高到低依次补到下一名**（每名最多吃满自身剩余额度
`y_eff - cap_i`），直到预算填满（见 §3「单票权重上限」、`selection_caps` 的 docstring
与 `tests/test_selection.py` 的两个用例）。

这里有一个**实测踩到的坑**，如实记录：填充逻辑最初写成「把缺口按各票剩余额度等比摊到
整个横截面」，结果在 `π` 极稀疏时把账面稠密化了——19 通道的对照运行
`results/_invalidated_extra_returns_prefixfillbug`（已作废，保留作证据）中，f2 折 24 个调仓期
**全部触发**填充（`cap_budget` 均值 0.589），`pi_support` 只有 9.5 只而 `book_n` 达到 119.0 只
（即上百个约 0.3% 的仓位），f3 折同样有 90.9% 的期次触发、账面 101.4 只，与「稀疏神经选股」直接矛盾。
改成「按分数放宽选择集」后，同一批期次的账面只覆盖分数最高的那一段，`book_n` 与 `pi_support`
同量级（见 §9.7 修复前后的对照）。硬约束 `0 ≤ y ≤ y_max` 在两种写法下都成立。

补平分支只有在 `cap_budget = Σ_i cap_i < 1` 时才进入。诊断把填充前的上限之和
（`cap_budget`）、缺口（`deficit`）与触发率（`fill_periods`）分别落盘后，把四个网格
（主网格 9 个神经网络实验 + 补充网格 + 19 通道 + 完整池）全部诊断完，实测结果是：

* **主网格（`csi300_20260910_235309`）有两处触发**：`full_nocost` f2 折 **13/24 期**
  （`cap_budget` 最低 0.589、最大缺口 0.411）、`full_quad` f2 折 **2/24 期**
  （最低 0.886、最大缺口 0.114）；其余折次 `cap_budget ≥ 1`，缺口恒为 0。
* **补充网格一处：`full_musigma` f3 折 4/55 期**（2022-04-01、2023-01-03、2023-06-01、2023-09-01），
  缺口分别为 6.7%、7.3%、6.7%、0.4%——**是真实缺口，不是浮点噪声**。这 4 期 `pi_support`
  只有 17–19 只，`π` 高度集中使得 `Σ min(1, k_eff·π_i)` 小于 1。
* **19 通道 `full` 的 f2/f3 折**（100% / 90.9% 的期次）与**完整池 `full` 的 f2 折**（12/24 期）
  也触发，这两处见 §9.7 与 §9.6。

（早先版本的本节文字写「主网格从不触发、只有 4 期浮点噪声」，那有两处错：
一是当时只诊断了 4 个实验，`full_nocost`/`full_quad` 还没进诊断表；
二是**读错了列**——`selection_diag.csv` 里记录的 `cap_sum` 是填充**之后**的和，
一旦填充跑过就恰好等于 1，所以它低于 1 只反映浮点噪声，无法用来判断触发与否。
新增的 `cap_budget`/`deficit` 两列才是触发判据，这也正是这次修正的副产品。）

触发与否对结果的实际影响由两组对照运行实测：把受影响作业用修复后的填充规则原样重跑
（[configs/csi300_fill_check.yaml](./configs/csi300_fill_check.yaml)、
[configs/csi300_fill_check_ablations.yaml](./configs/csi300_fill_check_ablations.yaml)、
[configs/csi300_fill_check_full_universe.yaml](./configs/csi300_fill_check_full_universe.yaml)），
并用同一份已训练权重在两种填充规则下各评一次以隔离规则本身的影响——见 §9.8。

普查口径（旧规则、已做过重放诊断的 45 个「实验×折」作业）：6 个作业触发过填充、
合计 105/1545 期；逐作业明细见 §9.8 的第一张表。
`runner` 把逐期 `cap_budget`/`deficit`/`fill_periods` 记进 `selection_diag.csv`，
可用一行复核触发次数：

```powershell
.venv\Scripts\python.exe -c "import glob,pandas as pd; d=pd.concat([pd.read_csv(f) for f in glob.glob('results/*/*/*/selection_diag.csv')]); print(len(d), int((d.deficit>0).sum()), float(d.cap_budget.min()))"
```

**（C）数据里没有行业信息，换手率只有代理变量。** 给定的 `findata` 面板只有 17 个价量特征
（`features.json`），既没有行业分类字段，也没有独立的换手率字段。
因此「行业 / 风格暴露约束」这一项无法实现，方案中提到的换手相关分析只能用
`volume_ratio`（成交量相对均值）与 `amihud`（非流动性）间接体现，
而真实的换手率由回测引擎的 `weights.csv` 直接算出
（那是交易层面的换手，不是特征层面的）。

此外还有三处「按方案下限执行」的选择：调试用的流动性截断取 120 名（方案建议 50–100 名，
留出更多截面以便 `y_max = 0.10` 可行），并且按方案 §5 的第二阶段要求，
另用 [configs/csi300_full_universe.yaml](./configs/csi300_full_universe.yaml) 在**不截断流动性**
（`top_liquidity: 0`，完整沪深 300）的池子上重跑了方案 §6 的核心对比（见 §9.6）；
`alpha = 0.95`、`gamma = 0.1` 等超参取方案默认值，全部写在 `configs/csi300.yaml` 里，可改。

**（D）输入特征通道与方案清单的差异。** 方案 §2 模块一给的输入清单是
「收益率与对数收益率；成交量、成交额、换手率；波动率；动量和反转指标；市场指数收益；行业信息」，
并明确用「其中**可以**包括」引入，属于开放建议。本仓库的默认 17 通道对这份清单的覆盖情况：

| 方案分组 | 对应通道 | 说明 |
| --- | --- | --- |
| 收益率与对数收益率 | `mom_5` / `mom_20` / `mom_60` | 动量通道本身就是收盘价收益率；**另有可选的 1 日收益率与对数收益率通道** |
| 成交量 / 成交额 / 换手率 | `log_volume`、`volume_ratio`、`amihud`、`vwap_gap`、`open_close`、`range` | 成交额以 `amihud`（额/波动）与 `vwap_gap` 体现；真实换手率由回测层从 `weights.csv` 计算 |
| 波动率 | `vol_20`、`vol_60`、`downside_vol_20`、`ret_skew_60` | |
| 动量与反转指标 | `mom_5`、`mom_20`、`mom_60`、`resid_mom_20`、`open_close` | `resid_mom_20` = 剔除市场后的残差动量（反转更敏感） |
| 市场指数收益 | `mkt_ret_1`、`mkt_ret_20`、`beta_60` | 指数日频收益来自 `index_daily.csv` |
| 行业信息 | `beta_60`、`resid_mom_20`（代理） | 数据里没有行业字段，只能用「相对市场」的暴露度代理，见（C） |

1 日简单收益率 `ret_1` 与 1 日对数收益率 `log_ret_1` 是**可选的**两个通道
（`features.include_daily_return: true`，17 → 19 通道），默认关闭：主网格所有结果表都基于
17 通道，把这个开关打开等于换一份输入，不应与已冻结的结果混着比较。
两个通道的构造有单测覆盖（`tests/test_data.py::test_daily_return_channels_are_optional_and_exactly_the_z_scored_return`
校验它与手算的 `close[t]/close[t−1]−1` 的截面 z-score 逐位一致、`log_ret_1` 与 `ret_1` 相关系数 > 0.999），
打开后的敏感性对照见 §9.7。

**（E）复刻 `D:\figures` 那 11 张参考图时的替代与口径差异。** 参考图是本仓库之外的材料
（文件名前缀 `LSTM_MCVaR_`），我们只照抄了它的**画法与画布规格**，图里的数值全部来自自己跑的 run。
四处必须说明的差异：① 样本区间：参考图横轴 2016–2023，我们的 walk-forward 折是
2018–2019 / 2020–2021 / 2022–2026，所以横轴是 2018–2026，**不能与参考图逐点比对**；
② Fig11 的市场：参考图用巴西 IBrX50，本地数据没有该指数，改用**中证 500**（同样起"换一个股票池"
的作用），并额外补了上证 50（Fig12）；③ Fig8/Fig9：参考图只有 RNN/LSTM/GRU 三条线，
所以熵扫描也只跑这三种结构，MLP/RBFN 的熵配置（`configs/csi300_entropy_sweep_ext.yaml`）
写好但未跑，需要时可直接补；④ 目标收益的 7 档 τ 扫描是为了对齐参考图 Fig2–Fig5 才补的，
主网格里的 μ 仍是软惩罚，两者不要混着比。逐张对照表见 §9.13。
