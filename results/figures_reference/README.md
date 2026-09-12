# 复刻图（用我们自己的实验结果画）

对照图在 `D:\figures`。每张图的画布尺寸、字号、图例位置、配色都照抄对照图，
**图里的每个数字都来自本仓库 `results/` 下自己跑的 run**，没有引用对照图的任何数值。

| 本图 | 对照图 | 画的是什么 | 数据来源（run 目录） |
|------|--------|-----------|---------------------|
| fig02_std_vs_target.png | Fig2 | 7 档目标收益 μ × 5 种网络结构的**年化波动** | `results/csi300_mu_sweep*` |
| fig03_sharpe_vs_target.png | Fig3 | 同上，纵轴为夏普比率 | `results/csi300_mu_sweep*` |
| fig04_tracking_error_vs_target.png | Fig4 | 同上，纵轴为相对基准的跟踪误差 | `results/csi300_mu_sweep*` |
| fig05_cumulative_return_mu020.png | Fig5 | μ=0.020 时 5 种结构的累计净值 | `results/csi300_mu_sweep*` |
| fig08_entropy_sharpe_ratio.png | Fig8 | 5 档熵正则 × **3 种循环结构**（RNN/LSTM/GRU）的夏普，横轴为**实际实现的组合熵** | `results/csi300_entropy_sweep*` |
| fig09_entropy_cumulative_return.png | Fig9 | 同上 3 种结构在熵正则 0 vs 默认档的累计净值（上下两块面板） | `results/csi300_entropy_sweep*` |
| fig10_out_of_sample_return.png | Fig10 | **同一 11 个月窗口**内 5 种结构的月度已实现收益（左右 = 目标收益最低 / 最高档） | `results/csi300_mu_sweep*` |
| fig11_csi500_cumulative_return.png | Fig11 | 换到 CSI 500 股票池、5 种结构的累计净值 | `results/csi500_arch_compare_*`（本仓库交付的图即由此生成；`results/csi500_reference_*` 只在上级目录缺失时用于回退） |
| fig12_sse50_cumulative_return.png | Fig12 | 换到上证 50、5 种结构的累计净值 | `results/sse50_arch_compare_*`（本仓库交付的图即由此生成；`results/sse50_reference_*` 只在上级目录缺失时用于回退） |
| fig13_adaboost_xgboost_mu020.png | Fig13 | μ=0.020 下 AdaBoost / XGBoost 与 LSTM 的对比（3 条线） | `results/csi300_mu_sweep*` + `results/csi300_tree_baselines*` |
| fig14_adaboost_xgboost_mu022.png | Fig14 | 同上，μ=0.022 | `results/csi300_mu_sweep*` + `results/csi300_tree_baselines*` |

## 配色 / 版式（逐像素测量对照图后照抄）

* 颜色就是 matplotlib 的默认色：蓝 `#0000ff` = RNN，红 `#ff0000` = LSTM，绿 `#008000` = GRU，
  橙 `#ffa500` = MLP，紫 `#800080` = RBFN；Fig5 的第一条线（RNN）在对照图里是**黑色**，
  Fig13/14 的图例顺序是 ADA(蓝) / LSTM(红) / XGB(绿)。
* 所有面板都不画内部网格线、图例都不带边框（`frameon=False`）。
* 柱状图（Fig2/3/4/8）图例在坐标区**内部顶行**、每组 5 根柱子从左到右即上面的配色顺序；
  折线图（Fig5/11/12/13/14）图例在左上角。
* 画布尺寸按对照图逐张量取（如 Fig2 = 13.24×6.03 in，Fig9 = 13.35×12.44 in）。
* 字号照抄对照图的实测值：刻度 17 pt、轴标题 20 pt、**折线图图例 18 pt、柱状图图例 17 pt**，
  图例色块长 = 2.0 个字号（对照图图例的行距 ≈ 1.5 × 字号，Fig5 实测 37.2 px = 18 pt，
  与我们的 37.1 px 一致）。
* 柱状图：柱簇总宽 = min(0.78, 0.18×系列数) 个横轴单位、柱子紧邻（间隙 0），
  x 轴范围 = (-0.5, 组数−0.5)，y 轴顶部 = max + 7% 极差、底部 = 0（全为正时柱子贴底）；
  图例 `loc='upper center'` 且贴住轴顶（`borderaxespad=0`）、列间距 `columnspacing=1.0`，
  且**不参与 `tight_layout`**（`set_in_layout(False)`）—— 否则居中的 5 列宽图例会把绘图区
  压窄 ≈160 px（对照图实测绘图区宽 1192 px，我们设了 `set_in_layout(False)` 后为 1193 px）。
* 折线图：y 轴沿用 matplotlib 默认的 5% 边距；**纵轴是 0 起算的累计收益**
  （Π(1+r)−1），不是从 1 起算的净值 —— 与对照图一致（对照图的刻度里有 0.0）。
* 横轴刻度沿用 matplotlib 的默认浮点格式（`0.016` … `0.028`；Fig8 是 `2.55` … `2.75`）；
  Fig10 的日期刻度为 `%b-%y` 且每 2 个月一格（Aug-23、Oct-23 …），与对照图一致。
* 四周留白由 `tight_layout(pad=0.7)` 控制（对照图实测约 16–18 px）；左右留白会随
  坐标轴两端的标签浮动 —— 折线图的横轴末端若被最后一个刻度标签压住，标签悬出轴外，
  留白就会变宽（详见下面第 11 条）。

## 与对照图的差别（诚实说明）

1. **样本区间**：对照图横轴是 2016–2023；我们的 walk-forward 测试折是
   2018–2019 / 2020–2021 / 2022–2026，三折首尾相接，所以横轴是 2018–2026。
2. **Fig11 的市场**：对照图用巴西 IBrX50，本地数据里没有该指数，改用
   **CSI 500**（同样的「换一个股票池」作用），并在图上与 README 中标明。
3. **Fig8 的横轴**：对照图直接以实现的组合熵为横轴，我们也一样——横轴刻度是
   每一档 λ 在 3 种结构（对照图只有 RNN/LSTM/GRU 三条线）上实现的平均熵（nats），
   而不是 λ 本身。我们另跑了 MLP/RBFN 的熵扫描（`csi300_entropy_sweep_ext`），
   但对照图只有 3 条线，所以这两条不进 Fig8/Fig9。
4. **Fig10 的两个面板**：对照图里左右两块的差别（窗口？标的？参数？）无法从图上看出来，
   我们改成**同一 11 个月窗口**、目标收益最低档（μ=0.016）与最高档（μ=0.028）的对比，
   并在 `panels.json` 的 `assumption` 字段里写明。
5. **Fig9 的小标题**：对照图每块面板下方有一行小标题，格式是
   `Cumulative return graphs for E = 2.55`（E 为该档实际实现的组合熵），我们照这个格式写，
   但把 E 换成我们这一档的实测平均熵；对照图第二块的小标题被画布裁掉了，我们两块都画。
6. **指标定义**：波动 / 夏普 / 跟踪误差都在「三折首尾相接的日度净收益」上重算，
   与 `report/pooled.csv` 的口径一致（`e2e_portfolio.metrics.compute_metrics`，
   年化因子 252；跟踪误差 = std(组合-基准)×√252）。
7. **纵轴口径**：对照图的折线图纵轴是 0 起算的累计收益，我们原先画的是从 1 起算的净值，
   现已改成同样的 0 起算口径（形状不变，刻度整体下移 1.0）。
8. **Fig8 的柱子**：对照图里所有夏普都为正、柱子贴住下边界；我们的夏普在三折上跨零
   （−0.32 … +0.30，主要来自 2020–2021 折的普遍为负，这是真实的样本外结果，
   见 §9.12.1），所以下边界落在最小值上、柱高有正有负。其他几何量（柱簇宽度、x 轴范围、
   顶部 7% 留白、图例 18 pt / 色块 2.0 字号）已按对照图对齐。
9. **对照图自身的两处不一致**（我们没有照抄）：a) Fig10 的字号明显小于其余图
   （实测约 11 pt 对比 17 pt），我们保持全套统一；b) Fig2/3/4 的柱簇整体右移了一个柱宽
   （柱簇中心落在 1.5、2.5 … 而不是 1、2 …），我们按 matplotlib 的正常做法让柱簇居中。
10. **数据量级**：对照图的累计收益到 +300% 量级、测试区间 2016–2023，我们的样本外区间
    是 2018–2026，曲线形状与终点因此不可比；本仓库只保证「同样的图、同样的口径、
    同样的版式」，数字本身来自我们自己的三折 walk-forward 结果。
11. **折线图的右侧留白（44 px vs 对照图 17 px）—— 已查明是数据端点造成的，不是版式差异**：
    我们的样本外区间止于 2026-08-20，matplotlib 默认的 5% 横轴边距把视窗推到 2027-01-01，
    于是最后一格「2027」年刻度正好落在坐标区右缘、标签有一半悬在轴外，`tight_layout`
    必须为它留宽（16.5 px pad + ≈27 px 悬出 = 44 px，与实测逐像素闭合）；对照图数据止于
    2023 年中，末刻度「2023」深处轴内，所以只需 17 px。两边除留白外的刻度要素实测一致：
    刻度定位（年度 `AutoDateLocator`）、刻度字号（末刻度墨迹宽 58 vs 60 px）、
    图例版面（左上角、单列 5 行、行距 37.1 vs 37.2 px）、无网格线。
12. **左侧留白相差 15 px（对照图坐标区左缘 x≈119，我们 x≈134）**：两边的纵轴标题都是
    `"Cumulative return"`（OCR 复核一致）、左缘都在画布 x≈15，差值来自纵轴刻度标签块的
    宽度（对照 103 px vs 我们 117 px）叠加 `tight_layout` 的分配，属数据驱动的派生量，
    同样不是版式差异。

## 重新生成

```powershell
# 1) 目标收益扫描（5 种结构 × 7 档 μ）
python scripts/02_run_experiments.py --config configs/csi300_mu_sweep.yaml --workers 4 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/csi300_mu_sweep_ext.yaml --workers 4 --threads-per-worker 2
# 2) 熵扫描（3 种结构 × 5 档 λ，口径同对照图 Fig8/Fig9 的三条线）
python scripts/02_run_experiments.py --config configs/csi300_entropy_sweep.yaml --workers 4 --threads-per-worker 2
#    可选：把 MLP / RBFN 也补进熵扫描（本轮未跑，故图里没有它们）
python scripts/02_run_experiments.py --config configs/csi300_entropy_sweep_ext.yaml --workers 4 --threads-per-worker 2
# 3) 树模型基线（AdaBoost / XGBoost，τ=0.020 与 0.022）
python scripts/02_run_experiments.py --config configs/csi300_tree_baselines.yaml --workers 2 --threads-per-worker 2
# 4) 另外两个股票池（5 种结构，Fig11/Fig12 的数据源）
python scripts/02_run_experiments.py --config configs/csi500_arch_compare.yaml --workers 4 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/sse50_arch_compare.yaml --workers 4 --threads-per-worker 2
#    （只有 LSTM 的旧 run 也能画，缺的结构会记进 panels.json 的 unsupported_architectures）
python scripts/02_run_experiments.py --config configs/csi500_reference.yaml --workers 2 --threads-per-worker 2
python scripts/02_run_experiments.py --config configs/sse50_reference.yaml --workers 2 --threads-per-worker 2
# 5) 画图（只写进一个独立文件夹）
python scripts/09_reference_figures.py --out results/figures_reference
```

`panels.json` 记录了每张图对应的实验名、指标定义、数据来源目录，便于逐条核对。
