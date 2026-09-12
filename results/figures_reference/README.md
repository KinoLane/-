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
5. **Fig9 的小标题**：对照图每块面板下方有一行 LaTeX 小标题（`(a) …`），我们用等价的
   文字放在同样的位置，正文内容换成本实验的说法。
6. **指标定义**：波动 / 夏普 / 跟踪误差都在「三折首尾相接的日度净收益」上重算，
   与 `report/pooled.csv` 的口径一致（`e2e_portfolio.metrics.compute_metrics`，
   年化因子 252；跟踪误差 = std(组合-基准)×√252）。

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
