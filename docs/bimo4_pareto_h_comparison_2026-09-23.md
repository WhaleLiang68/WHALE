# BiMO4 Pareto-H 接受准则对照记录

## 基线版本

- 基线模块：`src.algorithms.ELP_DRL_BiMO4`
- 基线含义：2026-09-23 开始修改接受公式前的当前 BiMO4 主版本。
- 相关局部搜索：`src/algorithms/BiMO4PaperLocalSearch.py`
- 基线提交：以本文件首次提交所在的 Git commit 为准。
- 目标：最小化 `MHC`，最大化 `CR`；最小化向量为 `[MHC, -CR]`。

基线对不可比较的 Pareto 候选继续使用标量 `fitness` 的历史增强能量：

```text
E_tilde(x) = fitness(x) + k_hist * H(fitness(x))
P_accept = min(1, exp((E_tilde(current) - E_tilde(candidate)) / T))
```

其中 `H` 是一维标量能量区间的访问次数。候选支配当前解时直接接受，候选被当前解支配时直接拒绝。

## 新实验版本

- 模块：`src.algorithms.ELP_DRL_BiMO4_ParetoH`
- 版本标签：`pareto_max_deterioration_grid_history_v1`
- 原 BiMO4 模块不修改，便于相同 seed 直接对照。

对两个可行且互不支配的解，按固定目标参考框架归一化后计算最大正向退化：

```text
D_obj = max_i(max(0, z_candidate_i - z_current_i))
h(x) = ln(1 + N(grid(x)))
P_accept = min(1, exp(-D_obj / tau + beta * (h(current) - h(candidate))))
```

二维网格直接建立在 `[MHC, -CR]` 的归一化目标空间中；网格坐标不裁剪，因此前沿扩展到初始参考框架之外时不会挤入边界格。候选支配当前解时仍直接接受，被支配时仍直接拒绝。Pareto 档案更新继续与“是否成为当前解”解耦。

默认参数：

```text
ELP_BIMO_PARETO_H_BETA=0.35
ELP_BIMO_PARETO_H_TAU_START=0.20
ELP_BIMO_PARETO_H_TAU_END=0.02
ELP_BIMO_PARETO_H_TAU_POWER=1.25
ELP_BIMO_PARETO_H_GRID_BINS=20
```

## 公平对照

原版：

```powershell
$env:ELP_EXP_ALGORITHM="ELP_DRL_BiMO4_BASELINE"; python -m src.algorithms.ELP_DRL_BiMO4
```

新公式版：

```powershell
$env:ELP_EXP_ALGORITHM="ELP_DRL_BiMO4_PARETO_H"; python -m src.algorithms.ELP_DRL_BiMO4_ParetoH
```

两组实验必须使用相同的实例、固定 seeds、`G`、`T_MAX`、初始温度、动作集、DQN 设置和局部搜索开关。重点比较合并非支配前沿的 `HV`、`IGD`、`Spacing`、档案规模、`MHC/CR` 极值及运行时间。
