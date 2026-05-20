# Du62 `ELP_DRL_Standard4.5` 消融结果记录

## 1. 实验范围
- 实例：`Du62`
- 训练预算：`G=600`，`T_MAX=240`，总步数 `144000`
- 固定种子：`20260328, 20260329, 20260330, 20260331, 20260332`
- 评估指标：
  - `平均适应度值`：主指标，越小越好
  - `平均运行时间（小时）`：越小越好
  - `适应度标准差`：衡量跨种子稳定性，越小越稳

## 2. 数据口径
- 数据来源：`files/expresults/Du62-ablation_standard45_*.csv`
- 统计口径：每组固定 `5` 个种子
- 去重规则：`A1_no_topk_guided` 因中断重跑产生 `1` 条重复记录，对重复 seed 仅保留最后一次结果

## 3. 结果表

| 组别 | 平均适应度值 | 相对基线成本变化 | 平均运行时间（小时） | 相对基线时间变化 | 适应度标准差 |
|---|---:|---:|---:|---:|---:|
| `B0_baseline` | 3685995.512 | +0.000% | 8.181 | +0.000% | 19190.349 |
| `A1_no_topk_guided` | 3738383.895 | +1.421% | 2.616 | -68.021% | 23079.391 |
| `A2_no_high_flow_warmstart` | 3691955.476 | +0.162% | 4.622 | -43.507% | 19142.667 |
| `A3_no_reheat` | 3685995.512 | +0.000% | 5.440 | -33.503% | 19190.349 |
| `A4_no_mid_structural_shot` | 3683441.463 | -0.069% | 6.377 | -22.052% | 15097.643 |
| `A5_no_elite_archive` | 3707580.436 | +0.586% | 7.285 | -10.951% | 11097.787 |
| `A6_no_archive_switch` | 3697866.799 | +0.322% | 6.250 | -23.605% | 19920.352 |
| `A7_no_final_elite_push` | 3682424.288 | -0.097% | 5.728 | -29.984% | 13542.879 |
| `A8_no_two_stage_heavy_actions` | 3667738.095 | -0.495% | 3.126 | -61.790% | 9521.194 |
| `A9_no_local_search_on_feasible_accept` | 3811994.003 | +3.418% | 1.959 | -76.060% | 19488.161 |
| `A10_no_segment_insert_light` | 3685995.512 | +0.000% | 5.817 | -28.895% | 19190.349 |
| `A11_random_policy_no_dqn` | 3681804.209 | -0.114% | 4.677 | -42.830% | 9130.846 |

## 4. 简短结论
- 最优综合组为 `A8_no_two_stage_heavy_actions`：平均适应度值相对基线下降 `0.495%`，平均运行时间下降 `61.790%`，同时稳定性也显著更好。这说明当前 `Two-stage heavy actions` 在 `Du62` 上没有兑现预期收益，反而拉高了时间成本并削弱了结果稳定性。
- `A1_no_topk_guided` 与 `A9_no_local_search_on_feasible_accept` 明显变差，分别劣化 `1.421%` 和 `3.418%`。这说明 `Top-k guided` 与“可行解接受后的局部强化”是当前算法的核心有效组件，不宜删除。
- `A3_no_reheat` 与 `A10_no_segment_insert_light` 在最终成本上与基线完全一致，但运行时间更短，说明这两个机制在当前配置下几乎没有形成有效增益，更可能只是增加了运行开销或被其他路径覆盖。
- `A11_random_policy_no_dqn` 略优于基线（`-0.114%`），且更稳、更快。结合 `A8` 的结果，当前主收益更可能来自动作本身及其后处理搜索，而不是 `DQN` 选动作或 `Two-stage heavy actions` 的复杂内层筛选。

## 5. 当前建议
- 后续版本应优先以 `A8` 所代表的“去除 `Two-stage heavy actions`”路径为新基线，保留 `Top-k guided` 与 `local search on feasible accept`。
- 若仍希望保留结构动作 `9/10/11/14/15`，建议先做轻量化重写，而不是继续堆叠当前的 `Two-stage heavy actions` 内层启发式。

## 6. `Standard4.6 light core` 验证结果

### 6.1 实验口径
- 代码入口：`src/algorithms/ELP_DRL_Standard4.6.py`
- 执行脚本：`scripts/run_standard46_light.ps1`
- 实例与训练预算：同消融实验，`Du62`、`G=600`、`T_MAX=240`、总步数 `144000`
- 固定种子：`20260328, 20260329, 20260330, 20260331, 20260332`
- 默认关闭机制：`Two-stage heavy actions`、`reheat`、`segment_insert_light`、`final_elite_push`、`mid_structural_shot`

### 6.2 关键结果对比

| 组别 | 平均适应度值 | 相对基线成本变化 | 相对 `A8` 成本变化 | 平均运行时间（小时） | 相对基线时间变化 | 相对 `A8` 时间变化 | 适应度标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0_baseline` | 3685995.512 | +0.000% | +0.498% | 8.181 | +0.000% | +161.710% | 19190.349 |
| `A8_no_two_stage_heavy_actions` | 3667738.095 | -0.495% | +0.000% | 3.126 | -61.790% | +0.000% | 9521.194 |
| `A11_random_policy_no_dqn` | 3681804.209 | -0.114% | +0.384% | 4.677 | -42.830% | +49.619% | 9130.846 |
| `Standard4.6_light_dqn` | 3681109.754 | -0.133% | +0.365% | 3.590 | -56.119% | +14.840% | 13214.038 |
| `Standard4.6_light_random` | 3682238.589 | -0.102% | +0.395% | 4.081 | -50.120% | +30.541% | 8715.724 |

### 6.3 结论
- `Standard4.6_light_dqn` 相对原始基线成本下降 `0.133%`，运行时间下降 `56.119%`，说明轻量化方向成立，但没有超过 `A8_no_two_stage_heavy_actions`。
- `A8_no_two_stage_heavy_actions` 仍是当前最优综合结果：平均成本最低、时间最低、稳定性也较好。说明在 `Du62` 单一算例上，仅移除 `Two-stage heavy actions` 比同时关闭多个弱贡献机制更优。
- `Standard4.6_light_random` 与 `Standard4.6_light_dqn` 成本非常接近，进一步支持前面的判断：当前收益主要来自动作与后处理机制本身，而不是 `DQN` 策略学习。
- 本轮 `Standard4.6` 结果 CSV 的 `备注` 字段继承了旧的 `A11_random_policy_no_dqn` 环境变量；但 `算法` 字段、日志目录与实际开关均指向 `Standard4.6`，不影响数值归属。`scripts/run_standard46_light.ps1` 已改为显式覆盖 `ELP_EXP_REMARK`，后续新跑结果不会再污染备注。

### 6.4 后续代码建议
- 不建议直接把 `Standard4.6 light core` 作为最终替代版本；它是一个验证性轻量版本，而不是当前最优配置。
- 更合理的新主线应以 `A8_no_two_stage_heavy_actions` 为核心：保留 `reheat`、`segment_insert_light`、`final_elite_push`、`mid_structural_shot` 的默认行为，仅彻底移除或默认关闭 `Two-stage heavy actions`。
- 已将 `src/algorithms/ELP_DRL_Standard4.6.py` 默认入口落实为 `standard46_a8` profile：默认只关闭 `Two-stage heavy actions`，其余四个外层机制保持开启。旧 `standard46_light` 仍通过 `ELP_STANDARD46_PROFILE=light` 和 `scripts/run_standard46_light.ps1` 保留，用作过度裁剪对照。
