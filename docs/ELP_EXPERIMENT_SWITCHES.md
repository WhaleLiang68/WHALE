# ELP 实验开关说明

这个文件只记录当前 `ELP_DRL_Standard.py`、`ELP_DRL_MO.py`、`ELP_DRL_MO2.py`、`ELP_DRL_MO4.py` 里**实际生效**且后续实验容易忘记的环境变量开关。

目标不是把所有变量机械抄一遍，而是回答下面四个问题：

1. 哪些变量是日常一定会用到的。
2. 哪些变量只在论文最终对比时该开。
3. 哪些变量是 `MO2` 特有的，什么时候需要重建参考系。
4. 哪些变量不要轻易碰，除非你明确在做消融或调参。

## 1. 先记住三条规则

### 1.1 随机性没有被删除，只是被控制了

当前代码会始终播种：

- `python random`
- `numpy`
- `torch`

所以现在的行为是：

- **同 seed** -> 同结果
- **不同 seed** -> 不同结果

这才是正确的实验状态。

### 1.2 `ELP_STRICT_DETERMINISM` 只控制“严格确定性”，不控制“是否随机”

- `ELP_STRICT_DETERMINISM=1`
  - 适合论文表格、最终对比、复现实验
  - 会打开 `torch` 的确定性算法约束
- `ELP_STRICT_DETERMINISM=0`
  - 适合日常训练、搜最好值
  - 仍然会播种，但不强制 `torch` 走严格确定性路径

### 1.3 `MO2` 改了参考系相关逻辑后，正式对比前最好重建一次参考系

尤其在下面几种情况里，建议先跑一次参考系校准：

- 换实例
- 改了 `MO2` 奖励结构
- 改了 `ELP_STRICT_DETERMINISM`
- 要做论文最终表格

---

## 2. 最常用的通用开关

这些变量适用于 `Standard / MO / MO2` 的主实验入口。

| 变量名 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ELP_EXP_INSTANCE` | `Du62` | 实例名，例如 `AB20-ar3` | 必设 |
| `ELP_EXP_ALGORITHM` | 脚本内默认值 | 结果文件名中的算法标签 | 建议显式设，避免覆盖 |
| `ELP_EXP_REMARK` | 脚本内默认值 | 结果备注 | 建议设 |
| `ELP_IS_EXP` | `1` | 是否按实验模式落盘 | 一般保持 `1` |
| `ELP_EXP_NUMBER` | `30` | 连续跑多少组 | 单次对比用 `1` |
| `ELP_G` | `1000` | 外层 episode 数 | 主要预算参数 |
| `ELP_T_MAX` | `300` | 每个 episode 的步数上限 | 主要预算参数 |
| `ELP_T_INITIAL` | `10000` | 初始温度 | 一般不改 |
| `ELP_K_HIST` | `10` | 历史温度/能量相关参数 | 一般不改 |
| `ELP_BASE_SEED` | `20260427` 或脚本内默认 | 基础 seed | 必设 |
| `ELP_STRICT_DETERMINISM` | `1` | 是否开启严格确定性 | 最终对比用 `1`，日常训练可设 `0` |
| `ELP_WALL_TIME_LIMIT_SECONDS` | `0` | MO/MO4 墙钟时间预算，`0` 表示不限制 | 同时间预算实验时设置 |

### 说明

- `ELP_EXP_NUMBER>1` 时，脚本会按 `base_seed + run_index` 自动递增。
- `MO/MO2` 当前**没有**接入 `ELP_FIXED_SEEDS`；只有 `ELP_DRL_Standard.py` 主入口支持该变量。
- `ELP_WALL_TIME_LIMIT_SECONDS` 只在 `MO/MO4` 当前主线中使用；触发后会停止训练循环，但仍正常保存 archive、HV/IGD/Spacing、CSV 和 run summary。

---

## 3. 可复现与训练搜索的推荐配置

### 3.1 论文最终对比

```powershell
$env:ELP_STRICT_DETERMINISM="1"
$env:ELP_EXP_NUMBER="1"
$env:ELP_BASE_SEED="20260507"
```

用途：

- 同实例
- 同预算
- 同 seed
- 结果可复现

### 3.2 日常训练 / 搜最好值

```powershell
$env:ELP_STRICT_DETERMINISM="0"
$env:ELP_EXP_NUMBER="5"
$env:ELP_BASE_SEED="20260508"
```

用途：

- 保留 seed 管理
- 放开 `torch` 的严格确定性约束
- 连续多 seed 搜索更强结果

---

## 4. `MO baseline` 相关开关

这些变量主要由 `ELP_DRL_MO.py` 和 `ELP_DRL_MO4.py` 的 baseline 模式读取。

| 变量名 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ELP_MO_BASELINE_ALGO` | 空 | baseline 算法名，可选 `nsga2 / moead / spea2 / pso` | 只在跑传统/群智能 baseline 时设 |
| `ELP_MO_BASELINE_POP` | `64` | baseline 种群规模 | baseline 调参时再改 |
| `ELP_MO_BASELINE_GEN` | `80` | baseline 代数 | baseline 调参时再改 |
| `ELP_MO_BASELINE_SEQ_LEN` | `t_max` | action sequence 长度 | 一般保持默认 |
| `ELP_MO_PSO_INERTIA` | `0.72` | PSO 惯性权重，只在 `ELP_MO_BASELINE_ALGO=pso` 时使用 | 一般不改 |
| `ELP_MO_PSO_C1` | `1.49` | PSO 个体最优学习因子 | 一般不改 |
| `ELP_MO_PSO_C2` | `1.49` | PSO 全局 guide 学习因子 | 一般不改 |
| `ELP_MO_PSO_VMAX_RATIO` | `0.50` | PSO 速度上限占动作索引范围的比例 | 一般不改 |
| `ELP_MO_PSO_MUTATION_PROB` | `1 / ELP_MO_BASELINE_SEQ_LEN` | PSO 动作 token 随机重采样概率 | 消融时再改 |

### 说明

- 不设 `ELP_MO_BASELINE_ALGO` 时，`MO.py` 跑的是 `ELP_DRL_MO`，`MO4.py` 跑的是 `ELP_DRL_MO4`。
- 设了 `ELP_MO_BASELINE_ALGO` 后，脚本会切到 baseline 分支。
- `pso` 使用和 NSGA2/MOEA-D/SPEA2 相同的动作序列编码；粒子位置是动作 token 序列，不是直接连续坐标布局。

---

## 5. `MO` 旧版搜索行为开关

这些变量主要影响 `ELP_DRL_MO.py` 的旧版启发式节奏。除非你明确在做消融，否则不要轻易动。

### 5.1 非支配接受率相关

| 变量名 | 默认值 |
| --- | --- |
| `ELP_MO_ND_ACCEPT_CAP_EARLY_FLOOR` | `0.55` |
| `ELP_MO_ND_ACCEPT_CAP_MID_FLOOR` | `0.25` |
| `ELP_MO_ND_ACCEPT_CAP_LATE_MAX_START` | `0.35` |
| `ELP_MO_ND_ACCEPT_CAP_LATE_MAX_END` | `0.18` |

### 5.2 后期非入档候选门控

| 变量名 | 默认值 |
| --- | --- |
| `ELP_MO_LATE_NOARCHIVE_GATE_ENABLE` | `0` |
| `ELP_MO_LATE_NOARCHIVE_GATE_START` | `0.88` |
| `ELP_MO_LATE_NOARCHIVE_PROB_SCALE_END` | `0.70` |
| `ELP_MO_LATE_NOARCHIVE_PROB_CAP_START` | `0.50` |
| `ELP_MO_LATE_NOARCHIVE_PROB_CAP_END` | `0.28` |
| `ELP_MO_LATE_NOARCHIVE_STAGNATION_WINDOWS` | `4` |
| `ELP_MO_LATE_NOARCHIVE_STAGNATION_RAMP_WINDOWS` | `4` |
| `ELP_MO_LATE_NOARCHIVE_STAGNATION_MIN_PROGRESS` | `0.70` |

### 5.3 局部搜索退火/回退

| 变量名 | 默认值 |
| --- | --- |
| `ELP_MO_LOCAL_SEARCH_COOLDOWN` | `24` |
| `ELP_MO_LOCAL_SEARCH_COOLDOWN_MAX` | `24*16` |
| `ELP_MO_LOCAL_SEARCH_MIN_REL_IMPROVE` | `0.01` |
| `ELP_MO_LOCAL_SEARCH_DISABLE_AFTER` | `0.80` |
| `ELP_MO_LOCAL_SEARCH_BACKOFF_ENABLE` | `1` |
| `ELP_MO_LOCAL_SEARCH_BACKOFF_EXP_CAP` | `4` |

### 5.4 Archive 质量门控

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `ELP_MO_ARCHIVE_QUALITY_GATE` | 脚本内默认 | archive 满时是否做质量门控 | 不做专项实验时别动 |
| `ELP_MO_ARCHIVE_HV_TOL` | 脚本内默认 | HV 容忍阈值 | 不建议日常改 |
| `ELP_MO_ARCHIVE_SPACING_TOL` | 脚本内默认 | Spacing 容忍阈值 | 不建议日常改 |

### 5.5 日志/辅助

| 变量名 | 默认值 |
| --- | --- |
| `ELP_MO_TRACE_INTERVAL` | 脚本内默认 |

---

## 5.6 `MO/MO4` 指标输出口径

`MO` 与 `MO4` 当前最终打印、CSV、`run_summary.json` 中的 `archive_hypervolume` 使用固定公共参考前沿口径：

- 先用实例级公共 reference front 的 `ideal/nadir` 归一化 archive。
- HV 的固定参考点为归一化坐标 `[1.1, 1.1, 1.1, 1.1]`。
- Spacing 也用同一公共 reference front 的 `ideal/nadir` 归一化。
- IGD 本来就是基于公共 reference front。

| 变量名 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ELP_MO_ARCHIVE_REFERENCE_ENABLE` | `1` | 是否启用实例级公共参考前沿 | 正式对比保持 `1` |
| `ELP_MO_ARCHIVE_FIXED_HV_MARGIN` | `0.1` | 固定 HV 参考点 margin，参考点为 `1 + margin` | 正式对比不要改 |

CSV 中会额外写入：

- `archive_hypervolume_mode=fixed_reference_front`
- `archive_hypervolume_reference_point=[1.1, 1.1, 1.1, 1.1]`

### 5.7 `AR` 第四目标口径

当前 `AR` 已按论文图 3-2 的纵横比满意度函数实现，不再是“满足宽高比约束即 1”的可行性分数：

- 单设施纵横比 `ratio = max(width, height) / min(width, height)`。
- 下界 `1.0`，默认上界 `2.5`。
- 最优纵横比 `1.5`，满意度为 `1`。
- `ratio=1.0` 和 `ratio=2.5` 的满意度为 `0`。
- 区间内线性插值，超出上界也为 `0`。
- 布局级 `AR` 为所有设施满意度均值。

注意：这个口径会改变第四目标，因此旧 archive/reference front 不能和新实验混用。新 archive 会写入 `objectiveDefinitionVersion=mo_objectives_ar_paper_triangular_v1`，公共参考前沿只会聚合同版本 archive；正式对比前仍需要重跑参与对比的算法。

如果要重算历史结果，用：

```powershell
& 'C:\Users\17122\AppData\Local\conda\conda\envs\tensorflow\python.exe' .\src\utils\recompute_mo_archive_metrics.py .\files\expresults\<结果文件>.csv --in-place
```

---

## 6. `MO2` 参考系相关开关

这是当前最重要的一组。`MO2` 和 `MO` 最大的使用差异就在这里。

| 变量名 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ELP_MO2_CALIBRATE_REFERENCE` | `0` | 是否进入参考系校准模式 | 只在校准时设 `1` |
| `ELP_MO2_CALIBRATION_RUNS` | `8` | 校准运行次数 | 正式校准建议保留 |
| `ELP_MO2_CALIBRATION_G` | `20` | 每次校准的 episode 数 | 正式校准建议保留 |
| `ELP_MO2_CALIBRATION_TMAX` | `40` | 每次校准的步数 | 正式校准建议保留 |
| `ELP_MO2_REFERENCE_REUSE` | `1` | 是否复用已有参考系缓存 | 正式实验一般 `1` |
| `ELP_MO2_REFERENCE_REBUILD` | `0` | 是否强制重建参考系 | 除非明确重建，否则保持 `0` |
| `ELP_MO2_REFERENCE_MIN_ARCHIVE_SIZE` | `max(12, archive_limit/4)` | 参考系缓存的最小 archive 门槛 | 一般不改 |
| `ELP_MO2_REFERENCE_MARGIN_RATIO` | `0.25` | 参考框架 margin 比例 | 一般不改 |
| `ELP_MO2_REFERENCE_MIN_SPAN_RATIO` | `0.40` | 参考框架最小 span 比例 | 一般不改 |

### 推荐流程

#### 先校准参考系

```powershell
$env:ELP_STRICT_DETERMINISM="1"
$env:ELP_EXP_INSTANCE="AB20-ar3"
$env:ELP_BASE_SEED="20260507"
$env:ELP_MO2_CALIBRATE_REFERENCE="1"
$env:ELP_MO2_CALIBRATION_RUNS="8"
$env:ELP_MO2_CALIBRATION_G="20"
$env:ELP_MO2_CALIBRATION_TMAX="40"
Remove-Item Env:ELP_MO2_REFERENCE_REBUILD -ErrorAction SilentlyContinue
```

#### 再跑正式 `MO2`

```powershell
$env:ELP_MO2_CALIBRATE_REFERENCE="0"
$env:ELP_MO2_REFERENCE_REBUILD="0"
```

---

## 7. `MO2` 奖励/质量主开关

这部分是 `MO2` 的主线调参区。建议把它们分成三层理解。

### 7.1 第一层：主线最常用

| 变量名 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ELP_MO2_ARCHIVE_QUALITY_GUARD` | `1` | archive 质量底线开关 | 主线保持开启 |
| `ELP_MO2_SPACING_REWARD_WEIGHT` | `0.70` | 均匀性奖励权重 | 常用调参项 |
| `ELP_MO2_DENSITY_REWARD_WEIGHT` | `0.18` | 局部密度协调奖励 | 常用调参项 |
| `ELP_MO2_HV_REWARD_WEIGHT` | `0.12` | archive HV 奖励 | 常用调参项 |
| `ELP_MO2_MARGINAL_HV_REWARD_WEIGHT` | `0.36` | 边际 HV 贡献奖励 | 常用调参项 |
| `ELP_MO2_ARCHIVE_CHANGE_BONUS` | `0.24` | 有效入档 bonus | 常用调参项 |
| `ELP_MO2_WEAK_ARCHIVE_PENALTY` | `0.14` | 弱入档惩罚 | 一般不大改 |

### 7.2 第二层：质量门槛

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ELP_MO2_QUALITY_REP_SLACK` | `0.08` | 相对 representative 的容忍度 |
| `ELP_MO2_QUALITY_MEDIAN_SLACK` | `0.05` | 相对 archive 中位数的容忍度 |
| `ELP_MO2_QUALITY_OVERRIDE_SCORE_BONUS` | `0.03` | override 时额外容忍分数 |
| `ELP_MO2_QUALITY_OVERRIDE_HV_REL` | `0.06` | 允许 override 的最小相对 HV 增益 |
| `ELP_MO2_QUALITY_OVERRIDE_SPACING_TOL` | `0.04` | override 时允许的 spacing 恶化 |
| `ELP_MO2_QUALITY_PENALTY_WEIGHT` | `0.28` | 低质量入档惩罚权重 |
| `ELP_MO2_USEFUL_HV_REL_THRESHOLD` | `0.010` | 认定为“有用入档”的边际 HV 阈值 |

### 7.3 第三层：轻量 core-quality 奖励

这部分已经从“强约束”改成了“后期轻奖励”，不要轻易再调大。

| 变量名 | 默认值 | 作用 | 建议 |
| --- | --- | --- | --- |
| `ELP_MO2_REP_REWARD_WEIGHT` | `0.12` | representative 质量轻奖励 | 不建议随便增大 |
| `ELP_MO2_TOPK_REWARD_WEIGHT` | `0.08` | archive top-k 质量轻奖励 | 不建议随便增大 |
| `ELP_MO2_TOPK_SCORE_K` | `8` | top-k 核心规模 | 一般不改 |
| `ELP_MO2_CORE_QUALITY_START_PROGRESS` | `0.62` | 后期开始启用 core-quality 奖励 | 一般不改 |
| `ELP_MO2_CORE_QUALITY_ONLY_USEFUL` | `1` | 只对 `useful_archive_update` 生效 | 主线保持开启 |

### 7.4 Archive 满时的 spacing 守卫

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `ELP_MO2_ARCHIVE_SPACING_GUARD` | `0` | 是否启用 archive 满时的 spacing guard |
| `ELP_MO2_ARCHIVE_SPACING_GUARD_REL_TOL` | `0.03` | spacing 允许恶化比例 |
| `ELP_MO2_ARCHIVE_SPACING_GUARD_HV_GAIN_REL` | `0.12` | 触发 guard override 的最小 HV 增益 |

### 实战建议

- 想保守提高前沿均匀性：
  - 先调 `ELP_MO2_SPACING_REWARD_WEIGHT`
  - 再调 `ELP_MO2_DENSITY_REWARD_WEIGHT`
- 想减少弱点污染：
  - 先调 `ELP_MO2_QUALITY_REP_SLACK`
  - 再调 `ELP_MO2_QUALITY_MEDIAN_SLACK`
- 想加强后期前沿核心质量：
  - 优先调 `ELP_MO2_CORE_QUALITY_START_PROGRESS`
  - 不要先把 `REP/TOPK` 权重调大

---

## 8. `Standard` 相关开关

这部分更多是基类层的控制。

| 变量名 | 默认值 | 作用 |
| --- | --- | --- |
| `ELP_RL_AGENT` | 脚本内默认 | 选择 `qlearning` 或 `dqn` |
| `ELP_TUNE_PROFILE` | `ED1` | 训练调优 profile |
| `ELP_RL_S` | 脚本内默认 | RL 相关模式开关 |
| `ELP_ENABLE_REHEAT_LOG` | 脚本内默认 | 是否输出 reheat 日志 |
| `ELP_FIXED_SEEDS` | 空 | 仅 `ELP_DRL_Standard.py` 主入口支持的固定 seed 列表 |

---

## 9. 建议你以后只记三套命令

### 9.1 最终公平对比

```powershell
$env:ELP_STRICT_DETERMINISM="1"
$env:ELP_EXP_NUMBER="1"
$env:ELP_BASE_SEED="20260507"
```

### 9.2 日常训练搜最好值

```powershell
$env:ELP_STRICT_DETERMINISM="0"
$env:ELP_EXP_NUMBER="5"
$env:ELP_BASE_SEED="20260508"
```

### 9.3 `MO2` 参考系重建

```powershell
$env:ELP_STRICT_DETERMINISM="1"
$env:ELP_MO2_CALIBRATE_REFERENCE="1"
$env:ELP_MO2_CALIBRATION_RUNS="8"
$env:ELP_MO2_CALIBRATION_G="20"
$env:ELP_MO2_CALIBRATION_TMAX="40"
```

---

## 10. 哪些变量不建议日常碰

下面这类变量如果你不是在做专门消融，建议别动：

- `ELP_MO2_REFERENCE_MARGIN_RATIO`
- `ELP_MO2_REFERENCE_MIN_SPAN_RATIO`
- `ELP_MO2_REFERENCE_MIN_ARCHIVE_SIZE`
- `ELP_MO2_ARCHIVE_SPACING_GUARD*`
- `ELP_MO2_QUALITY_OVERRIDE_*`
- `ELP_MO_ARCHIVE_HV_TOL`
- `ELP_MO_ARCHIVE_SPACING_TOL`
- `ELP_MO_ARCHIVE_FIXED_HV_MARGIN`
- `ELP_MO_LOCAL_SEARCH_BACKOFF_*`
- `ELP_MO_ND_ACCEPT_CAP_*`

原因不是这些变量没用，而是它们对系统行为的影响更“结构性”，一旦改动，往往需要重新做一轮完整对照，不适合临时微调。

---

## 11. 当前建议的主线

截至目前，建议你把这两个策略分开：

- **最终论文对比主线**
  - `MO baseline`
  - `MO2 lightcore`
  - `ELP_STRICT_DETERMINISM=1`

- **继续搜更强结果的训练主线**
  - `MO2 lightcore`
  - `ELP_STRICT_DETERMINISM=0`
  - 多 seed

如果后面又新增了新的环境变量，优先把它们补进这个文件，而不是继续散落在聊天记录里。

---

## 12. `MO4_Paper` 论文口径专用线

用途：

- 用 `Du62` 复现“定量 + 定性”双目标测试口径；
- 有效优化目标只保留：
  - `MHC`，越小越好；
  - `f3_proxy`，越大越好；
- `f1` 面积利用率按实例常量记录，不参与 Pareto 优化；
- 评价指标改为论文【21】定义的 `PR / SP / OPS`，不再使用 `HV / IGD / Spacing`；
- 正式评价解集改为末段解集 `S_final`，默认取最后 `10%` 搜索阶段内观测到的唯一可行解，而不是整次运行的历史解池。

新增文件：

- `src/algorithms/ELP_DRL_MO4_Paper.py`
- `src/utils/MO_FBSUtil_Paper.py`
- `src/utils/MO_PaperPreferenceUtil.py`
- `src/utils/MO_PaperMetricsUtil.py`

固定偏好矩阵：

- 文件：`src/utils/data/Du62_paper_preference_matrix.pkl`
- 版本：`paper_proxy_preference_v1`
- 生成规则：
  - 使用固定 seed；
  - 生成稀疏非物流偏好边；
  - `f3_proxy = 100 * sum(w_ij / (1 + d_ij)) / sum(w_ij)`；
  - 偏好设施对越近，得分越高。

建议运行：

```powershell
$env:ELP_STRICT_DETERMINISM="1"
$env:ELP_EXP_INSTANCE="Du62"
$env:ELP_EXP_ALGORITHM="ELP_DRL_MO4_PAPER"
$env:ELP_EXP_REMARK="paper MHC + f3_proxy with PR/SP/OPS"
$env:ELP_IS_EXP="1"
$env:ELP_EXP_NUMBER="5"
$env:ELP_BASE_SEED="20260516"
$env:ELP_G="80"
$env:ELP_T_MAX="80"
$env:ELP_T_INITIAL="10000"
$env:ELP_K_HIST="10"
$env:ELP_PAPER_FINAL_WINDOW_RATIO="0.10"

& 'C:\Users\17122\AppData\Local\conda\conda\envs\tensorflow\python.exe' .\src\algorithms\ELP_DRL_MO4_Paper.py
```

结果解释：

- `paperPr`：`|APF(S_final)| / |S_final|`，越大越好；
- `paperSp` / `paperSpRaw`：论文【21】原始尺度下的 `SP`，越小越好；
- `paperSpNorm`：按 `S_final` 各目标范围归一化后的 `SP`，用于横向比较不同尺度目标下的前沿均匀性，越小越好；
- `paperOps`：论文【21】的 `OPS`，越大越好；
- `paperSolutionCount`：`S_final` 中的唯一可行解数量；
- `paperParetoCount`：`S_final` 中的非支配解数量；
- `paperHistorySolutionCount` / `paperHistoryParetoCount`：整次运行历史解池的诊断计数，只用于排查，不作为正式论文指标；
- `paperFinalWindowEmpty`：若为 `true`，说明末段没有采集到可行唯一解，此次结果不应直接用于论文对比。

正式解集口径：

- 群体算法 `NSGA-II` / `MOEA-D` / `SPEA2` / `PSO` 统一按终局候选集评价；其中前三者使用最终种群，`PSO` 使用最后一代粒子群；
- `MO4_Paper` 是单轨迹搜索，没有“最终种群”，因此用运行末段窗口近似终局候选集，默认取最后 `10%` 的唯一可行解；
- 这不是对郭文“最后一代”定义的逐字复刻，但可以避免整段历史解池把 `PR/SP/OPS` 人为抬高，已足够支持同类静态代理实验的近似比较；
- 结果中会记录 `paperSolutionSetMode` 与 `paperSolutionSetDetail`，后续汇总时应只比较这些字段含义一致或可解释对应的结果。

论文评价指标口径：

- `PR = |APF(S)| / |S|`，其中 `S` 按解集而非目标值集合计数；当前实现按布局编码 `(permutation, bay)` 去重，同目标值但不同布局仍视为不同解；
- `SP` 采用论文【21】式 (25)-(26) 的原始目标尺度 L1 最近邻距离标准差，正式字段为 `paperSp` / `paperSpRaw`；
- `OPS` 采用论文【21】式 (27)-(28)，分母使用同一 `S` 内各目标的最优/最差值跨度；
- `paperSpNorm` 仅作诊断，不能与郭文表 6 的 `SP` 直接对比。

论文实例口径：

- `O7`、`O9`：`MO4_Paper` 使用 `pkl` 中的原始上三角流量矩阵，不再沿用默认环境的对称补全；
- `AB20-ar3`：`MO4_Paper` 固定使用 `data/AB20(1963).csv`；
- `SC30`、`SC35`：郭文正文只明确采用最大长宽比约束，并未单列这两个实例的 `AR` 数值；当前论文线按常用基准口径使用 `SC30=5.0`、`SC35=4.0`；
- 结果 JSON/CSV 会额外记录 `paperFlowProfileVersion`、`paperFlowSource`、`paperFlowSourcePath`、`paperConstraintMode`、`paperAspectRatioLimit`、`paperConstraintProfileNote`，用于区分默认实验口径和论文对比口径。

注意：

- `MO4_Paper` 是独立实验线，不替代当前 `MO4`；
- 这里的 `f3_proxy` 是静态可复现代理，不等同于真实人机交互评分；
- 如果后续要声称“严格复现郭广颂论文”，还需要真实交互评分机制，而不是继续增强这个静态代理版本。
