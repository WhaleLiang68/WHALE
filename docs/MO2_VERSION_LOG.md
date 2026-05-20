# MO2 版本记录

这个文件用于记录 `ELP_DRL_MO2.py` 的关键实验版本，避免后续继续改动后丢失比较基线。

---

## 2026-05-07 `final_clean`

### 版本定位

这是在完成以下修复/整理后的 `MO2` 版本：

- 统一 `random / numpy / torch` 播种
- 增加 `ELP_STRICT_DETERMINISM` 开关
- `MO2` 采用 `lightcore` 路线：
  - `quality guard` 保留
  - `rep/top-k` 奖励改为后期轻量触发
  - 继续使用固定参考系缓存

### 对比目的

和 `MO baseline` 做同实例、同预算、同 seed 的最终公平对比。

### 实验条件

- 实例：`AB20-ar3`
- 预算：`G=80`
- 步数：`T_MAX=80`
- 初始温度：`T_INITIAL=10000`
- 历史参数：`K_HIST=10`
- seed：`20260507`
- 严格确定性：`ELP_STRICT_DETERMINISM=1`

### 使用文件

- `MO baseline`
  - 结果文件：[AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean.csv](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean.csv)
  - archive 文件：[AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean-20260507_111717_967472.json](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/pareto_archives/AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean-20260507_111717_967472.json)
- `MO2`
  - 结果文件：[AB20-ar3-ELP_DRL_MO2_AB20-ar3_G80_T80_final_clean.csv](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/AB20-ar3-ELP_DRL_MO2_AB20-ar3_G80_T80_final_clean.csv)
  - archive 文件：[AB20-ar3-ELP_DRL_MO2_AB20-ar3_G80_T80_final_clean-20260507_112653_526031.json](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/pareto_archives/AB20-ar3-ELP_DRL_MO2_AB20-ar3_G80_T80_final_clean-20260507_112653_526031.json)

### 当前 `MO2` 默认主开关

对应代码位置：[ELP_DRL_MO2.py:196](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:196)

- `ELP_MO2_SPACING_REWARD_WEIGHT = 0.70`
- `ELP_MO2_DENSITY_REWARD_WEIGHT = 0.18`
- `ELP_MO2_HV_REWARD_WEIGHT = 0.12`
- `ELP_MO2_MARGINAL_HV_REWARD_WEIGHT = 0.36`
- `ELP_MO2_ARCHIVE_CHANGE_BONUS = 0.24`
- `ELP_MO2_WEAK_ARCHIVE_PENALTY = 0.14`
- `ELP_MO2_ARCHIVE_QUALITY_GUARD = 1`
- `ELP_MO2_QUALITY_REP_SLACK = 0.08`
- `ELP_MO2_QUALITY_MEDIAN_SLACK = 0.05`
- `ELP_MO2_QUALITY_OVERRIDE_HV_REL = 0.06`
- `ELP_MO2_QUALITY_PENALTY_WEIGHT = 0.28`
- `ELP_MO2_REP_REWARD_WEIGHT = 0.12`
- `ELP_MO2_TOPK_REWARD_WEIGHT = 0.08`
- `ELP_MO2_CORE_QUALITY_START_PROGRESS = 0.62`
- `ELP_MO2_CORE_QUALITY_ONLY_USEFUL = 1`

### CSV 原始指标

#### `MO baseline`

- `decision_score = 0.1421753063298136`
- `stable_decision_score = 0.10432073560838162`
- `archive_hypervolume = 0.10570330507530903`
- `archive_spacing = 0.1276422493919171`
- `pareto_size = 64`
- representative raw objectives:
  - `MHC = 8514.434775297126`
  - `CR = 487.66580614968177`
  - `DR = 9164.27981124304`
  - `AR = 1.0`

#### `MO2`

- `decision_score = 0.2606909900961536`
- `stable_decision_score = 0.13446734429326546`
- `archive_hypervolume = 0.008520016595133902`
- `archive_spacing = 0.02436268934834301`
- `pareto_size = 64`
- `mo2_reference_source = cache`
- representative raw objectives:
  - `MHC = 9382.692524559661`
  - `CR = 608.4774320445124`
  - `DR = 8504.20896153192`
  - `AR = 1.0`

### 公共尺度重算结果

说明：下面的比较不直接使用各自 CSV 里的 `decision_score/HV/Spacing` 做横比，而是把两份 archive 放到同一套 `ideal/nadir` 下重算。

#### `MO baseline`

- `common_hv = 0.7508310590426625`
- `common_spacing = 0.11431677765167261`
- `common_rep_score = 0.14779694705304128`
- `common_avg_score = 0.19392256718742712`
- `nd_contrib = 60`
- `cross_dominated = 4`

#### `MO2`

- `common_hv = 0.6073624821260905`
- `common_spacing = 0.049281298109728526`
- `common_rep_score = 0.19439163398546122`
- `common_avg_score = 0.2741227600138363`
- `nd_contrib = 34`
- `cross_dominated = 30`

### 结论

这次 `final_clean` 的正式公平对比中：

- `MO2` 只在 `Spacing` 上更好，说明前沿更均匀。
- `MO baseline` 在以下维度更强：
  - `HV`
  - representative 解质量
  - archive 平均质量
  - 并集非支配贡献

因此，这一版 `MO2` 不能作为“超过 `MO baseline`”的最终主线。

### 根因判断

这版 `MO2` 的主要问题不是随机性，也不是参考系不稳定，而是：

- 过于强调前沿均匀性
- 对高质量边界点的保留压力仍然不够
- 结果表现为：
  - `Spacing` 改善
  - 但 `HV`、代表解质量、archive 质量下降

### 后续改造方向

下一轮应该直接瞄准“超过 `MO baseline`”的主矛盾：

1. 降低 `MO2` 的均匀性压力
2. 增强高质量边界点保留
3. 把 `quality guard` 从“只挡弱点”升级为“保强边界点”

后续新版本请继续追加到本文件，而不是覆盖这条记录。

---

## 2026-05-07 `MO3 final`

### 版本定位

这是以 `MO` 为骨架、只迁移少量已验证安全改动后的 `MO3` 版本：

- 保留 `MO` 原始搜索主骨架
- 默认开启 `candidate-retained` 语义修复
- 增加轻量 `spacing guard`
- 增加轻量 `boundary bonus`

对应文件：

- [ELP_DRL_MO3.py](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO3.py)

### 对比目的

验证“以 `MO` 为骨架、只做少量安全增强”是否能稳定超过 `MO baseline`。

### 实验条件

- 实例：`AB20-ar3`
- 预算：`G=80`
- 步数：`T_MAX=80`
- 初始温度：`T_INITIAL=10000`
- 历史参数：`K_HIST=10`
- seed：`20260507`
- 严格确定性：`ELP_STRICT_DETERMINISM=1`

### 使用文件

- `MO baseline`
  - 结果文件：[AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean.csv](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean.csv)
  - archive 文件：[AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean-20260507_111717_967472.json](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/pareto_archives/AB20-ar3-ELP_DRL_MO_AB20-ar3_G80_T80_final_clean-20260507_111717_967472.json)
- `MO3`
  - 结果文件：[AB20-ar3-ELP_DRL_MO3_AB20-ar3_G80_T80_final.csv](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/AB20-ar3-ELP_DRL_MO3_AB20-ar3_G80_T80_final.csv)
  - archive 文件：[AB20-ar3-ELP_DRL_MO3_AB20-ar3_G80_T80_final-20260507_144044_939719.json](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/pareto_archives/AB20-ar3-ELP_DRL_MO3_AB20-ar3_G80_T80_final-20260507_144044_939719.json)

### 核心结果

`MO3` 和 `MO baseline` 在这次实验中得到的结果完全一致：

- representative 解完全一致
- `decision_score` 完全一致
- `stable_decision_score` 完全一致
- `archive_hypervolume` 完全一致
- `archive_spacing` 完全一致
- archive `items` 完全一致

唯一差异来自运行元数据：

- `algorithm`
- `generatedAt`
- 输出路径

### 运行代价

- `MO baseline runtimeSeconds = 565.219258`
- `MO3 runtimeSeconds = 5277.357871`

这说明：

- `MO3` 当前引入的轻量增强没有改变搜索轨迹
- 但新增的 archive 反馈计算显著增加了运行开销

### 结论

这次 `MO3 final` 不能算“优于 `MO`”，也不能算“劣于 `MO`”。

更准确地说：

- 当前 `MO3` 在行为上等价于 `MO`
- 只是增加了额外计算成本

### 根因判断

高概率原因不是逻辑错误，而是当前这些增强在这组阈值下**基本没有形成有效选择压力**：

- `spacing guard` 没有实质改变 archive 演化
- `boundary bonus` 没有推动策略走向与 `MO` 不同的轨迹

### 后续方向

如果继续深耕 `MO3`，优先级应该是：

1. 先加触发计数和行为遥测，确认新逻辑到底触发了多少次
2. 如果触发接近 0，说明当前阈值设计过于保守
3. 只有在拿到触发数据后，再决定是调阈值还是改机制

### `MO3 final telemetry` 补充结论

结果文件：

- [AB20-ar3-ELP_DRL_MO3_AB20-ar3_G80_T80_final_telemetry.csv](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/files/expresults/AB20-ar3-ELP_DRL_MO3_AB20-ar3_G80_T80_final_telemetry.csv)

关键遥测：

- `mo3FeasibleObserveCalls = 2798`
- `mo3ArchivePreviewInsertions = 239`
- `mo3ArchiveChanged = 239`
- `mo3CandidateRetainedRejects = 0`
- `mo3SpacingGuardRejects = 0`
- `mo3BoundaryBonusTriggers = 20`
- `mo3BoundaryHvRelGainAvg = 0.2577`
- `mo3BoundaryRepScoreDeltaAvg = -0.0295`

事件分解：

- `mo3_boundary_bonus` 共触发 `20` 次
- 这 `20` 次全部发生在：
  - `phase = elite`
  - `action = facility_swap`

这说明：

1. `candidate-retained` 在这次正式运行中没有成为有效差分
2. `spacing guard` 在这次正式运行中完全没有形成约束
3. 真正介入搜索的只有 `boundary bonus`
4. 但它全部发生在 `elite` 强化阶段，而不是主策略阶段

因此，这次 `MO3` 与 `MO` 结果完全一致，更像是：

- 不是单纯“坏 seed”
- 也不主要是“预算太少”
- 而是 `MO3` 当前真正起作用的新增机制，**信用分配位置不对**

下一步应优先修正：

- 让 `MO3` 的 reward bonus 只归因给主策略动作
- 不要把 `elite` 阶段带来的 archive 改善混进主步 reward
