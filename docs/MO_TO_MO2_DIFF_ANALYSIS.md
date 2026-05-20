# `MO -> MO2` 差分归因分析

这个文件的目标不是证明 `MO2` “理念错误”，而是回答一个更直接的问题：

> `MO2` 相对 `MO` 到底改了什么，哪些差分值得保留，哪些差分大概率就是把结果拉弱的主因。

结论先给：

- `MO2` 不是整体都该回退。
- 但它目前把**正确性修复**、**实验可复现性改进**、**算法新假设**、**搜索行为删减**混在了一起。
- 当前版本没超过 `MO baseline`，高概率不是某一个单点 bug，而是**多条高风险差分叠加**。

---

## 1. 先看结果，不先看代码

参见：[MO2_VERSION_LOG.md](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/docs/MO2_VERSION_LOG.md)

在 `2026-05-07 final_clean` 这组严格公平对比里：

- `MO2` 只在 `Spacing` 上更好
- `MO` 在以下方面更强：
  - `HV`
  - representative 解质量
  - archive 平均质量
  - 并集非支配贡献

所以这次差分归因的目标很清楚：

> 找出 `MO2` 哪些改动把搜索从“强覆盖、高质量前沿”推成了“更均匀但偏弱的前沿”。

---

## 2. `MO2` 相对 `MO` 的差分分组

按性质分，`MO2` 相对 `MO` 的变化可以拆成 5 组：

1. **实验/工程正确性修复**
2. **状态与策略学习链路改造**
3. **奖励与 archive 门控改造**
4. **参考系改造**
5. **搜索强化机制删减**

真正影响输赢的，主要是后 4 组。

---

## 3. 第一组：工程正确性修复

这一组不是性能风险，应该保留。

### 3.1 全局随机源统一播种

相关位置：

- [ELP_DRL_Standard.py:73](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_Standard.py:73)
- [ELP_DRL_MO.py:2201](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:2201)
- [ELP_DRL_MO2.py:1355](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:1355)

作用：

- 统一控制 `random / numpy / torch`
- 让同 seed 结果可复现

判断：

- **保留**

原因：

- 这是实验可靠性的底线，不属于算法能力差分。

### 3.2 archive 候选必须最终保留

作用：

- 避免“候选解被 trim 掉了，但还算 archive 更新”这种伪更新

判断：

- **保留**

原因：

- 这是 archive 语义修正，不该回退。

---

## 4. 第二组：状态与策略学习链路改造

这是 `MO2` 最显眼的变化，但不一定是最该优先回退的。

### 4.1 `MO` 的状态/策略链路

`MO` 使用的是离散 band 编码状态：

- [ELP_DRL_MO.py:1199](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:1199)

然后用 `StandardDQNAgent` 学一个 16384 状态空间上的 Q 值：

- [ELP_DRL_MO.py:1830](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:1830)

### 4.2 `MO2` 的状态/策略链路

`MO2` 改成了：

- 24 维连续状态向量
- 4 维 preference 向量
- 条件化 DQN

相关位置：

- 网络结构：[ELP_DRL_MO2.py:34](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:34)
- 条件化 agent：[ELP_DRL_MO2.py:52](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:52)
- 偏好采样：[ELP_DRL_MO2.py:249](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:249)
- 连续状态编码：[ELP_DRL_MO2.py:924](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:924)

### 4.3 归因判断

判断：

- **重做，不建议直接回退**

原因：

1. 连续状态本身不一定坏。
2. 真正的高风险不是“连续状态”，而是：
   - **条件化 preference 采样**
   - 在相同预算下，把同一条搜索轨迹拆成了多个方向的学习任务
3. 最终评价仍然看 representative / Pareto 前沿，而 `MO2` 训练时却在多偏好间分散样本。

换句话说：

> `MO2` 不是因为“连续状态”输，而更像是因为“把单任务学习改成了多条件任务学习，但预算没同步放大”。

### 4.4 建议

- 连续状态可以保留
- **条件化 preference 采样不应当作为下一轮主差分**

建议下一轮做法：

- 回到单一等权目标方向训练
- 先验证连续状态是否仍能超过 `MO`

---

## 5. 第三组：奖励与 archive 门控改造

这一组是当前最可能把 `MO2` 拉弱的核心来源。

### 5.1 `MO` 的奖励

`MO` 的奖励相对简单：

- 支配关系
- proxy 改善
- archive 是否变化
- 可行性/约束改善

相关位置：

- [ELP_DRL_MO.py:1411](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:1411)

### 5.2 `MO2` 的奖励

`MO2` 的奖励现在同时包含：

- preference 加权目标改进
- spacing
- density
- hv
- marginal hv
- quality penalty
- representative 改进
- top-k 改进
- archive bonus

相关位置：

- [ELP_DRL_MO2.py:1004](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:1004)

### 5.3 archive 门控

`MO2` 的 `_observe_feasible_state` 还额外引入了：

- tentative archive
- candidate retained 检查
- quality guard
- quality override
- marginal HV 判断
- rep/top-k before/after 分析

相关位置：

- [ELP_DRL_MO2.py:683](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:683)
- [ELP_DRL_MO2.py:604](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:604)

### 5.4 归因判断

判断：

- **部分保留，部分回退**

具体拆分：

#### 应保留

- `candidate must be retained`
- 基础 `quality guard`
- `marginal_hv` 反馈

#### 应回退/弱化

- 把 `spacing` 当主奖励驱动项
- 把 `density` 当主奖励驱动项
- 用过多奖励项同时拉 policy

### 5.5 为什么这是主风险

从结果看，`MO2` 一直更容易得到：

- 更小的 `Spacing`
- 但更弱的 `HV`
- 更差的 representative 质量

这和当前奖励结构完全一致：

> 它更像在学“怎样把点排匀”，而不是“怎样把高质量前沿做强”。

### 5.6 建议

下一轮应把奖励结构收缩成：

1. 基础目标改进
2. marginal HV
3. 低质量点惩罚
4. spacing 只保留成“恶化惩罚”或“护栏”

而不是继续让 spacing/density 主导。

---

## 6. 第四组：参考系改造

### 6.1 `MO`

`MO` 用当前 archive 动态更新的 `ideal/nadir`：

- [ELP_DRL_MO.py:870](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:870)

### 6.2 `MO2`

`MO2` 引入了固定参考系：

- 加载/保存/校准：多处新方法
- refresh 时优先用固定参考：
  - [ELP_DRL_MO2.py:467](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:467)

### 6.3 归因判断

判断：

- **保留基础设施，但重做用途边界**

原因：

- 固定参考系对**跨 run 可比性**是有价值的
- 但把它直接接进：
  - 在线状态
  - 在线奖励
  - archive 门控
  这会把“评估参考框架”变成“搜索压力来源”

这两者不应该强耦合。

### 6.4 建议

下一轮建议：

- 固定参考系继续保留给：
  - 结果记录
  - 后处理比较
  - representative 稳定评分
- 在线训练主链路里，尽量减少它对 reward / gating 的直接主导

---

## 7. 第五组：搜索强化机制删减

这是当前最被低估、但实际最可能影响结果的一组。

### 7.1 `MO` 运行时会触发的强化链路

`MO` 在主循环里有这些动作：

- greedy local search
  - [ELP_DRL_MO.py:1914](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:1914)
- elite intensification
  - [ELP_DRL_MO.py:1959](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:1959)
- reheating
  - [ELP_DRL_MO.py:2006](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:2006)
- diversification
  - [ELP_DRL_MO.py:2007](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO.py:2007)

### 7.2 `MO2` 的做法

`MO2` 在初始化里直接关掉了这些强干预：

- [ELP_DRL_MO2.py:231](/c:/Users/17122/PycharmProjects/pythonProject/ua-flp-LSA2/src/algorithms/ELP_DRL_MO2.py:231)

具体是：

- `reheat_enabled = False`
- `local_search_disable_after_progress = 0.0`
- `diversify_trigger_no_improve = 10**9`

而且 `MO2._run_impl()` 中确实没有再调用：

- `_greedy_local_search`
- `_attempt_reheating`
- `_attempt_diversification`

### 7.3 归因判断

判断：

- **高概率主因，应优先回迁**

原因：

1. `MO` 能赢的恰恰是：
   - `HV`
   - representative 质量
   - archive 非支配贡献
2. 这些指标通常正是：
   - 局部强化
   - reheating
   - diversification
   带来的收益最明显
3. `MO2` 一边把学习任务变复杂，一边又把 `MO` 原本强力的强化链路关掉，等于双重削弱。

换句话说：

> `MO2` 不是在 `MO` 的强骨架上做增量改进，而更像是“换了更难的学习问题，同时拆掉了 MO 本来最能出结果的部分”。 

---

## 8. 最终分类：保留 / 回退 / 重做

### 8.1 建议保留

- 全局随机源统一播种
- `ELP_STRICT_DETERMINISM`
- archive 候选必须最终保留
- 参考系校准/缓存基础设施（仅作为评估工具）
- richer logging / recorder / trace

### 8.2 建议回退到 `MO` 行为

- 后期持续强化的 `spacing` 主奖励
- 多 preference 条件化训练主线
- 关闭 local search / reheating / diversification 的做法

### 8.3 建议重做

- 固定参考系在在线训练中的作用边界
- archive quality guard 的目标
  - 从“尽量排匀”
  - 改成“挡弱点，同时保强边界点”
- 连续状态表征
  - 保留连续状态
  - 但先回到单目标/单偏好训练，再验证它是否真有增益

---

## 9. 下一轮最合理的版本方向

如果目标是做一个真正可能超过 `MO baseline` 的新版本，建议不是继续在当前 `MO2` 上加机制，而是做一个：

### `MO + selective carry-over`

保留：

- `MO` 的主搜索骨架
- local search / elite / reheat / diversification
- `MO` 的简单 reward 主线

只迁移少数高价值修正：

1. 全局 seed 修复
2. archive retain 语义修复
3. 部分 quality guard
4. 如有必要，再单独测试连续状态编码

不要一次性迁移：

- preference-conditioned DQN
- fixed reference 主导训练
- spacing/density 主奖励
- 重型 core-quality 奖励

---

## 10. 一句话总结

当前 `MO2` 输给 `MO`，不是因为“`MO` 更老所以更差”这个常识失效了，而是因为：

> `MO2` 同时做了三件高风险事情：把学习任务变难、把奖励变复杂、把 `MO` 原本强的强化机制关掉了。  
>  
> 所以下一轮正确方向不是继续堆 `MO2`，而是回到 `MO` 的强骨架，只迁移那些已经被证明是工程正确性或局部有效的改动。
