# ua-flp-LSA2

面向不等面积设施布局问题（Unequal Area Facility Layout Problem, UA-FLP）的实验仓库，包含单目标与多目标布局算法、结果评估脚本，以及本地实验结果可视化 dashboard。

## 仓库内容

- `src/algorithms/`: 布局求解算法实现，包括 ELP/DRL、MO4、BiMO4、GRASP 等变体
- `src/utils/`: 实例加载、流量矩阵处理、布局评价、Pareto 指标与 benchmark 工具
- `src/dashboard/`: 本地结果看板与布局可视化服务
- `scripts/`: 批量实验、参数扫描、矩阵生成等 PowerShell/Python 脚本
- `tools/`: 对比分析、指标回填、实验结果汇总工具
- `tests/`: 回归测试与样例数据
- `data/`、`files/`: 实例数据、实验结果、评估输入输出

## 环境要求

- Python 3.10+
- Windows PowerShell（仓库内已有多个 `.ps1` 实验脚本）
- 建议使用独立虚拟环境，避免全局 `numpy` / `scipy` 二进制依赖冲突

安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 常用入口

### 1. 运行基准评估

评估候选集相对于固定 benchmark package 的指标：

```powershell
python evaluate.py --instance A-10-10 --input path\to\candidates.json
```

按论文口径评估 archive：

```powershell
python paper_exact_evaluate.py --instance A-10-10 --archive path\to\archive.json
```

### 2. 启动本地 dashboard

```powershell
python -m src.dashboard.server --host 127.0.0.1 --port 8765
```

启动后访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)。

默认会读取 `files/expresults/` 下最新的 `.csv` 结果文件，也可以显式指定：

```powershell
python -m src.dashboard.server --csv files/expresults/AB20-ar3-ELP_RL_Standard.csv
```

### 3. 批量实验

批量运行实例：

```powershell
.\scripts\run_all_instances.ps1
```

部分附加脚本：

- `scripts/run_standard46_a8.ps1`: 运行 Standard4.6 相关实验
- `scripts/sweep_standard46_true_cost.ps1`: 扫描 true-cost 配置
- `tools/run_bimo4_comparison.py`: 运行 BiMO4 对比
- `tools/analyze_bimo4_comparison.py`: 分析 BiMO4 对比结果
- `tools/summarize_bimo4_comparison.py`: 汇总 BiMO4 对比结果

## 测试

运行全部测试：

```powershell
pytest
```

运行单个测试文件：

```powershell
pytest tests/test_flow_matrix_loading.py
```

如果在测试收集阶段看到 `numpy` C 扩展加载失败，通常不是业务代码问题，而是当前 Python 环境中的 `numpy` / `scipy` 二进制安装损坏或与解释器版本不匹配。优先重建虚拟环境并重新安装依赖，不要用 fallback 掩盖环境问题。

## 数据与结果文件约定

- `data/` 与 `src/utils/data/` 中包含实例相关数据与缓存矩阵
- `files/expresults/` 用于保存实验结果 CSV
- `files/expresults/pareto_archives/` 可用于保存 Pareto archive JSON
- `backup/`、`tmp/`、临时 CSV/PKL 通常不应直接提交到主线

## 开发建议

- 提交前优先区分代码改动与实验产物，避免把临时数据、备份文件和子模块工作区状态一并提交
- 修改涉及评估口径、流量矩阵或布局指标的逻辑后，应同步补充或更新 `tests/` 中的回归测试
- 若需要复现实验，请固定随机种子、实例名和结果输出路径，避免覆盖历史结果
