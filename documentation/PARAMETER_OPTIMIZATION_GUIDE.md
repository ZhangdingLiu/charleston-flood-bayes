# 贝叶斯网络参数优化系统使用指南

## 🎯 概述

本系统为贝叶斯网络洪水预测模型提供全面的参数优化和可视化分析功能。通过网格搜索和约束条件筛选，帮助你找到最适合你需求的参数配置。

## 🚀 快速开始

### 1. 完整参数优化

运行完整的参数网格搜索（4,320个参数组合）：

```bash
python run_parameter_optimization.py
```

**预计耗时**: 2-4小时  
**内存需求**: 建议 8GB+

### 2. 快速测试验证

运行小规模测试（128个参数组合）：

```bash
python test_parameter_optimization.py
```

**预计耗时**: 5-10分钟  
**用途**: 验证系统功能，快速获得初步结果

## 📋 系统架构

```
├── run_parameter_optimization.py      # 主控脚本
├── test_parameter_optimization.py     # 测试脚本
├── analysis/
│   └── comprehensive_parameter_grid_search.py  # 网格搜索核心
├── visualization/
│   └── parameter_analysis_visualizer.py        # 可视化模块
└── results/
    └── parameter_optimization_YYYYMMDD_HHMMSS/  # 结果目录
```

## 🎛️ 参数配置

### 网络构建参数

| 参数 | 含义 | 默认搜索范围 | 建议值 |
|------|------|-------------|--------|
| `occ_thr` | 道路最小出现次数 | [2, 3, 4, 5] | 3-4 |
| `edge_thr` | 边最小共现次数 | [1, 2, 3] | 2-3 |
| `weight_thr` | 边权重阈值 | [0.2, 0.3, 0.4, 0.5] | 0.3-0.4 |

### 评估参数

| 参数 | 含义 | 默认搜索范围 | 建议值 |
|------|------|-------------|--------|
| `evidence_count` | 证据道路数量 | [1, 2, 3, 4] | 2-3 |
| `pred_threshold` | 预测阈值 | [0.1, 0.2, 0.3, 0.4, 0.5] | 0.2-0.3 |

### 负样本策略参数

| 参数 | 含义 | 默认搜索范围 | 建议值 |
|------|------|-------------|--------|
| `neg_pos_ratio` | 负正样本比例 | [1.0, 1.5, 2.0] | 1.0-1.5 |
| `marginal_prob_threshold` | 边际概率阈值 | [0.03, 0.05, 0.08] | 0.05 |

## 🎯 约束条件设置

在 `run_parameter_optimization.py` 中修改 `constraints` 字典：

```python
constraints = {
    'min_precision': 0.8,    # 精确度 ≥ 0.8
    'min_recall': 0.8,       # 召回率 ≥ 0.8
    'min_f1_score': 0.7,     # F1分数 ≥ 0.7
    'min_samples': 30,       # 测试样本数 ≥ 30
    'min_accuracy': 0.7      # 准确率 ≥ 0.7 (可选)
}
```

### 常见约束条件组合

**高精度场景**：
```python
constraints = {'min_precision': 0.9, 'min_recall': 0.6}
```

**高召回场景**：
```python
constraints = {'min_precision': 0.6, 'min_recall': 0.9}
```

**平衡场景**：
```python
constraints = {'min_precision': 0.7, 'min_recall': 0.7, 'min_f1_score': 0.7}
```

## 📊 输出文件说明

### 结果目录结构

```
results/parameter_optimization_YYYYMMDD_HHMMSS/
├── complete_results.csv              # 完整结果
├── parameter_recommendations.json    # 推荐配置
├── optimization_report.md           # 分析报告
├── experiment_config.json           # 实验配置
├── performance_summary.json         # 性能摘要
└── visualizations/                  # 可视化图表
    ├── precision_recall_f1_3d.png       # 3D性能分布
    ├── parameter_heatmaps.png            # 参数热图
    ├── parameter_sensitivity.png         # 敏感性分析
    ├── pareto_frontier.png              # Pareto前沿
    └── constraint_filtering.png         # 约束筛选
```

### CSV结果文件格式

`complete_results.csv` 包含以下列：

| 列名 | 含义 |
|------|------|
| `occ_thr`, `edge_thr`, `weight_thr` | 网络构建参数 |
| `evidence_count`, `pred_threshold` | 评估参数 |
| `neg_pos_ratio`, `marginal_prob_threshold` | 负样本策略 |
| `precision`, `recall`, `f1_score`, `accuracy` | 性能指标 |
| `tp`, `fp`, `tn`, `fn` | 混淆矩阵元素 |
| `total_samples`, `network_nodes` | 统计信息 |
| `runtime_seconds` | 运行时间 |

## 🎨 可视化图表

### 1. 3D性能分布图 (`precision_recall_f1_3d.png`)
- **用途**: 展示所有参数组合在 Precision-Recall-F1 三维空间中的分布
- **解读**: 颜色越深表示F1分数越高

### 2. 参数热图 (`parameter_heatmaps.png`)
- **用途**: 显示不同参数组合对F1分数的影响
- **解读**: 红色区域表示高性能参数组合

### 3. 参数敏感性分析 (`parameter_sensitivity.png`)
- **用途**: 分析单个参数变化对性能的影响
- **解读**: 斜率越大表示参数越敏感

### 4. Pareto前沿分析 (`pareto_frontier.png`)
- **用途**: 展示 Precision-Recall 权衡关系
- **解读**: 右上角的点代表最优的权衡配置

### 5. 约束筛选结果 (`constraint_filtering.png`)
- **用途**: 显示满足约束条件的参数分布
- **解读**: 蓝色点表示满足约束的配置

## 💡 推荐配置解释

系统会生成5种推荐配置：

### 1. 最佳F1分数配置
- **特点**: F1分数最高
- **适用**: 综合性能要求高的场景

### 2. 最高精确度配置
- **特点**: 精确度最高，减少误报
- **适用**: 对误报敏感的场景

### 3. 最高召回率配置
- **特点**: 召回率最高，减少漏报
- **适用**: 不能遗漏真实事件的场景

### 4. 最平衡配置
- **特点**: Precision 和 Recall 最接近
- **适用**: 需要平衡性能的场景

### 5. 最鲁棒配置
- **特点**: 参数变化时性能最稳定
- **适用**: 生产环境部署

## 🔧 高级用法

### 1. 自定义参数网格

修改 `run_parameter_optimization.py` 中的参数网格：

```python
custom_param_grid = {
    'occ_thr': [3, 4, 5],           # 自定义范围
    'edge_thr': [2],                # 固定值
    'weight_thr': [0.3, 0.35, 0.4], # 更细粒度
    # ... 其他参数
}

optimizer = ParameterOptimizer(
    constraints=constraints, 
    param_grid=custom_param_grid
)
```

### 2. 单独运行网格搜索

```python
from analysis.comprehensive_parameter_grid_search import ParameterGridSearcher

searcher = ParameterGridSearcher()
results, result_dir = searcher.run_grid_search()
```

### 3. 单独运行可视化

```python
import pandas as pd
from visualization.parameter_analysis_visualizer import ParameterVisualizer

# 加载现有结果
df = pd.read_csv("results/parameter_optimization_XXX/complete_results.csv")

# 创建可视化
visualizer = ParameterVisualizer(df, "my_visualizations")
visualizer.generate_all_visualizations()
```

## ⚠️ 注意事项

### 系统要求
- **Python**: 3.7+
- **内存**: 建议 8GB+
- **存储**: 需要 1-2GB 空间保存结果

### 依赖库
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install pgmpy networkx  # 项目特定依赖
```

### 运行时间估算
- **完整优化** (4,320组合): 2-4小时
- **测试版本** (128组合): 5-10分钟
- **单个组合**: 约30-60秒

### 常见问题

**Q: 搜索过程中出现大量失败怎么办？**
A: 这是正常现象。某些参数组合可能导致网络构建失败或评估样本不足。成功率在30-70%是正常的。

**Q: 没有参数组合满足约束条件怎么办？**
A: 尝试放宽约束条件，或者增加参数搜索范围。

**Q: 如何选择最适合的配置？**
A: 根据应用场景选择：
- 预警系统：选择高召回率配置
- 决策支持：选择高精确度配置
- 研究分析：选择最佳F1配置

## 📞 支持

遇到问题时：
1. 查看生成的 `optimization_report.md`
2. 检查 `experiment_config.json` 确认配置
3. 运行 `test_parameter_optimization.py` 验证系统功能

## 🔄 更新日志

**v1.0** (2025-01-21)
- 初始版本发布
- 支持全参数网格搜索
- 多种可视化分析
- 约束条件筛选
- 5种推荐策略