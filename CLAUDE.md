# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Bayesian Network analysis project for Charleston flood prediction using road closure data. The project analyzes patterns in historical flood events to build predictive networks for flood-prone road segments.

## Project Structure

**清洁项目结构 (Clean Structure):**
```
├── src/                            # 唯一源代码目录 ⭐
│   ├── models/                     # 核心模型
│   │   ├── model.py                # FloodBayesNetwork核心类 ⭐
│   │   ├── main.py                 # 基础网络构建入口 ⭐
│   │   └── Road_Closures_2024.csv  # 主数据文件 (2015-2024, 923条记录)
│   ├── evaluation/                 # 评估脚本
│   │   ├── latest/                 # 最新推荐版本
│   │   │   ├── validation_focused_evaluation.py  # 主评估管道 ⭐
│   │   │   └── pilot_conservative_evaluation.py  # 快速评估
│   │   ├── conservative_strategies/ # 保守策略评估
│   │   ├── precision_focused/      # 精度优化评估
│   │   ├── experimental/           # 实验性评估
│   │   └── [测试脚本]              # 各种测试脚本
│   ├── analysis/                   # 分析工具
│   │   ├── comprehensive_parameter_grid_search.py ⭐ # 网格搜索引擎
│   │   ├── run_parameter_optimization.py ⭐        # 参数优化主控 (4,320组合)
│   │   ├── test_parameter_optimization.py          # 快速测试 (128组合)
│   │   └── [其他分析工具]          # 阈值优化、数据分析等
│   ├── visualization/              # 可视化工具
│   │   ├── parameter_analysis_visualizer.py ⭐     # 参数分析可视化
│   │   ├── network_visualization.py               # 网络结构图
│   │   └── [其他可视化工具]        # 生成图表、报告等
│   └── data_processing/            # 数据处理 (预留)
├── experiments/                    # 实验和专项分析
│   ├── 2025_validation/            # 2025年新数据验证 ⭐
│   │   ├── 2025 0822 reliable bayes test _ for final defence/ ⭐ # 答辩实验 (Sept 2025)
│   │   │   ├── pure_python_prediction.py          # 纯Python实时预测脚本
│   │   │   └── realtime_window_*.json             # 16个JSON结果文件 (2次运行)
│   │   └── [其他验证脚本]          # 其他2025数据验证
│   ├── parameter_tuning/           # 测试集评估实验
│   │   ├── evaluate_top_configs_on_test_set.py ⭐  # 最佳配置测试
│   │   └── [其他测试集评估]        # 灵活参数、指定参数测试
│   ├── flood_specific/             # 特定洪水事件分析
│   └── debug/                      # 调试脚本
├── documentation/                  # 项目文档 📚
│   ├── CLAUDE.md                   # 本文件
│   ├── PARAMETER_OPTIMIZATION_GUIDE.md # 参数优化指南
│   └── [技术报告]                  # 各种分析报告
├── results/                        # 结果文件 📊
│   ├── latest/                     # 最新结果
│   │   └── validation_focused_results_20250714_190013/ ⭐ # 最佳结果
│   ├── parameter_optimization_*/    # 参数优化结果
│   └── [历史结果]                  # 其他历史结果
├── archive/                        # 归档区域 🗄️
│   ├── deprecated_scripts/         # 废弃脚本
│   ├── old_results/                # 旧结果文件
│   └── core_backup_*/              # 重要文件备份
└── .gitignore                      # Git忽略配置
```

## Quick Start

**环境设置:**
```bash
pip install pgmpy networkx pandas numpy scikit-learn matplotlib seaborn
```

**核心功能 - 3步开始:**
```bash
# 1. 基础网络构建  
python src/models/main.py

# 2. 参数优化 (推荐) ⭐  
python src/analysis/run_parameter_optimization.py

# 3. 主评估管道
python src/evaluation/latest/validation_focused_evaluation.py
```

## Key Features

### 🧠 **Bayesian Network Core**
- **FloodBayesNetwork**: 核心模型类，构建基于道路依赖关系的DAG
- **时间分割验证**: 避免数据泄露的严格时间分割 (2015-2021训练, 2022-2024测试)
- **条件概率估计**: 拉普拉斯平滑处理稀疏数据

### ⚡ **Parameter Optimization System** 
- **4,320参数组合**: 全面网格搜索 + 约束条件筛选
- **多策略推荐**: 最佳F1、高精度、高召回、平衡、鲁棒5种配置
- **可视化分析**: 3D散点图、热图、敏感性分析、Pareto前沿

### 🔬 **Comprehensive Evaluation**
- **测试集评估**: 独立测试集验证最佳配置性能
- **2025数据验证**: 最新洪水数据的模型验证
- **答辩实验 (Sept 2025)**: 纯Python实时累积预测系统 (无pandas依赖)
- **特定事件分析**: 历史重大洪水事件案例研究

## Development Commands

**推荐开发流程:**
```bash
# 1. 测试核心功能
python src/evaluation/test_validation_script.py

# 2. 运行主评估管道 (推荐使用)
python src/evaluation/latest/validation_focused_evaluation.py

# 3. 参数优化 (新功能) ⭐
python src/analysis/run_parameter_optimization.py

# 4. 快速参数优化测试
python src/analysis/test_parameter_optimization.py

# 5. 生成可视化
python src/visualization/generate_visualizations.py results/latest/[结果文件夹]

# 6. 基础网络构建 (传统方法)
python src/models/main.py
```

**特定功能命令:**
```bash
# 参数优化系统 ⭐
python src/analysis/run_parameter_optimization.py                    # 完整参数优化 (4,320组合)
python src/analysis/test_parameter_optimization.py                   # 快速测试 (128组合)

# 测试集评估 ⭐
python experiments/parameter_tuning/evaluate_top_configs_on_test_set.py              # 最佳配置测试集评估
python experiments/parameter_tuning/evaluate_specified_params_on_test.py             # 指定参数测试评估  
python experiments/parameter_tuning/evaluate_focused_flexible_params.py              # 灵活参数评估
python experiments/parameter_tuning/evaluate_flexible_params_on_test.py              # 灵活参数测试评估

# 数据分析
python src/analysis/detailed_analysis_fixed.py              # 数据质量分析
python src/analysis/threshold_analysis_and_optimization.py  # 阈值优化
python src/analysis/comprehensive_parameter_grid_search.py  # 网格搜索

# 可视化
python src/visualization/network_visualization.py           # 网络结构图
python src/visualization/parameter_analysis_visualizer.py   # 参数分析图

# 参数验证
python src/analysis/verify_best_config.py

# 测试套件
python src/evaluation/test_improved_eval.py
python src/evaluation/test_visualizations.py
python src/evaluation/test_validation_script.py                # 验证脚本测试

# 2025年数据验证
python experiments/2025_validation/validate_2025_flood_data_fixed.py

# 答辩实验 - 实时累积预测 (Sept 2025) ⭐
cd "experiments/2025_validation/2025 0822 reliable bayes test _ for final defence"
python pure_python_prediction.py                # 纯Python实时预测 (生成16个JSON结果)
```

## Core Architecture

**主要组件:**
- `src/models/model.py`: FloodBayesNetwork类 - 贝叶斯网络构建与推理核心
- `src/models/main.py`: 基础入口点 - 网络构建和参数搜索
- `Road_Closures_2024.csv`: Charleston道路封闭数据 (2015-2024年)

**数据处理流程:**
1. **数据预处理**: 时间聚合和道路共现分析
2. **网络构建**: 基于道路依赖关系构建有向无环图
3. **参数估计**: 计算条件概率表 (CPTs)
4. **模型评估**: 使用多种阈值在测试数据上验证

**推荐使用的评估脚本:**
- **主要**: `src/evaluation/latest/validation_focused_evaluation.py` - 最完整的评估管道
- **快速**: `src/evaluation/latest/pilot_conservative_evaluation.py` - 简化版本，适合快速测试

## Data Handling

**数据分割方法:**
- **时间分割** (推荐): `src/evaluation/latest/validation_focused_evaluation.py`中实现
  - 训练集: 2015-2021
  - 测试集: 2022-2024
  - 避免同日洪水事件的数据泄露
- **随机分割**: `src/models/main.py`中使用 (有时间泄露问题)

**数据结构:**
- 时间窗口: 日聚合 (`t_window="D"`)
- 随机种子: 42 (保持一致性)
- 分割比例: 70% 训练 / 30% 测试
- 关键列: `time_create`, `STREET`, `OBJECTID`, `REASON`

## Model Parameters

**核心阈值:**
- `occ_thr`: 道路纳入的最小洪水发生次数 (默认: 10)
- `edge_thr`: 边创建的最小共现次数 (默认: 3)
- `weight_thr`: 边的最小条件概率 (默认: 0.4)
- `prob_thr`: 预测概率阈值 (通常 0.3-0.7)

**网络约束:**
- 最大节点数: ~20-157 (取决于参数)
- DAG结构: 不允许循环
- 拉普拉斯平滑: 避免零概率

## Results Analysis

**最佳结果位置:**
- **`results/latest/validation_focused_results_20250714_190013/`** - 包含完整的PDF/PNG可视化、参数敏感性分析和性能比较

**结果文件类型:**
- **模型**: `*.pkl` 训练好的贝叶斯网络
- **指标**: JSON格式的评估指标
- **可视化**: 网络图、混淆矩阵、性能图表
- **数据**: CSV格式的详细结果

## Dependencies

**核心依赖:**
```bash
pip install pgmpy networkx pandas numpy scikit-learn matplotlib seaborn
```

## Common Issues & Solutions

**兼容性**: 脚本处理pgmpy的`BayesianNetwork`和`DiscreteBayesianNetwork`导入
**数据泄露**: 使用`validation_focused_evaluation.py`避免时间信息泄露  
**稀疏数据**: 拉普拉斯平滑处理边缘情况
**零概率**: 自动应用平滑防止数值问题

## Parameter Optimization System ⭐

**新功能**: 全面的参数网格搜索和可视化分析系统

**核心文件**:
- `src/analysis/run_parameter_optimization.py`: 主控脚本 (4,320个参数组合)
- `src/analysis/test_parameter_optimization.py`: 快速测试版本 (128个组合)
- `src/analysis/comprehensive_parameter_grid_search.py`: 网格搜索引擎
- `src/visualization/parameter_analysis_visualizer.py`: 可视化分析

**约束条件支持**: 
- 精确度 (Precision) ≥ 0.8
- 召回率 (Recall) ≥ 0.8  
- F1分数 ≥ 0.7
- 测试样本数 ≥ 30

**输出内容**:
- 完整结果CSV (所有参数组合性能)
- 5种推荐策略 (最佳F1、高精度、高召回、平衡、鲁棒)
- 多种可视化图表 (3D散点图、热图、敏感性分析、Pareto前沿)
- 详细分析报告 (Markdown格式)

**使用场景**: 
- 答辩展示参数选择的科学性
- 不同应用场景的参数配置
- 模型性能的全面分析

**详细文档**: 参见 `PARAMETER_OPTIMIZATION_GUIDE.md`

## Validation & Test Dataset Guide ⭐

### 📊 **核心Validation/Test文件**

**主要验证代码:**
- `src/evaluation/latest/validation_focused_evaluation.py` ⭐ - **最重要的验证脚本** (时间分割，避免泄露)
- `src/evaluation/latest/pilot_conservative_evaluation.py` - 保守策略验证
- `src/evaluation/test_validation_script.py` - 验证脚本测试

**测试集评估系统:**
- `experiments/parameter_tuning/evaluate_top_configs_on_test_set.py` ⭐ - **最佳配置测试集评估** (推荐)
- `experiments/parameter_tuning/evaluate_specified_params_on_test.py` - 指定参数测试
- `experiments/parameter_tuning/evaluate_focused_flexible_params.py` - 灵活参数测试
- `experiments/parameter_tuning/evaluate_flexible_params_on_test.py` - 全测试集评估

### 📁 **关键结果位置**

**最重要的结果文件夹:**
1. `results/latest/validation_focused_results_20250714_190013/` ⭐ - **最完整验证结果**
2. `results/parameter_optimization_[timestamp]/test_set_evaluation/` - **当前测试集评估**
3. `results/parameter_optimization_[timestamp]/focused_flexible_test_evaluation/` - 灵活测试评估

**结果文件类型:**
- `*_test_results.csv` - 测试集性能指标
- `TEST_*.md` - 测试评估报告 
- `experiment_config.json` - 实验配置
- `performance_summary.json` - 性能摘要

### 🔄 **Validation vs Test工作流程**

**验证阶段 (Validation):**
```bash
# 1. 运行时间分割验证 (避免数据泄露)
python src/evaluation/latest/validation_focused_evaluation.py

# 2. 快速验证测试
python src/evaluation/latest/pilot_conservative_evaluation.py
```

**测试阶段 (Test):**
```bash
# 1. 评估最佳配置 (推荐)
python experiments/parameter_tuning/evaluate_top_configs_on_test_set.py

# 2. 评估指定参数
python experiments/parameter_tuning/evaluate_specified_params_on_test.py

# 3. 灵活参数测试
python experiments/parameter_tuning/evaluate_focused_flexible_params.py
```

### ⚠️ **数据分割注意事项**

**推荐: 时间分割** (src/evaluation/latest/validation_focused_evaluation.py)
- ✅ 训练集: 2015-2021  
- ✅ 测试集: 2022-2024
- ✅ 避免同日洪水事件泄露

**避免: 随机分割** (src/models/main.py)
- ❌ 可能存在时间信息泄露
- ❌ 同日洪水事件可能被分到训练和测试集

### 📈 **性能指标重点关注**

**约束条件筛选:**
- Precision ≥ 0.8 (高精度要求)
- Recall ≥ 0.8 (高召回要求)
- F1 Score ≥ 0.7 (平衡性能)
- Test Samples ≥ 30 (统计可靠性)

## Defence Experiment (Sept 2025) 🎓

### 📍 **实验位置**
`experiments/2025_validation/2025 0822 reliable bayes test _ for final defence/`

### 🎯 **实验目的**
为2025年9月12日答辩准备的实时累积洪水预测演示系统

### ⭐ **核心特性**
- **纯Python实现**: 无pandas依赖，简化部署
- **实时累积预测**: 每个时间窗口累积之前所有证据进行推理
- **10分钟时间窗口**: 模拟实时洪水监测场景
- **可靠贝叶斯网络**: 27节点网络 (occ_thr=5, edge_thr=3, weight_thr=0.4)
- **2025年真实数据**: 测试最新洪水事件数据

### 📁 **文件结构**
```
2025 0822 reliable bayes test _ for final defence/
├── pure_python_prediction.py              # 主脚本 (340行)
└── realtime_window_*.json                 # 16个JSON结果文件
    ├── realtime_window_01_*_204233.json   # 第1次运行 (9个窗口)
    └── realtime_window_01_*_204753.json   # 第2次运行 (9个窗口)
```

### 🚀 **使用方法**
```bash
# 进入答辩实验目录
cd "experiments/2025_validation/2025 0822 reliable bayes test _ for final defence"

# 运行实时预测脚本
python pure_python_prediction.py

# 输出: 9个JSON文件 (对应9个10分钟时间窗口)
# - Window 1: 12:19-12:29 PM
# - Window 2: 12:29-12:39 PM
# ...
# - Window 9: 13:39-13:49 PM
```

### 📊 **JSON结果文件结构**
每个JSON文件包含:
```json
{
  "experiment_metadata": {
    "experiment_name": "Real-Time Cumulative Flood Prediction (Pure Python)",
    "timestamp": "2025-09-06 20:42:33",
    "description": "Using 27-node reliable network...",
    "random_seed": 42
  },
  "training_data_info": {
    "total_records": 923,
    "unique_streets": 27,
    "data_source": "Road_Closures_2024.csv"
  },
  "bayesian_network": {
    "parameters": {"occ_thr": 5, "edge_thr": 3, "weight_thr": 0.4},
    "statistics": {"total_nodes": 27, "total_edges": 89},
    "all_nodes": ["AMERICA_ST", "ASHLEY_AVE", ...]
  },
  "current_window": {
    "window_id": 1,
    "window_label": "12:19-12:29",
    "evidence": {
      "cumulative_evidence_roads": [...],  # 累积证据道路
      "network_evidence_count": 5
    },
    "predictions": [
      {"road": "AMERICA_ST", "probability": 0.65, "is_evidence": false},
      ...
    ],
    "summary_stats": {
      "average_prediction_probability": 0.423,
      "high_risk_roads_count": 12
    }
  }
}
```

### 🔑 **关键技术点**
1. **累积证据机制**: 每个窗口保留之前所有观测到的淹水道路作为证据
2. **贝叶斯推理**: 基于累积证据计算未观测道路的淹水概率
3. **无依赖设计**: 仅使用Python标准库 (json, csv, datetime, collections)
4. **简化网络**: 使用SimpleBayesianNetwork类替代pgmpy，降低复杂度

### 📈 **实验数据**
- **训练数据**: Road_Closures_2024.csv (923条记录, 2015-2024)
- **测试数据**: archive/old_results/2025_flood_processed.csv
- **网络规模**: 27个关键道路节点, 89条边
- **时间跨度**: 约1.5小时 (12:19 PM - 13:49 PM)
- **运行次数**: 2次 (验证结果一致性)

### 💡 **答辩展示要点**
1. ✅ 展示纯Python实现的简洁性和可部署性
2. ✅ 强调实时累积预测的实用价值
3. ✅ 说明10分钟时间窗口适合实时监测
4. ✅ 展示JSON结果的结构化和可读性
5. ✅ 验证多次运行结果的一致性 (随机种子=42)

## Development Tips

1. **参数优化优先**: 使用新的参数优化系统找到最佳配置
2. **优先使用最新脚本**: `src/evaluation/latest/`中的脚本代表最新最佳实践
3. **关注最新结果**: 重点分析`results/latest/`中的输出
4. **测试驱动**: 使用`src/evaluation/`中的脚本验证功能正常
5. **可视化优先**: 使用`src/visualization/`工具理解模型行为
6. **约束筛选**: 根据应用需求设置合适的性能约束条件
7. **时间分割验证**: 优先使用`src/evaluation/latest/validation_focused_evaluation.py`避免数据泄露
8. **测试集评估**: 使用`experiments/parameter_tuning/evaluate_top_configs_on_test_set.py`获得可靠的测试性能

## Summary

这是一个**清洁、轻便**的Charleston洪水预测贝叶斯网络项目:
- ✅ **单一源码目录** (`src/`) - 避免重复和混乱
- ✅ **Git友好** - 实体文件而非符号链接
- ✅ **功能完整** - 参数优化、评估验证、2025数据测试、答辩实验
- ✅ **文档齐全** - 清晰的使用指南和最佳实践
- ✅ **可复现** - 严格的时间分割和随机种子控制
- ✅ **答辩就绪** - 纯Python实时预测演示系统 (Sept 2025)

**核心命令速查**:
```bash
python src/models/main.py                                         # 基础网络
python src/analysis/run_parameter_optimization.py                # 参数优化
python src/evaluation/latest/validation_focused_evaluation.py    # 主评估
python experiments/parameter_tuning/evaluate_top_configs_on_test_set.py  # 测试集评估

# 答辩实验 (Sept 2025)
cd "experiments/2025_validation/2025 0822 reliable bayes test _ for final defence"
python pure_python_prediction.py                                 # 实时预测演示
```