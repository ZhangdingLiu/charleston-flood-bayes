# Charleston洪水数据贝叶斯网络参数优化总结

## 项目概述
基于Charleston洪水数据的质量和特征分析，优化贝叶斯网络参数以处理数据稀疏性问题。

## 数据分析发现

### 数据稀疏性问题
- **总数据量**: 923条洪水记录，涉及157条道路
- **严重稀疏性**: 54.1%的道路只出现1次
- **时间跨度**: 2015-2024年，143个洪水日
- **连接密度**: 仅20.17%的可能道路对有实际共现关系

### 道路频次分布
```
频次    道路数    累计道路数
>=20      6        6      (高频核心道路)
>=15      9        9      (频繁道路)  
>=10     18       18      (中高频道路)
>=5      32       32      (中频道路)
>=2      54       54      (有效道路)
=1       63      117      (稀疏道路)
```

### 高频洪水道路(TOP 10)
1. HAGOOD AVE (66次)
2. WASHINGTON ST (61次) 
3. ASHLEY AVE (59次)
4. S MARKET ST (50次)
5. CALHOUN ST (47次)
6. RUTLEDGE AVE (34次)
7. FISHBURNE ST (31次)
8. PRESIDENT ST (27次)
9. AMERICA ST (22次)
10. GADSDEN ST (22次)

## 技术问题修复

### 发现的Bug
在原始`model.py`中发现关键bug：
```python
# 问题代码：
graph.add_nodes_from(df["link_id"].unique())  # 添加所有道路作为节点

# 修复代码：
eligible_roads = [road for road, count in occurrence.items() if count >= occ_thr]
graph.add_nodes_from(eligible_roads)  # 只添加满足频次阈值的道路
```

这个bug导致无论`occ_thr`设置多高，网络都包含所有117个道路节点。

## 参数优化结果

### 测试的策略组合
1. **Ultra Conservative**: occ_thr=20, edge_thr=5, weight_thr=0.6
   - 结果: 6节点, 0边 (太严格)

2. **High Selective**: occ_thr=15, edge_thr=4, weight_thr=0.5  
   - 结果: 9节点, 4边 (偏小)

3. **Moderate Selective**: occ_thr=10, edge_thr=3, weight_thr=0.4
   - 结果: 18节点, 33边 ✅ **推荐策略**

4. **Targeted 20 Nodes**: occ_thr=12, edge_thr=3, weight_thr=0.45
   - 结果: 15节点, 18边 ✅ 可选方案

5. **Targeted 15 Nodes**: occ_thr=15, edge_thr=3, weight_thr=0.4
   - 结果: 9节点, 10边 (太小)

### 最优参数设置 ⭐

```python
# 推荐的最优参数组合
occ_thr = 10      # 道路出现频次阈值
edge_thr = 3      # 共现次数阈值  
weight_thr = 0.4  # 条件概率阈值
max_parents = 2   # 最大父节点数(保持不变)
```

### 网络质量指标
- **节点数**: 18 (在理想范围10-30内)
- **边数**: 33 (适度连接)
- **网络密度**: 0.1078 (适中)
- **平均度**: 3.67 (良好连通性)
- **权重均值**: 0.491 (合理的条件概率)
- **孤立节点**: 1个 (最小化)

## evaluate_bn函数优化

### 稀疏数据处理改进
1. **Evidence选择策略**: 优先选择频次≥2的道路作为evidence
2. **错误处理**: 增加推理失败的异常处理
3. **样本平衡**: 记录正负样本比例
4. **性能指标**: 增加样本数量和正样本比例统计

### 新增评估指标
- **Samples**: 评估样本数量
- **Pos_ratio**: 正样本比例
- **BSS**: Brier技能分数
- **ΔH(bits)**: 信息熵改善

## 文件清单

### 核心脚本
- `data_diagnosis.py`: 数据质量和特征分析
- `parameter_optimization.py`: 完整参数优化(包含性能评估)
- `quick_test.py`: 快速网络构建测试
- `targeted_optimization.py`: 针对性参数调优
- `fixed_model.py`: 修复版本的模型类

### 修复的文件
- `main.py`: 更新为使用最优参数 (occ_thr=10)
- `model.py`: 修复了occ_thr参数的bug

### 输出文件
- `flood_analysis_results.json`: 数据分析结果
- `parameter_optimization_results.json`: 参数优化结果
- `parameter_optimization_report.txt`: 详细优化报告

## 主要改进

### 1. 数据理解
- 深入分析了道路频次分布
- 识别了数据稀疏性的严重程度
- 发现了高频核心道路群体

### 2. 技术修复
- 修复了关键的节点过滤bug
- 确保occ_thr参数正确工作
- 优化了边的添加逻辑

### 3. 参数调优
- 从原始的occ_thr=10→2, edge_thr=3→3, weight_thr=0.5→0.4
- 网络规模从117节点降低到18节点(符合理想范围)
- 保持了良好的连接性和权重质量

### 4. 性能优化
- 改进了evaluate_bn函数处理稀疏数据
- 增加了错误处理和性能诊断
- 优化了evidence选择策略

## 使用建议

### 立即可用的设置
```python
# 在main.py中使用
flood_net.build_network_by_co_occurrence(
    train_df,
    occ_thr=10,
    edge_thr=3, 
    weight_thr=0.4,
    report=False,
    save_path="charleston_flood_network.csv"
)
```

### 进一步实验
如果需要调整网络大小：
- **更小网络**: 增加occ_thr到12-15
- **更大网络**: 降低occ_thr到7-8
- **更严格连接**: 增加weight_thr到0.5
- **更多连接**: 降低edge_thr到2

## 预期效果

使用优化后的参数，贝叶斯网络将具有：
1. **合适的规模**: 18个核心洪水易发道路
2. **良好的连通性**: 平均每个节点有3.67个连接
3. **高质量的依赖关系**: 条件概率均值0.491
4. **实用的推理能力**: 适合实时洪水预警应用

这个优化解决了原始数据的稀疏性问题，创建了一个既精简又有效的贝叶斯网络模型。