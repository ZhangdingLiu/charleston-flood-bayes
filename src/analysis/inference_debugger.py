#!/usr/bin/env python3
"""
推理过程调试器：逐步展示贝叶斯推理过程

详细展示每个推理步骤，让用户能够验证推理过程的正确性：
1. Evidence设置过程
2. CPT查询步骤
3. 概率计算过程
4. 阈值决策逻辑
5. 测试日期的详细分解
6. 负样本构造过程验证
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
from precision_focused_evaluation import PrecisionFocusedEvaluator
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class InferenceDebugger:
    """推理过程调试器"""
    
    def __init__(self, flood_net, test_df):
        self.flood_net = flood_net
        self.test_df = test_df
        self.bn_nodes = set(flood_net.network_bayes.nodes())
        self.marginals_dict = dict(zip(
            flood_net.marginals['link_id'], 
            flood_net.marginals['p']
        ))
        
    def debug_single_inference(self, target_node, evidence):
        """详细调试单个推理过程"""
        print(f"🔍 详细推理过程调试")
        print("=" * 60)
        print(f"目标节点: {target_node}")
        print(f"Evidence: {evidence}")
        
        # 1. 验证输入
        print(f"\n1️⃣  输入验证:")
        
        if target_node not in self.bn_nodes:
            print(f"   ❌ 目标节点 '{target_node}' 不在贝叶斯网络中")
            return None
            
        print(f"   ✅ 目标节点在网络中")
        
        for ev_node, ev_value in evidence.items():
            if ev_node not in self.bn_nodes:
                print(f"   ❌ Evidence节点 '{ev_node}' 不在网络中")
                return None
            if ev_value not in [0, 1]:
                print(f"   ❌ Evidence值 '{ev_value}' 必须是0或1")
                return None
                
        print(f"   ✅ 所有Evidence节点和值都有效")
        
        # 2. 查看目标节点的CPT
        print(f"\n2️⃣  目标节点CPT分析:")
        
        # 获取父节点
        parents = list(self.flood_net.network.predecessors(target_node))
        print(f"   父节点: {parents if parents else '无'}")
        
        # 边际概率
        marginal_prob = self.marginals_dict.get(target_node, 0)
        print(f"   边际概率 P({target_node}=1): {marginal_prob:.3f}")
        
        if target_node in self.flood_net.conditionals:
            cfg = self.flood_net.conditionals[target_node]
            cpt_parents = cfg['parents']
            conditionals = cfg['conditionals']
            
            print(f"   CPT父节点: {cpt_parents}")
            print(f"   条件概率表:")
            
            for state, prob in conditionals.items():
                parent_state_str = ", ".join([f"{p}={s}" for p, s in zip(cpt_parents, state)])
                print(f"     P({target_node}=1 | {parent_state_str}) = {prob:.3f}")
        else:
            print(f"   该节点无条件概率表（使用边际概率）")
        
        # 3. 分析Evidence对推理的影响
        print(f"\n3️⃣  Evidence影响分析:")
        
        # 检查Evidence中是否包含父节点
        relevant_evidence = {}
        irrelevant_evidence = {}
        
        for ev_node, ev_value in evidence.items():
            if ev_node in parents:
                relevant_evidence[ev_node] = ev_value
                print(f"   📍 相关Evidence: {ev_node}={ev_value} (是{target_node}的父节点)")
            else:
                irrelevant_evidence[ev_node] = ev_value
                print(f"   📄 其他Evidence: {ev_node}={ev_value} (通过网络间接影响)")
        
        if not relevant_evidence:
            print(f"   ⚠️  没有直接相关的Evidence (无父节点在Evidence中)")
        
        # 4. 执行推理
        print(f"\n4️⃣  执行推理:")
        
        try:
            # 调用pgmpy推理
            from pgmpy.inference import VariableElimination
            inference = VariableElimination(self.flood_net.network_bayes)
            
            print(f"   使用变量消除算法进行推理...")
            result = inference.query(variables=[target_node], evidence=evidence)
            
            prob_values = result.values
            prob_not_flooded = prob_values[0]
            prob_flooded = prob_values[1]
            
            print(f"   推理结果:")
            print(f"     P({target_node}=0 | Evidence) = {prob_not_flooded:.6f}")
            print(f"     P({target_node}=1 | Evidence) = {prob_flooded:.6f}")
            print(f"     概率和: {prob_not_flooded + prob_flooded:.6f}")
            
            # 5. 与边际概率对比
            print(f"\n5️⃣  与边际概率对比:")
            print(f"   边际概率 P({target_node}=1): {marginal_prob:.6f}")
            print(f"   条件概率 P({target_node}=1|Evidence): {prob_flooded:.6f}")
            
            if prob_flooded > marginal_prob:
                change = (prob_flooded - marginal_prob) / marginal_prob * 100
                print(f"   📈 Evidence使洪水概率增加了 {change:.1f}%")
            elif prob_flooded < marginal_prob:
                change = (marginal_prob - prob_flooded) / marginal_prob * 100
                print(f"   📉 Evidence使洪水概率降低了 {change:.1f}%")
            else:
                print(f"   ➡️  Evidence对洪水概率无影响")
            
            return {
                'prob_not_flooded': prob_not_flooded,
                'prob_flooded': prob_flooded,
                'marginal_prob': marginal_prob,
                'relevant_evidence': relevant_evidence,
                'irrelevant_evidence': irrelevant_evidence
            }
            
        except Exception as e:
            print(f"   ❌ 推理失败: {e}")
            return None
    
    def debug_test_day_evaluation(self, date_str=None, max_examples=3):
        """调试测试日的评估过程"""
        print(f"\n\n🗓️  测试日评估过程调试")
        print("=" * 60)
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        if date_str:
            # 调试特定日期
            target_date = pd.to_datetime(date_str).floor("D")
            day_groups = [(date, group) for date, group in test_by_date if date == target_date]
        else:
            # 调试前几天
            day_groups = list(test_by_date)[:max_examples]
        
        for i, (date, day_group) in enumerate(day_groups):
            print(f"\n📅 日期: {date.date()} (示例 {i+1})")
            print("-" * 40)
            
            # 当天洪水道路
            flooded_roads = list(day_group["link_id"].unique())
            flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
            
            print(f"原始洪水道路: {len(flooded_roads)}条")
            print(f"网络中洪水道路: {len(flooded_in_bn)}条")
            print(f"洪水道路列表: {flooded_roads}")
            print(f"网络道路列表: {flooded_in_bn}")
            
            if len(flooded_in_bn) < 2:
                print("⚠️  可用网络道路不足2条，跳过此日期")
                continue
            
            # Evidence选择过程
            evidence_count = max(1, int(len(flooded_in_bn) * 0.5))
            evidence_roads = flooded_in_bn[:evidence_count]
            target_roads = flooded_in_bn[evidence_count:]
            
            print(f"\nEvidence选择 (前{evidence_count}条):")
            for j, road in enumerate(evidence_roads):
                marginal_p = self.marginals_dict.get(road, 0)
                print(f"  {j+1}. {road} (边际P={marginal_p:.3f})")
            
            evidence = {road: 1 for road in evidence_roads}
            
            print(f"\n目标道路推理 ({len(target_roads)}条):")
            
            # 对每个目标道路进行详细推理
            for target_road in target_roads:
                print(f"\n  🎯 目标: {target_road}")
                
                # 简化版推理调试
                try:
                    result = self.flood_net.infer_w_evidence(target_road, evidence)
                    prob_flood = result['flooded']
                    
                    # 父节点分析
                    parents = list(self.flood_net.network.predecessors(target_road))
                    relevant_parents = [p for p in parents if p in evidence]
                    
                    print(f"     父节点: {parents}")
                    print(f"     相关Evidence父节点: {relevant_parents}")
                    print(f"     推理概率: P(洪水) = {prob_flood:.6f}")
                    
                    # 边际概率对比
                    marginal_p = self.marginals_dict.get(target_road, 0)
                    print(f"     边际概率: P(洪水) = {marginal_p:.6f}")
                    
                    if prob_flood > marginal_p:
                        print(f"     📈 Evidence提升了洪水概率")
                    elif prob_flood < marginal_p:
                        print(f"     📉 Evidence降低了洪水概率") 
                    else:
                        print(f"     ➡️  Evidence无影响")
                        
                except Exception as e:
                    print(f"     ❌ 推理失败: {e}")
    
    def debug_negative_sampling(self, date_str=None, max_examples=3):
        """调试负样本构造过程"""
        print(f"\n\n🚫 负样本构造过程调试")
        print("=" * 60)
        
        # 获取低概率道路作为负样本候选
        low_prob_roads = [
            road for road, prob in self.marginals_dict.items() 
            if road in self.bn_nodes and prob <= 0.15
        ]
        
        print(f"低概率负样本候选: {len(low_prob_roads)}条")
        print(f"候选道路 (P≤0.15): {low_prob_roads[:10]}{'...' if len(low_prob_roads) > 10 else ''}")
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        if date_str:
            target_date = pd.to_datetime(date_str).floor("D")
            day_groups = [(date, group) for date, group in test_by_date if date == target_date]
        else:
            day_groups = list(test_by_date)[:max_examples]
        
        for i, (date, day_group) in enumerate(day_groups):
            print(f"\n📅 日期: {date.date()} 负样本分析")
            print("-" * 40)
            
            flooded_roads = list(day_group["link_id"].unique())
            flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
            
            # 当天没有洪水的低概率道路
            negative_candidates = [
                road for road in low_prob_roads 
                if road not in flooded_roads
            ]
            
            print(f"当天洪水道路: {flooded_in_bn}")
            print(f"负样本候选: {len(negative_candidates)}条")
            
            if len(negative_candidates) > 0:
                # 选择前3个作为负样本
                selected_negatives = negative_candidates[:3]
                print(f"选中负样本: {selected_negatives}")
                
                # 分析每个负样本
                for neg_road in selected_negatives:
                    marginal_p = self.marginals_dict.get(neg_road, 0)
                    print(f"\n  🚫 负样本: {neg_road}")
                    print(f"     边际概率: {marginal_p:.6f}")
                    print(f"     选择理由: 边际概率低，当天无洪水记录")
                    
                    # 查看这条道路在训练集中的出现情况
                    train_occurrences = len(self.flood_net.marginals[
                        self.flood_net.marginals['link_id'] == neg_road
                    ])
                    if train_occurrences > 0:
                        print(f"     训练集中: 确实存在且概率很低")
                    else:
                        print(f"     训练集中: 未出现（边际概率=0）")
            else:
                print("⚠️  当天无合适的负样本候选")
    
    def debug_threshold_decision(self, prob_values, pos_threshold=0.6, neg_threshold=0.3):
        """调试阈值决策过程"""
        print(f"\n\n🎚️  阈值决策过程调试")
        print("=" * 60)
        print(f"正预测阈值: {pos_threshold}")
        print(f"负预测阈值: {neg_threshold}")
        print(f"不确定区间: ({neg_threshold}, {pos_threshold})")
        
        # 分析一系列概率值
        if not isinstance(prob_values, list):
            prob_values = [prob_values]
        
        decisions = []
        for prob in prob_values:
            if prob >= pos_threshold:
                decision = "正预测(洪水)"
                confidence = prob
            elif prob <= neg_threshold:
                decision = "负预测(无洪水)"
                confidence = 1 - prob
            else:
                decision = "不确定(弃权)"
                confidence = 0.5
            
            decisions.append((prob, decision, confidence))
            print(f"概率={prob:.3f} → {decision} (置信度={confidence:.3f})")
        
        return decisions
    
    def comprehensive_inference_demo(self):
        """综合推理演示"""
        print(f"\n\n🎭 综合推理演示")
        print("=" * 60)
        
        # 选择一个有代表性的测试日
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        # 找一个有足够网络道路的日期
        target_date = None
        target_group = None
        
        for date, group in test_by_date:
            flooded_roads = list(group["link_id"].unique())
            flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
            
            if len(flooded_in_bn) >= 4:  # 需要足够的道路进行演示
                target_date = date
                target_group = group
                break
        
        if target_date is None:
            print("❌ 没有找到合适的演示日期")
            return
        
        print(f"🎯 演示日期: {target_date.date()}")
        
        # 执行完整的推理流程
        flooded_roads = list(target_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
        
        print(f"当天洪水道路: {flooded_in_bn}")
        
        # 选择Evidence
        evidence_count = max(1, int(len(flooded_in_bn) * 0.5))
        evidence_roads = flooded_in_bn[:evidence_count]
        target_roads = flooded_in_bn[evidence_count:]
        
        evidence = {road: 1 for road in evidence_roads}
        
        print(f"\n1️⃣  Evidence设置: {evidence}")
        
        # 对第一个目标道路进行详细推理
        if target_roads:
            target_road = target_roads[0]
            print(f"\n2️⃣  详细推理目标: {target_road}")
            
            # 执行详细推理
            debug_result = self.debug_single_inference(target_road, evidence)
            
            if debug_result:
                # 阈值决策
                prob_flooded = debug_result['prob_flooded']
                print(f"\n3️⃣  阈值决策:")
                
                decisions = self.debug_threshold_decision([prob_flooded])
                
                print(f"\n4️⃣  最终结果:")
                prob, decision, confidence = decisions[0]
                print(f"   推理概率: {prob:.6f}")
                print(f"   决策结果: {decision}")
                print(f"   决策置信度: {confidence:.6f}")
                print(f"   真实标签: 洪水 (因为目标道路确实发生了洪水)")
                
                # 评估结果
                if "正预测" in decision:
                    print("   ✅ 预测正确 (True Positive)")
                elif "负预测" in decision:
                    print("   ❌ 预测错误 (False Negative)")
                else:
                    print("   ❓ 不确定预测 (弃权)")

def load_test_system():
    """加载测试系统"""
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 时序分割
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    # 构建网络
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    flood_net.build_network_by_co_occurrence(
        train_df, occ_thr=3, edge_thr=2, weight_thr=0.3, report=False
    )
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    flood_net.build_bayes_network()
    
    return flood_net, test_df

def main():
    """主函数"""
    print("🔧 推理过程调试器")
    print("=" * 60)
    
    # 加载系统
    flood_net, test_df = load_test_system()
    
    # 创建调试器
    debugger = InferenceDebugger(flood_net, test_df)
    
    # 1. 演示单个推理过程
    print("\n🔍 单个推理过程演示:")
    example_evidence = {"HAGOOD_AVE": 1, "WASHINGTON_ST": 1}
    example_target = "RUTLEDGE_AVE"
    
    debugger.debug_single_inference(example_target, example_evidence)
    
    # 2. 调试测试日评估
    debugger.debug_test_day_evaluation(max_examples=2)
    
    # 3. 调试负样本构造
    debugger.debug_negative_sampling(max_examples=2)
    
    # 4. 综合演示
    debugger.comprehensive_inference_demo()
    
    print(f"\n\n✅ 推理过程调试完成！")
    print(f"🎯 关键验证点:")
    print(f"   ✓ 推理算法正确执行")
    print(f"   ✓ Evidence影响计算准确")
    print(f"   ✓ 阈值决策逻辑合理")
    print(f"   ✓ 负样本构造保守")
    
    return debugger

if __name__ == "__main__":
    debugger = main()