import pandas as pd
import numpy as np
from model import FloodBayesNetwork
from sklearn.model_selection import train_test_split
import random
from collections import Counter

# 设置随机种子
random.seed(0)
np.random.seed(0)

def analyze_road_frequency():
    """分析道路频次分布来指导参数设置"""
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    
    train_df, _ = train_test_split(df, test_size=0.3, random_state=42)
    
    # 统计道路频次
    road_counts = Counter(train_df['link_id'])
    freq_dist = Counter(road_counts.values())
    
    print("道路频次分布分析:")
    print("频次\t道路数\t累计道路数")
    cumulative = 0
    for freq in sorted(freq_dist.keys(), reverse=True):
        cumulative += freq_dist[freq]
        print(f"{freq}\t{freq_dist[freq]}\t{cumulative}")
    
    # 推荐occ_thr设置
    print(f"\n推荐参数设置:")
    for target_nodes in [10, 15, 20, 25, 30]:
        for threshold in sorted(road_counts.values(), reverse=True):
            retained = sum(1 for count in road_counts.values() if count >= threshold)
            if retained <= target_nodes:
                print(f"目标{target_nodes}节点: occ_thr >= {threshold}")
                break
    
    return road_counts

def test_targeted_parameters(road_counts):
    """基于道路频次分析测试更精确的参数组合"""
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # 基于频次分析设计参数组合
    strategies = {
        "high_freq_conservative": {
            "occ_thr": 15,  # 只保留高频道路
            "edge_thr": 5,
            "weight_thr": 0.6,
            "description": "高频保守策略 - 只保留最频繁出现的道路"
        },
        "high_freq_moderate": {
            "occ_thr": 10,
            "edge_thr": 4,
            "weight_thr": 0.5,
            "description": "高频中等策略 - 保留较频繁的道路"
        },
        "medium_freq_conservative": {
            "occ_thr": 7,
            "edge_thr": 4,
            "weight_thr": 0.5,
            "description": "中频保守策略 - 平衡频次和网络大小"
        },
        "medium_freq_moderate": {
            "occ_thr": 5,
            "edge_thr": 3,
            "weight_thr": 0.4,
            "description": "中频中等策略"
        },
        "targeted_optimal": {
            "occ_thr": 8,  # 基于分析调整
            "edge_thr": 3,
            "weight_thr": 0.45,
            "description": "目标优化策略 - 针对10-30节点设计"
        }
    }
    
    results = {}
    
    for name, params in strategies.items():
        print(f"\n=== 测试 {name} ===")
        print(f"参数: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, weight_thr={params['weight_thr']}")
        
        try:
            # 预估节点数
            estimated_nodes = sum(1 for count in road_counts.values() if count >= params['occ_thr'])
            print(f"预估节点数: {estimated_nodes}")
            
            if estimated_nodes < 5:
                print("⚠️ 预估节点数太少，跳过")
                results[name] = {"success": False, "error": "too_few_nodes", "estimated_nodes": estimated_nodes}
                continue
            
            # 创建网络
            flood_net = FloodBayesNetwork(t_window="D")
            flood_net.fit_marginal(train_df)
            
            flood_net.build_network_by_co_occurrence(
                train_df,
                occ_thr=params["occ_thr"],
                edge_thr=params["edge_thr"],
                weight_thr=params["weight_thr"],
                report=False
            )
            
            # 网络统计
            n_nodes = flood_net.network.number_of_nodes()
            n_edges = flood_net.network.number_of_edges()
            
            if n_nodes == 0:
                print("❌ 网络为空")
                results[name] = {"success": False, "error": "empty_network"}
                continue
            
            # 度分布
            degrees = dict(flood_net.network.degree())
            in_degrees = dict(flood_net.network.in_degree())
            out_degrees = dict(flood_net.network.out_degree())
            
            # 边权重统计
            if n_edges > 0:
                weights = [flood_net.network[u][v]['weight'] for u, v in flood_net.network.edges()]
                weight_stats = {
                    "mean": np.mean(weights),
                    "std": np.std(weights),
                    "min": np.min(weights),
                    "max": np.max(weights),
                    "median": np.median(weights)
                }
            else:
                weight_stats = {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
            
            # 计算网络密度和连通性
            max_edges = n_nodes * (n_nodes - 1)
            density = n_edges / max_edges if max_edges > 0 else 0
            
            # 孤立节点
            isolated_nodes = sum(1 for node in flood_net.network.nodes() if degrees[node] == 0)
            
            results[name] = {
                "success": True,
                "params": params,
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_degree": np.mean(list(degrees.values())),
                "max_degree": max(degrees.values()) if degrees else 0,
                "isolated_nodes": isolated_nodes,
                "weight_stats": weight_stats,
                "in_target_range": 10 <= n_nodes <= 30
            }
            
            print(f"✅ 成功: {n_nodes}节点, {n_edges}边")
            print(f"   密度: {density:.4f}, 平均度: {np.mean(list(degrees.values())):.2f}")
            print(f"   权重: {weight_stats['mean']:.3f}±{weight_stats['std']:.3f}")
            print(f"   在目标范围内: {'是' if 10 <= n_nodes <= 30 else '否'}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    return results

def generate_optimization_report(results):
    """生成优化报告"""
    print("\n" + "="*80)
    print("目标化参数优化报告")
    print("="*80)
    
    # 成功的策略
    successful = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful:
        print("❌ 所有策略都失败了")
        return
    
    print("\n【成功策略概览】")
    for name, result in successful.items():
        in_range = "✅" if result["in_target_range"] else "❌"
        print(f"{name:25s}: {result['nodes']:2d}节点, {result['edges']:2d}边, "
              f"密度={result['density']:.4f} {in_range}")
    
    # 筛选目标范围内的策略
    target_strategies = {k: v for k, v in successful.items() if v["in_target_range"]}
    
    if target_strategies:
        print(f"\n【目标范围内策略(10-30节点)】")
        
        # 按不同指标排序
        best_balanced = min(target_strategies.keys(), 
                           key=lambda x: abs(target_strategies[x]["nodes"] - 20))
        best_connected = max(target_strategies.keys(), 
                            key=lambda x: target_strategies[x]["edges"])
        best_quality = max(target_strategies.keys(), 
                          key=lambda x: target_strategies[x]["weight_stats"]["mean"])
        
        print(f"🎯 最平衡策略: {best_balanced}")
        result = target_strategies[best_balanced]
        print(f"   - 网络规模: {result['nodes']}节点, {result['edges']}边")
        print(f"   - 参数: occ_thr={result['params']['occ_thr']}, "
              f"edge_thr={result['params']['edge_thr']}, "
              f"weight_thr={result['params']['weight_thr']}")
        
        if best_connected != best_balanced:
            print(f"🔗 最多连接策略: {best_connected} ({target_strategies[best_connected]['edges']}边)")
        
        if best_quality != best_balanced:
            print(f"⭐ 最高权重策略: {best_quality} "
                  f"(权重均值={target_strategies[best_quality]['weight_stats']['mean']:.3f})")
        
        # 推荐设置
        recommended = target_strategies[best_balanced]
        print(f"\n【推荐参数设置】")
        print(f"occ_thr = {recommended['params']['occ_thr']}")
        print(f"edge_thr = {recommended['params']['edge_thr']}")
        print(f"weight_thr = {recommended['params']['weight_thr']}")
        print(f"max_parents = 2  # 保持不变")
        
    else:
        print(f"\n⚠️ 没有策略在目标范围内，最接近的策略:")
        closest = min(successful.keys(), 
                     key=lambda x: abs(successful[x]["nodes"] - 20))
        result = successful[closest]
        print(f"   {closest}: {result['nodes']}节点")
        print(f"   建议进一步调整occ_thr参数")

def main():
    print("开始目标化参数优化...")
    
    # 分析道路频次
    road_counts = analyze_road_frequency()
    
    # 测试目标参数
    results = test_targeted_parameters(road_counts)
    
    # 生成报告
    generate_optimization_report(results)
    
    return results

if __name__ == "__main__":
    main()