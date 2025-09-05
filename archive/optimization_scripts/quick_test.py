import pandas as pd
import numpy as np
from model import FloodBayesNetwork
from sklearn.model_selection import train_test_split
import random

# 设置随机种子
random.seed(0)
np.random.seed(0)

def quick_network_test():
    """快速测试网络构建和基本统计"""
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 数据预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 训练测试集分割
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    print(f"数据加载完成: 训练集{len(train_df)}, 测试集{len(test_df)}")
    
    # 测试三种策略
    strategies = {
        "conservative": {"occ_thr": 3, "edge_thr": 4, "weight_thr": 0.5},
        "moderate": {"occ_thr": 2, "edge_thr": 3, "weight_thr": 0.4},
        "relaxed": {"occ_thr": 2, "edge_thr": 3, "weight_thr": 0.3}
    }
    
    results = {}
    
    for name, params in strategies.items():
        print(f"\n=== 测试 {name} 策略 ===")
        try:
            # 创建网络
            flood_net = FloodBayesNetwork(t_window="D")
            flood_net.fit_marginal(train_df)
            
            # 构建网络
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
            
            # 度分布
            in_degrees = dict(flood_net.network.in_degree())
            out_degrees = dict(flood_net.network.out_degree())
            
            avg_in = np.mean(list(in_degrees.values())) if in_degrees else 0
            avg_out = np.mean(list(out_degrees.values())) if out_degrees else 0
            max_in = max(in_degrees.values()) if in_degrees else 0
            max_out = max(out_degrees.values()) if out_degrees else 0
            
            # 密度
            max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 0
            density = n_edges / max_edges if max_edges > 0 else 0
            
            # 边权重统计
            if n_edges > 0:
                weights = [flood_net.network[u][v]['weight'] for u, v in flood_net.network.edges()]
                weight_stats = {
                    "mean": np.mean(weights),
                    "std": np.std(weights),
                    "min": np.min(weights),
                    "max": np.max(weights)
                }
            else:
                weight_stats = {"mean": 0, "std": 0, "min": 0, "max": 0}
            
            results[name] = {
                "success": True,
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_in_degree": avg_in,
                "avg_out_degree": avg_out,
                "max_in_degree": max_in,
                "max_out_degree": max_out,
                "weight_stats": weight_stats
            }
            
            print(f"✅ 成功: {n_nodes}节点, {n_edges}边, 密度={density:.4f}")
            print(f"   平均入度: {avg_in:.2f}, 平均出度: {avg_out:.2f}")
            print(f"   权重: 均值={weight_stats['mean']:.3f}, 范围=[{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # 生成简化报告
    print("\n" + "="*60)
    print("参数策略比较结果")
    print("="*60)
    
    for name, result in results.items():
        if result["success"]:
            print(f"{name:12s}: {result['nodes']:3d}节点, {result['edges']:3d}边, "
                  f"密度={result['density']:.4f}, 权重均值={result['weight_stats']['mean']:.3f}")
        else:
            print(f"{name:12s}: 失败 - {result['error']}")
    
    # 推荐策略
    print(f"\n【推荐分析】")
    successful = {k: v for k, v in results.items() if v["success"]}
    
    if successful:
        # 筛选合适大小的网络 (10-30节点)
        suitable_size = {k: v for k, v in successful.items() if 10 <= v["nodes"] <= 30}
        
        if suitable_size:
            # 按边数排序 (更多连接可能意味着更丰富的依赖关系)
            best = max(suitable_size.keys(), key=lambda x: suitable_size[x]["edges"])
            print(f"✅ 推荐策略: {best}")
            print(f"   理由: 网络大小适中({suitable_size[best]['nodes']}节点)且连接丰富({suitable_size[best]['edges']}边)")
        else:
            print("⚠️ 没有策略产生理想网络大小(10-30节点)")
            # 找到最接近的
            closest = min(successful.keys(), key=lambda x: abs(successful[x]["nodes"] - 20))
            print(f"💡 最接近理想大小的策略: {closest} ({successful[closest]['nodes']}节点)")
    
    return results

if __name__ == "__main__":
    quick_network_test()