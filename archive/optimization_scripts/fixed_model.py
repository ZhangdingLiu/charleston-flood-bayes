import pandas as pd
import numpy as np
from model import FloodBayesNetwork
import networkx as nx

class FixedFloodBayesNetwork(FloodBayesNetwork):
    """修复版本的FloodBayesNetwork，正确处理occ_thr参数"""
    
    def build_network_by_co_occurrence(
            self,
            df: pd.DataFrame,
            weight_thr: float = 0.5,
            edge_thr: int = 2,
            occ_thr: int = 5,
            report: bool = False,
            save_path: str = None
    ):
        """
        修复版本的网络构建函数，正确过滤低频道路
        """
        # 预处理数据
        _, occurrence, co_occurrence = self.process_raw_flood_data(df.copy())
        
        # ★ 修复：只添加满足occ_thr条件的道路作为节点
        eligible_roads = [road for road, count in occurrence.items() if count >= occ_thr]
        
        graph = nx.DiGraph()
        graph.add_nodes_from(eligible_roads)  # 只添加合格的道路
        
        print(f"occ_thr={occ_thr}过滤后: {len(eligible_roads)}个节点")
        
        # 添加节点属性
        for n in graph.nodes:
            graph.nodes[n]["occurrence"] = occurrence.get(n, 0)
        
        # 构建边并过滤
        edges_added = 0
        for (a, b), count in co_occurrence.items():
            # 确保源节点和目标节点都在合格节点列表中
            if a not in eligible_roads or b not in eligible_roads:
                continue
                
            # 过滤 1：A 的出现次数足够多 (已经通过eligible_roads保证)
            if occurrence[a] < occ_thr:
                continue
            
            # 过滤 2：A 与 B 共现次数足够多
            if count < edge_thr:
                continue
            
            # 计算条件概率
            weight = count / occurrence[a]
            
            # 过滤 3：条件概率足够大
            if weight < weight_thr:
                continue
            
            # 通过所有过滤 → 加边
            graph.add_edge(
                a, b,
                weight=weight,
                count=count,
                occ_a=occurrence[a]
            )
            edges_added += 1
        
        print(f"添加了{edges_added}条边")
        
        # 去环处理
        if hasattr(self, 'remove_min_weight_feedback_arcs'):
            graph, _ = self.remove_min_weight_feedback_arcs(graph)
        
        # 保存网络
        self.network = graph
        
        if save_path:
            edge_df = pd.DataFrame([
                dict(source=u, target=v, weight=d["weight"], count=d["count"], occ_a=d["occ_a"])
                for u, v, d in graph.edges(data=True)
            ])
            edge_df.to_csv(save_path, index=False)
        
        return graph

def test_fixed_parameters():
    """测试修复后的参数"""
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # 测试修复后的参数组合
    strategies = {
        "ultra_conservative": {
            "occ_thr": 20,
            "edge_thr": 5,
            "weight_thr": 0.6,
            "description": "超保守 - 只保留最高频道路"
        },
        "high_selective": {
            "occ_thr": 15,
            "edge_thr": 4,
            "weight_thr": 0.5,
            "description": "高选择性 - 严格筛选"
        },
        "moderate_selective": {
            "occ_thr": 10,
            "edge_thr": 3,
            "weight_thr": 0.4,
            "description": "适度选择 - 平衡质量和规模"
        },
        "targeted_20_nodes": {
            "occ_thr": 12,
            "edge_thr": 3,
            "weight_thr": 0.45,
            "description": "目标20节点 - 针对理想规模优化"
        },
        "targeted_15_nodes": {
            "occ_thr": 15,
            "edge_thr": 3,
            "weight_thr": 0.4,
            "description": "目标15节点 - 小型高质量网络"
        }
    }
    
    results = {}
    
    for name, params in strategies.items():
        print(f"\n{'='*60}")
        print(f"测试策略: {name}")
        print(f"参数: {params}")
        print(f"{'='*60}")
        
        try:
            # 使用修复版本的模型
            flood_net = FixedFloodBayesNetwork(t_window="D")
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
            
            if n_nodes == 0:
                print("❌ 网络为空")
                results[name] = {"success": False, "error": "empty_network"}
                continue
            
            # 度分布统计
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
                    "max": np.max(weights)
                }
            else:
                weight_stats = {"mean": 0, "std": 0, "min": 0, "max": 0}
            
            # 网络密度
            max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 0
            density = n_edges / max_edges if max_edges > 0 else 0
            
            # 孤立节点
            isolated = sum(1 for node in flood_net.network.nodes() if degrees[node] == 0)
            
            results[name] = {
                "success": True,
                "params": params,
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_degree": np.mean(list(degrees.values())),
                "max_degree": max(degrees.values()) if degrees else 0,
                "isolated_nodes": isolated,
                "weight_stats": weight_stats,
                "in_target_range": 10 <= n_nodes <= 30
            }
            
            # 输出结果
            print(f"✅ 成功构建网络:")
            print(f"   节点数: {n_nodes}")
            print(f"   边数: {n_edges}")
            print(f"   密度: {density:.4f}")
            print(f"   平均度: {np.mean(list(degrees.values())):.2f}")
            print(f"   最大度: {max(degrees.values()) if degrees else 0}")
            print(f"   孤立节点: {isolated}")
            print(f"   权重: 均值={weight_stats['mean']:.3f}, 范围=[{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
            print(f"   在目标范围(10-30节点): {'✅ 是' if 10 <= n_nodes <= 30 else '❌ 否'}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # 生成总结报告
    print(f"\n{'='*80}")
    print("修复后参数优化总结报告")
    print(f"{'='*80}")
    
    successful = {k: v for k, v in results.items() if v.get("success", False)}
    target_range = {k: v for k, v in successful.items() if v["in_target_range"]}
    
    print(f"\n【所有成功策略】")
    for name, result in successful.items():
        status = "✅" if result["in_target_range"] else "❌"
        print(f"{name:20s}: {result['nodes']:2d}节点, {result['edges']:2d}边, 密度={result['density']:.4f} {status}")
    
    if target_range:
        print(f"\n【目标范围内策略 (10-30节点)】")
        
        # 找到最佳策略
        best_balanced = min(target_range.keys(), 
                           key=lambda x: abs(target_range[x]["nodes"] - 20))
        best_connected = max(target_range.keys(), 
                            key=lambda x: target_range[x]["edges"])
        
        print(f"🎯 推荐策略: {best_balanced}")
        best = target_range[best_balanced]
        print(f"   网络规模: {best['nodes']}节点, {best['edges']}边")
        print(f"   推荐参数: occ_thr={best['params']['occ_thr']}, "
              f"edge_thr={best['params']['edge_thr']}, weight_thr={best['params']['weight_thr']}")
        print(f"   网络质量: 密度={best['density']:.4f}, 权重均值={best['weight_stats']['mean']:.3f}")
        
        if best_connected != best_balanced:
            print(f"🔗 最多连接: {best_connected} ({target_range[best_connected]['edges']}边)")
    else:
        print(f"\n⚠️ 仍然没有策略在目标范围内")
        if successful:
            closest = min(successful.keys(), 
                         key=lambda x: abs(successful[x]["nodes"] - 20))
            print(f"   最接近的策略: {closest} ({successful[closest]['nodes']}节点)")
    
    return results

if __name__ == "__main__":
    test_fixed_parameters()