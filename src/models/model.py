# -*- coding: utf-8 -*-
"""
Created on 5/28/2025 8:54 PM

@author: zliu952
"""

import numpy as np
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.mixture import GaussianMixture
import networkx as nx


class FloodBayesNetwork:
    def __init__(self, t_window: str = 'D'):    #, max_parents=4, laplace_alpha=1.0
        """
        :param t_window: the unit for processing flood data, default is day

        Purpose: Initializes the class with a time granularity for analysis ('D' = daily).

        Variables:
        self.network: Co-occurrence graph (road dependencies).
        self.marginals: Marginal flood probabilities per road.
        self.conditionals: Conditional probabilities given parent roads.
        self.network_bayes: Final Bayesian Network object (pgmpy).
        """
        self.t_window = t_window
        self.network = None
        self.marginals = None
        self.conditionals = None
        self.network_bayes = None
        # self.max_parents = max_parents  # NEW
        # self.alpha = laplace_alpha  # NEW

    def fit_marginal(self, df: pd.DataFrame, if_vis=False):
        """
        df should contain time_create, id, and link_id col
        time_create col is the start time of the road closure
        link_id col is the id of the road
        id col is the id of the closure

        Input: Historical road closure data.
        Output: Estimates marginal flood probability 𝑃(𝐹𝑖) for each road i.
        Bayesian Concept: Marginal probability distribution for prior knowledge.
        """
        from pandas.api.types import is_datetime64_ns_dtype
        from pandas.api.types import is_string_dtype
        assert 'time_create' in df.columns, 'Column time_create not found in the DataFrame'
        assert is_datetime64_ns_dtype(df['time_create']), 'Column time_create is not in type datetime64[ns]'
        assert 'link_id' in df.columns, 'Column link_id not found in the DataFrame'
        assert is_string_dtype(df['link_id']), 'Column link_id is not of string type'
        assert 'id' in df.columns, 'Column id not found in the DataFrame'
        assert is_string_dtype(df['id']), 'Column id is not of string type'

        # if if_vis:
        #     vis.bar_closure_count_over_time(df.copy())
        # pass

        df.loc[:, 'time_bin'] = df['time_create'].dt.floor(self.t_window)
        df = df.drop_duplicates(subset=['link_id', 'time_bin'])

        flooding_day_count = df['time_bin'].nunique()
        closure_count = df.groupby('link_id').count()['id']

        closure_count_p = closure_count / flooding_day_count
        closure_count_p = closure_count_p.reset_index().rename(columns={'id': 'p'})
        self.marginals = closure_count_p
        return

    def compute_all_edge_weights(self, df: pd.DataFrame):
        _, occurrence, co_occurrence = self.process_raw_flood_data(df.copy())
        edge_stats = []
        for (a, b), count in co_occurrence.items():
            weight = count / occurrence[a]
            edge_stats.append((a, b, weight, count, occurrence[a]))
        return edge_stats

    def build_network_by_co_occurrence(
            self,
            df: pd.DataFrame,
            weight_thr: float = 0.5,  # ① 条件概率下限
            edge_thr: int = 2,  # ② 共现次数下限
            occ_thr: int = 5,  # ③ 源路段出现次数下限
            report: bool = False,
            save_path: str = None
    ):
        """
        Builds a co-occurrence-based dependency graph with three filters:
        1. occ_thr   : keep A only if occurrence[A] >= occ_thr
        2. edge_thr  : keep edge A→B only if co_occurrence(A,B) >= edge_thr
        3. weight_thr: keep edge only if weight = co_occ / occurrence[A] >= weight_thr
        """

        # ---------- 预处理 ----------
        _, occurrence, co_occurrence = self.process_raw_flood_data(df.copy())

        # ★ 修复：只添加满足occ_thr条件的道路作为节点
        eligible_roads = [road for road, count in occurrence.items() if count >= occ_thr]
        
        graph = nx.DiGraph()
        graph.add_nodes_from(eligible_roads)

        # ---------- 添加节点属性 ----------
        for n in graph.nodes:
            graph.nodes[n]["occurrence"] = occurrence.get(n, 0)

        # ---------- 构建边并三重过滤 ----------
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

            # 通过三重过滤 → 加边
            graph.add_edge(
                a,
                b,
                weight=weight,
                count=count,
                occ_a=occurrence[a]
            )

        # ---------- 保证 DAG（去环） ----------
        graph, _ = self.remove_min_weight_feedback_arcs(graph)

        # ---------- 可选报告 ----------
        if report:
            for node, indeg in graph.in_degree():
                print(f"Node {node}: in-degree = {indeg}")

        # ---------- 保存 ----------
        self.network = graph

        if save_path:
            edge_df = pd.DataFrame([
                dict(source=u, target=v, weight=d["weight"], count=d["count"], occ_a=d["occ_a"])
                for u, v, d in graph.edges(data=True)
            ])
            edge_df.to_csv(save_path, index=False)
            print(f"✅ Network saved to {save_path}")
        return

    def fit_conditional(self, df: pd.DataFrame, max_parents: int = 3, alpha: int = 1):
        """
        Purpose: Calculates conditional probabilities 𝑃(𝐹𝑖∣parents(𝐹𝑖)) for each road node.
        Uses: Empirical frequency from co-flood events.
        Bayesian Concept: Conditional Probability Table (CPT) estimation.

        df should contain time_create and link_id col
        time_create col is the start time of the road closure
        link_id col is the id of the road

        zd add：
        Estimate P(F_i | parents(F_i)) with:
        - parent truncation (top-k by edge weight)
        - Laplace smoothing (alpha)
        """
        from collections import defaultdict
        from itertools import product

        time_groups, _, _ = self.process_raw_flood_data(df.copy())
        conditionals = defaultdict(dict)

        for node in self.network.nodes:
            parents_full = list(self.network.predecessors(node))
            if not parents_full:
                continue

            # ---- 1️⃣  按边权排序，截取前 k 个父节点 ----
            parents = sorted(parents_full,
                             key=lambda p: self.network[p][node]['weight'],
                             reverse=True)[:max_parents]

            # warn if CPT size too large ( > 256 rows )
            if len(parents) > 4:
                import warnings
                warnings.warn(
                    f"CPT for {node} has {len(parents)} rows; "
                    f"consider decreasing max_parents or raising thresholds."
                )

            parent_states = list(product([0, 1], repeat=len(parents)))
            counts = {s: {'co': 0, 'occ': 0} for s in parent_states}

            # ---- 2️⃣  遍历每天的洪水集合统计计数 ----
            for flood_set in time_groups:
                for state in parent_states:
                    if all(((p in flood_set) == bool(s)) for p, s in zip(parents, state)):
                        counts[state]['occ'] += 1
                        if node in flood_set:
                            counts[state]['co'] += 1

            # consistency sanity-check
            total_occ = sum(v['occ'] for v in counts.values())
            assert total_occ == len(time_groups), "Count inconsistent."

            # ---- 3️⃣  计算条件概率（拉普拉斯平滑） ----
            cond_probs = {}
            for state, val in counts.items():
                p_cond = (val['co'] + alpha) / (val['occ'] + 2 * alpha)
                cond_probs[state] = p_cond

            conditionals[node] = {
                'parents': parents,
                'conditionals': cond_probs
            }

        self.conditionals = conditionals

        # trim graph edges that exceed max_parents   剪枝：限制最大父节点数
        for child, cfg in conditionals.items():
            keep = set(cfg["parents"])
            for pred in list(self.network.predecessors(child)):
                if pred not in keep:
                    self.network.remove_edge(pred, child)
        return


    def build_bayes_network(self):
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from itertools import product

        # 1. Create network with edges, then add isolated nodes
        bn = DiscreteBayesianNetwork(list(self.network.edges()))
        bn.add_nodes_from(self.network.nodes())  # ← adds isolated nodes too

        for node in self.network.nodes:
            parents = list(self.network.predecessors(node))

            # --- case A: no parents OR node missing in self.conditionals
            if not parents or node not in self.conditionals:
                p_flood = float(
                    self.marginals.loc[self.marginals["link_id"] == node, "p"].values[0]
                )
                cpd = TabularCPD(node, 2, [[1 - p_flood], [p_flood]])
                bn.add_cpds(cpd)
                continue

            # --- case B: node with parents (already truncated)
            cfg = self.conditionals[node]
            parents = cfg["parents"]
            parent_states = list(product([0, 1], repeat=len(parents)))

            p1 = [cfg["conditionals"][st] for st in parent_states]
            p0 = [1 - p for p in p1]

            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                evidence=parents,
                evidence_card=[2] * len(parents),
                values=[p0, p1],
            )
            bn.add_cpds(cpd)

        # 2. Validate model consistency
        bn.check_model()

        self.network_bayes = bn
        return

    def check_bayesian_network(self):
        """
        Checks: Ensures marginal flood probability computed from the network matches the data-derived marginal.

        Check if the inferred p of flooding for nodes is consistent with the p calculated as marginals.
        """
        from pgmpy.inference import VariableElimination

        inference = VariableElimination(self.network_bayes)
        for node in self.network_bayes.nodes():
            if self.network_bayes.in_degree(node) == 0:
                continue
            p = inference.query(variables=[node]).values

            if self.marginals.loc[self.marginals['link_id'] == node, 'p'].values[0] != p[1]:
                import warnings
                warnings.warn(
                    """
                        Probability in Bayesian Network and fitted marginals should be the same 
                        before any observations. There could be small differences becasue of numerical issue.
                        Check that manually if they are not the same.
                    """
                )
                print(self.marginals.loc[self.marginals['link_id'] == node, 'p'].values[0], p[1])


    def infer_w_evidence(self, target_node: str, evidence: dict):
        """
        Purpose: Performs probabilistic inference using Variable Elimination.
        Bayesian Concept: Posterior inference:  𝑃(𝐹𝑖∣evidence).

        Get the probability of flooding for roads.
        Example input:
            target_node = 'A'
            evidence = {'B': 1, 'C': 0}, where 1 = flooded, 0 = not flooded
        """
        from pgmpy.inference import VariableElimination
        assert target_node in self.network_bayes.nodes, f"{target_node} is not in the network."
        assert isinstance(evidence, dict), "evidence must be a dictionary."
        for node, state in evidence.items():
            assert isinstance(node, str), f"Evidence key '{node}' must be a string."
            assert node in self.network_bayes.nodes, f"Evidence node '{node}' is not in the network."
            assert state in [0, 1], f"Evidence value for '{node}' must be 0 or 1, got {state}."

        inference = VariableElimination(self.network_bayes)
        result = inference.query(variables=[target_node], evidence=evidence)
        p = result.values
        return {'not_flooded': p[0], 'flooded': p[1]}

    def infer_node_states(self, node, node_value, thr_flood, thr_not_flood):
        """
        Given the observation of a node, get other nodes with a flooding probability above the threshold.
        """
        flooded_nodes = []
        not_flooded_nodes = []
        if node in self.network_bayes.nodes():
            for n in self.network_bayes.nodes():

                if self.network_bayes.in_degree(n) == 0:
                    continue
                if n == node:
                    continue

                p = self.infer_w_evidence(n, {node: node_value})
                if p['flooded'] >= thr_flood:
                    flooded_nodes.append(n)
                if p['not_flooded'] >= thr_not_flood:
                    not_flooded_nodes.append(n)

        assert not set(flooded_nodes) & set(not_flooded_nodes), """
        At least one road is regarded both flooded and not flooded
        """
        output = {i: 'flooded' for i in flooded_nodes}
        output.update({i: 'not flooded' for i in not_flooded_nodes})
        return output

    @staticmethod
    def remove_min_weight_feedback_arcs(graph: nx.DiGraph()):
        # Purpose: Ensures acyclic graph (requirement for Bayesian Networks).
        # Method: Removes weakest edge in feedback loops.
        graph = graph.copy()
        removed_edges = []

        while not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = next(nx.simple_cycles(graph))
            except StopIteration:
                break

            min_weight = float('inf')
            min_edge = None

            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                w = graph[u][v].get('weight',)
                if w < min_weight:
                    min_weight = w
                    min_edge = (u, v)

            graph.remove_edge(*min_edge)
            removed_edges.append(min_edge)

        return graph, removed_edges

    def process_raw_flood_data(self, df: pd.DataFrame):
        """
        Returns:
            Co - occurrence  counts.
            Time - binned groupings of flooded roads.
        Used in: Structure and CPT construction.
        """
        from pandas.api.types import is_datetime64_ns_dtype
        from pandas.api.types import is_string_dtype
        from collections import defaultdict
        from itertools import permutations

        assert 'time_create' in df.columns, 'Column time_create not found in the DataFrame'
        assert is_datetime64_ns_dtype(df['time_create']), 'Column time_create is not in type datetime64[ns]'
        assert 'link_id' in df.columns, 'Column link_id not found in the DataFrame'
        assert is_string_dtype(df['link_id']), 'Column link_id is not of string type'

        df.loc[:, 'time_bin'] = df['time_create'].dt.floor(self.t_window)
        time_groups = df.groupby('time_bin')['link_id'].apply(set)
        occurrence = defaultdict(int)
        co_occurrence = defaultdict(int)
        for links in time_groups:
            for a in links:
                occurrence[a] += 1
            for a, b in permutations(links, 2):
                co_occurrence[(a, b)] += 1

        return time_groups, occurrence, co_occurrence





