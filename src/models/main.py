import random
import numpy as np
random.seed(0)
np.random.seed(0)
import pandas as pd
from model import FloodBayesNetwork
from sklearn.model_selection import train_test_split
from visualization import visualize_flood_network
import sys
import os
from itertools import product


# Step 1: Load and preprocess data
df = pd.read_csv("Road_Closures_2024.csv")
df = df[df["REASON"].str.upper() == "FLOOD"].copy()

# 将 START 列转换为 datetime 类型，自动处理时区
df["time_create"] = pd.to_datetime(df["START"], utc=True)
df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
df["link_id"] = df["link_id"].astype(str)
df["id"] = df["OBJECTID"]
df["id"] = df["id"].astype(str)

# Step 2: Split into train (build network) and test (inference validation)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


# Step 3: Construct Bayesian Flood Network
flood_net = FloodBayesNetwork(t_window="D")
flood_net.fit_marginal(train_df)

# Compute edge stats and visualize distributions  看看数据分布， 之后可以注释
# 这样你可以在构建网络前观察 weight 分布是否偏小，从而更科学地设定 weight_thr 和 edge_thr
# edge_stats = flood_net.compute_all_edge_weights(train_df)
# visualize_edge_stats(edge_stats)
# sys.exit("🎯 All network variants saved. Terminate script.")

# Step 3: Construct Bayesian Flood Network

"""
# Grid search over parameters,  codes are below:


# ------------------------------------------------------------
# Parameter grids
occ_thr_list    = [2, 3]              # appearance threshold
edge_thr_list   = [2, 3]              # co-occurrence threshold
weight_thr_list = [0.2, 0.3, 0.4]        # conditional-probability threshold
# ------------------------------------------------------------
output_dir = "results_network_variation"
fig_dir    = os.path.join(output_dir, "figs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
summary = []

for occ_thr, edge_thr, weight_thr in product(occ_thr_list, edge_thr_list, weight_thr_list):
    tag       = f"occ{occ_thr}_edge{edge_thr}_w{weight_thr:.2f}"
    csv_path  = os.path.join(output_dir, f"network_{tag}.csv")
    fig_path  = os.path.join(fig_dir,  f"network_{tag}.png")

    # Build network
    net = FloodBayesNetwork(t_window="D")
    net.fit_marginal(train_df)
    net.build_network_by_co_occurrence(
        train_df,
        occ_thr=occ_thr,
        edge_thr=edge_thr,
        weight_thr=weight_thr,
        report=False,
        save_path=csv_path
    )

    # Visualize
    visualize_flood_network(net.network, save_path=fig_path)

    # Collect stats
    stats = {
        "tag": tag,
        "nodes": net.network.number_of_nodes(),
        "edges": net.network.number_of_edges(),
        "avg_in":   sum(dict(net.network.in_degree()).values()) / max(1, net.network.number_of_nodes()),
        "max_in":   max(dict(net.network.in_degree()).values(), default=0),
        "avg_out":  sum(dict(net.network.out_degree()).values()) / max(1, net.network.number_of_nodes()),
        "max_out":  max(dict(net.network.out_degree()).values(), default=0),
    }
    summary.append(stats)
    print(f"✅ Finished {tag}: {stats['nodes']} nodes, {stats['edges']} edges")

# Save summary table for quick comparison
import pandas as pd
pd.DataFrame(summary).to_csv(os.path.join(output_dir, "network_summary.csv"), index=False)

# Optional: stop script after generating variants
sys.exit("🎯 All network variants generated and saved. See results_network_variation/")

"""

# gridsearch result find best: occ_thr = 3, edge_thr = 3, weight_thr = 0.30 (tag = occ3_edge3_w0.30)
#  如果参数已经确定好， 可以使用的代码：

flood_net.build_network_by_co_occurrence(
    train_df,
    occ_thr=10,
    edge_thr=3,
    weight_thr=0.4,
    report=False,
    save_path="charleston_flood_network.csv"
)

visualize_flood_network(flood_net.network, save_path="figs/flood_network_refine.png")

flood_net.fit_conditional(train_df, max_parents=2, alpha=1)
# 对每个节点，统计它在其父节点特定状态下发生洪水的条件概率

# 停止程序运行
# sys.exit("Visualization complete. Stopping script.")

flood_net.build_bayes_network()
flood_net.check_bayesian_network()


# # Step 4: Run inference on test data (simulate real-time partial observation)
# # Example: randomly pick observed roads from test_df and infer another

# observed = test_df.sample(n=3, random_state=1)
# evidence = {row["link_id"]: 1 for _, row in observed.iterrows()}
#
# print("🧠 Evidence used:", evidence)
#
# # Pick one unobserved road to infer
# unobserved = test_df[~test_df["link_id"].isin(evidence.keys())]
# if not unobserved.empty:
#     target_node = unobserved.iloc[0]["link_id"]
#     result = flood_net.infer_w_evidence(target_node, evidence)
#     print(f"🔍 Prediction for {target_node}:", result)
# else:
#     print("No unobserved road found for inference.")

# ------------------------------------------------------------------
#  VALIDATION HELPERS  (keep them close to the end of main.py)
# ------------------------------------------------------------------
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    average_precision_score, accuracy_score, f1_score,
    recall_score, precision_score
)
import math, warnings
from collections import defaultdict

def _binary_entropy(p: float) -> float:
    if p in (0, 1):
        return 0.0
    return -(p*math.log2(p) + (1-p)*math.log2(1-p))

def evaluate_bn(flood_net, test_df, evidence_size=3, k=5, prob_thr=0.5):
    """
    One full pass over the test set.
    evidence_size : # roads we pretend to observe per day
    k             : top-K alarm list for Recall@k
    prob_thr      : probability threshold for hard 0/1 decisions
    """
    bn_nodes = set(flood_net.network_bayes.nodes())
    y_true, y_prob = [], []

    hits_at_k = total_at_k = 0
    h_clim, h_post = 0.0, 0.0             #  entropy

    # ----- iterate by day -----
    for day, group in test_df.groupby(test_df["time_create"].dt.floor("D")):
        flooded = [r for r in group["link_id"] if r in bn_nodes]

        # choose first N flooded roads as evidence (swap for random/heuristic)
        evidence = {r: 1 for r in flooded[:evidence_size]}

        # entropy baseline (marginal)
        for r in bn_nodes:
            p0 = float(flood_net.marginals.loc[
                       flood_net.marginals["link_id"] == r, "p"].values[0])
            h_clim += _binary_entropy(p0)

        # forecast each non-evidence road
        for r in bn_nodes:
            if r in evidence:
                continue
            p_flood = flood_net.infer_w_evidence(r, evidence)["flooded"]
            y_prob.append(p_flood)
            y_true.append(1 if r in flooded else 0)
            h_post += _binary_entropy(p_flood)

        # ---------- Recall@k ----------
        topk = sorted(
            ((r, flood_net.infer_w_evidence(r, evidence)['flooded'])
             for r in bn_nodes if r not in evidence),
            key=lambda x: x[1], reverse=True
        )[:k]

        hits_at_k += sum(1 for r, _ in topk if r in flooded)
        total_at_k += min(k, len(flooded))

    # -------- metrics ----------
    y_hat = (np.array(y_prob) >= prob_thr).astype(int)
    metrics = {
        "Brier"       : brier_score_loss(y_true, y_prob),
        "LogLoss"     : log_loss(y_true, y_prob),
        "ROC-AUC"     : roc_auc_score(y_true, y_prob),
        "PR-AUC"      : average_precision_score(y_true, y_prob),
        "Accuracy"    : accuracy_score(y_true, y_hat),
        "Precision": precision_score(y_true, y_hat, zero_division=0),  # ← NEW
        "F1"          : f1_score(y_true, y_hat),
        f"Recall@{k}" : hits_at_k / total_at_k if total_at_k else float("nan"),
        "ΔH(bits)"    : h_clim - h_post,
    }
    #  Brier-skill
    clim = sum(y_true)/len(y_true) if y_true else 0
    clim_bs = brier_score_loss(y_true, [clim]*len(y_true))
    metrics["BSS"] = 1 - metrics["Brier"]/clim_bs if clim_bs else float("nan")
    return metrics

# ------------------------------------------------------------------
#  VALIDATION EXPERIMENTS
# ------------------------------------------------------------------
print("\n=====  VALIDATION METRICS (baseline) =====")
baseline = evaluate_bn(flood_net, test_df)
# print(f"Precision : {baseline['Precision']:.4f}")
# print(f"F1        : {baseline['F1']:.4f}")
# print(f"Recall@5  : {baseline['Recall@5']:.4f}")
for key in ["Precision", "F1", f"Recall@5", "ROC-AUC", "PR-AUC", "Brier", "LogLoss", "ΔH(bits)", "BSS"]:
        print(f"{key:>10}: {baseline[key]:.4f}")
sys.exit("first try; Terminate script.")

# ---------- Threshold sweep ----------
print("\n--- Threshold sweep (Precision / F1 / Recall@5) ---")
for thr in np.linspace(0.5, 1.0, 5):
    m = evaluate_bn(flood_net, test_df, prob_thr=thr)
    print(f"thr={thr:.2f}  P={m['Precision']:.3f}  F1={m['F1']:.3f}  R@5={m['Recall@5']:.3f}")

# ---------- Parent / smoothing grid ----------
print("\n--- Parent-limit & Laplace α grid (Precision / F1 / Recall@5) ---")
grid_results = []
for k_parent in [1, 2, 3]:
    for alpha in [0.1, 1]:
        flood_net.fit_conditional(train_df, max_parents=k_parent, alpha=alpha)
        flood_net.build_bayes_network()
        m = evaluate_bn(flood_net, test_df)
        grid_results.append((k_parent, alpha, m["Precision"], m["F1"], m["Recall@5"]))
        print(f"k={k_parent}, α={alpha}  P={m['Precision']:.3f}  F1={m['F1']:.3f}  R@5={m['Recall@5']:.3f}")

best_cfg = max(grid_results, key=lambda x: x[4])
print(f"\n>>> Best config (Precision) = k={best_cfg[0]}, α={best_cfg[1]}  "
      f"P={best_cfg[2]:.3f}, F1={best_cfg[3]:.3f}, R@5={best_cfg[4]:.3f}")


# •	Run the above validation loop for each parameter combination in your grid search.
# •	Compare Brier / LogLoss / AUC.
# •	Select the network with the best predictive score and acceptable sparsity (edge count, max-degree).


# old version 1    codes for validation
# Validation loop skeleton；
# from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
# import random
#
# all_preds, all_true = [], []
# bn_nodes = set(flood_net.network_bayes.nodes())
#
# for t, grp in test_df.groupby('time_create'):
#     flooded_roads = list(grp['link_id'].unique())
#     evidence_roads = flooded_roads[:3]           # or any strategy
#     evidence = {r:1 for r in evidence_roads if r in bn_nodes}
#
#     # positive samples
#     for r in flooded_roads:
#         if r in evidence or r not in bn_nodes:
#             continue
#         all_preds.append(flood_net.infer_w_evidence(r, evidence)['flooded'])
#         all_true.append(1)
#
#     # negative samples
#     neg_pool = list(bn_nodes - set(flooded_roads))
#     sample_neg = random.sample(neg_pool, min(len(flooded_roads), len(neg_pool)))
#     for r in sample_neg:
#         all_preds.append(flood_net.infer_w_evidence(r, evidence)['flooded'])
#         all_true.append(0)
#
# print("Brier:", brier_score_loss(all_true, all_preds))
# print("LogLoss:", log_loss(all_true, all_preds))
# print("AUC:", roc_auc_score(all_true, all_preds))




# old version 2    codes for validation

# ----------------------------------------------------------------------
# # VALIDATION BLOCK  –– append *after* flood_net.check_bayesian_network()
# # ----------------------------------------------------------------------
# from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score,
#                              average_precision_score, accuracy_score,
#                              f1_score)
# import math, warnings
# from collections import defaultdict
#
# def entropy(p):
#     """Binary entropy in bits; p is flood-probability."""
#     if p in (0, 1):
#         return 0.0
#     return -(p*math.log2(p) + (1-p)*math.log2(1-p))
#
# def evaluate_bn(flood_net, test_df, evidence_size=3, k=5):
#     bn_nodes = set(flood_net.network_bayes.nodes())
#     y_true, y_prob = [], []
#
#     # keep a few extra tallies ↓
#     hits_at_k = total_at_k = 0
#     entropy_baseline = entropy_model = 0
#
#     # --- iterate over each day in the test set ---
#     for day, grp in test_df.groupby(test_df["time_create"].dt.floor("D")):
#         flooded = list(grp["link_id"].unique())   # 今天所有淹水路段
#         flooded_in_bn = [r for r in flooded if r in bn_nodes] # 只保留出现在网络中的路段
#
#         # choose evidence (first N flooded roads, or sample)
#         evidence_roads = flooded_in_bn[:evidence_size]
#         evidence = {r: 1 for r in evidence_roads}
#
#         # baseline: entropy without evidence (marginals)
#         for r in bn_nodes:
#             p0 = float(
#                 flood_net.marginals.loc[
#                     flood_net.marginals["link_id"] == r, "p"
#                 ].values[0]
#             )
#             entropy_baseline += entropy(p0)
#
#         # forecast every other road
#         for r in bn_nodes:
#             if r in evidence:
#                 continue
#             prob_flood = flood_net.infer_w_evidence(r, evidence)["flooded"]
#
#             y_prob.append(prob_flood)
#             y_true.append(1 if r in flooded else 0)
#
#             entropy_model += entropy(prob_flood)
#
#         # Recall@k -----------------
#         topk = sorted(
#             [(r, flood_net.infer_w_evidence(r, evidence)["flooded"])
#              for r in bn_nodes if r not in evidence],
#             key=lambda x: x[1],
#             reverse=True
#         )[:k]
#         hits_at_k += sum(1 for r, _ in topk if r in flooded)
#         total_at_k += min(k, len(flooded))
#
#     # --------- METRICS ----------
#     y_hat = [1 if p >= 0.5 else 0 for p in y_prob]
#     metrics = {
#         "Brier"        : brier_score_loss(y_true, y_prob),
#         "LogLoss"      : log_loss(y_true, y_prob),
#         "ROC-AUC"      : roc_auc_score(y_true, y_prob),
#         "PR-AUC"       : average_precision_score(y_true, y_prob),
#         "Accuracy"     : accuracy_score(y_true, y_hat),
#         "F1"           : f1_score(y_true, y_hat),
#         f"Recall@{k}"  : hits_at_k / total_at_k if total_at_k else float("nan"),
#         "ΔH (bits)"    : entropy_baseline - entropy_model,
#     }
#
#     # climatology Brier (skill score)
#     climatology = sum(y_true)/len(y_true) if y_true else 0
#     climatology_bs = brier_score_loss(
#         y_true, [climatology]*len(y_true)
#     )
#     metrics["BSS"] = 1 - metrics["Brier"]/climatology_bs if climatology_bs else float("nan")
#     return metrics
#
#
# # ----------------------------------------------------------------------
# # run validation
# # ----------------------------------------------------------------------
# results = evaluate_bn(flood_net, test_df,
#                       evidence_size=3,   # how many roads “observed” each day
#                       k=5)              # size for Recall@k
# print("\n=====  VALIDATION METRICS  =====")
# for m,v in results.items():
#     print(f"{m:>10}: {v:.4f}")
# # ----------------------------------------------------------------------



