# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import scipy.io as sio

import os
BASE_DIR = Path(__file__).resolve().parent
# File paths
# Input data files are in the 'data/' folder
# Output files will be saved to the 'results/' folder

# Configuration
# K_VALUE = 1           # Cluster number (1-8)

# Constants
GENE_SIZE = 100       # Feature dimension per node (t_wavelet-all-L has 500 columns)
OUTPUT_DIM = 200      # Output dimension  
HIDDEN_DIM = 300      # Hidden layer dimension
DROPOUT = 0.3         # Dropout rate
ROUNDS = 3            # Number of runs
EPOCHS = 1000         # Training epochs
LEARNING_RATE = 0.001 # Learning rate
WEIGHT_DECAY = 5e-6   # Weight decay
#WEIGHT_DECAY = 1e-5   # Weight decay
THRESHOLD_CANDIDATES = [0.4, 0.6, 0.8]  # Synchronous thresholds: prior==bin
REPORT_BIN_THRESHOLD = 0.6

# Composite score weights (sum to 1.0)
W_STABILITY = 0.20
W_DENSITY_MATCH = 0.35
W_CONNECTIVITY = 0.25
W_NON_ISOLATED = 0.20

# Load data from .mat (use t_data directly, not xlsx)
data_path = BASE_DIR / 't_wavelet-all-R100列.mat'
mat = sio.loadmat(data_path)
if 't_data' not in mat:
    raise KeyError(f"'t_data' not found in {data_path}")
node_features = np.asarray(mat['t_data'], dtype=np.float32)
scaler = StandardScaler()


node_features = scaler.fit_transform(node_features)  # Normalize

# Normalize adjacency matrix
def normalize_adj(adj):
    """Symmetric normalization of adjacency matrix"""
    adj = adj + np.eye(adj.shape[0])  # Add self-connections
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt)

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_tensor = torch.FloatTensor(node_features).to(device)
n_nodes = node_features.shape[0]
possible_edges = n_nodes * (n_nodes - 1) // 2

# GCN layer
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        if adj.is_sparse:
            support = torch.matmul(x, self.weight)
            output = torch.sparse.mm(adj, support)
        else:
            support = torch.mm(x, self.weight)
            output = torch.mm(adj, support)
        return output + self.bias

# GCN model
class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_output_dim, dropout=DROPOUT):
        super().__init__()
        self.gc1 = GraphConv(n_feat, n_hid)
        self.gc2 = GraphConv(n_hid, n_output_dim)
        self.dropout = dropout
        self.decoder_weight = nn.Parameter(torch.Tensor(n_output_dim, n_output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.decoder_weight)

    def forward(self, adj, x):
        x = F.relu(self.gc1(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(adj, x))
        score = x @ self.decoder_weight @ x.t()
        return score

# Training function
def train(epoch, model, optimizer, adj_tensor, target_tensor):
    model.train()
    optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    score = model(adj_tensor, feature_tensor)
    loss = F.mse_loss(score, target_tensor.float()) + \
        0.05*F.binary_cross_entropy(torch.sigmoid(score), target_tensor.float())
    loss.backward()
    optimizer.step()
    
    return loss.item(), score.detach().cpu().clone()


def binarize_from_prob(prob_matrix: np.ndarray, threshold: float) -> np.ndarray:
    adj = (prob_matrix > threshold).astype(np.int32)
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    return adj


def upper_triangle_edge_set(adj: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.where(np.triu(adj, k=1) == 1)
    return set(zip(rows.tolist(), cols.tolist()))


def connected_component_stats(adj: np.ndarray) -> tuple[int, int]:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    largest = 0
    components = 0

    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        size = 0

        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.where(adj[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        largest = max(largest, size)

    return largest, components


def normalize_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return pd.Series(np.ones(len(values)) * 0.5, index=values.index)
    norm = (values - vmin) / (vmax - vmin)
    return norm if higher_is_better else (1 - norm)


def evaluate_thresholds(
    prob_runs: list[np.ndarray],
    thresholds: list[float],
    possible_edge_count: int,
    target_density: float,
) -> pd.DataFrame:
    records = []

    for t in thresholds:
        run_edges = []
        run_densities = []
        run_lcc_ratios = []
        run_non_isolated_ratios = []

        for run_prob in prob_runs:
            adj_pred = binarize_from_prob(run_prob, t)
            edge_set = upper_triangle_edge_set(adj_pred)
            edge_count = len(edge_set)

            degrees = adj_pred.sum(axis=1)
            non_isolated_ratio = float(np.mean(degrees > 0))

            largest_cc, _ = connected_component_stats(adj_pred)
            lcc_ratio = largest_cc / adj_pred.shape[0] if adj_pred.shape[0] > 0 else 0.0
            density = edge_count / possible_edge_count if possible_edge_count > 0 else 0.0

            run_edges.append(edge_set)
            run_densities.append(density)
            run_lcc_ratios.append(lcc_ratio)
            run_non_isolated_ratios.append(non_isolated_ratio)

        pairwise_jaccards = []
        if len(run_edges) >= 2:
            for i, j in combinations(range(len(run_edges)), 2):
                union = len(run_edges[i] | run_edges[j])
                inter = len(run_edges[i] & run_edges[j])
                pairwise_jaccards.append(inter / union if union > 0 else 1.0)
        else:
            pairwise_jaccards = [1.0]

        mean_density = float(np.mean(run_densities))
        density_gap = abs(mean_density - target_density)

        records.append(
            {
                "threshold": t,
                "stability_jaccard_mean": float(np.mean(pairwise_jaccards)),
                "stability_jaccard_std": float(np.std(pairwise_jaccards)),
                "density_mean": mean_density,
                "density_std": float(np.std(run_densities)),
                "density_gap_to_prior": density_gap,
                "lcc_ratio_mean": float(np.mean(run_lcc_ratios)),
                "non_isolated_ratio_mean": float(np.mean(run_non_isolated_ratios)),
            }
        )

    return pd.DataFrame(records)

all_records = []
avg_prob_by_threshold = {}

for threshold in THRESHOLD_CANDIDATES:
    print(f"\n=== THRESHOLD={threshold:.2f} (prior=bin) ===")

    prior_adj = (np.abs(np.corrcoef(node_features)) > threshold).astype(np.int32)
    np.fill_diagonal(prior_adj, 0)
    prior_adj[prior_adj == 2] = 1

    processed_adj = normalize_adj(prior_adj)
    adj_tensor = torch.FloatTensor(processed_adj).to(device)
    prior_adj_tensor = torch.FloatTensor(prior_adj).to(device)

    prior_edge_count = int(np.triu(prior_adj, k=1).sum())
    prior_density = prior_edge_count / possible_edges if possible_edges > 0 else 0.0

    score_collection = []
    for run in range(ROUNDS):
        model = GCN(n_feat=GENE_SIZE, n_hid=HIDDEN_DIM, n_output_dim=OUTPUT_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        final_score = None

        for epoch in range(EPOCHS):
            loss, score_snapshot = train(epoch, model, optimizer, adj_tensor, prior_adj_tensor)
            if epoch == EPOCHS - 1:
                final_score = score_snapshot
            print(f'THRESHOLD {threshold:.2f} | Run {run+1} | Epoch {epoch}, Loss: {loss:.4f}')

        score_collection.append(final_score)

    avg_score = torch.mean(torch.stack(score_collection), dim=0)
    avg_score = (avg_score + avg_score.T) / 2
    avg_score.fill_diagonal_(0)

    prob_runs = [torch.sigmoid(s).numpy() for s in score_collection]
    avg_prob_by_threshold[threshold] = torch.sigmoid(avg_score).numpy()

    df_eval_single = evaluate_thresholds(
        prob_runs,
        [threshold],
        possible_edges,
        prior_density,
    )
    df_eval_single = df_eval_single.rename(columns={"threshold": "sync_threshold"})
    df_eval_single["prior_edge_count"] = prior_edge_count
    df_eval_single["prior_density"] = prior_density
    all_records.append(df_eval_single)

df_all = pd.concat(all_records, ignore_index=True)

# Use natural 0-1 bounds (avoid penalizing 0.99 to 0.0 just because it's the minimum)
df_all["score_stability"] = df_all["stability_jaccard_mean"]
df_all["score_density_match"] = 1.0 - df_all["density_gap_to_prior"]
df_all["score_connectivity"] = df_all["lcc_ratio_mean"]
df_all["score_non_isolated"] = df_all["non_isolated_ratio_mean"]

df_all["composite_score"] = (
    W_STABILITY * df_all["score_stability"]
    + W_DENSITY_MATCH * df_all["score_density_match"]
    + W_CONNECTIVITY * df_all["score_connectivity"]
    + W_NON_ISOLATED * df_all["score_non_isolated"]
)

df_all = df_all.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
df_all["rank"] = np.arange(1, len(df_all) + 1)
df_all.to_csv(BASE_DIR / 'threshold_sync_eval-k1-100列.csv', index=False)

df_t06 = df_all[np.isclose(df_all["sync_threshold"], REPORT_BIN_THRESHOLD)].copy()
# df_t06.to_csv(BASE_DIR / 'threshold_sync_eval-t0.6-k1-100列.csv', index=False)

global_best = df_all.loc[df_all["composite_score"].idxmax()]
best_threshold = float(global_best["sync_threshold"])
print(
    f"Global best synchronous threshold: t={best_threshold:.2f}, "
    f"score={global_best['composite_score']:.4f}"
)

best_prob = avg_prob_by_threshold[best_threshold]
best_adj = (best_prob > best_threshold).astype(int)
np.fill_diagonal(best_adj, 0)
pd.DataFrame(best_adj).to_excel(
    BASE_DIR / 'L-averaged_final_adj-sync-best-100列.xlsx',
    index=False,
    header=False,
)

print(f'Processing completed for cluster k=')
#print(f'Results saved to: R-averaged_final_adj100列-A2.xlsx')