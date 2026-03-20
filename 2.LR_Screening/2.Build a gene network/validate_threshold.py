"""
Threshold Validation for Gene Network Construction
====================================================
验证 THRESHOLD=0.6 的合理性，从四个维度设计指标：
  1. GCN 重建质量（AUC-ROC / F1 / Precision-Recall曲线）
  2. 网络拓扑合理性（幂律度分布、网络密度、最大连通分量）
  3. 阈值敏感性分析（扫描 0.3~0.9，观察指标变化曲线）
  4. Bootstrap 稳定性（同一阈值下网络的跨轮次一致性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ── 基础配置 ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
GENE_SIZE  = 100
HIDDEN_DIM = 300
OUTPUT_DIM = 200
DROPOUT    = 0.3
EPOCHS     = 500          # 验证时可适当减少轮次
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 5e-6
TARGET_THRESHOLD = 0.6    # 待验证的目标阈值
SWEEP_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
BOOTSTRAP_ROUNDS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 数据加载 ──────────────────────────────────────────────────────────────────
data_path = BASE_DIR / 't_wavelet-all-L100列.mat'
mat = sio.loadmat(data_path)
if 't_data' not in mat:
    raise KeyError(f"'t_data' not found in {data_path}")
raw_features = np.asarray(mat['t_data'], dtype=np.float32)
n_genes = raw_features.shape[0]
print(f"[Data] Loaded: {n_genes} genes × {raw_features.shape[1]} features")

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def build_adj(features: np.ndarray, threshold: float) -> np.ndarray:
    """根据阈值构建相关系数邻接矩阵"""
    adj = (np.abs(np.corrcoef(features)) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj

def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """对称归一化"""
    adj_hat = adj + np.eye(adj.shape[0])
    rowsum   = np.array(adj_hat.sum(1))
    d_inv_sq = np.power(rowsum, -0.5).flatten()
    d_inv_sq[np.isinf(d_inv_sq)] = 0.0
    D = np.diag(d_inv_sq)
    return adj_hat.dot(D).T.dot(D)

def prepare_tensors(features: np.ndarray, adj: np.ndarray):
    scaler = StandardScaler()
    feat_norm = scaler.fit_transform(features)
    feat_t = torch.FloatTensor(feat_norm).to(device)
    adj_norm_t = torch.FloatTensor(normalize_adj(adj)).to(device)
    adj_t = torch.FloatTensor(adj).to(device)
    return feat_t, adj_norm_t, adj_t


# ── GCN 模型定义 ──────────────────────────────────────────────────────────────
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias   = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        return torch.mm(adj, torch.mm(x, self.weight)) + self.bias

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_out, dropout=DROPOUT):
        super().__init__()
        self.gc1 = GraphConv(n_feat, n_hid)
        self.gc2 = GraphConv(n_hid, n_out)
        self.dropout = dropout
        self.decoder_weight = nn.Parameter(torch.Tensor(n_out, n_out))
        nn.init.xavier_uniform_(self.decoder_weight)

    def forward(self, adj, x):
        x = F.relu(self.gc1(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(adj, x))
        return x @ self.decoder_weight @ x.t()

def train_gcn(features: np.ndarray, threshold: float, epochs: int = EPOCHS):
    """训练GCN并返回最终输出的sigmoid分数矩阵（numpy）"""
    adj = build_adj(features, threshold)
    feat_t, adj_norm_t, adj_t = prepare_tensors(features, adj)

    model = GCN(n_feat=GENE_SIZE, n_hid=HIDDEN_DIM, n_out=OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        score = model(adj_norm_t, feat_t)
        loss  = F.mse_loss(score, adj_t) + \
                0.05 * F.binary_cross_entropy(torch.sigmoid(score), adj_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        score = model(adj_norm_t, feat_t)
        pred_prob = torch.sigmoid(score).cpu().numpy()

    # 对称化，清零对角线
    pred_prob = (pred_prob + pred_prob.T) / 2
    np.fill_diagonal(pred_prob, 0)
    return pred_prob, adj  # (预测概率矩阵, 真实邻接矩阵)


# ══════════════════════════════════════════════════════════════════════════════
# 维度一：GCN 重建质量（取上三角展开后计算）
# ══════════════════════════════════════════════════════════════════════════════
def compute_reconstruction_metrics(pred_prob: np.ndarray, true_adj: np.ndarray,
                                   output_threshold: float = TARGET_THRESHOLD) -> dict:
    """计算 AUC-ROC / AUC-PR / F1 / Precision / Recall"""
    idx = np.triu_indices(pred_prob.shape[0], k=1)
    y_score = pred_prob[idx]
    y_true  = true_adj[idx]

    # 防止全0/全1导致指标无意义
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auc_roc": np.nan, "auc_pr": np.nan, "f1": np.nan,
                "precision": np.nan, "recall": np.nan}

    y_pred = (y_score >= output_threshold).astype(int)
    return {
        "auc_roc":   roc_auc_score(y_true, y_score),
        "auc_pr":    average_precision_score(y_true, y_score),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 维度二：网络拓扑分析
# ══════════════════════════════════════════════════════════════════════════════
def compute_topology_metrics(adj: np.ndarray) -> dict:
    """
    分析网络拓扑：
     - 边密度
     - 最大连通分量比例
     - 幂律指数（γ），生物基因网络通常 γ ∈ [2, 3]
     - 平均聚类系数
    """
    G = nx.from_numpy_array(adj)
    n = G.number_of_nodes()
    density = nx.density(G)

    components = list(nx.connected_components(G))
    lcc_ratio  = max(len(c) for c in components) / n if components else 0

    degrees = np.array([d for _, d in G.degree()])
    degrees = degrees[degrees > 0]

    # 幂律指数估计（MLE for discrete power-law）
    d_min = 1
    if len(degrees) > 1:
        gamma = 1 + len(degrees) / np.sum(np.log(degrees / (d_min - 0.5)))
    else:
        gamma = np.nan

    avg_clustering = nx.average_clustering(G)

    return {
        "n_edges":       G.number_of_edges(),
        "density":       density,
        "lcc_ratio":     lcc_ratio,
        "powerlaw_gamma": gamma,
        "avg_clustering": avg_clustering,
        "n_components":  len(components),
    }


def powerlaw_fit_quality(adj: np.ndarray) -> float:
    """
    用 KS 统计量检验度分布的幂律拟合优度（越小越符合幂律）
    返回 KS 统计量 p 值（p > 0.05 表示不能拒绝幂律假设）
    """
    G = nx.from_numpy_array(adj)
    degrees = np.array([d for _, d in G.degree()])
    degrees = degrees[degrees > 0]
    if len(degrees) < 5:
        return np.nan
    log_d = np.log(degrees)
    # 用指数分布近似检验（对数正态 vs 幂律通过 Kolmogorov-Smirnov）
    _, p_value = stats.kstest(log_d, 'norm', args=(log_d.mean(), log_d.std()))
    return p_value


# ══════════════════════════════════════════════════════════════════════════════
# 维度三：阈值敏感性分析
# ══════════════════════════════════════════════════════════════════════════════
def threshold_sensitivity_analysis(features: np.ndarray,
                                    thresholds: list,
                                    epochs: int = EPOCHS) -> pd.DataFrame:
    """
    扫描多个阈值，为每个阈值同时计算：
      - GCN 重建指标（AUC-ROC, F1）
      - 网络拓扑指标（density, lcc_ratio, powerlaw_gamma）
    """
    records = []
    for thr in thresholds:
        print(f"  [Sweep] threshold={thr:.2f} ...")
        pred_prob, true_adj = train_gcn(features, thr, epochs)
        recon = compute_reconstruction_metrics(pred_prob, true_adj,
                                               output_threshold=thr)
        topo  = compute_topology_metrics(true_adj)
        records.append({
            "threshold":       thr,
            **{f"recon_{k}": v for k, v in recon.items()},
            **{f"topo_{k}":  v for k, v in topo.items()},
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 维度四：Bootstrap 稳定性（边重叠率）
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_stability(features: np.ndarray,
                        threshold: float,
                        rounds: int = BOOTSTRAP_ROUNDS,
                        subsample_ratio: float = 0.8,
                        epochs: int = EPOCHS) -> dict:
    """
    每次随机选取 subsample_ratio 比例的基因子集，独立构图并训练，
    计算不同轮次之间边重叠的 Jaccard 系数（越高越稳定）。
    """
    n = features.shape[0]
    k = int(n * subsample_ratio)
    edge_sets = []

    for r in range(rounds):
        idx = np.random.choice(n, k, replace=False)
        sub_feat = features[idx]
        pred_prob, _ = train_gcn(sub_feat, threshold, epochs)
        binary_adj = (pred_prob >= threshold).astype(int)
        # 将局部索引映射到边集合（用排序索引对表示）
        tri = np.triu_indices(k, k=1)
        edges = set(
            (idx[i], idx[j])
            for i, j in zip(tri[0], tri[1])
            if binary_adj[i, j] == 1
        )
        edge_sets.append(edges)
        print(f"    [Bootstrap] round {r+1}/{rounds}, edges={len(edges)}")

    # 两两计算 Jaccard
    jaccards = []
    for i in range(rounds):
        for j in range(i + 1, rounds):
            s1, s2 = edge_sets[i], edge_sets[j]
            union = len(s1 | s2)
            jac   = len(s1 & s2) / union if union > 0 else 0.0
            jaccards.append(jac)

    return {
        "jaccard_mean": np.mean(jaccards),
        "jaccard_std":  np.std(jaccards),
        "jaccard_min":  np.min(jaccards),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════
def plot_sensitivity(df: pd.DataFrame, save_dir: Path):
    """绘制阈值敏感性曲线（四合一图）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Threshold Sensitivity Analysis", fontsize=14)

    ax = axes[0, 0]
    ax.plot(df["threshold"], df["recon_auc_roc"], 'b-o', label="AUC-ROC")
    ax.plot(df["threshold"], df["recon_f1"],      'r-s', label="F1")
    ax.axvline(x=TARGET_THRESHOLD, color='gray', linestyle='--', label=f'θ={TARGET_THRESHOLD}')
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Reconstruction Quality"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["threshold"], df["topo_density"], 'g-o')
    ax.axvline(x=TARGET_THRESHOLD, color='gray', linestyle='--')
    ax.set_xlabel("Threshold"); ax.set_ylabel("Edge Density")
    ax.set_title("Network Density"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["threshold"], df["topo_lcc_ratio"], 'm-o')
    ax.axvline(x=TARGET_THRESHOLD, color='gray', linestyle='--')
    ax.set_xlabel("Threshold"); ax.set_ylabel("LCC / N")
    ax.set_title("Largest Connected Component Ratio"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["threshold"], df["topo_powerlaw_gamma"], 'c-o')
    ax.axhline(y=2.0, color='green', linestyle=':', label='γ=2 (bio lower)')
    ax.axhline(y=3.0, color='green', linestyle=':', label='γ=3 (bio upper)')
    ax.axvline(x=TARGET_THRESHOLD, color='gray', linestyle='--')
    ax.set_xlabel("Threshold"); ax.set_ylabel("Power-law γ")
    ax.set_title("Degree Distribution Power-law Exponent"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "threshold_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out}")


def plot_precision_recall(pred_prob: np.ndarray, true_adj: np.ndarray, save_dir: Path):
    """绘制 PR 曲线和 ROC 曲线，标注 θ=0.6 工作点"""
    idx = np.triu_indices(pred_prob.shape[0], k=1)
    y_score = pred_prob[idx]
    y_true  = true_adj[idx]

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("[Plot] Skipping PR/ROC: trivial labels")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"GCN Reconstruction at threshold={TARGET_THRESHOLD}", fontsize=12)

    # PR 曲线
    precision, recall, thr_pr = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax = axes[0]
    ax.plot(recall, precision, 'b-', label=f"AP={ap:.3f}")
    # 标注工作点
    wp_mask = np.abs(thr_pr - TARGET_THRESHOLD).argmin()
    ax.scatter(recall[wp_mask], precision[wp_mask], color='red', zorder=5,
               label=f"θ={TARGET_THRESHOLD} (P={precision[wp_mask]:.2f}, R={recall[wp_mask]:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(True, alpha=0.3)

    # ROC 曲线
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ax = axes[1]
    ax.plot(fpr, tpr, 'b-', label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--')
    wp_roc = np.abs(thr_roc - TARGET_THRESHOLD).argmin()
    ax.scatter(fpr[wp_roc], tpr[wp_roc], color='red', zorder=5,
               label=f"θ={TARGET_THRESHOLD}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "pr_roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out}")


def plot_degree_distribution(adj: np.ndarray, threshold: float, save_dir: Path):
    """绘制度分布对数直方图，检验幂律性"""
    G = nx.from_numpy_array(adj)
    degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
    if len(degrees) < 3:
        return

    unique, counts = np.unique(degrees, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(unique, counts, 'bo', alpha=0.7)
    # 线性拟合（对数空间）
    log_k = np.log(unique); log_c = np.log(counts)
    coef  = np.polyfit(log_k, log_c, 1)
    ax.loglog(unique, np.exp(np.polyval(coef, log_k)), 'r--',
              label=f"γ={-coef[0]:.2f}")
    ax.set_xlabel("Degree k"); ax.set_ylabel("Count P(k)")
    ax.set_title(f"Degree Distribution (θ={threshold:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3, which='both')
    out = save_dir / f"degree_dist_thr{threshold:.2f}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    save_dir = BASE_DIR / "threshold_validation_results"
    save_dir.mkdir(exist_ok=True)

    # ── 步骤1：在目标阈值下训练并评估 ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Step 1: Evaluate GCN at target threshold={TARGET_THRESHOLD}")
    print("="*60)
    pred_prob, true_adj = train_gcn(raw_features, TARGET_THRESHOLD, EPOCHS)
    recon_metrics = compute_reconstruction_metrics(pred_prob, true_adj, TARGET_THRESHOLD)
    topo_metrics  = compute_topology_metrics(true_adj)

    print(f"\n[Reconstruction Metrics @ θ={TARGET_THRESHOLD}]")
    for k, v in recon_metrics.items():
        print(f"  {k:20s}: {v:.4f}" if not np.isnan(v) else f"  {k:20s}: NaN")

    print(f"\n[Topology Metrics @ θ={TARGET_THRESHOLD}]")
    for k, v in topo_metrics.items():
        print(f"  {k:20s}: {v}")

    # 保存目标阈值结果
    pd.DataFrame([{**recon_metrics, **topo_metrics, "threshold": TARGET_THRESHOLD}]).to_csv(
        save_dir / f"metrics_thr{TARGET_THRESHOLD}.csv", index=False)

    # ── 步骤2：PR 曲线 / ROC 曲线 ─────────────────────────────────────────────
    print(f"\nStep 2: Plot PR/ROC curves ...")
    plot_precision_recall(pred_prob, true_adj, save_dir)

    # ── 步骤3：度分布可视化（幂律检验）────────────────────────────────────────
    print(f"\nStep 3: Degree distribution (power-law check) ...")
    plot_degree_distribution(true_adj, TARGET_THRESHOLD, save_dir)

    # ── 步骤4：阈值敏感性扫描 ─────────────────────────────────────────────────
    print(f"\nStep 4: Threshold sensitivity sweep {SWEEP_THRESHOLDS} ...")
    df_sweep = threshold_sensitivity_analysis(raw_features, SWEEP_THRESHOLDS, EPOCHS)
    df_sweep.to_csv(save_dir / "sensitivity_sweep.csv", index=False)
    plot_sensitivity(df_sweep, save_dir)

    print("\n[Sensitivity Sweep Summary]")
    cols_show = ["threshold", "recon_auc_roc", "recon_f1", "topo_density",
                 "topo_lcc_ratio", "topo_powerlaw_gamma"]
    print(df_sweep[cols_show].to_string(index=False, float_format="%.4f"))

    # ── 步骤5：Bootstrap 稳定性 ────────────────────────────────────────────────
    print(f"\nStep 5: Bootstrap stability @ θ={TARGET_THRESHOLD} ...")
    stability = bootstrap_stability(raw_features, TARGET_THRESHOLD,
                                    rounds=BOOTSTRAP_ROUNDS, epochs=EPOCHS)
    print(f"\n[Bootstrap Stability @ θ={TARGET_THRESHOLD}]")
    for k, v in stability.items():
        print(f"  {k:20s}: {v:.4f}")
    pd.DataFrame([stability]).to_csv(save_dir / "bootstrap_stability.csv", index=False)

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Target threshold : {TARGET_THRESHOLD}")
    print(f"AUC-ROC          : {recon_metrics['auc_roc']:.4f}  (>0.7 good, >0.85 excellent)")
    print(f"F1 Score         : {recon_metrics['f1']:.4f}")
    print(f"AUC-PR           : {recon_metrics['auc_pr']:.4f}")
    print(f"Edge Density     : {topo_metrics['density']:.4f}  (0.01~0.1 typical for gene nets)")
    print(f"LCC Ratio        : {topo_metrics['lcc_ratio']:.4f}  (>0.8 well-connected)")
    print(f"Power-law γ      : {topo_metrics['powerlaw_gamma']:.4f}  (2~3 scale-free)")
    print(f"Bootstrap Jaccard: {stability['jaccard_mean']:.4f} ± {stability['jaccard_std']:.4f}  (>0.5 stable)")
    print(f"\nResults saved to: {save_dir}")
