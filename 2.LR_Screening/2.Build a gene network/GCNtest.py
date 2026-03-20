# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
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
THRESHOLD = 0.6       # Binarization threshold

# Load data from .mat (use t_data directly, not xlsx)
data_path = BASE_DIR / 't_wavelet-all-L100列.mat'
mat = sio.loadmat(data_path)
if 't_data' not in mat:
    raise KeyError(f"'t_data' not found in {data_path}")
node_features = np.asarray(mat['t_data'], dtype=np.float32)
scaler = StandardScaler()


# Create initial adjacency matrix
your_adj_matrix_AM = (np.abs(np.corrcoef(node_features)) > THRESHOLD).astype(int)
node_features = scaler.fit_transform(node_features)  # Normalize

# Process adjacency matrix
np.fill_diagonal(your_adj_matrix_AM, 0)  # Remove self-loops
your_adj_matrix_AM[your_adj_matrix_AM == 2] = 1  # Ensure binary

# Normalize adjacency matrix
def normalize_adj(adj):
    """Symmetric normalization of adjacency matrix"""
    adj = adj + np.eye(adj.shape[0])  # Add self-connections
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt)

processed_adj_AM = normalize_adj(your_adj_matrix_AM)

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_tensor = torch.FloatTensor(node_features).to(device)
adj_tensor_AM = torch.FloatTensor(processed_adj_AM).to(device)
your_adj_matrix_AM_tensor = torch.FloatTensor(your_adj_matrix_AM).to(device)

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
def train(epoch):
    model.train()
    optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    score = model(adj_tensor_AM, feature_tensor)
    loss = F.mse_loss(score, your_adj_matrix_AM_tensor.float()) + \
           0.05*F.binary_cross_entropy(torch.sigmoid(score), your_adj_matrix_AM_tensor.float())
    loss.backward()
    optimizer.step()
    
    return loss.item(), score.detach().cpu().clone()

# Main training loop
score_collection = []
for run in range(ROUNDS):
    model = GCN(n_feat=GENE_SIZE, n_hid=HIDDEN_DIM, n_output_dim=OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_score = None
    
    for epoch in range(EPOCHS):
        loss, score_snapshot = train(epoch)
        if epoch == EPOCHS - 1:
            final_score = score_snapshot
        print(f'Run {run+1} | Epoch {epoch}, Loss: {loss:.4f}')

    score_collection.append(final_score)

# Process final results
avg_score = torch.mean(torch.stack(score_collection), dim=0)
avg_score = (avg_score + avg_score.T) / 2  # Symmetrize
avg_score.fill_diagonal_(0)  # Zero diagonal

# Save results
df_avg = pd.DataFrame(torch.sigmoid(avg_score).numpy())
df_avg = (df_avg > THRESHOLD).astype(int)
df_avg.to_excel(BASE_DIR /  'L-averaged_final_adj-k1-100列.xlsx', index=False, header=False)

print(f'Processing completed for cluster k=')
#print(f'Results saved to: R-averaged_final_adj100列-A2.xlsx')