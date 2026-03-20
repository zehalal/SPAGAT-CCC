import os
import time
from pathlib import Path

import psutil
import csv
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import commot as ct

current_dir = os.getcwd()
print("current path:", current_dir)

def show_info(start):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024/1024/1024
    return memory

start = show_info('strat')
start_time = time.time()

base_dir = Path(__file__).resolve().parent
benchmark_dir = base_dir.parent.parent
data_dir = base_dir.parent / "数据"

raw_dataset_dir = data_dir / "Breast_Cancer_Block_A_Section_1"
preprocess_dir = data_dir / "run_all_outputs_quan0.98_20260119_213616" / "preprocess"
celltype_file = preprocess_dir / "celltype_predictions.csv"

if not raw_dataset_dir.exists():
    raise FileNotFoundError(f"Visium folder not found: {raw_dataset_dir}")
if not celltype_file.exists():
    raise FileNotFoundError(f"Cell type predictions not found: {celltype_file}")

res_path = base_dir / "result"
res_path.mkdir(parents=True, exist_ok=True)

print(f"Loading Visium data from {raw_dataset_dir}")
# Use filtered gene subset to speed up COMMOT
filtered_dir = raw_dataset_dir / "filtered_feature_bc_matrix_combo"
spatial_dir = raw_dataset_dir / "spatial"
if not filtered_dir.exists():
    raise FileNotFoundError(f"Filtered feature matrix folder not found: {filtered_dir}")
if not spatial_dir.exists():
    raise FileNotFoundError(f"Spatial metadata folder not found: {spatial_dir}")

adata = sc.read_10x_mtx(
    path=str(filtered_dir),
    var_names="gene_symbols",
    make_unique=True,
)

positions_path = spatial_dir / "tissue_positions_list.csv"
positions = pd.read_csv(
    positions_path,
    header=None,
    names=[
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ],
)
positions.set_index("barcode", inplace=True)
positions = positions.reindex(adata.obs_names)
missing_positions = positions["pxl_row_in_fullres"].isna().sum()
if missing_positions:
    raise ValueError(f"Missing spatial coordinates for {missing_positions} barcodes in {positions_path}")
adata.obsm['spatial'] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy().astype(float)

celltype_probs = pd.read_csv(celltype_file, index_col=0)
predicted_celltypes = celltype_probs.idxmax(axis=0).rename("celltype")
predicted_celltypes = predicted_celltypes.reindex(adata.obs_names)
missing_mask = predicted_celltypes.isna()
if missing_mask.any():
    missing_count = int(missing_mask.sum())
    print(f"Dropping {missing_count} spots without cell type predictions")
    adata = adata[~missing_mask].copy()
    predicted_celltypes = predicted_celltypes[~missing_mask]

adata.obs['celltype'] = predicted_celltypes.astype('category')

# Preprocessing the data
adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

adata_dis500 = adata.copy()
print(adata_dis500)

db_name = 'custom'

# load custom LR list (1610 combos): format Ligand__Sender|Receptor__Receiver
combo_path = benchmark_dir.parent / "5.Model_Training" / "combo_only 100列.csv"
if not combo_path.exists():
    raise FileNotFoundError(f"Custom LR list not found: {combo_path}")
combo_df = pd.read_csv(combo_path)
ligands = []
receptors = []
for combo in combo_df['combo']:
    ligand_part, receptor_part = combo.split('|')
    ligands.append(ligand_part.split('__')[0])
    receptors.append(receptor_part.split('__')[0])

df_custom = pd.DataFrame({
    'ligand': ligands,
    'receptor': receptors,
    'pathway_name': ['CUSTOM'] * len(ligands),
    'annotation': ['Custom'] * len(ligands),
})
print('custom LR database shape:', df_custom.shape)

# spatial communication inference using custom database (no filtering to keep all pairs)
ct.tl.spatial_communication(adata_dis500, database_name=db_name,
                            df_ligrec=df_custom,
                            dis_thr=500,
                            heteromeric=True,
                            pathway_sum=True)

ct.tl.communication_direction(adata_dis500, database_name=db_name, pathway_name=None, k=5)  # type: ignore[arg-type]
adata_dis500.write(res_path / 'adata_pw.h5ad')
lr_keys = list(zip(ligands, receptors))

result = []
for lr in lr_keys:
    print('calculate the communication score of', lr)
    ct.tl.cluster_communication(adata_dis500, lr_pair=lr, database_name=db_name, pathway_name=None,  # type: ignore[arg-type]
                                clustering='celltype',
                                n_permutations=100)
    comm_mtx = adata_dis500.uns[f'commot_cluster-celltype-{db_name}-{lr[0]}-{lr[1]}']['communication_matrix']
    comm_mtx = comm_mtx.reset_index()
    comm_mtx.rename(columns={'index': 'Sender'}, inplace=True)
    comm_mtx_df = comm_mtx.melt(id_vars='Sender', var_name='Receiver', value_name='score')
    comm_mtx_df['Ligand'] = lr[0]
    comm_mtx_df['Receptor'] = lr[1]
    result.append(comm_mtx_df)

result = pd.concat(result, ignore_index=True)
end = show_info('end')
end_time = time.time()
run_time = (end_time - start_time) / 60
print(f"Training time is: {run_time} mins")
print('total memory used '+str(end-start) + 'GB')
adata_dis500.write(res_path / 'adata_pw_new.h5ad')
result.to_csv(res_path / 'result.csv', index=False)

summary_name = 'commot-'+db_name+'-sum-'+'receiver'
summary_abrv = 'r'
lr_pair: tuple = ('total','total')
comm_sum = np.asarray(
    adata_dis500.obsm[summary_name][summary_abrv+'-'+lr_pair[0]+'-'+lr_pair[1]]
).reshape(-1,1)
cell_weight = np.ones_like(comm_sum).reshape(-1,1)

np.savetxt(res_path / 'comm_sum.csv', comm_sum, delimiter=',', fmt='%.6f')
np.savetxt(res_path / 'cell_weight.csv', cell_weight, delimiter=',', fmt='%d')


