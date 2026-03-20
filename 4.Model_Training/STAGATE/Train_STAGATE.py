import numpy as np
import scipy.sparse as sp
from .STAGATE_multihead import STAGATE
#from .STAGATE import STAGATE
import tensorflow.compat.v1 as tf
import pandas as pd
import scanpy as sc
from pathlib import Path
import tempfile
import gzip

def _parse_pair_string(pair_str):
    """Parse pair string like 'FCER2__Bcell|ICAM3__Stroma' -> ('FCER2__Bcell', 'ICAM3__Stroma')."""

    try:
        left, right = pair_str.split('|', 1)
        return left, right
    except Exception:
        return None, None


def _load_lr_score_map(lr_score_path, score_column="mean_nonzero"):
    """Load LR scores from CSV/TSV or (optionally gzipped) RDS and return {(lig, rec): score}."""

    path = Path(lr_score_path)
    if not path.exists():
        raise FileNotFoundError(f"LR score file not found: {path}")

    # CSV / TSV fast path
    if path.suffix.lower() in [".csv", ".tsv", ".txt"]:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        lower_cols = {c.lower(): c for c in df.columns}
        # Case: ligand / receptor columns
        if "ligand" in lower_cols and "receptor" in lower_cols:
            lig_col = lower_cols["ligand"]
            rec_col = lower_cols["receptor"]
            score_col = score_column if score_column.lower() in lower_cols else next(
                (lower_cols.get(c) for c in ["mean_nonzero", "score", "lr_score", "value", "mean"] if c in lower_cols),
                None,
            )
            if score_col is None:
                raise ValueError("CSV LR score file must contain a score column")
            tidy = df[[lig_col, rec_col, score_col]].rename(
                columns={lig_col: "Ligand", rec_col: "Receptor", score_col: "Score"}
            )
            tidy = tidy.dropna(subset=["Score"])
            return {(str(l), str(r)): float(s) for l, r, s in tidy.itertuples(index=False)}
        # Case: pair column
        if "pair" in lower_cols:
            pair_col = lower_cols["pair"]
            score_col = score_column if score_column.lower() in lower_cols else next(
                (lower_cols.get(c) for c in ["mean_nonzero", "score", "lr_score", "value", "mean"] if c in lower_cols),
                None,
            )
            if score_col is None:
                raise ValueError("CSV LR score file must contain a score column")
            parsed = df[[pair_col, score_col]].copy()
            parsed[["Ligand", "Receptor"]] = parsed[pair_col].apply(lambda x: pd.Series(_parse_pair_string(str(x))))
            parsed = parsed.dropna(subset=["Ligand", "Receptor", score_col])
            return {(str(l), str(r)): float(s) for l, r, s in parsed[["Ligand", "Receptor", score_col]].itertuples(index=False)}
        raise ValueError("CSV/TSV LR score file must include ligand+receptor or pair column")

    # RDS path (with optional gzip)
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError("pyreadr is required to read RDS LR score files") from exc

    tmp_path = None
    with open(path, "rb") as f_in:
        magic = f_in.read(3)
    target_path = path
    if magic.startswith(b"\x1f\x8b\x08"):
        tmp = tempfile.NamedTemporaryFile(suffix=".rds", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        with open(path, "rb") as f_in, gzip.open(f_in, "rb") as gz, open(tmp_path, "wb") as f_out:
            for chunk in iter(lambda: gz.read(1024 * 1024), b""):
                f_out.write(chunk)
        target_path = tmp_path

    try:
        res = pyreadr.read_r(str(target_path))
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    if len(res) == 0:
        raise ValueError(f"No objects found in {path}")
    obj = list(res.values())[0]
    if hasattr(obj, "to_dataframe"):
        df = obj.to_dataframe()
    else:
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)

    df = df.copy()
    lower_cols = {c.lower(): c for c in df.columns}
    def _pick_score(col_map):
        if score_column and score_column.lower() in col_map:
            return col_map[score_column.lower()]
        for cand in ["mean_nonzero", "score", "lr_score", "value", "mean"]:
            if cand in col_map:
                return col_map[cand]
        for c in df.columns:
            if c.lower() not in ["ligand", "ligand_gene", "receptor", "receptor_gene", "pair"]:
                return c
        return None

    if "ligand_gene" in lower_cols and "receptor_gene" in lower_cols:
        lig_col = lower_cols["ligand_gene"]
        rec_col = lower_cols["receptor_gene"]
        score_col = _pick_score(lower_cols)
        if score_col is None:
            raise ValueError("LR score table must contain a score column")
        tidy = df[[lig_col, rec_col, score_col]].rename(
            columns={lig_col: "Ligand", rec_col: "Receptor", score_col: "Score"}
        )
    elif "pair" in lower_cols:
        pair_col = lower_cols["pair"]
        score_col = _pick_score(lower_cols)
        if score_col is None:
            raise ValueError("LR score table must contain a score column")
        parsed = df[[pair_col, score_col]].copy()
        parsed[["Ligand", "Receptor"]] = parsed[pair_col].apply(lambda x: pd.Series(_parse_pair_string(str(x))))
        tidy = parsed.rename(columns={score_col: "Score"})
        tidy = tidy.dropna(subset=["Ligand", "Receptor", "Score"])
    else:
        tidy = df.rename_axis("Ligand").reset_index().melt(id_vars=["Ligand"], var_name="Receptor", value_name="Score")

    tidy = tidy.dropna(subset=["Score"])
    lr_map = {(str(l), str(r)): float(s) for l, r, s in tidy.itertuples(index=False)}
    if len(lr_map) == 0:
        raise ValueError(f"Parsed LR score map is empty for {path}")
    return lr_map


def _build_lr_weights(cross_df, section_lookup, ligand_section, receptor_section, lr_score_map):
    """Return LR weights aligned to cross_df rows using section to decide orientation."""

    sec_series = None
    if section_lookup is not None:
        sec_series = section_lookup if isinstance(section_lookup, pd.Series) else pd.Series(section_lookup)

    def _resolve_section(val):
        if isinstance(val, pd.Series):
            return val.iloc[0]
        return val

    weights = []
    for row in cross_df.itertuples(index=False):
        cell1, cell2 = row.Cell1, row.Cell2
        lig, rec = cell1, cell2
        if sec_series is not None and ligand_section is not None and receptor_section is not None:
            sec1 = _resolve_section(sec_series.get(cell1, None))
            sec2 = _resolve_section(sec_series.get(cell2, None))
            if sec1 == ligand_section and sec2 == receptor_section:
                lig, rec = cell1, cell2
            elif sec1 == receptor_section and sec2 == ligand_section:
                lig, rec = cell2, cell1
        weight = lr_score_map.get((str(lig), str(rec)), 0.0)
        weights.append(float(weight))
    return np.asarray(weights, dtype=np.float32)

#def train_STAGATE(adata, hidden_dims=[512, 30], alpha=0, n_epochs=500, lr=0.0001, key_added='STAGATE',
def train_STAGATE(adata, hidden_dims=[512, 30], num_heads=4, alpha=0, n_epochs=500, lr=0.0001, key_added='STAGATE',
                gradient_clipping=5, nonlinear=True, weight_decay=0.0001,verbose=True, 
                random_seed=2021, pre_labels=None, pre_resolution=0.2,
                save_attention=False, save_loss=False, save_reconstrction=False,
                use_corr_loss=False, sigma_expr=1.0, lr_score_rds=None,
                ligand_section=None, receptor_section=None, key_section='Section_id',
                lr_score_column="mean_nonzero"):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    alpha
        The weight of cell type-aware spatial neighbor network.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    nonlinear
        If True, the nonlinear avtivation is performed.
    weight_decay
        Weight decay for AdamOptimizer.
    pre_labels
        The key in adata.obs for the manually designate the pre-clustering results. Only used when alpha>0.
    pre_resolution
        The resolution parameter of sc.tl.louvain for the pre-clustering. Only used when alpha>0 and per_labels==None.
    save_attention
        If True, the weights of the attention layers are saved in adata.uns['STAGATE_attention']
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].

    Returns
    -------
    AnnData
    """

    tf.reset_default_graph()
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    edge_weights = G_df['Weight'] if 'Weight' in G_df.columns else np.ones(G_df.shape[0], dtype=np.float32)
    G = sp.coo_matrix((edge_weights, (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G_tf = prepare_graph_data(G)

    # Optional corr target aligned to edge order (including self-loops from prepare_graph_data)
    corr_target = None
    if use_corr_loss:
        expr = X.values
        section_lookup = adata.obs[key_section] if key_section in adata.obs.columns else None
        lr_score_map = _load_lr_score_map(lr_score_rds, score_column=lr_score_column) if lr_score_rds is not None else None
        cross_mask = Spatial_Net['SNN'].astype(str).str.contains('-') if 'SNN' in Spatial_Net.columns else None
        cross_df = Spatial_Net if cross_mask is None else Spatial_Net.loc[cross_mask]

        if cross_df.shape[0] == 0:
            corr_target = np.zeros(G_tf[0].shape[0], dtype=np.float32)
        else:
            rows = cross_df['Cell1'].map(cells_id_tran).values
            cols = cross_df['Cell2'].map(cells_id_tran).values
            diff = expr[rows] - expr[cols]
            expr_dist_sq = np.sum(diff * diff, axis=1)
            #expr_sim = np.exp(-expr_dist_sq / (2.0 * sigma_expr * sigma_expr))
            if sigma_expr is None or sigma_expr <= 0:
                expr_sim = np.ones_like(expr_dist_sq, dtype=np.float32)
            else:
                expr_sim = np.exp(-expr_dist_sq / (2.0 * sigma_expr * sigma_expr))

            if lr_score_map is not None:
                if (ligand_section is None or receptor_section is None) and section_lookup is not None:
                    unique_sections = list(pd.unique(section_lookup))
                    if len(unique_sections) >= 2:
                        ligand_section = ligand_section or unique_sections[0]
                        receptor_section = receptor_section or unique_sections[1]
                lr_weights = _build_lr_weights(cross_df, section_lookup, ligand_section, receptor_section, lr_score_map)
            else:
                lr_weights = np.ones_like(expr_sim, dtype=np.float32)

            prior = expr_sim * lr_weights

            max_prior = np.max(prior)
            prior = prior / (max_prior + 1e-6) if max_prior > 0 else np.zeros_like(prior)

            prior_map = {(r, c): v for r, c, v in zip(rows, cols, prior)}
            aligned = []
            for r, c in zip(G_tf[0][:, 1], G_tf[0][:, 0]):
                aligned.append(prior_map.get((r, c), 0.0))
            corr_target = np.asarray(aligned, dtype=np.float32)

    #trainer = STAGATE(hidden_dims=[X.shape[1]] + hidden_dims,  alpha=alpha, 
   
    trainer = STAGATE(hidden_dims=[X.shape[1]] + hidden_dims, num_heads=num_heads, alpha=alpha, 
                    n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping, 
                    nonlinear=nonlinear,weight_decay=weight_decay, verbose=verbose, 
                    random_seed=random_seed, use_corr_loss=use_corr_loss)
    if alpha == 0:
        trainer(G_tf, G_tf, X, corr_target)
        embeddings, attentions, loss, ReX= trainer.infer(G_tf, G_tf, X, corr_target)
    else:
        G_df = Spatial_Net.copy()
        if pre_labels==None:
            if verbose:
                print('------Pre-clustering using louvain with resolution=%.2f' %pre_resolution)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata)
            sc.tl.louvain(adata, resolution=pre_resolution, key_added='expression_louvain_label')
            pre_labels = 'expression_louvain_label'
        prune_G_df = prune_spatial_Net(G_df, adata.obs[pre_labels])
        prune_G_df['Cell1'] = prune_G_df['Cell1'].map(cells_id_tran)
        prune_G_df['Cell2'] = prune_G_df['Cell2'].map(cells_id_tran)
        prune_G = sp.coo_matrix((np.ones(prune_G_df.shape[0]), (prune_G_df['Cell1'], prune_G_df['Cell2'])))
        prune_G_tf = prepare_graph_data(prune_G)
        prune_G_tf = (prune_G_tf[0], prune_G_tf[1], G_tf[2])
        trainer(G_tf, prune_G_tf, X, corr_target)
        embeddings, attentions, loss, ReX = trainer.infer(G_tf, prune_G_tf, X, corr_target)
    cell_reps = pd.DataFrame(embeddings)
    cell_reps.index = cells

    adata.obsm[key_added] = cell_reps.loc[adata.obs_names, ].values
    if save_attention:
        adata.uns['STAGATE_attention'] = attentions
    if save_loss:
        adata.uns['STAGATE_loss'] = loss
    if save_reconstrction:
        ReX = pd.DataFrame(ReX, index=X.index, columns=X.columns)
        ReX[ReX<0] = 0
        adata.layers['STAGATE_ReX'] = ReX.values
    return adata


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)# self-loop
    #data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)

def recovery_Imputed_Count(adata, size_factor):
    assert('ReX' in adata.uns)
    temp_df = adata.uns['ReX'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count'] = temp_df
    return adata
