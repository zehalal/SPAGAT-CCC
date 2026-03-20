inp <- tryCatch({
  src <- sys.frame(1)$ofile
  if (!is.null(src)) dirname(normalizePath(src)) else getwd()  # use script dir when sourced
}, error = function(...) getwd())
ligs <- readRDS(file.path(inp, "10X_bc_Ligs_up_yll.rds"))
recs <- readRDS(file.path(inp, "10X_bc_Recs_expr.rds"))
env <- new.env()
load(file.path(inp, "output_bc.RData"), envir = env)
cnt <- env[["de_count"]]
ct <- env[["de_cell_type"]]
coords <- env[["de_coords"]]
ct_map <- split(ct$Barcode, ct$Cluster)
celltypes <- names(ct_map)

if (!is.null(coords)) {
  coords_df <- as.data.frame(coords)
  if (!is.null(rownames(coords_df)) && !"Barcode" %in% colnames(coords_df)) {
    coords_df <- cbind(Barcode = rownames(coords_df), coords_df)
  }

  if (!is.null(ct) && all(c("Barcode", "Cluster") %in% colnames(ct)) && "Barcode" %in% colnames(coords_df)) {
    ct_map_df <- unique(ct[, c("Barcode", "Cluster"), drop = FALSE])
    coords_df <- merge(coords_df, ct_map_df, by = "Barcode", all.x = TRUE, sort = FALSE)
  }

  write.csv(coords_df, file.path(inp, "de_coords.csv"), row.names = FALSE)
  cat("written de_coords.csv with", nrow(coords_df), "rows\n")
} else {
  cat("de_coords not found in output_bc.RData; skipped de_coords.csv\n")
}

lig_map <- stack(ligs); colnames(lig_map) <- c("gene", "celltype")
rec_map <- stack(recs); colnames(rec_map) <- c("gene", "celltype")
lig_multi <- table(lig_map$gene) > 1
rec_multi <- table(rec_map$gene) > 1
cat("lig genes shared by >1 cell type:", sum(lig_multi), "of", length(lig_multi), "\n")
cat("rec genes shared by >1 cell type:", sum(rec_multi), "of", length(rec_multi), "\n")

make_expr <- function(genes) {
  genes <- unique(genes)
  genes_in <- intersect(genes, rownames(cnt))
  mat <- matrix(0, nrow = length(genes), ncol = length(celltypes),
                dimnames = list(genes, celltypes))
  if (length(genes_in) > 0) {
    expr_sub <- cnt[genes_in, , drop = FALSE]
    for (cty in celltypes) {
      cells <- ct_map[[cty]]
      vals <- expr_sub[, cells, drop = FALSE]
      mat[genes_in, cty] <- rowMeans(vals)
    }
  }
  mat
}

lig_expr <- make_expr(lig_map$gene)
rec_expr <- make_expr(rec_map$gene)

# write.csv(cbind(gene = rownames(lig_expr), lig_expr),
#           file.path(inp, "ligand_expr_by_celltype.csv"), row.names = FALSE)
# write.csv(cbind(gene = rownames(rec_expr), rec_expr),
#           file.path(inp, "receptor_expr_by_celltype.csv"), row.names = FALSE)

# cat("written ligand_expr_by_celltype.csv with", nrow(lig_expr), "genes\n")
# cat("written receptor_expr_by_celltype.csv with", nrow(rec_expr), "genes\n")

# 按“细胞类型__基因”行、所有细胞列展开配体表达矩阵；只在对应细胞类型列填真实表达，其余填0
cells_all <- colnames(cnt)
lig_rows <- lig_map[lig_map$gene %in% rownames(cnt), , drop = FALSE]
if (nrow(lig_rows) > 0) {
  lig_cell_mat <- matrix(0,
                         nrow = nrow(lig_rows),
                         ncol = length(cells_all),
                         dimnames = list(paste(lig_rows$gene, lig_rows$celltype, sep = "__"),
                                         cells_all))
  for (cty in names(ct_map)) {
    idx <- which(lig_rows$celltype == cty)
    if (length(idx) == 0) next
    genes_cty <- lig_rows$gene[idx]
    cells_cty <- ct_map[[cty]]
    lig_cell_mat[idx, cells_cty] <- as.matrix(cnt[genes_cty, cells_cty, drop = FALSE])
  }
  write.csv(cbind(feature = rownames(lig_cell_mat), lig_cell_mat),
            file.path(inp, "ligand_expr_by_cell.csv"), row.names = FALSE)
  cat("written ligand_expr_by_cell.csv with", nrow(lig_cell_mat), "rows (celltype__gene) and",
      ncol(lig_cell_mat), "cells\n")
} else {
  cat("no ligand genes found in count matrix; skipped ligand_expr_by_cell.csv\n")
}

# 按“基因__细胞类型”行、所有细胞列展开受体表达矩阵；只在对应细胞类型列填真实表达，其余填0
rec_rows <- rec_map[rec_map$gene %in% rownames(cnt), , drop = FALSE]
if (nrow(rec_rows) > 0) {
  rec_cell_mat <- matrix(0,
                         nrow = nrow(rec_rows),
                         ncol = length(cells_all),
                         dimnames = list(paste(rec_rows$gene, rec_rows$celltype, sep = "__"),
                                         cells_all))
  for (cty in names(ct_map)) {
    idx <- which(rec_rows$celltype == cty)
    if (length(idx) == 0) next
    genes_cty <- rec_rows$gene[idx]
    cells_cty <- ct_map[[cty]]
    rec_cell_mat[idx, cells_cty] <- as.matrix(cnt[genes_cty, cells_cty, drop = FALSE])
  }
  write.csv(cbind(feature = rownames(rec_cell_mat), rec_cell_mat),
            file.path(inp, "receptor_expr_by_cell.csv"), row.names = FALSE)
  cat("written receptor_expr_by_cell.csv with", nrow(rec_cell_mat), "rows (gene__celltype) and",
      ncol(rec_cell_mat), "cells\n")
} else {
  cat("no receptor genes found in count matrix; skipped receptor_expr_by_cell.csv\n")
}
