library(Seurat)
library(scriabin)
library(ComplexHeatmap)
library(cowplot)
library(magrittr)
library(tibble)
library(Matrix)
library(readr)
library(dplyr)
library(lobstr)

rm(list = ls())
gc()

# ============================================================================
# L-R Interaction Scoring using CellPhoneDB-like product
#
# Inputs: deconvolved ligand_expr_by_cell_filtered 100列.csv,
#         receptor_expr_by_cell_filtered 100列.csv,
#         combo_only 100列.csv (custom L-R pairs with cell-type suffix)
# Metadata: OtherMethods/数据/preprocess/celltype_predictions.csv,
#           OtherMethods/数据/de_coords.csv,
#           OtherMethods/数据/Breast_Cancer_Block_A_Section_1 (for consistency checks)
# Goal: keep suffix (e.g., ADAM17__Bcell) and score only provided combos.
# ============================================================================

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(script_path) > 0) {
    return(dirname(normalizePath(script_path[1])))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE)))
  }
  getwd()
}

script_dir <- get_script_dir()
benchmark_root <- normalizePath(file.path(script_dir, "..", "..", ".."), mustWork = TRUE)
setwd(script_dir)

ligand_file <- file.path(benchmark_root, "5.Model_Training", "ligand_expr_by_cell_filtered 100列.csv")
receptor_file <- file.path(benchmark_root, "5.Model_Training", "receptor_expr_by_cell_filtered 100列.csv")
combo_file <- file.path(benchmark_root, "5.Model_Training", "combo_only 100列.csv")
celltype_file <- file.path(benchmark_root, "6.Results", "OtherMethods", "数据", "run_all_outputs_quan0.98_20260119_213616", "preprocess", "celltype_predictions.csv")
coords_file <- file.path(benchmark_root, "1.Preprocessing", "de_coords.csv")

ptm <- Sys.time()

cat("\n=== Loading deconvolved expression data ===\n")
ligand_expr <- read_csv(ligand_file, show_col_types = FALSE) |> as.data.frame()
rownames(ligand_expr) <- ligand_expr$feature
ligand_expr$feature <- NULL

receptor_expr <- read_csv(receptor_file, show_col_types = FALSE) |> as.data.frame()
rownames(receptor_expr) <- receptor_expr$feature
receptor_expr$feature <- NULL

common_cells <- intersect(colnames(ligand_expr), colnames(receptor_expr))
cat("Common cells:", length(common_cells), "\n")
ligand_expr <- ligand_expr[, common_cells]
receptor_expr <- receptor_expr[, common_cells]

overlap_genes <- intersect(rownames(ligand_expr), rownames(receptor_expr))
if (length(overlap_genes) > 0) {
  cat("Removing", length(overlap_genes), "overlapping genes from receptor matrix\n")
  receptor_expr <- receptor_expr[setdiff(rownames(receptor_expr), overlap_genes), ]
}

cat("\nKEEPING cell type suffix in gene names (e.g., ADAM17__Bcell)\n")
expr_combined <- rbind(ligand_expr, receptor_expr)
cat("Combined expression matrix:", nrow(expr_combined), "features x", ncol(expr_combined), "cells\n")

cat("\nLoading spatial coordinates from:", coords_file, "\n")
coords_df <- read_csv(coords_file, show_col_types = FALSE) |> as.data.frame()
rownames(coords_df) <- coords_df[, 1]
coords_df <- coords_df[, c("x", "y")]

cells_with_coords <- intersect(rownames(coords_df), common_cells)
cat("Cells with coordinates:", length(cells_with_coords), "\n")
expr_combined <- expr_combined[, cells_with_coords]
coords_df <- coords_df[cells_with_coords, ]

cat("Loading cell type predictions from:", celltype_file, "\n")
celltype_probs <- read_csv(celltype_file, show_col_types = FALSE) |> as.data.frame()
rownames(celltype_probs) <- celltype_probs[, 1]
celltype_probs <- celltype_probs[, -1]
celltype_probs <- t(celltype_probs)
predicted_celltypes <- colnames(celltype_probs)[apply(celltype_probs, 1, which.max)]
names(predicted_celltypes) <- rownames(celltype_probs)

common_spots <- intersect(names(predicted_celltypes), cells_with_coords)
cat("Common spots between predictions and expression:", length(common_spots), "\n")
if (length(common_spots) == 0) {
  stop("No overlap between cell type predictions and expression data. Check cell naming!")
}

expr_combined <- expr_combined[, common_spots]
coords_df <- coords_df[common_spots, ]
predicted_celltypes <- predicted_celltypes[common_spots]

cat("\nFinal data:", nrow(expr_combined), "features x", ncol(expr_combined), "cells\n")
cat("Cell types:", unique(predicted_celltypes), "\n")

cat("\nLoading custom L-R pairs from:", combo_file, "\n")
combo_df <- read_csv(combo_file, show_col_types = FALSE)
ligands <- c(); receptors <- c()
for (combo in combo_df$combo) {
  parts <- strsplit(combo, "\\|")[[1]]
  if (length(parts) != 2) {
    warning(paste("Skipping malformed combo:", combo))
    next
  }
  ligands <- c(ligands, parts[1])
  receptors <- c(receptors, parts[2])
}

lr_pairs <- data.frame(
  ligand = ligands,
  receptor = receptors,
  combo_key = combo_df$combo,
  stringsAsFactors = FALSE
)

normalize_feature_name <- function(x) gsub("_", "-", x, fixed = TRUE)
lr_pairs$ligand_seurat <- normalize_feature_name(lr_pairs$ligand)
lr_pairs$receptor_seurat <- normalize_feature_name(lr_pairs$receptor)

rownames(expr_combined) <- normalize_feature_name(rownames(expr_combined))

genes_in_data <- rownames(expr_combined)
lr_pairs_filtered <- lr_pairs[lr_pairs$ligand_seurat %in% genes_in_data & lr_pairs$receptor_seurat %in% genes_in_data, ]
cat("Filtered L-R pairs:", nrow(lr_pairs_filtered), "/", nrow(lr_pairs), "\n")
if (nrow(lr_pairs_filtered) == 0) {
  stop("No L-R pairs left after filtering. Gene names in combo_only do not match expression data.")
}

cat("\n=== Creating Seurat object ===\n")
expr_sparse <- as(as.matrix(expr_combined), "sparseMatrix")
metadata <- data.frame(celltype = predicted_celltypes, row.names = colnames(expr_combined))
options(Seurat.object.feature.name.validate = FALSE,
  Seurat.object.assay.validate = FALSE)
ser <- CreateSeuratObject(counts = expr_sparse, meta.data = metadata, assay = "Spatial", min.cells = 0, min.features = 0)
ser@meta.data$x <- coords_df[colnames(ser), "x"]
ser@meta.data$y <- coords_df[colnames(ser), "y"]

cat("Seurat object created with", ncol(ser), "cells and", nrow(ser), "features\n")

cat("\n=== Normalizing data ===\n")
DefaultAssay(ser) <- "Spatial"
ser <- NormalizeData(ser, verbose = FALSE)
ser <- FindVariableFeatures(ser, verbose = FALSE)
for (layer in c("counts", "data", "scale.data")) {
  tryCatch({
    mat <- GetAssayData(ser, assay = "Spatial", slot = layer)
    if (nrow(mat) > 0) {
      ser <- SetAssayData(ser, assay = "Spatial", slot = layer, new.data = mat)
    }
  }, error = function(e) {})
}

if (!dir.exists("./result")) dir.create("./result", recursive = TRUE)

cat("\n=== CellPhoneDB-style scoring with custom database ===\n")
expr_norm <- GetAssayData(ser, assay = "Spatial", slot = "data")

compute_cellphonedb_score <- function(ligand_expr, receptor_expr) {
  mean(ligand_expr) * mean(receptor_expr)
}

result_list <- list()
cat("Computing scores for", nrow(lr_pairs_filtered), "L-R pairs...\n")
pb <- txtProgressBar(min = 0, max = nrow(lr_pairs_filtered), style = 3)

for (i in seq_len(nrow(lr_pairs_filtered))) {
  ligand_name <- lr_pairs_filtered$ligand_seurat[i]
  receptor_name <- lr_pairs_filtered$receptor_seurat[i]
  if (!(ligand_name %in% rownames(expr_norm))) next
  if (!(receptor_name %in% rownames(expr_norm))) next
  ligand_vec <- as.numeric(expr_norm[ligand_name, ])
  receptor_vec <- as.numeric(expr_norm[receptor_name, ])
  lr_score <- compute_cellphonedb_score(ligand_vec, receptor_vec)
  result_list[[i]] <- data.frame(
    Ligand = lr_pairs_filtered$ligand[i],
    Receptor = lr_pairs_filtered$receptor[i],
    LRscore = lr_score,
    ligand_mean_expr = mean(ligand_vec),
    receptor_mean_expr = mean(receptor_vec),
    ligand_pct_expr = sum(ligand_vec > 0) / length(ligand_vec) * 100,
    receptor_pct_expr = sum(receptor_vec > 0) / length(receptor_vec) * 100,
    stringsAsFactors = FALSE
  )
  setTxtProgressBar(pb, i)
}
close(pb)

results_df <- bind_rows(result_list)
if (nrow(results_df) == 0) {
  stop("No communication scores were calculated successfully")
}

# Sort by LRscore descending
results_df <- results_df %>% arrange(desc(LRscore))

write_csv(results_df, "./result/result.csv")
cat("Saved results to ./result/result.csv\n")

cat("\n=== Completed ===\n")
cat("Runtime (min):", round(as.numeric(difftime(Sys.time(), ptm, units = "mins")), 2), "\n")
cat("Peak mem (GB):", mem_used() / 1024^3, "\n")
