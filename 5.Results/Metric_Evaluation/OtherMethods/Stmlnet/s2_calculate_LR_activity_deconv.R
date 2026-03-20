#############################################################
# stMLnet L-R Activity Calculation using Deconvolved Data
# 
# Uses stMLnet's scoring formula (direct implementation)
# Formula: LR_score[i] = Ligand[i] * Σ_j(Receptor[j] * (1/dist[i,j]))
# 
# Input:
#   - ligand_expr_by_cell_filtered 100列.csv (deconvolved)
#   - receptor_expr_by_cell_filtered 100列.csv (deconvolved)
#   - combo_only 100列.csv (L-R pairs)
#   - de_coords.csv (spatial coordinates)
#############################################################

rm(list = ls())
gc()

library(dplyr)
library(readr)
library(Matrix)

# Set paths
script_dir <- getwd()
bc1_dir <- normalizePath(file.path(script_dir, "..", "..", ".."), mustWork = FALSE)
setwd(script_dir)

ligand_file <- file.path(bc1_dir, "5.Model_Training", "ligand_expr_by_cell_filtered 100列.csv")
receptor_file <- file.path(bc1_dir, "5.Model_Training", "receptor_expr_by_cell_filtered 100列.csv")
combo_file <- file.path(bc1_dir, "5.Model_Training", "combo_only 100列.csv")
coords_file <- file.path(bc1_dir, "1.Preprocessing", "de_coords.csv")

# Output directory
output_dir <- file.path(getwd(), "result_deconv")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

ptm <- Sys.time()

cat("\n=== stMLnet L-R Scoring (stMLnet Formula + Deconvolved Data) ===\n\n")

# ============================================================================
# 1. Load Data
# ============================================================================

cat("Step 1: Loading data...\n")

# Load ligand expression
cat("  Loading ligand expression...\n")
ligand_expr <- read_csv(ligand_file, show_col_types = FALSE)
ligand_expr <- as.data.frame(ligand_expr)
rownames(ligand_expr) <- ligand_expr$feature
ligand_expr$feature <- NULL

# Load receptor expression
cat("  Loading receptor expression...\n")
receptor_expr <- read_csv(receptor_file, show_col_types = FALSE)
receptor_expr <- as.data.frame(receptor_expr)
rownames(receptor_expr) <- receptor_expr$feature
receptor_expr$feature <- NULL

# Get common spots
common_spots <- intersect(colnames(ligand_expr), colnames(receptor_expr))
cat("  Common spots:", length(common_spots), "\n")

ligand_expr <- ligand_expr[, common_spots]
receptor_expr <- receptor_expr[, common_spots]

# Combine expression matrix
exprMat <- rbind(ligand_expr, receptor_expr)
cat("  Combined expression matrix:", nrow(exprMat), "features x", ncol(exprMat), "spots\n")

# Load spatial coordinates
cat("  Loading spatial coordinates...\n")
coords <- read_csv(coords_file, show_col_types = FALSE)
coords_df <- as.data.frame(coords)
rownames(coords_df) <- coords_df[[1]]
coords_df[[1]] <- NULL

# Align coordinates with expression
coords_df <- coords_df[common_spots, c("x", "y")]
cat("  Spots with coordinates:", nrow(coords_df), "\n")

# Load L-R pairs
cat("  Loading L-R pairs from combo_only...\n")
combo_df <- read_csv(combo_file, show_col_types = FALSE)
combo_df$ligand <- sapply(strsplit(combo_df$combo, "\\|"), `[`, 1)
combo_df$receptor <- sapply(strsplit(combo_df$combo, "\\|"), `[`, 2)

# Filter to available pairs
lr_pairs <- combo_df[
  combo_df$ligand %in% rownames(exprMat) &
  combo_df$receptor %in% rownames(exprMat),
]

cat("  L-R pairs to score:", nrow(lr_pairs), "/", nrow(combo_df), "\n")

# ============================================================================
# 2. Calculate Distance Matrix
# ============================================================================

cat("\nStep 2: Computing spatial distance matrix...\n")
distMat <- as.matrix(dist(coords_df))
cat("  Distance matrix:", nrow(distMat), "x", ncol(distMat), "\n")

# Create distance weight matrix: 1/distance
cat("  Applying stMLnet weighting: 1/distance\n")
distMat_weight <- 1 / (distMat + 1e-6)  # Add epsilon to avoid Inf
diag(distMat_weight) <- 0  # No self-interaction

# ============================================================================
# 3. Compute L-R Scores Using stMLnet Formula
# ============================================================================

cat("\nStep 3: Computing L-R scores using stMLnet formula...\n")
cat("  Formula: LR_score[i] = Ligand[i] × Σ_j(Receptor[j] × (1/dist[i,j]))\n\n")

result_list <- list()

cat("  Processing", nrow(lr_pairs), "L-R pairs...\n")
pb <- txtProgressBar(min = 0, max = nrow(lr_pairs), style = 3)

for (i in seq_len(nrow(lr_pairs))) {
  ligand_name <- lr_pairs$ligand[i]
  receptor_name <- lr_pairs$receptor[i]
  
  # Get expression vectors
  ligand_vec <- as.numeric(exprMat[ligand_name, ])
  receptor_vec <- as.numeric(exprMat[receptor_name, ])
  
  # Calculate L-R scores using stMLnet formula
  # For each spot i: LR_score[i] = Ligand[i] × Σ_j(Receptor[j] × distWeight[i,j])
  weighted_receptor <- distMat_weight %*% receptor_vec  # Matrix-vector multiplication
  lr_scores <- ligand_vec * as.numeric(weighted_receptor)
  
  # Store results
  result_list[[i]] <- data.frame(
    ligand = ligand_name,
    receptor = receptor_name,
    combo = paste(ligand_name, receptor_name, sep = "|"),
    mean_score = mean(lr_scores, na.rm = TRUE),
    median_score = median(lr_scores, na.rm = TRUE),
    max_score = max(lr_scores, na.rm = TRUE),
    sum_score = sum(lr_scores, na.rm = TRUE),
    sd_score = sd(lr_scores, na.rm = TRUE),
    ligand_mean_expr = mean(ligand_vec, na.rm = TRUE),
    receptor_mean_expr = mean(receptor_vec, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  
  setTxtProgressBar(pb, i)
}

close(pb)

# Combine results
result_df <- do.call(rbind, result_list)
rownames(result_df) <- NULL

# Sort by mean score
result_df <- result_df[order(-result_df$mean_score), ]

# ============================================================================
# 4. Save Results
# ============================================================================

cat("\n\nStep 4: Saving results...\n")

used.time <- Sys.time() - ptm

result_record <- list(
  result = result_df,
  method = "stMLnet formula (direct implementation)",
  formula = "LR_score[i] = Ligand[i] * Σ_j(Receptor[j] * (1/dist[i,j]))",
  database = "combo_only 100列.csv",
  n_spots = ncol(exprMat),
  n_lr_pairs = nrow(result_df),
  used_time = used.time
)

# Save RDS
output_rds <- file.path(output_dir, "LR_activity_scores.rds")
saveRDS(result_record, file = output_rds)
cat("  Saved RDS:", output_rds, "\n")

# Save CSV
output_csv <- file.path(output_dir, "LR_activity_scores.csv")
write_csv(result_df, output_csv)
cat("  Saved CSV:", output_csv, "\n")

# ============================================================================
# 5. Summary
# ============================================================================

cat("\n=== Results Summary ===\n")
cat("Total L-R pairs scored:", nrow(result_df), "\n")
cat("Scoring method: stMLnet formula (1/distance weighting)\n")
cat("Time used:", format(used.time), "\n\n")

cat("Top 20 L-R interactions by mean score:\n")
print(head(result_df[, c("ligand", "receptor", "mean_score", "ligand_mean_expr", "receptor_mean_expr")], 20))

cat("\n\nScore statistics (mean_score):\n")
cat("  Min:", min(result_df$mean_score, na.rm = TRUE), "\n")
cat("  Max:", max(result_df$mean_score, na.rm = TRUE), "\n")
cat("  Mean:", mean(result_df$mean_score, na.rm = TRUE), "\n")
cat("  Median:", median(result_df$mean_score, na.rm = TRUE), "\n")

cat("\n=== Analysis Complete ===\n")
cat("Output directory:", output_dir, "\n")
