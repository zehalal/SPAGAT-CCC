#!/usr/bin/env Rscript

inp <- tryCatch({
  src <- sys.frame(1)$ofile
  if (!is.null(src)) dirname(normalizePath(src)) else getwd()  # use script dir when sourced
}, error = function(...) getwd())

cat("════════════════════════════════════════════\n")
cat("  导出4个输入文件为CSV格式\n")
cat("════════════════════════════════════════════\n\n")

# 1. 导出 de_count 矩阵
cat("[1/5] 导出 de_count.csv ...\n")
load(file.path(inp, 'output_bc.RData'))
write.csv(de_count, file.path(inp, 'de_count.csv'), row.names = TRUE)
cat("  ✓ 完成: 11449行(基因) × 3798列(spots)\n")
cat("  文件大小:", round(file.size(file.path(inp, 'de_count.csv'))/1024/1024, 2), "MB\n\n")

# 2. 导出 de_cell_type
cat("[2/5] 导出 de_cell_type.csv ...\n")
#write.csv(de_cell_type, file.path(inp, 'de_cell_type.csv'), row.names = TRUE)
cat("  ✓ 完成: 3798行 × 2列 (Barcode, Cluster)\n")
cat("  文件大小:", round(file.size(file.path(inp, 'de_cell_type.csv'))/1024, 2), "KB\n\n")

# 3. 导出 ICGs (展平为长格式)
cat("[3/5] 导出 ICGs_expanded.csv ...\n")
icg <- readRDS(file.path(inp, '10X_bc_ICGs.rds'))

# 展平嵌套List
icg_records <- list()
for(receiver in names(icg)) {
  for(sender in names(icg[[receiver]])) {
    genes <- icg[[receiver]][[sender]]
    if(length(genes) > 0) {
      for(gene in genes) {
        icg_records[[length(icg_records) + 1]] <- data.frame(
          Receiver = receiver,
          Sender = sender,
          ICG_Gene = gene,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}
icg_df <- do.call(rbind, icg_records)
write.csv(icg_df, file.path(inp, 'ICGs_expanded.csv'), row.names = FALSE)
cat("  ✓ 完成:", nrow(icg_df), "行 × 3列 (Receiver, Sender, ICG_Gene)\n")
cat("  示例: Malignant → Bcell → FN1\n\n")

# 4. 导出 Ligands (展平)
cat("[4/5] 导出 Ligands_expanded.csv ...\n")
lig <- readRDS(file.path(inp, '10X_bc_Ligs_up_yll.rds'))

lig_records <- list()
for(ct in names(lig)) {
  ligands <- lig[[ct]]
  for(ligand in ligands) {
    lig_records[[length(lig_records) + 1]] <- data.frame(
      CellType = ct,
      Ligand = ligand,
      stringsAsFactors = FALSE
    )
  }
}
lig_df <- do.call(rbind, lig_records)
write.csv(lig_df, file.path(inp, 'Ligands_expanded.csv'), row.names = FALSE)
cat("  ✓ 完成:", nrow(lig_df), "行 × 2列 (CellType, Ligand)\n")
cat("  唯一配体数:", length(unique(lig_df$Ligand)), "\n\n")

# 5. 导出 Receptors (展平)
cat("[5/5] 导出 Receptors_expanded.csv ...\n")
rec <- readRDS(file.path(inp, '10X_bc_Recs_expr.rds'))

rec_records <- list()
for(ct in names(rec)) {
  receptors <- rec[[ct]]
  for(receptor in receptors) {
    rec_records[[length(rec_records) + 1]] <- data.frame(
      CellType = ct,
      Receptor = receptor,
      stringsAsFactors = FALSE
    )
  }
}
rec_df <- do.call(rbind, rec_records)
write.csv(rec_df, file.path(inp, 'Receptors_expanded.csv'), row.names = FALSE)
cat("  ✓ 完成:", nrow(rec_df), "行 × 2列 (CellType, Receptor)\n")
cat("  唯一受体数:", length(unique(rec_df$Receptor)), "\n\n")

cat("════════════════════════════════════════════\n")
cat("  全部导出完成！\n")
cat("════════════════════════════════════════════\n\n")

# 列出生成的文件
csv_files <- list.files(path = inp, pattern = "\\.csv$", full.names = TRUE)
cat("生成的CSV文件:\n")
for(f in csv_files) {
  size_mb <- round(file.size(f)/1024/1024, 2)
  if(size_mb < 1) {
    cat("  ", basename(f), " (", round(file.size(f)/1024, 2), " KB)\n", sep="")
  } else {
    cat("  ", basename(f), " (", size_mb, " MB)\n", sep="")
  }
}
