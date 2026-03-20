library(readr)
library(dplyr)
library(tidyr)

# Resolve paths
args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path[1]))
} else {
  # Support source() by checking sys.frames for ofile
  frame_files <- lapply(sys.frames(), function(x) x$ofile)
  frame_files <- Filter(Negate(is.null), frame_files)
  if (length(frame_files) > 0) {
    script_dir <- dirname(normalizePath(frame_files[[length(frame_files)]]))
  } else {
    script_dir <- getwd()
  }
}

combo_path <- normalizePath(file.path(script_dir, "..", "..", "5.Model_Training", "combo_only 100列.csv"), mustWork = FALSE)
cellchat_rds_path <- file.path(script_dir, "CellChatV2", "result", "result.rds")
commot_csv_path <- file.path(script_dir, "COMMOT", "result", "result.csv")

# Output paths
cellchat_out <- file.path(script_dir, "CellChatV2", "result", "result_combo_only.csv")
commot_out <- file.path(script_dir, "COMMOT", "result", "result_combo_only.csv")

# 1) Load combo definitions (1610 rows)
combo_df <- read_csv(combo_path, show_col_types = FALSE, col_types = cols()) %>%
  separate(combo, into = c("ligand_part", "receptor_part"), sep = "\\|") %>%
  mutate(
    Ligand   = sub("__.*", "", ligand_part),
    Receptor = sub("__.*", "", receptor_part),
    Sender   = sub(".*__", "", ligand_part),
    Receiver = sub(".*__", "", receptor_part)
  ) %>%
  select(Sender, Receiver, Ligand, Receptor)

stopifnot(nrow(combo_df) == 1610)

# Helper to ensure full combo coverage
fill_missing <- function(df, score_col) {
  df %>%
    right_join(combo_df, by = c("Sender", "Receiver", "Ligand", "Receptor")) %>%
    mutate({{ score_col }} := replace_na({{ score_col }}, 0)) %>%
    # Preserve original combo order
    mutate(.combo_id = row_number()) %>%
    arrange(.combo_id) %>%
    select(-.combo_id)
}

# 2) CellChat extraction
if (!file.exists(cellchat_rds_path)) {
  stop("CellChat result.rds not found at ", cellchat_rds_path)
}
cellchat_record <- readRDS(cellchat_rds_path)
cellchat_df <- cellchat_record$result %>%
  select(Sender, Receiver, Ligand, Receptor, LRscore)

cellchat_combo <- fill_missing(cellchat_df, LRscore)
write_csv(cellchat_combo, cellchat_out)
cat("Saved CellChat combo scores to: ", cellchat_out, "\n")
cat("Rows: ", nrow(cellchat_combo), "  Missing combos filled: ", sum(cellchat_combo$LRscore == 0), "\n")

# 3) COMMOT extraction
if (!file.exists(commot_csv_path)) {
  stop("COMMOT result.csv not found at ", commot_csv_path)
}
commot_df <- read_csv(commot_csv_path, show_col_types = FALSE, col_types = cols()) %>%
  rename(LRscore = score) %>%
  group_by(Sender, Receiver, Ligand, Receptor) %>%
  summarise(LRscore = max(LRscore, na.rm = TRUE), .groups = "drop")

commot_combo <- fill_missing(commot_df, LRscore)
write_csv(commot_combo, commot_out)
cat("Saved COMMOT combo scores to: ", commot_out, "\n")
cat("Rows: ", nrow(commot_combo), "  Missing combos filled: ", sum(commot_combo$LRscore == 0), "\n")
