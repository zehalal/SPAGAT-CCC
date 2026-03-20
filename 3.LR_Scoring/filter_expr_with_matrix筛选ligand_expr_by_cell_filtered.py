import pandas as pd
from pathlib import Path
from typing import Set, Tuple


ROOT = Path(__file__).resolve().parent
COMBO_PATH = ROOT / "combo_only 100列.csv"
LIGAND_IN = ROOT / "ligand_expr_by_cell.csv"
RECEPTOR_IN = ROOT / "receptor_expr_by_cell.csv"
LIGAND_OUT = ROOT / "ligand_expr_by_cell_filtered 100列.csv"
RECEPTOR_OUT = ROOT / "receptor_expr_by_cell_filtered 100列.csv"


def load_feature_sets(combo_path: Path) -> Tuple[Set[str], Set[str]]:
    lig_set: Set[str] = set()
    rec_set: Set[str] = set()

    with combo_path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            combo = line.strip()
            if not combo or "|" not in combo:
                continue
            # Keep the full feature token (gene + cell type) so it matches the matrices directly.
            left, right = combo.split("|", 1)
            lig_set.add(left.strip())
            rec_set.add(right.strip())
    return lig_set, rec_set


def filter_matrix(csv_path: Path, genes: Set[str], out_path: Path) -> int:
    df = pd.read_csv(csv_path)
    feature_col = "feature" if "feature" in df.columns else df.columns[0]
    filtered = df[df[feature_col].isin(genes)]
    filtered.to_csv(out_path, index=False)
    return filtered.shape[0]


def main() -> None:
    lig_set, rec_set = load_feature_sets(COMBO_PATH)
    lig_rows = filter_matrix(LIGAND_IN, lig_set, LIGAND_OUT)
    rec_rows = filter_matrix(RECEPTOR_IN, rec_set, RECEPTOR_OUT)
    print(f"Ligand features kept: {lig_rows} / {len(lig_set)}")
    print(f"Receptor features kept: {rec_rows} / {len(rec_set)}")


if __name__ == "__main__":
    main()
