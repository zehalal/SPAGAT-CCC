import gzip
import shutil
from pathlib import Path

base = Path(__file__).resolve().parent
for name in ["matrix.mtx", "features.tsv", "barcodes.tsv"]:
    src = base / name
    dst = base / f"{name}.gz"
    with src.open("rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"gzipped {src.name} -> {dst.name}")
