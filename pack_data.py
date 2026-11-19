"""Freeze the reviewed split into a ZIP for uploading to Colab."""
import argparse
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data/yolo")
args = parser.parse_args()
root = Path(args.root)
if not (root / "manifest.csv").exists():
    raise SystemExit("Run prepare_data.py and review the split first.")
target = Path("data-yolo.zip")
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
    for file in sorted(root.rglob("*")):
        if file.is_file():
            archive.write(file, (Path("data/yolo") / file.relative_to(root)).as_posix())
print(target.resolve())
