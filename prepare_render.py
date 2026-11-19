"""Prepare a minimal, repeatable Render deployment bundle from this project."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
FILES = [
    "app.py", "detector.py", "alerts.py", "requirements.txt", "Dockerfile",
    ".dockerignore", ".gitignore", "render.yaml",
    "web/index.html", "web/favicon.svg", "supabase/alerts.sql", ".env.example",
    "HUONG_DAN_CANH_BAO.md", "telegram_setup.py", "assets/model.onnx",
]


def main():
    missing = [name for name in FILES if not (BASE / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    output = BASE / "deploy-render"
    output.mkdir(exist_ok=True)
    records = []
    for name in FILES:
        source, target = BASE / name, output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({"path": name, "bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {"files": records,
                "note": "Local packaging only; no Cloud deployment or accuracy claim."}
    (output / "deployment-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    archive = BASE / "deploy-render.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in [*FILES, "deployment-manifest.json"]:
            zipped.write(output / name, arcname=name)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        for record in records:
            actual = hashlib.sha256(zipped.read(record["path"])).hexdigest()
            if actual != record["sha256"]:
                raise RuntimeError(f"Archive checksum mismatch: {record['path']}")
    print(json.dumps({"folder": str(output), "archive": str(archive),
                      "archive_bytes": archive.stat().st_size,
                      "model": records[-1], "archive_verified": True}, indent=2))


if __name__ == "__main__":
    main()
