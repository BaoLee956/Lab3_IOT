"""Convert the slide's Kaggle VOC dataset to YOLO; group before splitting."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), "big")


def read_items(root):
    items = []
    for xml in sorted((root / "annotations").glob("*.xml")):
        doc = ET.parse(xml).getroot()
        image_path = root / "images" / Path(doc.findtext("filename")).name
        with Image.open(image_path) as image:
            width, height = image.size
            if (width, height) != (int(doc.findtext("size/width")), int(doc.findtext("size/height"))):
                raise ValueError(f"Image/XML size mismatch: {xml}")
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"Normalize EXIF and update boxes first: {image_path}")
            fingerprint = dhash(image)
        labels, counts = [], [0, 0, 0]
        for obj in doc.findall("object"):
            label = obj.findtext("name")
            if label not in NAMES:
                raise ValueError(f"Unknown class {label} in {xml}")
            cls = NAMES.index(label)
            coords = [float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
            x1, y1, x2, y2 = coords
            # This Kaggle export is treated as pixel-edge coordinates.
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid box in {xml}: {coords}")
            labels.append(f"{cls} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} "
                          f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
            counts[cls] += 1
        items.append({"path": image_path, "labels": labels, "counts": counts,
                      "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "dhash": fingerprint})
    if not items:
        raise ValueError("No annotations/*.xml. Check the extracted dataset root.")
    return items


def group_items(items, threshold, manual=None):
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        parent[find(b)] = find(a)

    pairs = []
    for i, a in enumerate(items):
        for j in range(i):
            b = items[j]
            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            if a["sha256"] == b["sha256"] or distance <= threshold:
                join(i, j)
                pairs.append([a["path"].name, b["path"].name, distance])
    if manual:
        groups = defaultdict(list)
        known = {x["path"].name for x in items}
        with Path(manual).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["image"] not in known:
                    raise ValueError(f"Unknown image in group CSV: {row['image']}")
                groups[row["group"]].append(row["image"])
        lookup = {x["path"].name: i for i, x in enumerate(items)}
        for names in groups.values():
            for name in names[1:]:
                join(lookup[names[0]], lookup[name])
    grouped = defaultdict(list)
    for i in range(len(items)):
        grouped[find(i)].append(i)
    return list(grouped.values()), pairs


def split_groups(items, groups, seed=42):
    """Try deterministic group assignments; select closest class and image ratios."""
    total = np.array([sum(x["counts"][c] for x in items) for c in range(3)])
    if (total == 0).any():
        raise ValueError(f"All three classes are required; found {total.tolist()}")
    ratios = np.array([.70, .15, .15])
    best = None
    rng = random.Random(seed)
    for _ in range(3000):
        assignment = rng.choices(range(3), weights=ratios, k=len(groups))
        counts = np.zeros((3, 3), dtype=int)
        sizes = np.zeros(3, dtype=int)
        for group, split in zip(groups, assignment):
            sizes[split] += len(group)
            counts[split] += np.sum([items[i]["counts"] for i in group], axis=0)
        if (counts == 0).any() or (sizes == 0).any():
            continue
        score = np.mean((counts / total - ratios[:, None]) ** 2)
        score += np.mean((sizes / len(items) - ratios) ** 2)
        if best is None or score < best[0]:
            best = score, assignment
    if best is None:
        raise ValueError("Cannot put all three classes in every split without breaking groups")
    return best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/downloads/face-mask-detection.zip")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/yolo")
    parser.add_argument("--groups", help="Optional CSV: image,group (same person/session/event)")
    parser.add_argument("--dhash-distance", type=int, default=5)
    args = parser.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    if (out / "manifest.csv").exists():
        raise SystemExit("Output already has a split. Use another --out to preserve the frozen test set.")
    if not (raw / "annotations").exists():
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as archive:
            base = raw.resolve()
            for member in archive.infolist():
                destination = (raw / member.filename).resolve()
                if not destination.is_relative_to(base):
                    raise ValueError("Unsafe path in ZIP")
            archive.extractall(raw)
    items = read_items(raw)
    groups, pairs = group_items(items, args.dhash_distance, args.groups)
    assignment = split_groups(items, groups)
    out.mkdir(parents=True, exist_ok=True)
    split_names = ["train", "val", "test"]
    stats = {s: {"images": 0, "objects": dict.fromkeys(NAMES, 0)} for s in split_names}
    for split in split_names:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "group", "sha256", "dhash"])
        for group_id, (group, split_id) in enumerate(zip(groups, assignment)):
            split = split_names[split_id]
            for i in group:
                item = items[i]
                shutil.copy2(item["path"], out / "images" / split / item["path"].name)
                (out / "labels" / split / (item["path"].stem + ".txt")).write_text(
                    "\n".join(item["labels"]) + "\n", encoding="utf-8")
                writer.writerow([item["path"].name, split, group_id, item["sha256"], hex(item["dhash"])])
                stats[split]["images"] += 1
                for cls, count in zip(NAMES, item["counts"]):
                    stats[split]["objects"][cls] += count
    with (out / "similar-pairs.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["image_a", "image_b", "dhash_distance"], *pairs])
    # Relative dataset path works on Windows and in Colab after relocating the project.
    config = "path: data/yolo\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    if out.as_posix() != "data/yolo":
        config = config.replace("path: data/yolo", "path: " + out.resolve().as_posix())
    config += "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    (out / "data.yaml").write_text(config, encoding="utf-8")
    report = {"seed": 42, "target_ratios": [.7, .15, .15], "group_count": len(groups),
              "similar_pair_count": len(pairs), "dhash_distance": args.dhash_distance,
              "splits": stats, "note": "dHash is a heuristic; manually review identity/session leakage."}
    (out / "split-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
