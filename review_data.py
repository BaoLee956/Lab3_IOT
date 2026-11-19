"""Create a local HTML review page with the real images and original VOC boxes."""
import csv
import html
import xml.etree.ElementTree as ET
from pathlib import Path

NAMES = {"with_mask": "Đeo đúng", "without_mask": "Không đeo", "mask_weared_incorrect": "Đeo sai"}


def preview(name):
    xml = Path("data/raw/annotations") / (Path(name).stem + ".xml")
    root = ET.parse(xml).getroot()
    width, height = int(root.findtext("size/width")), int(root.findtext("size/height"))
    rects = []
    for obj in root.findall("object"):
        x1, y1, x2, y2 = [float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
        title = html.escape(NAMES[obj.findtext("name")])
        rects.append(f'<span title="{title}" style="left:{x1/width*100}%;top:{y1/height*100}%;'
                     f'width:{(x2-x1)/width*100}%;height:{(y2-y1)/height*100}%"></span>')
    return f'<figure><div class="photo"><img src="../data/raw/images/{html.escape(name)}">{"".join(rects)}</div><figcaption>{html.escape(name)}</figcaption></figure>'


def main():
    with Path("data/yolo/similar-pairs.csv").open(encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    sections = ['<h1>Rà soát dữ liệu trước huấn luyện</h1><p>Di chuột lên khung để xem nhãn. Các cặp phía dưới đã được giữ cùng nhóm. Kiểm tra thêm những ảnh cùng người/cùng sự kiện nhưng khác góc chụp.</p><h2>Cặp ảnh gần giống nhau</h2>']
    for pair in pairs:
        sections.append('<div class="pair">' + preview(pair["image_a"]) + preview(pair["image_b"]) + '</div>')
    sections.append('<h2>Mẫu có nhãn đeo sai</h2>')
    count = 0
    for xml in sorted(Path("data/raw/annotations").glob("*.xml")):
        root = ET.parse(xml).getroot()
        if any(obj.findtext("name") == "mask_weared_incorrect" for obj in root.findall("object")):
            sections.append(preview(root.findtext("filename")))
            count += 1
            if count == 10:
                break
    style = 'body{font:16px system-ui;margin:24px;max-width:1100px}.pair{display:flex;gap:16px;margin-bottom:24px}figure{margin:0;width:440px}.photo{position:relative}.photo img{width:100%;display:block}.photo span{position:absolute;border:2px solid #dc2626;box-sizing:border-box}figcaption{font-size:14px;padding:6px}'
    target = Path("evidence/data-review.html")
    target.parent.mkdir(exist_ok=True)
    target.write_text(f'<!doctype html><html lang="vi"><meta charset="utf-8"><title>Rà soát dataset</title><style>{style}</style>{"".join(sections)}</html>', encoding="utf-8")
    print(target.resolve())


if __name__ == "__main__":
    main()
