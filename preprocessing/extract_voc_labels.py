"""从 VOC2007 官方 Annotations 中提取评估子集图片的类别与 bbox。

用法:
    python preprocessing/extract_voc_labels.py

输出:
    data/voc_labels.csv  (image_id, class_id, class_name, bbox_x1y1x2y2, image_path)
"""
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "voc", "raw")
XML_DIR = os.path.join(DATA_DIR, "voc", "VOCdevkit", "VOC2007", "Annotations")
OUT_CSV = os.path.join(DATA_DIR, "voc_labels.csv")

# VOC2007 标准 20 类(顺序即 class_id)
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NAME2ID = {n: i for i, n in enumerate(VOC_CLASSES)}


def extract_label(xml_path):
    """取 XML 中第一个 object 的类别与 bbox。"""
    tree = ET.parse(xml_path)
    obj = tree.find("object")
    name = obj.findtext("name")
    bbox = obj.find("bndbox")
    box = [int(bbox.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
    return NAME2ID.get(name, -1), name, box


def main():
    rows, missing = [], []
    for img in sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        xml = os.path.join(XML_DIR, stem + ".xml")
        if not os.path.exists(xml):
            missing.append(stem)
            continue
        class_id, name, box = extract_label(xml)
        rows.append({
            "image_id": stem,
            "class_id": class_id,
            "class_name": name,
            "bbox_x1y1x2y2": box,
            "image_path": os.path.relpath(img, DATA_DIR),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"提取完成: {len(df)} 张有标签, 输出 -> {os.path.relpath(OUT_CSV)}")
    print(f"类别分布:\n{df['class_name'].value_counts().to_string()}")
    if missing:
        print(f"警告: {len(missing)} 张图片找不到对应 XML,如 {missing[:3]}")


if __name__ == "__main__":
    main()
