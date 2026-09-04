"""下载 VOC2007 trainval 数据集(主要为 Annotations 标注,与资源里的 500 张子集图片配套)。

用法:
    python preprocessing/download_voc.py

说明:
    官方包约 440MB,下载后解压到 data/voc/VOCdevkit/,
    其中 Annotations/ 存放每张图的类别与 bbox 标注。
"""
import os

import torchvision.datasets as datasets

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VOC_ROOT = os.path.join(DATA_DIR, "voc")

if __name__ == "__main__":
    datasets.VOCDetection(root=VOC_ROOT, year="2007", image_set="trainval", download=True)
    print(f"VOC2007 trainval 就绪: {VOC_ROOT}")
