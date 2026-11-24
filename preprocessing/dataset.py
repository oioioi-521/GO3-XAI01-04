"""全组统一的数据加载入口。

约定(见 data/README.md):
    - 输入统一 224x224,ImageNet 均值方差归一化
    - CHNCXR 灰度图自动 L->RGB 三通道复制

用法:
    from preprocessing.dataset import load_image, get_metadata, get_split
"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
META_CSV = os.path.join(DATA_DIR, "metadata.csv")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_image(image_path):
    """读图 -> tensor (3, 224, 224),已归一化。灰度图自动复制为三通道。"""
    # metadata uses Windows separators; normalizing both kinds keeps shared
    # CSV files portable across Windows and Linux runners.
    image_path = str(image_path).replace("\\", os.sep).replace("/", os.sep)
    if not os.path.isabs(image_path):
        image_path = os.path.join(DATA_DIR, image_path)
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _transform(img)


def get_metadata():
    """返回 metadata.csv 全表。"""
    return pd.read_csv(META_CSV)


def get_split(dataset, split):
    """取某个数据集的 eval/debug 子集清单。"""
    meta = get_metadata()
    return meta[(meta["dataset"] == dataset) & (meta["split"] == split)].reset_index(drop=True)


class MetadataDataset(Dataset):
    """Torch dataset backed by the frozen ``metadata.csv`` manifest."""

    def __init__(self, dataset, split="debug", limit=None):
        self.rows = get_split(dataset, split)
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError("limit must be positive")
            self.rows = self.rows.iloc[: int(limit)].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"no metadata rows for dataset={dataset!r}, split={split!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return {
            "image": load_image(row["image_path"]),
            "image_id": str(row["image_id"]),
            "target": int(row["class_id"]),
            "class_name": str(row["class_name"]),
            "image_path": str(Path(str(row["image_path"].replace("\\", "/")))),
        }
