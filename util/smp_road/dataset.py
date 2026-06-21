"""YOLO polygon segmentation 라벨을 binary mask 로 변환하는 Dataset.

cctv-roadseg-dataset 는 YOLO-seg 포맷(`class xn yn ...` 정규화 폴리곤)으로 되어 있다.
U-Net 학습에는 픽셀 단위 mask 가 필요하므로 라벨 폴리곤을 `cv2.fillPoly` 로 래스터화한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_image_paths(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: Path) -> Path:
    """`.../images/foo.jpg` -> `.../labels/foo.txt` (YOLO 규약)."""
    labels_dir = image_path.parent.parent / "labels"
    return labels_dir / (image_path.stem + ".txt")


def rasterize_yolo_polygons(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """YOLO-seg 라벨 파일을 0/1 binary mask(H,W) 로 변환한다.

    단일 클래스(road) 데이터셋이므로 class id 는 무시하고 모든 폴리곤을 합친다.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + 최소 3점(6값)
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        if coords.size % 2 != 0:
            coords = coords[:-1]
        pts = coords.reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


class RoadSegDataset(Dataset):
    """(image_tensor, mask_tensor) 쌍을 반환한다.

    transform 은 albumentations Compose(image+mask) 를 받는다. 반환 텐서는
    transform 의 ToTensorV2 결과(이미지 float CHW, mask uint8 HW)이다.
    """

    def __init__(self, images_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")
        self.image_paths = list_image_paths(self.images_dir)
        if not self.image_paths:
            raise RuntimeError(f"no images under {self.images_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask = rasterize_yolo_polygons(label_path_for(image_path), h, w)
        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self.load_pair(idx)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t, mask_t = out["image"], out["mask"]
            return image_t, mask_t.unsqueeze(0).float()
        return image, mask
