from pathlib import Path
import numpy as np
import cv2

class ImageDataset:
    def __init__(self, root_dir: Path, pattern: str = "*.png"):
        self.root_dir = root_dir
        self.pattern = pattern
        self.images_path = self._scan_files()

        def _scan_files(self):
            return sorted(self.root_dir.glob(self.pattern))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")
        return img

