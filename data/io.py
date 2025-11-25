from pathlib import Path
import numpy as np

class DataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def save_start_centroids(self, image_id: str, centroids: np.ndarray):
        out =self.base_dir / "star_centroids" / f"{image_id}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, centroids)
