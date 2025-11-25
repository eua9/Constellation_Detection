from dataclasses import dataclass
from pathlib import Path

@dataclass
class DetectionConfig:
    data_root: Path
    output_root: Path
    log_level: str = "INFO"