from dataclasses import dataclass

import numpy as np


@dataclass
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int

    def normalized(self, width: int, height: int) -> "Roi":
        x0, x1 = sorted((int(round(self.x0)), int(round(self.x1))))
        y0, y1 = sorted((int(round(self.y0)), int(round(self.y1))))
        x0 = int(np.clip(x0, 0, max(0, width - 1)))
        x1 = int(np.clip(x1, x0 + 1, width))
        y0 = int(np.clip(y0, 0, max(0, height - 1)))
        y1 = int(np.clip(y1, y0 + 1, height))
        return Roi(x0, y0, x1, y1)


@dataclass
class CorrectionLayer:
    layer_id: int
    family: str
    enabled: bool
    roi: Roi
    component: dict
    params: dict
    score: float
    period: float
    correction_path: str
    noise_path: str
    weight_path: str
