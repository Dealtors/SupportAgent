import joblib
import numpy as np
from typing import Optional

class HDBSCANPredictor:
    def __init__(self, path: str = "./models/hdbscan_model.joblib"):
        try:
            self.model = joblib.load(path)
        except Exception:
            self.model = None

    def predict(self, emb: np.ndarray) -> Optional[int]:
        if self.model is None:
            return None
        try:
            from hdbscan import prediction
            lbl, _ = prediction.approximate_predict(self.model, [emb])
            return int(lbl[0])
        except Exception:
            return None
