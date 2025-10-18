import joblib
import numpy as np
from typing import Tuple

class Classifier:
    def __init__(self, path: str = "./models/classifier.joblib"):
        try:
            obj = joblib.load(path)
            self.model = obj["model"]
            self.labels = obj["labels"]
        except Exception:
            self.model = None
            self.labels = []

    def predict(self, emb: np.ndarray) -> Tuple[str, float]:
        if self.model is None:
            return "Unknown", 0.0
        prob = self.model.predict_proba([emb])[0]
        idx = int(np.argmax(prob))
        return self.labels[idx], float(prob[idx])
