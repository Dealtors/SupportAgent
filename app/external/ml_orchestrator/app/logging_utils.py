import os, time
import ujson as json
from app.config import Cfg

def log_case(text: str, label: str, conf: float, answer: str):
    os.makedirs(os.path.dirname(Cfg.LOGS_PATH), exist_ok=True)
    rec = {
        "ts": int(time.time()),
        "query": text,
        "label": label,
        "confidence": conf,
        "answer": answer
    }
    with open(Cfg.LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
