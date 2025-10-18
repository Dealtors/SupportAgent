import os
import faiss
import numpy as np
import ujson as json
from typing import List, Tuple
from app.config import Cfg

class Retriever:
    def __init__(self, index_path: str = None, kb_path: str = None):
        self.index_path = index_path or Cfg.FAISS_INDEX_PATH
        self.kb_path = kb_path or Cfg.KB_JSONL
        self.index = None
        self.meta = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.kb_path):
            with open(self.kb_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self.meta.append(obj.get("text") or obj.get("content") or "")
                    except Exception:
                        continue

    def search(self, emb: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        x = np.array([emb]).astype("float32")
        D, I = self.index.search(x, top_k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    def fetch_context(self, ids: List[int]) -> str:
        parts = []
        for idx in ids:
            if 0 <= idx < len(self.meta):
                parts.append(self.meta[idx].strip())
        return "\n---\n".join(parts)

def faiss_search(emb, top_k: int = None) -> str:
    r = Retriever()
    res = r.search(emb, top_k or Cfg.TOP_K_CONTEXT)
    ids = [i for i, _ in res]
    return r.fetch_context(ids)
