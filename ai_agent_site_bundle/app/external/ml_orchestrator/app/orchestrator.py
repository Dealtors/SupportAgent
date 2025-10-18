import numpy as np
from app.embedder import Embedder
from app.classifier import Classifier
from app.clustering import HDBSCANPredictor
from app.retriever import faiss_search
from app.llm import qwen_generate
from app.logging_utils import log_case
from app.config import Cfg

_embedder = Embedder()
_classifier = Classifier()
_clusterer = HDBSCANPredictor()

def process_ticket(text: str) -> str:
    emb = _embedder.encode(text)
    label, conf = _classifier.predict(emb)
    cluster = None
    if conf < Cfg.MIN_CLASS_CONFIDENCE:
        cluster = _clusterer.predict(emb)
        if cluster is not None and label == "Unknown":
            label = f"Cluster_{cluster}"
    context = faiss_search(emb)
    answer = qwen_generate(text, context)
    log_case(text, label, conf, answer)
    return label, conf, answer
