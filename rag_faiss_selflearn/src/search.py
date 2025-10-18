"""
search.py — модуль для создания и использования FAISS-индекса
по смысловым эмбеддингам документов базы знаний.

Входные файлы:
    embeddings/base_embeddings.npy   # эмбеддинги документов
    models/metadata.pkl              # метаданные (соответствие id -> текст, source)

Выходные файлы:
    models/faiss_index.bin           # бинарный FAISS-индекс

Пример использования:
-----------------------------------
from search import SemanticSearch

search = SemanticSearch()
search.build_faiss_index()  # только при первом запуске
results = search.search_similar_by_text("ошибка деплоя docker", k=3)
print(results)
-----------------------------------
"""

import os
import pickle
import numpy as np
import faiss
import json
from datetime import datetime

# === Импорт модели эмбеддингов (ML1 должен предоставить embeddings_loader.py) ===
try:
    from embeddings_loader import embed_text
except ImportError:
    raise ImportError("Не найден модуль embeddings_loader.py. Убедитесь, что ML1 добавил функцию embed_text.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # rag_faiss_selflearn/
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "base_embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.pkl")
INDEX_PATH = os.path.join(BASE_DIR, "models", "faiss_index.bin")
LOG_PATH = os.path.join(BASE_DIR, "data", "logs_faiss.jsonl")


def log_event(event_type: str, message: str, extra: dict = None):
    """Записывает событие в jsonl-лог."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "message": message
    }
    if extra:
        record.update(extra)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class SemanticSearch:
    def __init__(self, index_path: str = INDEX_PATH):
        self.index_path = index_path
        self.index = None
        self.metadata = None
        self._load_metadata()
        self._load_index()

    def _load_metadata(self):
        """Загрузка метаданных (id -> текст, путь к файлу и т.п.)."""
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Метаданные не найдены: {METADATA_PATH}")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        log_event("info", "Метаданные загружены", {"path": METADATA_PATH})

    def _load_index(self):
        """Загрузка FAISS индекса, если он уже существует."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            log_event("info", "FAISS индекс успешно загружен", {"path": self.index_path})
        else:
            log_event("warning", "FAISS индекс не найден, необходимо построить", {"path": self.index_path})

    def build_faiss_index(self, use_hnsw: bool = False):
        """
        Создание и сохранение FAISS-индекса из эмбеддингов.
        use_hnsw = True — использовать HNSW (более быстрый при большом количестве документов),
        по умолчанию False — обычный IndexFlatIP.
        """
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Файл эмбеддингов не найден: {EMBEDDINGS_PATH}")

        embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
        dim = embeddings.shape[1]  # размер эмбеддинга

        log_event("info", "Начало построения FAISS индекса", {"dim": int(dim), "count": int(embeddings.shape[0])})

        if use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 — размер графа
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
        else:
            index = faiss.IndexFlatIP(dim)  # dot product similarity

        index.add(embeddings)

        # Сохраняем
        faiss.write_index(index, self.index_path)
        self.index = index
        log_event("success", "FAISS индекс успешно построен и сохранён", {"path": self.index_path})

    def search_similar(self, query_embedding: np.ndarray, k: int = 3):
        """
        Выполняет поиск ближайших документов по эмбеддингу.
        query_embedding должен быть np.ndarray формы (1, dim).
        Возвращает список словарей: id, text, source, score.
        """
        if self.index is None:
            raise ValueError("Индекс не загружен. Построй или загрузить индекс через build_faiss_index()")

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1).astype("float32")

        D, I = self.index.search(query_embedding, k)  # D – distance, I – indices
        results = []
        for rank, (doc_id, score) in enumerate(zip(I[0], D[0]), start=1):
            meta = self.metadata["documents"][doc_id]
            results.append({
                "rank": rank,
                "id": int(doc_id),
                "score": float(score),
                "source": meta.get("path", ""),
                "text": meta.get("text", "")[:500]  # первые 500 символов для краткости
            })
        log_event("info", "Поиск выполнен", {"query_score_top": results[0]["score"] if results else None})
        return results

    def search_similar_by_text(self, query: str, k: int = 3):
        """
        Выполняет поиск ближайших документов по текстовому запросу.
        1. Преобразует запрос в эмбеддинг (embed_text)
        2. Выполняет поиск через FAISS
        """
        if not query or not isinstance(query, str):
            raise ValueError("Запрос должен быть строкой")

        try:
            query_vec = embed_text(query).astype("float32")
        except Exception as e:
            log_event("error", "Ошибка эмбеддинга", {"exception": str(e), "query": query})
            raise RuntimeError(f"Ошибка преобразования текста в эмбеддинг: {str(e)}")

        return self.search_similar(query_vec, k)


# Пример самостоятельного запуска
if __name__ == "__main__":
    search = SemanticSearch()

    # Построение индекса (при первом запуске)
    if search.index is None:
        search.build_faiss_index(use_hnsw=False)

    # Пример поиска
    query = "Ошибка деплоя docker"
    results = search.search_similar_by_text(query, k=3)
    for r in results:
        print(f"{r['rank']}. [{r['score']:.4f}] {r['source']}: {r['text'][:200]}...")
