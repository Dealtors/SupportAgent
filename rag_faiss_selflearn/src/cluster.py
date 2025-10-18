"""
cluster.py — модуль для кластеризации обращений (тикетов) с помощью HDBSCAN.

Входные файлы:
    embeddings/ticket_embeddings.npy  # эмбеддинги обращений (получает ML1)
    data/logs.jsonl                  # журнал (source для привязки id)

Выходные файлы:
    models/hdbscan_model.pkl         # обученная кластерная модель
    models/clusters.pkl              # mapping: ticket_index → cluster_id

Цель:
 - сгруппировать обращения по смыслу
 - использовать для fallback-контекста, если классификатор не уверен

Пример использования:
-----------------------------------
from cluster import ClusterModel
cm = ClusterModel()
cm.build_clusters()
cluster_id = cm.get_cluster_for_query(embedding_vector)
print("Кластер:", cluster_id)
-----------------------------------
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
import hdbscan

# Пути к директориям
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EMB_PATH = os.path.join(BASE_DIR, "embeddings", "ticket_embeddings.npy")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hdbscan_model.pkl")
CLUSTERS_PATH = os.path.join(BASE_DIR, "models", "clusters.pkl")
LOG_PATH = os.path.join(BASE_DIR, "data", "logs_cluster.jsonl")


def log_event(event_type: str, message: str, extra: dict = None):
    """Запись события в лог (формат JSONL)."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "message": message
    }
    if extra:
        record.update(extra)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ClusterModel:
    def __init__(self):
        self.model = None
        self.clusters = None

    def build_clusters(self, min_cluster_size: int = 3):
        """
        Строит кластерную модель HDBSCAN по эмбеддингам тикетов.
        min_cluster_size — минимальный размер кластера.
        """
        if not os.path.exists(EMB_PATH):
            raise FileNotFoundError(f"Файл эмбеддингов не найден: {EMB_PATH}")

        embeddings = np.load(EMB_PATH).astype("float32")
        log_event("info", "Начало кластеризации", {"count": embeddings.shape[0]})

        # Обучаем HDBSCAN
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = self.model.fit_predict(embeddings)

        # Сохраняем: id тикета → кластер
        self.clusters = {int(i): int(label) for i, label in enumerate(cluster_labels)}

        # Сохранение модели
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(CLUSTERS_PATH, "wb") as f:
            pickle.dump(self.clusters, f)

        log_event(
            "success",
            "Кластеры построены и сохранены",
            {
                "unique_clusters": len(set(cluster_labels)),
                "num_noise": sum(1 for x in cluster_labels if x == -1)
            }
        )
        return self.clusters

    def load_clusters(self):
        """Загружает сохранённые кластер и модель."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLUSTERS_PATH):
            raise FileNotFoundError("Кластерная модель не найдена. Сначала запусти build_clusters().")

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(CLUSTERS_PATH, "rb") as f:
            self.clusters = pickle.load(f)

        log_event("info", "Модель кластеризации загружена", {"clusters_count": len(self.clusters)})

    def get_cluster_for_vector(self, embedding: np.ndarray) -> int:
        """
        Определяет кластер нового обращения на основе его эмбеддинга.
        embedding должен быть numpy массивом формы (dim,) или (1, dim)
        """
        if self.model is None:
            self.load_clusters()

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        try:
            label = self.model.approximate_predict(embedding)
            return int(label[0])
        except Exception:
            # Если не поддерживается approximate_predict
            label = self.model.predict(embedding)
            return int(label[0])


# Пример самостоятельного запуска
if __name__ == "__main__":
    cm = ClusterModel()
    clusters = cm.build_clusters(min_cluster_size=3)
    print("Кластеры:", clusters)
