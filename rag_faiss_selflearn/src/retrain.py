"""
retrain.py — Production-модуль полного переобучения (Full Retraining)

◾ Использует только dataset_train.csv
◾ Никаких ручных review
◾ Перезапускает эмбеддинги, FAISS, HDBSCAN
◾ Учитывает версии модели
◾ Можно запускать:
    - retrain_models(force=False) → автопроверка объема данных
    - retrain_models(force=True) → принудительно
"""

import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_DATASET = os.path.join(BASE_DIR, "data", "dataset_train.csv")
MODEL_VERSION_FILE = os.path.join(BASE_DIR, "models", "model_version.txt")
LAST_RETRAIN_COUNT = os.path.join(BASE_DIR, "models", "last_retrain_count.txt")

# Порог для автоматического обучения
MIN_NEW_SAMPLES = int(os.getenv("MIN_NEW_SAMPLES", "50"))

# Импорт ML-компонентов
from search import SemanticSearch
from cluster import ClusterModel

try:
    from embeddings_loader import rebuild_embeddings
except ImportError:
    raise ImportError("Функция rebuild_embeddings() не найдена (нужно реализовать в ML1)")

def log_event(message: str):
    print(f"[RETRAIN] {datetime.utcnow().isoformat()} - {message}")

def _read_version() -> int:
    if not os.path.exists(MODEL_VERSION_FILE):
        return 0
    try:
        return int(open(MODEL_VERSION_FILE).read().strip())
    except:
        return 0

def _write_version(version: int):
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write(str(version))

def _read_last_count() -> int:
    if not os.path.exists(LAST_RETRAIN_COUNT):
        return 0
    try:
        return int(open(LAST_RETRAIN_COUNT).read().strip())
    except:
        return 0

def _write_last_count(count: int):
    with open(LAST_RETRAIN_COUNT, "w") as f:
        f.write(str(count))

def _count_train_rows() -> int:
    if not os.path.exists(TRAIN_DATASET):
        return 0
    df = pd.read_csv(TRAIN_DATASET)
    return len(df)

def retrain_models(force: bool = False) -> bool:
    """
    Основной цикл retrain.
    """
    log_event("=== SELF-LEARNING START ===")

    if not os.path.exists(TRAIN_DATASET):
        log_event("dataset_train.csv не найден. Обучение невозможно.")
        return False

    total_rows = _count_train_rows()
    last_rows = _read_last_count()
    new_rows = total_rows - last_rows

    log_event(f"Всего обучающих примеров: {total_rows}")
    log_event(f"Новых примеров с последнего retrain: {new_rows} / {MIN_NEW_SAMPLES}")

    if not force and new_rows < MIN_NEW_SAMPLES:
        log_event("Недостаточно данных → retrain пропущен.")
        return False

    # === Шаг 1: Обновление эмбеддингов ===
    log_event("Запуск rebuild_embeddings() ...")
    rebuild_embeddings()

    # === Шаг 2: Перестроение FAISS ===
    log_event("Перестроение FAISS индекса ...")
    search = SemanticSearch()
    search.build_faiss_index()

    # === Шаг 3: Перестроение кластерной модели ===
    log_event("Перестроение HDBSCAN кластеров ...")
    cm = ClusterModel()
    cm.build_clusters()

    # === Шаг 4: Обновление версии ===
    current_version = _read_version()
    new_version = current_version + 1
    _write_version(new_version)
    _write_last_count(total_rows)

    log_event(f"✅ SELF-LEARNING COMPLETED. Новая версия: v{new_version}")
    return True

if __name__ == "__main__":
    retrain_models(force=True)
