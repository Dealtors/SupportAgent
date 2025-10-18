"""
retrain.py — модуль для реализации цикла самообучения (Self-Learning Loop)
в ЖЁСТКОМ режиме (Full Retraining).

Шаги:
1. Чтение логов обращений
2. Отбор успешных и неуверенных кейсов
3. Обновление dataset_train.csv и dataset_review.csv
4. Вызов функции обновления эмбеддингов (из ML1)
5. Перестройка FAISS индекса
6. Перестройка HDBSCAN кластеризации
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(BASE_DIR, "data", "logs.jsonl")
TRAIN_DATASET = os.path.join(BASE_DIR, "data", "dataset_train.csv")
REVIEW_DATASET = os.path.join(BASE_DIR, "data", "dataset_review.csv")

# Модули твоего проекта
from search import SemanticSearch
from cluster import ClusterModel

# Важно: ML1 должен предоставить эту функцию!
try:
    from embeddings_loader import rebuild_embeddings
except ImportError:
    raise ImportError("Функция rebuild_embeddings из ML1 не найдена. Убедитесь, что ML1 реализовал её.")


def log_event(message: str):
    """Простой лог событий retrain."""
    print(f"[RETRAIN] {datetime.utcnow().isoformat()} - {message}")


def read_logs() -> list:
    """Читает логи и возвращает список записей."""
    if not os.path.exists(LOGS_PATH):
        log_event("Лог-файл не найден, retrain невозможен")
        return []

    records = []
    with open(LOGS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue
    log_event(f"Прочитано записей логов: {len(records)}")
    return records


def filter_training_cases(records: list):
    """Отбираем кейсы для обучения и для review."""
    train_data = []
    review_data = []

    for r in records:
        conf = r.get("confidence")
        feedback = r.get("feedback")
        text = r.get("text")
        label = r.get("predicted_class")

        if not text or not label:
            continue

        # Успешно подтвержденные пользователем кейсы
        if conf and conf > 0.8 and feedback == "OK":
            train_data.append((text, label))

        # Неуверенные кейсы
        elif conf and conf < 0.6:
            review_data.append((text, label))

    log_event(f"Отобрано для обучения: {len(train_data)}, для review: {len(review_data)}")
    return train_data, review_data


def update_dataset(train_data, review_data):
    """Обновление train и review датасетов."""
    if train_data:
        df_train = pd.DataFrame(train_data, columns=["text", "class"])
        if os.path.exists(TRAIN_DATASET):
            old = pd.read_csv(TRAIN_DATASET)
            df_train = pd.concat([old, df_train], ignore_index=True)
        df_train.to_csv(TRAIN_DATASET, index=False)

    if review_data:
        df_review = pd.DataFrame(review_data, columns=["text", "class"])
        if os.path.exists(REVIEW_DATASET):
            old = pd.read_csv(REVIEW_DATASET)
            df_review = pd.concat([old, df_review], ignore_index=True)
        df_review.to_csv(REVIEW_DATASET, index=False)

    log_event("Датасеты обновлены")


def retrain_models():
    """
    Главная функция полного переобучения.
    """
    log_event("=== СТАРТ SELF-LEARNING CYCLE ===")

    # 1. Чтение логов
    records = read_logs()
    if not records:
        log_event("Нет данных для retrain")
        return False

    # 2. Отбор кейсов
    train_data, review_data = filter_training_cases(records)

    if not train_data:
        log_event("Нет новых подтвержденных кейсов. Обучение пропущено.")
        return False

    # 3. Обновление датасетов
    update_dataset(train_data, review_data)

    # 4. Пересоздание эмбеддингов (ML1 делает это)
    log_event("Вызов обновления эмбеддингов (ML1)")
    rebuild_embeddings()  # из ML1

    # 5. Перестроение FAISS
    log_event("Перестроение FAISS индекса")
    search = SemanticSearch()
    search.build_faiss_index()

    # 6. Перестроение HDBSCAN
    log_event("Перестроение кластерной модели")
    cm = ClusterModel()
    cm.build_clusters()

    log_event("=== SELF-LEARNING COMPLETED ===")
    return True


# Пример запуска
if __name__ == "__main__":
    success = retrain_models()
    if success:
        print("✅ Retrain завершен успешно")
    else:
        print("⚠️ Обновление пропущено")
