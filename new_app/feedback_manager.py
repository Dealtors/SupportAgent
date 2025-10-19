# === C:\Users\ntise\AI agent\new_app\feedback_manager.py ===
from __future__ import annotations

import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional

# Конфиг — оставил твои имена, проверь пути!
from .config import (
    LOGS_PATH, TRAIN_DATASET,
    REJECTS_TO_ESCALATE,
)
from .escalation_manager import EscalationManager
from .operator_router import OperatorRouter
from ml_core.classifier.enhanced_classifier import EnhancedClassifierAgent

def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _append_train(text: str, label: str):
    os.makedirs(os.path.dirname(TRAIN_DATASET), exist_ok=True)
    row = pd.DataFrame([{"text": text, "class": label}])
    if os.path.exists(TRAIN_DATASET):
        row.to_csv(TRAIN_DATASET, mode="a", header=False, index=False)
    else:
        row.to_csv(TRAIN_DATASET, index=False)

class FeedbackManager:
    """
    Управляет:
      - регистрацией предсказаний,
      - добавлением фидбека в обучающий датасет,
      - эскалациями к оператору.
    НЕ запускает обучение напрямую — возвращает флаг `trigger_retrain`,
    который должен обработать внешний менеджер (например, RetrainManager).
    """
    def __init__(self):
        self.escalation = EscalationManager()
        self.router = OperatorRouter()
        # Проверь путь до модели, если у тебя другой — поправь
        self.classifier = EnhancedClassifierAgent(model_path="models/trained_classifier.joblib")

    def register_prediction(self, text: str, ticket_id: Optional[str] = None):
        now = datetime.utcnow()
        if not ticket_id:
            from .ticket_id import generate_ticket_id
            ticket_id = generate_ticket_id(text, int(now.timestamp()))

        predicted_class, confidence = self.classifier.predict(text)
        self.escalation.start(ticket_id, predicted_class, confidence, now)

        _append_jsonl(LOGS_PATH, {
            "timestamp": now.isoformat(),
            "event": "prediction",
            "ticket_id": ticket_id,
            "text": text,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

        return {
            "ticket_id": ticket_id,
            "class": predicted_class,
            "confidence": confidence,
            "need_feedback": True
        }

    def feedback_ok(self, ticket_id: str, text: str, label: str):
        st = self.escalation.get(ticket_id)
        if not st:
            return {"ok": False, "reason": "unknown_ticket"}

        _append_train(text, label)
        _append_jsonl(LOGS_PATH, {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "feedback_ok",
            "ticket_id": ticket_id,
            "label": label,
            "confidence": st.confidence
        })
        self.escalation.mark_finished(ticket_id)

        return {"ok": True, "action": "train_added", "trigger_retrain": True}

    def feedback_corrected(self, ticket_id: str, text: str, correct_label: str):
        st = self.escalation.get(ticket_id)
        if not st:
            return {"ok": False, "reason": "unknown_ticket"}

        _append_train(text, correct_label)
        _append_jsonl(LOGS_PATH, {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "feedback_corrected",
            "ticket_id": ticket_id,
            "correct_label": correct_label
        })
        self.escalation.mark_finished(ticket_id)

        return {"ok": True, "action": "train_added", "trigger_retrain": True}

    def feedback_reject(self, ticket_id: str, text: str):
        st = self.escalation.get(ticket_id)
        if not st:
            return {"ok": False, "reason": "unknown_ticket"}

        self.escalation.add_reject(ticket_id)
        _append_jsonl(LOGS_PATH, {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "feedback_reject",
            "ticket_id": ticket_id,
            "rejects": st.rejects
        })

        if st.rejects >= REJECTS_TO_ESCALATE:
            return self._escalate(ticket_id, text)

        return {"ok": True, "action": "await_more_feedback"}

    def feedback_operator_request(self, ticket_id: str, text: str):
        return self._escalate(ticket_id, text)

    def _escalate(self, ticket_id: str, text: str):
        st = self.escalation.get(ticket_id)
        payload = {
            "ticket_id": ticket_id,
            "text": text,
            "predicted_class": st.predicted_class,
            "confidence": st.confidence
        }
        resp = self.router.send(payload)
        _append_jsonl(LOGS_PATH, {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "escalated",
            "ticket_id": ticket_id,
            "payload": payload
        })
        self.escalation.mark_escalated(ticket_id)
        return {"ok": True, "action": "escalated", "router_response": resp}
