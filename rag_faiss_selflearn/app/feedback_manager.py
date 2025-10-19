from classifier_agent import create_classifier_agent  # твой модуль
from retrain import retrain_models                   # твой retrain.py
from .escalation_manager import EscalationManager
from .operator_router import OperatorRouter
from .config import LOGS_PATH, TRAIN_DATASET, REJECTS_TO_ESCALATE, INACTIVITY_MINUTES
import pandas as pd
import os
import json
from datetime import datetime

def _append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _append_train(text, label):
    os.makedirs(os.path.dirname(TRAIN_DATASET), exist_ok=True)
    df = pd.DataFrame([{"text": text, "class": label}])
    if os.path.exists(TRAIN_DATASET):
        df.to_csv(TRAIN_DATASET, mode="a", header=False, index=False)
    else:
        df.to_csv(TRAIN_DATASET, index=False)

class FeedbackManager:
    def __init__(self, auto_retrain=True, auto_reload=True):
        self.escalation = EscalationManager()
        self.router = OperatorRouter()
        self.auto_retrain = auto_retrain
        self.auto_reload = auto_reload
        self.classifier = create_classifier_agent()  # ← теперь понятно откуда

    def _log(self, event, data):
        _append_jsonl(LOGS_PATH, {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            **data
        })

    def _trigger_retrain(self):
        if not self.auto_retrain:
            return
        result = retrain_models(force=False)
        if result and self.auto_reload:
            try:
                self.classifier.load_model()  # ← подгружает новую модель
                print("[FeedbackManager] 🔄 Hot reload модели выполнен")
            except Exception as e:
                print("[FeedbackManager] Ошибка hot reload:", e)

    def register_prediction(self, ticket_id, text, predicted_class, confidence):
        now = datetime.utcnow()
        self.escalation.start(ticket_id, predicted_class, confidence, now)
        self._log("prediction", {
            "ticket_id": ticket_id,
            "text": text,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    def feedback_ok(self, ticket_id, text, label):
        st = self.escalation.get(ticket_id)
        if not st:
            return {"ok": False, "reason": "unknown_ticket"}

        _append_train(text, label)
        self._log("feedback_ok", {
            "ticket_id": ticket_id,
            "label": label,
            "confidence": st.confidence
        })
        self.escalation.mark_finished(ticket_id)
        self._trigger_retrain()
        return {"ok": True, "action": "trained"}

    def feedback_corrected(self, ticket_id, text, correct_label):
        st = self.escalation.get(ticket_id)
        if not st:
            return {"ok": False, "reason": "unknown_ticket"}

        _append_train(text, correct_label)
        self._log("feedback_corrected", {
            "ticket_id": ticket_id,
            "correct_label": correct_label,
            "confidence": st.confidence
        })
        self.escalation.mark_finished(ticket_id)
        self._trigger_retrain()
        return {"ok": True, "action": "trained"}

    def feedback_reject(self, ticket_id, text):
        st = self.escalation.get(ticket_id)
        if not st:
            return {"ok": False, "reason": "unknown_ticket"}

        self.escalation.add_reject(ticket_id)
        self._log("feedback_reject", {
            "ticket_id": ticket_id,
            "rejects": st.rejects,
            "confidence": st.confidence
        })

        if st.rejects >= REJECTS_TO_ESCALATE:
            return self._escalate(ticket_id, text, "double_reject")

        return {"ok": True, "action": "continue_dialog"}

    def feedback_operator_request(self, ticket_id, text):
        return self._escalate(ticket_id, text, "user_requested_operator")

    def check_inactivity(self, ticket_id, text, label_if_ok):
        st = self.escalation.get(ticket_id)
        if not st or st.finished or st.escalated:
            return

        now = datetime.utcnow()
        if self.escalation.due_inactivity(ticket_id, now, INACTIVITY_MINUTES):
            if st.rejects == 0:
                _append_train(text, label_if_ok)
                self._log("auto_confirm", {
                    "ticket_id": ticket_id,
                    "label": label_if_ok
                })
                self.escalation.mark_finished(ticket_id)
                self._trigger_retrain()
                return {"ok": True, "action": "trained_auto"}
            else:
                return self._escalate(ticket_id, text, "timeout_after_reject")

    def operator_resolution(self, ticket_id, text, ground_truth_label):
        st = self.escalation.get(ticket_id)
        _append_train(text, ground_truth_label)
        self._log("operator_resolution", {
            "ticket_id": ticket_id,
            "ground_truth": ground_truth_label
        })
        if st:
            self.escalation.mark_finished(ticket_id)
        self._trigger_retrain()
        return {"ok": True, "action": "trained_from_operator"}

    def _escalate(self, ticket_id, text, reason):
        st = self.escalation.get(ticket_id)
        payload = {
            "ticket_id": ticket_id,
            "text": text,
            "predicted_class": st.predicted_class,
            "confidence": st.confidence,
            "reason": reason
        }
        resp = self.router.send(payload)
        self._log("escalated", {
            "ticket_id": ticket_id,
            "reason": reason,
            "crm_ok": resp.get("ok", False)
        })
        self.escalation.mark_escalated(ticket_id)
        return {"ok": True, "action": "escalated", "crm": resp}
