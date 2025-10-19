import os
import sys

# Ensure project root is on sys.path so local package imports work when
# running this script directly from the `ml_orchestrator` folder.
# Assumption: repository root (where `rag_faiss_selflearn` lives) is the
# parent of the `ml_orchestrator` directory.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from rag_faiss_selflearn.src.enhanced_classifier import EnhancedClassifierAgent
from rag_faiss_selflearn.app.feedback_manager import FeedbackManager
from rag_faiss_selflearn.app.action_executor import ActionExecutor
from rag_faiss_selflearn.app.ticket_id import generate_ticket_id
from datetime import datetime

class Orchestrator:
    def __init__(self):
        self.classifier = EnhancedClassifierAgent(model_path="models/trained_classifier.joblib")
        self.feedback_manager = FeedbackManager(auto_retrain=True)
        self.actions = ActionExecutor()

    def process_user_message(self, text: str, session_ts: int | None = None):
        # 1. Создаём ticket_id
        ticket_id = generate_ticket_id(text, session_ts)

        # 2. Модель предсказывает
        predicted_class, confidence = self.classifier.predict(text)

        # 3. Регистрируем в Feedback Manager
        self.feedback_manager.register_prediction(
            ticket_id=ticket_id,
            text=text,
            predicted_class=predicted_class,
            confidence=confidence
        )

        # 4. Если intent → выполняем action
        action_result = self.actions.route_and_execute(text)
        if action_result["ok"]:
            response = action_result["message"]
            return {
                "ticket_id": ticket_id,
                "response": response,
                "type": "action"
            }

        # 5. Иначе — это класификация / FAQ / RAG
        return {
            "ticket_id": ticket_id,
            "response": f"Система считает, что это {predicted_class} (уверенность {confidence:.2f})",
            "type": "classification"
        }

    def process_feedback(self, ticket_id: str, text: str, feedback_type: str, correct_label: str = None):
        """
        feedback_type:
            - ok
            - corrected
            - reject
            - operator
        """
        if feedback_type == "ok":
            return self.feedback_manager.feedback_ok(ticket_id, text, correct_label or "model_label")
        elif feedback_type == "corrected":
            return self.feedback_manager.feedback_corrected(ticket_id, text, correct_label)
        elif feedback_type == "reject":
            return self.feedback_manager.feedback_reject(ticket_id, text)
        elif feedback_type == "operator":
            return self.feedback_manager.feedback_operator_request(ticket_id, text)
        else:
            return {"ok": False, "error": "unknown_feedback_type"}


"""orch = Orchestrator()

# Пользователь отправил сообщение
result = orch.process_user_message("хочу сменить пароль")

# Если юзер отвечает "да, всё верно"
orch.process_feedback(result["ticket_id"], "хочу сменить пароль", feedback_type="ok")

# Если юзер говорит "нет, это не то"
orch.process_feedback(result["ticket_id"], "хочу сменить пароль", feedback_type="reject")
"""