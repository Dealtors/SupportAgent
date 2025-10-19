import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# === Настройка путей ===
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ml_core.parsers.document_parser import DocumentParser
from ml_core.retrain.search import SemanticSearch
from ml_core.classifier.enhanced_classifier import EnhancedClassifierAgent

from new_app.feedback_manager import FeedbackManager
from new_app.retrain_manager import RetrainManager
from new_app.config import CONFIDENCE_THRESHOLD

DATA_DIR = ROOT / "data"
BK_DIR = DATA_DIR / "base_knoweledge"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)
log = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self):
        self.parser = DocumentParser()
        self.search = SemanticSearch()
        self.classifier = EnhancedClassifierAgent(model_path="models/trained_classifier.joblib")
        self.feedback = FeedbackManager()
        self.retrain = RetrainManager()

    async def _run_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # ======================
    #   ПОЛНЫЙ PIPELINE
    # ======================
    async def build_base_knowledge(self):
        log.info("Старт индексации базы знаний")
        parsed_docs = await self._run_async(self.parser.batch_parse, str(BK_DIR), False)
        texts, sources = [], []
        for doc in parsed_docs:
            for page in doc.get("pages", []):
                text = (page.get("text") or "").strip()
                if text:
                    texts.append(text)
                    sources.append(doc.get("file", "unknown"))

        if not texts:
            return {"ok": False, "reason": "empty_base_knowledge"}

        result = await self._run_async(self.search.build_embeddings_and_index, texts, sources)
        log.info(f"База знаний проиндексирована: {result}")
        return result

    async def train_classifier(self):
        dataset_path = DATA_DIR / "dataset_train.csv"
        if not dataset_path.exists():
            return {"ok": False, "error": "dataset_missing"}

        report = await self._run_async(self.classifier.train_from_csv, str(dataset_path))
        log.info(f"Классификатор обучен: {report}")
        return {"ok": True, "report": report}

    async def full_pipeline(self):
        return await asyncio.gather(
            self.build_base_knowledge(),
            self.train_classifier()
        )

    # ======================
    #   ОСНОВНОЙ ИНТЕРФЕЙС
    # ======================
    async def process_user_message(self, text: str, ticket_id: str = None):
        """
        Полный цикл:
        - Генерация ticket_id (если новый диалог)
        - Классификация
        - Если низкая уверенность → подсказки из RAG
        - Сохранение в лог
        - Возврат: ticket_id, ответ модели, действия
        """
        result = await self._run_async(self.feedback.register_prediction, text, ticket_id)
        ticket_id = result["ticket_id"]
        model_class = result["class"]
        confidence = result["confidence"]

        response = {
            "ticket_id": ticket_id,
            "class": model_class,
            "confidence": confidence,
            "need_feedback": True,
            "suggestions": []
        }

        if confidence < CONFIDENCE_THRESHOLD:
            rag_results = await self._run_async(self.search.search_similar_by_text, text, 3)
            response["suggestions"] = rag_results

        return response

    async def process_feedback(self, ticket_id: str, text: str, label: str, kind: str):
        """
        Обработчик фидбека:
        kind: ok / correct / reject / operator
        """
        actions = await self._run_async(self.feedback_router, ticket_id, text, label, kind)

        # retrain trigger
        if actions.get("trigger_retrain"):
            await self._run_async(self.retrain.maybe_retrain)

        return actions

    def feedback_router(self, ticket_id, text, label, kind):
        """Синхронный маршрут (в executor)"""
        if kind == "ok":
            return self.feedback.feedback_ok(ticket_id, text, label)
        if kind == "correct":
            return self.feedback.feedback_corrected(ticket_id, text, label)
        if kind == "reject":
            return self.feedback.feedback_reject(ticket_id, text)
        if kind == "operator":
            return self.feedback.feedback_operator_request(ticket_id, text)
        return {"ok": False, "reason": "unknown_feedback"}


# ============ DEMO ============
async def main():
    orch = Orchestrator()
    await orch.full_pipeline()

    # Пользователь написал
    res = await orch.process_user_message("мне нужно изменить пароль")
    print("Ответ модели:", res)

    # Пользователь подтвердил
    feedback = await orch.process_feedback(
        ticket_id=res["ticket_id"],
        text="мне нужно изменить пароль",
        label="password_change",
        kind="ok"
    )
    print("Ответ фидбека:", feedback)

if __name__ == "__main__":
    asyncio.run(main())
