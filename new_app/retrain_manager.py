# === C:\Users\ntise\AI agent\new_app\retrain_manager.py ===
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Optional

# ВАЖНО: путь к корню репо = "<...>\AI agent"
# Тут предполагается, что этот файл лежит в: <root>\new_app\retrain_manager.py
# Если путь иной — поправь BASE как нужно.
BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "data" / "dataset_train.csv"
STATE_PATH = BASE / "logs" / "retrain_state.json"

try:
    # твоя функция полного обучения
    from ml_core.retrain.retrain import retrain_models
except Exception:
    # если у тебя другой импорт — поправь здесь
    from ml_core.retrain.retrain import retrain_models  # noqa: F401


class RetrainManager:
    """
    Следит за приростом обучающих данных и триггерит полное переобучение
    через retrain_models(force=True), когда дельта превысит порог.
    """
    def __init__(self, dataset_path: Optional[Path] = None, state_path: Optional[Path] = None,
                 threshold_new_rows: int = 50):
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET
        self.state_path = Path(state_path) if state_path else STATE_PATH
        self.threshold = int(threshold_new_rows)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"last_rows": 0, "last_retrain_ts": None}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _count_rows(self) -> int:
        if not self.dataset_path.exists():
            return 0
        # быстрая оценка строк (без чтения всего CSV в память)
        count = 0
        with self.dataset_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                count += 1
        # если есть заголовок
        return max(0, count - 1)

    def maybe_retrain(self, force: bool = False) -> bool:
        rows = self._count_rows()
        delta = rows - int(self._state.get("last_rows", 0))
        if force or delta >= self.threshold:
            ok = retrain_models(force=True)
            if ok:
                self._state["last_rows"] = rows
                self._state["last_retrain_ts"] = datetime.utcnow().isoformat()
                self._save_state()
            return bool(ok)
        return False
