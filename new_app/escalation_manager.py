from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict

@dataclass
class DialogState:
    ticket_id: str
    started_at: datetime
    last_user_at: datetime
    predicted_class: str
    confidence: float
    rejects: int = 0
    escalated: bool = False
    finished: bool = False

class EscalationManager:
    """
    В памяти хранит состояние диалогов (ticket_id → DialogState).
    В проде можно вынести в Redis/БД.
    """
    def __init__(self):
        self._state: Dict[str, DialogState] = {}

    def start(self, ticket_id: str, predicted_class: str, confidence: float, now: datetime):
        if ticket_id not in self._state:
            self._state[ticket_id] = DialogState(
                ticket_id=ticket_id,
                started_at=now,
                last_user_at=now,
                predicted_class=predicted_class,
                confidence=confidence
            )

    def touch_user(self, ticket_id: str, now: datetime):
        if ticket_id in self._state:
            self._state[ticket_id].last_user_at = now

    def add_reject(self, ticket_id: str):
        if ticket_id in self._state:
            self._state[ticket_id].rejects += 1

    def mark_escalated(self, ticket_id: str):
        if ticket_id in self._state:
            self._state[ticket_id].escalated = True

    def mark_finished(self, ticket_id: str):
        if ticket_id in self._state:
            self._state[ticket_id].finished = True

    def get(self, ticket_id: str) -> DialogState | None:
        return self._state.get(ticket_id)

    def due_inactivity(self, ticket_id: str, now: datetime, inactivity_minutes: int) -> bool:
        st = self._state.get(ticket_id)
        if not st or st.finished: return False
        return (now - st.last_user_at) >= timedelta(minutes=inactivity_minutes)
