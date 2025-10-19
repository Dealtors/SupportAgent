from .action_registry import ACTION_REGISTRY
from .action_rules import detect_action_by_text

class ActionExecutor:
    """
    Выполняет действие на основе текста пользователя.
    """
    def route_and_execute(self, user_text: str, data: dict = None) -> dict:
        data = data or {}
        action_name = detect_action_by_text(user_text)

        if not action_name:
            return {"ok": False, "reason": "no_action_detected", "details": "Текст не совпадает ни с одним действием"}

        action = ACTION_REGISTRY.get(action_name)
        if not action:
            return {"ok": False, "reason": "action_not_implemented", "action": action_name}

        return action.execute(data)
