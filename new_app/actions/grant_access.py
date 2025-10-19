from .base_action import BaseAction

class GrantAccessAction(BaseAction):
    action_name = "grant_access"

    def execute(self, data: dict) -> dict:
        username = data.get("username")
        access_level = data.get("access_level", "default")

        if not username:
            return {
                "ok": False,
                "error": "username не указан",
                "hint": "ожидается data={'username': '...', 'access_level': '...'}"
            }

        return {
            "ok": True,
            "action": self.action_name,
            "message": f"[STUB] Пользователю '{username}' был бы выдан доступ уровня '{access_level}'.",
            "integration": "TODO: подключение IAM API"
        }
