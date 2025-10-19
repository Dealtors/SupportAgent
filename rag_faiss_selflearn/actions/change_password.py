from .base_action import BaseAction

class ChangePasswordAction(BaseAction):
    action_name = "change_password"

    def execute(self, data: dict) -> dict:
        username = data.get("username")
        new_password = data.get("new_password")

        if not username or not new_password:
            return {
                "ok": False,
                "error": "username или new_password не переданы",
                "hint": "ожидается data={'username': '...', 'new_password': '...'}"
            }

        return {
            "ok": True,
            "action": self.action_name,
            "message": f"[STUB] Пароль пользователя '{username}' был бы изменён.",
            "integration": "TODO: подключить систему управления пользователями"
        }
