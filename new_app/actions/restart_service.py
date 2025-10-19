from .base_action import BaseAction

class RestartServiceAction(BaseAction):
    action_name = "restart_service"

    def execute(self, data: dict) -> dict:
        service_name = data.get("service_name", "unknown_service")

        return {
            "ok": True,
            "action": self.action_name,
            "message": f"[STUB] Сервис '{service_name}' был бы перезапущен.",
            "integration": "TODO: интеграция с системой управления"
        }
