from .base_action import BaseAction

class CheckTicketStatusAction(BaseAction):
    action_name = "check_ticket_status"

    def execute(self, data: dict) -> dict:
        ticket_id = data.get("ticket_id")

        if not ticket_id:
            return {
                "ok": False,
                "error": "ticket_id не указан",
                "hint": "ожидается data={'ticket_id': '...'}"
            }

        return {
            "ok": True,
            "action": self.action_name,
            "message": f"[STUB] Статус заявки '{ticket_id}': В обработке.",
            "integration": "TODO: CRM статус заявки"
        }
