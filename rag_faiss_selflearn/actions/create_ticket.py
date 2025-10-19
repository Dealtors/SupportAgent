from .base_action import BaseAction

class CreateTicketAction(BaseAction):
    action_name = "create_ticket"

    def execute(self, data: dict) -> dict:
        description = data.get("description")

        if not description:
            return {
                "ok": False,
                "error": "description не указан",
                "hint": "ожидается data={'description': '...'}"
            }

        return {
            "ok": True,
            "action": self.action_name,
            "ticket_id": "STUB-TICKET-123",
            "message": f"[STUB] Заявка создана: '{description}'. ID: STUB-TICKET-123",
            "integration": "TODO: создать заявку через CRM API"
        }
