ACTION_RULES = [
    {"keywords": ["сменить пароль", "сброс пароля"], "action": "change_password"},
    {"keywords": ["проверить статус", "статус заявки"], "action": "check_ticket_status"},
    {"keywords": ["создать заявку", "отправить заявку", "новая заявка"], "action": "create_ticket"},
    {"keywords": ["перезапусти", "перезапустить сервис", "рестарт"], "action": "restart_service"},
    {"keywords": ["выдать доступ", "добавить в группу"], "action": "grant_access"},
]

def detect_action_by_text(text: str) -> str | None:
    text = text.lower()
    for rule in ACTION_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            return rule["action"]
    return None
