class OperatorRouter:
    def send(self, payload: dict) -> dict:
        print(f"[STUB] ➡ Отправка оператору: {payload}")
        return {"ok": True, "message": "Оператор уведомлен (заглушка)"}
