import os
import httpx

# Используем 127.0.0.1, а не "localhost", и отключаем доверие к системным прокси
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2:7b-instruct-q4_K_M")

async def chat(system: str, user: str) -> str:
    """
    Возвращает один готовый ответ от модели.
    1) Пытаемся /api/chat (без стрима).
    2) Если не вышло, фоллбэк на /api/generate (prompt = system + user).
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # без стрима, чтобы не ловить обрывы
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }

    # trust_env=False — игнорировать системные HTTP(S)_PROXY
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        # 1) /api/chat
        try:
            r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content", "")
            if content:
                return content.strip()
        except Exception:
            pass

        # 2) fallback: /api/generate
        try:
            gen = {
                "model": CHAT_MODEL,
                "prompt": f"{system}\n\n{user}",
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 4096},
            }
            r = await client.post(f"{OLLAMA_BASE}/api/generate", json=gen)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            # Последний шанс: вернём диагностическое сообщение
            return f"Не удалось получить ответ от модели: {e}"
