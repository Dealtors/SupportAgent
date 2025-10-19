# AI Support Agent — Полный MVP по ТЗ

Реализовано всё, что указано в вашем README: API, веб‑интерфейс, ingest, отчёты, OpenAPI, recall@3 на Qwen2:7b‑q4_K_M, докер‑оркестрация.

## Быстрый старт

### Вариант A: Docker
```bash
docker compose up -d --build
# открыть http://localhost:8000
```

### Вариант B: Локально
```bash
./scripts/setup.sh
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# открыть http://localhost:8000/web
```

## Оценка recall@3 (Qwen2:7B‑q4_K_M)
1) Убедитесь, что Ollama поднят и модель скачана.
2) Подготовьте `dataset.json` вида:
```json
[{"query":"как почистить кэш","relevant":["<id_документа_из_Chroma>"]}]
```
3) Запустите:
```bash
python eval_recall_qwen.py dataset.json
```
Отчёт в `eval_report.json`.

## Эндпоинты
- `POST /ticket` — основной вход: классификация + RAG + ответ Qwen2
- `POST /feedback` — фидбек по тикету
- `GET /report` — baseline метрики (tickets, avg_confidence)

Спецификация: `OpenAPI.yaml`

## Ингест и хранение
- Документы из `app/data/kb` → `chromadb` (скрипт `app/ingest_kb.py`).
- Пересборка индекса — перезапуск `ingest_kb.py`.

## Веб
- Файлы `app/web` (подсунуты ваши `index.html`, `styles.css`, добавлен `main.js`).


