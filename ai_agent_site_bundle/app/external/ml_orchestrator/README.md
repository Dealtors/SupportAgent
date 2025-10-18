# ML Orchestrator: Qwen2-7B + CLI + Streamlit

## Установка
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Запуск CLI
```bash
python -m app.cli
```

## Генерация отчёта
```bash
python -m app.report
```

## Streamlit UI
```bash
streamlit run app/webui_streamlit.py
```

## Примечания
- Если нет индекса FAISS и БЗ, контекст будет пустым, но всё остальное работает.
- Для локального OpenAI-совместимого сервера выставьте `LLM_BACKEND=openai` и переменные в `.env`.
