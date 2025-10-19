# AI Support Agent

Интеллектуальный агент для автоматизации техподдержки. Классифицирует запросы, находит ответы в базе знаний и распределяет сложные задачи операторам.

Цель: снизить нагрузку на специалистов и ускорить обработку обращений с помощью NLP и AI.

## Задачи проекта

### Подготовка данных
- **Разметка и проверка датасета** (2 ч)
- Проверка корректности разметки классов обращений
- Валидация наличия минимум 5 примеров для каждого класса

### Оценка качества  
- **Сбор baseline-метрик** (2 ч)
- Метрики классификатора: accuracy, average confidence
- recall@3 для FAISS-поиска
- Метрики включаются в итоговый отчёт и презентацию

### Документация
- **Подготовка README.md** (2 ч)
- Цель проекта
- Инструкция по запуску (docker compose up или python cli.py)  
- Описание API-эндпоинтов (POST /api/ticket)
- Описание цикла самообучения

### API спецификация
- **Составить OpenAPI.yaml** (1 ч)
- Эндпоинты: POST /ticket, GET /report, POST /feedback
- Документ для наглядности структуры API

### Презентация
- **Подготовка презентации** (3 ч)
- 6 слайдов: цель, архитектура, pipeline, самообучение, метрики, перспективы
- Используется для защиты и демонстрации решения


## Обзор архитектуры

```mermaid
flowchart TD
    subgraph AUTOLEARN[" "]
        R["Retrain LogisticRegression (новые данные)"]
        M["Добавить кейс в dataset_train.csv"]
        S["Пересчёт кластеров HDBSCAN"]
        O["Добавить кейс в dataset_review.csv"]
        T["Обновление FAISS-индекса (новые эмбеддинги)"]
        U["Обновление словаря кластеров (NEW_TOPIC → Named Topic)"]
        V["Система обновлена и готова к новым тикетам"]
    end
    
    A["Пользователь прислал тикет"] --> B["SentenceTransformer → эмбеддинг"]
    B --> C["LogisticRegression → класс + confidence"]
    C --> D{"conf < 0.8 ?"}
    D -- да --> E["Кластеризация (HDBSCAN) → cluster_id = #7 или NEW_TOPIC"]
    D -- нет --> F["Используем предсказанный класс"]
    E --> G["Определены: класс + кластер"]
    F --> G
    G --> H["FAISS → поиск похожих документов (RAG)"]
    H --> I["Qwen2-7B → ответ + план действий"]
    I --> J["Ответ пользователю"]
    J --> K["Сохранить журнал (текст, класс, кластер, conf, ответ, feedback)"]
    K --> L{"conf > 0.8 ∧ feedback = 'помогло' ?"}
    L -- да --> M
    L -- нет --> N{"conf < 0.7 ∨ cluster = NEW_TOPIC ?"}
    N -- да --> O
    N -- нет --> P["Игнорировать (низкий приоритет)"]
    M --> R
    O --> S
    R --> T
    S --> T
    T --> U
    U --> V
    P --> V
```
## Бизнес логика retrain_loop
```mermaid
flowchart TD

%% ==== ENTRY POINT ====
A["Новый пользовательский запрос"] --> B["Модель выполняет предсказание"]
B --> C{"Существует ticket_id?"}
C -- "Нет" --> D["Создать ticket_id: hash текста + timestamp начала сессии"]
C -- "Да" --> E["Обновить состояние тикета"]

%% ==== FEEDBACK ====
D --> F{"Получен фидбек пользователя?"}
E --> F

%% === ВЕТВЛЕНИЯ ПО ФИДБЕКУ ===
F -- "Подтверждение" --> G1["Добавить (text, label) в dataset_train.csv"]
F -- "Исправление" --> G2["Добавить (text, correct_label) в dataset_train.csv"]
F -- "Отказ" --> G3["reject_count += 1"]
F -- "Запрос оператора" --> G4["Эскалация в OperatorRouter"]
F -- "Нет ответа" --> G5{"Проверка таймаута"}

%% === TIMEOUT LOGIC ===
G5 -- "Нет активности и нет отказов" --> G6["Auto-confirm → добавить (text, label)"]
G5 -- "Нет активности, но были отказы" --> G7["Эскалация (timeout_after_reject)"]

%% === ESCALATION CONDITIONS ===
G3 -- "reject_count >= 2" --> G8["Эскалация (двойной отказ)"]
G3 -- "reject_count < 2" --> H1["Продолжить диалог (fallback)"]

%% === ОТВЕТ ОПЕРАТОРА ===
G4 --> I1["Оператор возвращает ground_truth_label"]
G8 --> I1
G7 --> I1
I1 --> I2["Добавить (text, ground_truth_label) в dataset_train.csv"]

%% === LOGGING ===
G1 --> L["Записать событие в logs.jsonl"]
G2 --> L
G3 --> L
G4 --> L
G5 --> L
G6 --> L
G7 --> L
I2 --> L

%% === RETRAIN + HOT RELOAD ===
L --> J{"auto_retrain=True?"}
J -- "Да" --> K["retrain_models(force=False)"]
K --> K1{"Успех retrain?"}
K1 -- "Да" --> K2["Обновление модели (hot reload)"]
K1 -- "Нет" --> K3["Логирование ошибки retrain"]

J -- "Нет" --> M["Отложенный retrain (cron, scheduler)"]

%% === END OUTPUT ===
K2 --> Z["Модель готова (новая версия)"]
K3 --> Z
H1 --> Z
M --> Z

```

## Диаграмма компонентов

```mermaid
flowchart LR
%% Увеличиваем расстояния
classDef block fill:#eef,stroke:#333,stroke-width:1px;
classDef storage fill:#ffe,stroke:#333,stroke-width:1px;
classDef agent fill:#efe,stroke:#333,stroke-width:1px;

subgraph User_Interface["Пользовательский интерфейс"]
    UI1["CLI / Chat Interface"]:::block
    UI2["Web UI"]:::block
end

subgraph Core_Orchestrator["ОРКЕСТРАТОР"]
    OR1["Router"]:::agent
    OR2["Security Filter"]:::agent
    OR3["Confidence Manager"]:::agent
end

subgraph Classification_Agent["Классификация"]
    CL1["SentenceTransformer"]:::agent
    CL2["Logistic Regression"]:::agent
    CL3["HDBSCAN"]:::agent
end

subgraph Knowledge_RAG["RAG"]
    KG1["Парсер документов"]:::block
    KG2["FAISS Index"]:::block
    KG3["База знаний"]:::block
end

subgraph Planning_Agent["Планирование"]
    PL1["SLM / Qwen"]:::agent
    PL2["Генератор ответа"]:::agent
end

subgraph Automation_Agent["Автоматизация"]
    AU1["Executor"]:::agent
    AU2["Mock Actions"]:::agent
end

subgraph Self_Learning["Самообучение"]
    SL1["Логирование"]:::agent
    SL2["Retrain Loop"]:::agent
    SL3["Перестроение индексов"]:::agent
end

subgraph Storage["Хранилище"]
    ST1["Документы"]:::storage
    ST2["Векторные индексы"]:::storage
    ST3["Модели"]:::storage
    ST4["Логи"]:::storage
end

UI1 --> OR1
UI2 --> OR1
OR1 --> OR2
OR2 --> CL1
CL1 --> CL2
CL2 --> OR3
OR3 --> CL3 & KG2
KG2 --> PL1
PL1 --> PL2
PL2 --> AU1
AU1 --> OR1 & SL1
SL1 --> SL2
SL2 --> SL3
SL3 --> ST2 & ST3

```

## Быстрый старт

Ссылка на ветку с ИИ-агентом: https://github.com/Dealtors/SupportAgent/tree/add-docusaurus-docs

### Вариант A: Docker
1. Установите Docker и включите BuildKit.
2. Запустите:
   ```bash
   docker compose up -d
   ```
3. Откройте: http://localhost:8000

### Вариант B: Хостовая машина (без контейнеров)
1. Установите **Ollama** и подтяните модели:
   ```bash
   ollama pull qwen2:7b-q4_K_M
   ollama pull nomic-embed-text
   ```
2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r app/requirements.txt
   ```
3. Разверните знания и индексацию:
   ```bash
   python app/ingest_kb.py
   ```
4. Запуск API и веба:
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Откройте: http://localhost:8000

## Что внутри
- `scripts/setup.sh` — клонирует `SupportAgent`, распаковывает `ml_orchestrator.zip` и `reglament_ops_bundle.zip`, подтягивает модели Ollama.
- `docker-compose.yml` — контейнеры: `ollama` и `app`.
- `app/api.py` — FastAPI: `/api/chat`, `/api/ingest`, корень отдает простую веб-страницу чата.
- `app/ingest_kb.py` — разбор и индексация `reglament_ops_bundle` с эмбеддингами Ollama в ChromaDB.
- `app/ollama_client.py` — тонкая обертка над Python-клиентом `ollama`.
- `app/web/index.html` — примитивный чат UI.
- `ml_orchestrator.zip`, `reglament_ops_bundle.zip` — включены как есть.

## Интеграция с `ml_orchestrator`
Если в архиве есть модуль с функцией `route_task(query, context)`, API будет пробовать дергать ее на каждом запросе чата, чтобы обогащать ответ инструментальным действием. Если модуль отсутствует или интерфейс иной — будет мягкий фолбэк.

## Переменные окружения
Скопируйте `.env.example` в `.env` при необходимости.

| Переменная | Значение по умолчанию |
|-----------|------------------------|
| OLLAMA_BASE_URL | http://ollama:11434 (в Docker) / http://localhost:11434 (локально) |
| OLLAMA_CHAT_MODEL | qwen2:7b-q4_K_M |
| OLLAMA_EMBED_MODEL | nomic-embed-text |
| CHROMA_DIR | ./app/storage/chroma |

## Лицензии
Код скелета — MIT. Репозиторий `SupportAgent` и ваши архивы — по их лицензиям.

*Собрано: 2025-10-18T19:51:02.014957Z*
