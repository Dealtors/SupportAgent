import os

# --- Пороги и поведение ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))  # >= ок
REJECTS_TO_ESCALATE = int(os.getenv("REJECTS_TO_ESCALATE", "2"))         # 2 отказа → оператор
INACTIVITY_MINUTES = int(os.getenv("INACTIVITY_MINUTES", "5"))           # таймаут диалога (мин)

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LOGS_PATH = os.path.join(DATA_DIR, "logs.jsonl")              # ваш общий лог
TRAIN_DATASET = os.path.join(DATA_DIR, "dataset_train.csv")   # автообучение
DIALOG_STATE_PATH = os.path.join(DATA_DIR, "dialog_state.jsonl")  # трекинг диалогов

CRM_API_URL = os.getenv("CRM_API_URL", "")     # задайте в ENV
CRM_API_TOKEN = os.getenv("CRM_API_TOKEN", "") # задайте в ENV
CRM_TIMEOUT = float(os.getenv("CRM_TIMEOUT", "8.0"))
CRM_VERIFY_SSL = os.getenv("CRM_VERIFY_SSL", "true").lower() == "true"
