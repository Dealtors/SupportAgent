import os

# --- Пороги и поведение ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))  # ниже — fallback
RETRAIN_MIN_SAMPLES = int(os.getenv("RETRAIN_MIN_SAMPLES", "20"))        # минимум новых примеров
HOLD_TIMEOUT_HOURS = int(os.getenv("HOLD_TIMEOUT_HOURS", "12"))          # ожидание ответа

# --- Пути ---
MODEL_DIR = os.getenv("MODEL_DIR", "models")
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "trained_classifier.joblib")
VERSION_FILE = os.path.join(MODEL_DIR, "model_version.txt")

RETRAIN_BUFFER = os.getenv("RETRAIN_BUFFER", "data/retrain_buffer.jsonl")
HOLD_BUFFER = os.getenv("HOLD_BUFFER", "data/hold_buffer.jsonl")

# --- CRM / ServiceDesk integration ---
CRM_API_URL = os.getenv("CRM_API_URL", "https://crm.example.com/api/tickets")
CRM_API_TOKEN = os.getenv("CRM_API_TOKEN", "")  # задайте через ENV
CRM_TIMEOUT = float(os.getenv("CRM_TIMEOUT", "8.0"))  # seconds
CRM_VERIFY_SSL = os.getenv("CRM_VERIFY_SSL", "true").lower() == "true"

# --- FAISS / CLUSTER fallback параметры ---
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "3"))
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "3"))

# --- FastAPI ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
