import os
from dotenv import load_dotenv

load_dotenv()

class Cfg:
    LLM_BACKEND = os.getenv("LLM_BACKEND", "transformers")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2-7B-Instruct")
    QWEN_DEVICE = os.getenv("QWEN_DEVICE", "auto")
    QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "512"))
    QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))

    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen2-7b-instruct")

    MIN_CLASS_CONFIDENCE = float(os.getenv("MIN_CLASS_CONFIDENCE", "0.7"))
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
    KB_JSONL = os.getenv("KB_JSONL", "./data/kb.jsonl")
    LOGS_PATH = os.getenv("LOGS_PATH", "./logs/logs.jsonl")
    TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))
