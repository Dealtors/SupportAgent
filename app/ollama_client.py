import os
import ollama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
if OLLAMA_BASE_URL:
    # configure client base url
    ollama.Client(host=OLLAMA_BASE_URL)

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2:7b-q4_K_M")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        r = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        out.append(r['embedding'])
    return out

def chat(messages: list[dict], temperature: float = 0.3, max_tokens: int | None = None) -> str:
    res = ollama.chat(model=CHAT_MODEL, messages=messages, options={ "temperature": temperature, **({ "num_predict": max_tokens } if max_tokens else {})})
    return res["message"]["content"]
