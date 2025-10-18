import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import chromadb
from .ollama_client import chat, embed
import importlib.util
from pathlib import Path

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./app/storage/chroma")
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# Vector store
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space":"cosine"})

# Try import ml_orchestrator entrypoint if present
def try_load_orchestrator():
    # look in app/external/ml_orchestrator for a python package-like entry file
    base = Path(__file__).resolve().parent / "external" / "ml_orchestrator"
    candidates = list(base.rglob("*.py"))
    for c in candidates:
        spec = importlib.util.spec_from_file_location("ml_orchestrator_dynamic", c)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "route_task"):
                    return mod
            except Exception:
                continue
    return None

ORCH = try_load_orchestrator()

app = FastAPI(title="Local AI Agent")

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).resolve().parent / "web" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.post("/api/ingest")
def api_ingest():
    # lazy import to reuse code
    from ingest_kb import main as ingest_main
    try:
        ingest_main()
        return { "status":"ok" }
    except Exception as e:
        return JSONResponse(status_code=500, content={ "error": str(e) })

@app.post("/api/chat")
async def api_chat(req: Request):
    data = await req.json()
    query = data.get("query","").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error":"Empty query"})
    # retrieve context
    embs = embed([query])[0]
    results = collection.query(query_embeddings=[embs], n_results=MAX_CONTEXT_DOCS)
    contexts = results.get("documents", [[]])[0]
    sources = results.get("metadatas", [[]])[0]

    system = (
        "Ты локальный ИИ-агент. Отвечай кратко и точно на русском. "
        "Только русски отвечай"
        "Используй знания из контекста, если они релевантны. "
        "Если информации нет, признай это и предложи следующую команду /api/ingest."
        "Если человек выражается или пропагандирует что-то, предупреди его чтобы прекратил так выражаться и снова написал нормальный запрос"
    )
    context_block = "\n\n".join([f"[{i+1}] {c[:1200]}" for i, c in enumerate(contexts)])
    user = (
        f"Вопрос: {query}\n\n"
        f"Контекст (фрагменты):\n{context_block}\n\n"
        f"Если контекст нерелевантен, игнорируй его."
    )
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":user}
    ]

    # Optional orchestrator call
    tool_result = ""
    if ORCH and hasattr(ORCH, "route_task"):
        try:
            tool_result = ORCH.route_task(query=query, context=contexts) or ""
            if tool_result:
                messages.append({"role":"user","content":f"Результат инструмента: {tool_result}"})
        except Exception:
            pass

    answer = chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return {
        "answer": answer,
        "sources": sources
    }
