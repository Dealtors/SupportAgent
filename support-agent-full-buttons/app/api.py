import os, asyncio, time, json, hashlib
from typing import List, Optional
import dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from ollama_client import chat as ollama_chat

dotenv.load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./app/storage/chroma")

app = FastAPI(title="SupportAgent API")

# storage
client = chromadb.PersistentClient(path=CHROMA_DIR)
kb = client.get_or_create_collection("kb", metadata={"hnsw:space":"cosine"})

class TicketIn(BaseModel):
    text: str
    ticket_id: Optional[str] = None

class FeedbackIn(BaseModel):
    ticket_id: str
    feedback: str  # "помогло" | "исправление:<label>" | "оператор" | "отказ"
    label: Optional[str] = None

logs_path = "./app/storage/logs.jsonl"
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

def log_event(evt: dict):
    with open(logs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False)+"\n")

def simple_classify(text: str):
    t = text.lower()
    classes = {
    "install_update": [
        "обнов", "апдейт", "патч", "не запускается", "упал", "вылет", "краш",
        "зависает", "зависло", "ошибка запуска", "прекратил работу", "установк",
        "инсталляц", "переустанов", "обновление зависло", "setup", "installer"
    ],
    "network_issue": [
        "сеть", "интернет", "соедин", "vpn", "wifi", "wi-fi", "подключен",
        "подключени", "сервер", "не отвечает", "восстановить соединение",
        "потеря сети", "ошибка подключения", "dns", "ping", "провайдер",
        "пакеты", "тайм-аут", "задержка", "network"
    ],
    "billing": [
        "оплат", "подписк", "чек", "счет", "платеж", "транзакц", "карта",
        "списан", "банковск", "перевод", "покупк", "invoice", "subscription",
        "balance", "биллинг", "оплачено", "не прошел платеж"
    ],
    "account": [
        "аккаунт", "парол", "вход", "логин", "авторизац", "аутентификац",
        "профиль", "учетн", "регистрац", "войти", "sign in", "sign-in",
        "login", "logout", "смена пароля", "восстановить доступ", "сброс пароля"
    ]
}
    best = ("other",0)
    for label, kws in classes.items():
        score = sum(1 for k in kws if k in t)
        if score > best[1]: best = (label, score)
    conf = min(1.0, best[1]/3)
    return {"label": best[0], "confidence": conf}

def retrieve(query: str, k=5):
    if kb.count()==0:
        return []
    res = kb.query(query_texts=[query], n_results=k)
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i]
        })
    return docs

@app.post("/ticket")
async def post_ticket(inp: TicketIn):
    ticket_id = inp.ticket_id or hashlib.md5((inp.text+str(time.time())).encode()).hexdigest()[:12]
    cls = simple_classify(inp.text)
    docs = retrieve(inp.text, k=3)
    plan = f"Класс: {cls['label']} (уверенность {cls['confidence']:.2f}). Найдено подсказок: {len(docs)}."
    context = "\n\n".join([f"- {d['meta'].get('name')}: {d['text']}" for d in docs])
    system = "Ты помощник техподдержки. Пиши кратко и по делу. Предлагай пошаговые действия."
    answer = await ollama_chat(system, f"Запрос: {inp.text}\nКонтекст:\n{context}\nСформируй ответ и план.")
    log_event({"ts": time.time(), "ticket_id": ticket_id, "text": inp.text, "cls": cls, "docs": [d['meta'] for d in docs], "answer": answer})
    return {"ticket_id": ticket_id, "classification": cls, "docs": [d["meta"] for d in docs], "answer": answer, "plan": plan}

@app.post("/feedback")
async def post_feedback(fb: FeedbackIn):
    log_event({"ts": time.time(), "ticket_id": fb.ticket_id, "feedback": fb.feedback, "label": fb.label})
    return {"ok": True}

@app.get("/report")
def get_report():
    # baseline: accuracy unknown without labels; report average confidence over last N events
    import statistics
    confs = []
    total = 0
    try:
        with open(logs_path,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "cls" in obj:
                        total += 1
                        confs.append(obj["cls"].get("confidence",0))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    avg_conf = statistics.mean(confs) if confs else 0.0
    return {"tickets": total, "avg_confidence": round(avg_conf,3)}

@app.post("/api/chat")
async def api_chat(inp: TicketIn):
    # convenience alias for web UI
    return await post_ticket(inp)

@app.get("/")
def root():
    return {"message": "See /api for endpoints. Web UI served from /web."}


# Serve static web
app.mount("/web", StaticFiles(directory="./app/web", html=True), name="web")

@app.get("/")
def serve_index():
    return FileResponse("./app/web/index.html")
