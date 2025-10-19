import os, json, asyncio, dotenv, chromadb
from ollama_client import chat as ollama_chat

dotenv.load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./app/storage/chroma")

client = chromadb.PersistentClient(path=CHROMA_DIR)
kb = client.get_or_create_collection("knowledge_base")

def retrieve(query, k=10):
    if kb.count()==0: return []
    res = kb.query(query_texts=[query], n_results=k)
    return [{"id": res["ids"][0][i], "title": res["metadatas"][0][i].get("name","doc"),
             "text": res["documents"][0][i]} for i in range(len(res["ids"][0]))]

async def score(q, doc):
    system = "Верни только число от 0 до 1. Оцени релевантность документа запросу."
    prompt = f"Запрос: {q}\nДокумент: {doc['title']}\nТекст: {doc['text']}\nОтвет:"
    s = await ollama_chat(system, prompt)
    try:
        return float(s.strip().replace(",",".").split()[0])
    except:
        return 0.0

async def eval_file(dataset_path):
    data = json.load(open(dataset_path,"r",encoding="utf-8"))
    hits = 0
    results = []
    for i,item in enumerate(data, 1):
        cands = retrieve(item["query"], k=10)
        scored = []
        for d in cands:
            s = await score(item["query"], d)
            scored.append((d["id"], s))
        scored.sort(key=lambda x:-x[1])
        top3 = [x[0] for x in scored[:3]]
        hit = any(doc_id in top3 for doc_id in item["relevant"])
        hits += 1 if hit else 0
        results.append({"query": item["query"], "top3": top3, "hit": hit})
        print(f"[{i}/{len(data)}] hit={hit}")
    recall = hits/len(data) if data else 0.0
    json.dump({"recall_at_3": recall, "results": results}, open("eval_report.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Recall@3:", round(recall,3))

if __name__ == "__main__":
    import sys, asyncio
    if len(sys.argv)<2:
        print("Usage: python eval_recall_qwen.py dataset.json")
        raise SystemExit(2)
    asyncio.run(eval_file(sys.argv[1]))
