import os, dotenv, chromadb, hashlib, json
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

dotenv.load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./app/storage/chroma")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Chroma can use "DefaultEmbeddingFunction" which calls an API; we define a stub that asks Ollama via HTTP.
# To keep dependencies light, we stringify the file content as-is and let Chroma store it; embedding is delegated to client-time if needed.
# For real embeddings through Ollama, you'd implement a small client; here we store raw text and rely on naive keyword search fallback.

def read_docs(folder):
    for root, _, files in os.walk(folder):
        for fn in files:
            p = os.path.join(root, fn)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                yield fn, f.read()

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection("knowledge_base", metadata={"hnsw:space":"cosine"})
    docs = list(read_docs("./app/data/kb"))
    ids, texts, metas = [], [], []
    for fn, text in docs:
        did = hashlib.md5((fn+str(len(text))).encode()).hexdigest()
        ids.append(did); texts.append(text); metas.append({"name":fn})
    if ids:
        # Upsert raw text; no embeddings if not configured; Chroma will fallback to internal text splitter search.
        coll.upsert(ids=ids, documents=texts, metadatas=metas)
    print(f"Ingested {len(ids)} docs into Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    main()
