import os, glob, re
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from ollama_client import embed

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./app/storage/chroma")
KB_DIR = "app/data/kb"

os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space": "cosine"})

def read_texts() -> list[tuple[str, str]]:
    exts = ("*.txt","*.md","*.markdown","*.rst","*.log")
    docs = []
    # Read plain text-like files
    for ext in exts:
        for p in glob.glob(f"{KB_DIR}/**/{ext}", recursive=True):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((p, f.read()))
    # Try reading other text-likes by stripping tags
    for p in glob.glob(f"{KB_DIR}/**/*.html", recursive=True):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                text = re.sub("<[^>]+>", " ", raw)
                docs.append((p, text))
        except Exception:
            pass
    return docs

def chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+size]
        chunks.append(" ".join(chunk_words))
        i += size - overlap
    return chunks

def main():
    docs = read_texts()
    if not docs:
        print("No KB docs found. Put files into app/data/kb/")
        return
    ids, texts, metadatas = [], [], []
    for path, content in docs:
        for j, ch in enumerate(chunk(content)):
            ids.append(f"{path}:{j}")
            texts.append(ch)
            metadatas.append({ "source": path })
    vectors = embed(texts)
    # Purge and re-add for idempotency
    collection.delete(where={})
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(texts)} chunks from {len(docs)} files into Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    main()
