import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data"
CHROMA = _ROOT / ".chroma_data"

MODEL = "all-MiniLM-L6-v2"
COL_POLICIES = "policies"
COL_PLAYBOOKS = "playbooks"


def _read_json(name: str):
    path = DATA / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run():
    print("Aurora Bean Roasters, toy helpdesk RAG, ingest")
    print(f"  Embeddings: {MODEL}")
    print(f"  Chroma: {CHROMA}")

    ef = SentenceTransformerEmbeddingFunction(model_name=MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA))

    for col in (COL_POLICIES, COL_PLAYBOOKS):
        try:
            client.delete_collection(col)
        except Exception:
            pass

    pol = client.create_collection(
        name=COL_POLICIES,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    pb = client.create_collection(
        name=COL_PLAYBOOKS,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    policies = _read_json("policies.json")
    p_ids, p_docs, p_meta = [], [], []
    for item in policies:
        p_ids.append(item["id"])
        p_docs.append(f"{item['title']}\n{item['text']}")
        p_meta.append(
            {
                "id": item["id"],
                "title": item["title"],
                "category": item["category"],
            }
        )
    pol.add(ids=p_ids, documents=p_docs, metadatas=p_meta)
    print(f"  Indexed {len(p_ids)} policy chunks.")

    playbooks = _read_json("playbooks.json")
    b_ids, b_docs, b_meta = [], [], []
    for item in playbooks:
        b_ids.append(item["id"])
        b_docs.append(f"{item['title']}\n{item['scenario']}\n{item['text']}")
        b_meta.append(
            {
                "id": item["id"],
                "title": item["title"],
                "scenario": item["scenario"],
            }
        )
    pb.add(ids=b_ids, documents=b_docs, metadatas=b_meta)
    print(f"  Indexed {len(b_ids)} playbook chunks.")
    print("Done.")


if __name__ == "__main__":
    run()
