from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from load_env import chroma_path

_ROOT = Path(__file__).resolve().parent.parent
MODEL = "all-MiniLM-L6-v2"
COL_POLICIES = "policies"
COL_PLAYBOOKS = "playbooks"

_client = None
_ef = None


def _get_collections():
    global _client, _ef
    if _client is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=MODEL)
        _client = chromadb.PersistentClient(path=chroma_path())
    pol = _client.get_collection(name=COL_POLICIES, embedding_function=_ef)
    pb = _client.get_collection(name=COL_PLAYBOOKS, embedding_function=_ef)
    return pol, pb


def two_stage_search(question: str, policy_k: int = 3, playbook_k: int = 3):
    pol, pb = _get_collections()

    p_res = pol.query(query_texts=[question], n_results=policy_k)
    policy_rows = []
    if p_res["ids"] and p_res["ids"][0]:
        for i, pid in enumerate(p_res["ids"][0]):
            policy_rows.append(
                {
                    "id": pid,
                    "title": (p_res["metadatas"][0][i] or {}).get("title", ""),
                    "category": (p_res["metadatas"][0][i] or {}).get("category", ""),
                    "text": p_res["documents"][0][i],
                    "distance": p_res.get("distances", [[]])[0][i]
                    if p_res.get("distances")
                    else None,
                }
            )

    context_bits = [question]
    for p in policy_rows:
        context_bits.append(p.get("text", ""))
    enhanced = "\n\n".join(context_bits)

    b_res = pb.query(query_texts=[enhanced], n_results=playbook_k)
    playbook_rows = []
    if b_res["ids"] and b_res["ids"][0]:
        for i, bid in enumerate(b_res["ids"][0]):
            meta = b_res["metadatas"][0][i] or {}
            playbook_rows.append(
                {
                    "id": bid,
                    "title": meta.get("title", ""),
                    "scenario": meta.get("scenario", ""),
                    "text": b_res["documents"][0][i],
                    "distance": b_res.get("distances", [[]])[0][i]
                    if b_res.get("distances")
                    else None,
                }
            )

    return {"policies": policy_rows, "playbooks": playbook_rows}
