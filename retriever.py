import os
import json
import difflib
from pathlib import Path
import html
import re
from dotenv import load_dotenv
load_dotenv()
TOP_K = int(os.getenv("TOP_K", "5"))
DATA_DIR = Path(__file__).parent.joinpath("data")
RETRIEVAL_PATH = DATA_DIR.joinpath("retrieval_results.json")
HOTPOT_PATH = DATA_DIR.joinpath("hotpot_clean.jsonl")
retrieval_index = {}
LAST_MATCH = type("X", (), {})()
LAST_MATCH.value = None
def _normalize_question(s):
    if not s:
        return ""
    s = s.strip()
    s = s.replace(r'\"', '"').replace(r"\'", "'")
    s = s.replace("\\n", " ").replace("\\t", " ")
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()
def _load_retrieval_file():
    global retrieval_index
    retrieval_index = {}
    if not RETRIEVAL_PATH.exists():
        return
    try:
        with open(RETRIEVAL_PATH, "r", encoding="utf-8") as f:
            rr = json.load(f)
        for item in rr:
            qraw = item.get("question", "")
            q = _normalize_question(qraw)
            retrieval_index[q] = item.get("retrieved", [])
    except Exception:
        retrieval_index = {}
_load_retrieval_file()
hotpot_snippets = []
if HOTPOT_PATH.exists():
    try:
        with open(HOTPOT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                title = obj.get("title") or obj.get("article_title") or obj.get("id") or "doc"
                contexts = []
                if "context" in obj and isinstance(obj["context"], list):
                    for c in obj["context"]:
                        contexts.append(c)
                elif "paragraphs" in obj and isinstance(obj["paragraphs"], list):
                    for p in obj["paragraphs"]:
                        if isinstance(p, dict) and "context" in p:
                            contexts.append(p["context"])
                for t in contexts:
                    hotpot_snippets.append({"source": title, "snippet": t})
    except Exception:
        hotpot_snippets = []
embedder = None
hotpot_embeddings = None
def ensure_embeddings():
    global embedder, hotpot_embeddings
    if embedder is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("Install sentence-transformers to enable semantic fallback")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [s["snippet"] for s in hotpot_snippets]
    if texts:
        hotpot_embeddings = embedder.encode(texts, convert_to_tensor=True)
def retrieve_from_hotpot(question, top_k):
    ensure_embeddings()
    from sentence_transformers import util
    q_emb = embedder.encode(question, convert_to_tensor=True)
    hits = util.cos_sim(q_emb, hotpot_embeddings)[0]
    import torch
    vals, idxs = torch.topk(hits, k=min(top_k, len(hotpot_snippets)))
    results = []
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        s = hotpot_snippets[int(idx)]
        results.append({"id": f"hotpot_{idx}", "source": s.get("source"), "snippet": s.get("snippet"), "score": float(score)})
    return results
def rerank_candidates(question, candidates, top_k=5):
    if not candidates:
        return candidates[:top_k]
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return candidates[:top_k]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(question, convert_to_tensor=True)
    texts = [c.get("snippet","") for c in candidates]
    t_emb = model.encode(texts, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, t_emb)[0].cpu().tolist()
    pairs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    out = []
    for i,_ in pairs[:top_k]:
        out.append(candidates[int(i)])
    return out
def retrieve(question, top_k=None):
    top_k = int(top_k or TOP_K)
    qnorm = _normalize_question(question)
    LAST_MATCH.value = None
    if retrieval_index:
        if qnorm in retrieval_index:
            out = retrieval_index[qnorm]
            LAST_MATCH.value = f"exact:{qnorm}"
        else:
            keys = list(retrieval_index.keys())
            best = difflib.get_close_matches(qnorm, keys, n=1, cutoff=0.7)
            if best:
                out = retrieval_index[best[0]]
                LAST_MATCH.value = f"fuzzy:{best[0]}"
            else:
                out = []
    else:
        out = []
    if not out:
        if hotpot_snippets:
            try:
                hot = retrieve_from_hotpot(question, top_k=top_k)
                LAST_MATCH.value = f"hotpot_fallback ({len(hot)} hits)"
                return hot[:top_k]
            except Exception as e:
                LAST_MATCH.value = f"hotpot_error:{e}"
                return []
        return []
    normalized = []
    for i, s in enumerate(out[:top_k]):
        if isinstance(s, dict):
            src = s.get("source") or s.get("id") or f"source_{i}"
            snip = s.get("snippet") or s.get("text") or ""
        else:
            src = f"source_{i}"
            snip = str(s)
        normalized.append({"id": f"pre_{i}", "source": src, "snippet": snip})
    return normalized


