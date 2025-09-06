from functools import lru_cache
@lru_cache(maxsize=1)
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        raise RuntimeError("sentence-transformers required")
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return m, util
def _split_into_sentences(text):
    import re
    text = text.replace("\n", " ").strip()
    parts = [p.strip() for p in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text) if p.strip()]
    if not parts:
        parts = [text]
    return parts
def _encode_texts(embedder, texts, batch_size=64):
    if not texts:
        return None
    tensors = embedder.encode(texts, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False)
    return tensors
def verify_claims(claims, retrieved, sim_threshold=0.65, top_k=3):
    try:
        embedder, util = get_embedder()
    except Exception:
        out = []
        for c in claims:
            out.append({"claim": c, "best_snippet": (retrieved[0] if retrieved else None), "best_sentence": None, "sim": 0.0, "prob_supported": 0.0, "supported": False, "top_evidence_idxs": []})
        return out
    snippets = [r.get("snippet","") for r in retrieved]
    if not snippets:
        out = []
        for c in claims:
            out.append({"claim": c, "best_snippet": None, "best_sentence": None, "sim": 0.0, "prob_supported": 0.0, "supported": False, "top_evidence_idxs": []})
        return out
    snippet_sentences = []
    snippet_index_map = []
    for i, s in enumerate(snippets):
        sents = _split_into_sentences(s)
        for sent in sents:
            snippet_sentences.append(sent)
            snippet_index_map.append(i)
    if snippet_sentences:
        snippet_embs = _encode_texts(embedder, snippet_sentences)
    else:
        snippet_embs = None
    results = []
    for claim in claims:
        claim_text = claim if isinstance(claim, str) else str(claim)
        claim_emb = embedder.encode(claim_text, convert_to_tensor=True)
        best_sim = -1.0
        best_idx = -1
        best_sent = None
        sims_by_snippet = {}
        if snippet_embs is not None:
            sims = util.cos_sim(claim_emb, snippet_embs)[0]
            sims_list = sims.cpu().tolist()
            for sent_idx, sim_val in enumerate(sims_list):
                snip_idx = snippet_index_map[sent_idx]
                prev = sims_by_snippet.get(snip_idx, [])
                prev.append((sim_val, sent_idx))
                sims_by_snippet[snip_idx] = prev
            per_snippet_best = {}
            for snip_idx, vals in sims_by_snippet.items():
                best_pair = max(vals, key=lambda x: x[0])
                per_snippet_best[snip_idx] = best_pair[0]
                if per_snippet_best[snip_idx] > best_sim:
                    best_sim = per_snippet_best[snip_idx]
                    best_idx = snip_idx
                    sent_idx = best_pair[1]
                    best_sent = snippet_sentences[sent_idx]
            sorted_items = sorted(per_snippet_best.items(), key=lambda x: x[1], reverse=True)
            top_idxs = [int(x[0]) for x in sorted_items[:top_k]]
        else:
            top_idxs = []
            best_sim = -1.0
            best_idx = -1
        if best_idx == -1:
            full_embs = _encode_texts(embedder, snippets)
            if full_embs is not None:
                sims_full = util.cos_sim(claim_emb, full_embs)[0].cpu().tolist()
                for i_sim, s_val in enumerate(sims_full):
                    if s_val > best_sim:
                        best_sim = s_val
                        best_idx = i_sim
                top_idxs = sorted(range(len(sims_full)), key=lambda i: sims_full[i], reverse=True)[:top_k]
            if best_idx >= 0 and best_idx < len(snippets):
                best_sent = None
        best_snippet = None
        if best_idx >= 0 and best_idx < len(retrieved):
            best_snippet = retrieved[best_idx]
        prob_supported = float(max(min(best_sim, 1.0), -1.0)) if best_sim is not None else 0.0
        if prob_supported < 0:
            prob_supported = 0.0
        supported = prob_supported >= sim_threshold
        results.append({
            "claim": claim_text,
            "best_snippet": best_snippet,
            "best_sentence": best_sent,
            "sim": float(prob_supported),
            "prob_supported": float(prob_supported),
            "supported": bool(supported),
            "top_evidence_idxs": top_idxs
        })
    return results
