from sentence_transformers import SentenceTransformer, util
EMB = SentenceTransformer("all-MiniLM-L6-v2")
class GNNWrapper:
    def __init__(self):
        pass
    def load_state_dict(self, sd):
        return
    def predict(self, claims, evidence, params):
        outs = []
        evid_texts = [e.get("snippet","") for e in evidence]
        if evid_texts:
            evid_embs = EMB.encode(evid_texts, convert_to_tensor=True)
        else:
            evid_embs = None
        for c in claims:
            claim_emb = EMB.encode(c, convert_to_tensor=True)
            prob = 0.0
            top_idxs = []
            top_sims = []
            if evid_embs is not None:
                sims = util.cos_sim(claim_emb, evid_embs)[0].cpu().numpy().tolist()
                best_idx = int(max(range(len(sims)), key=lambda i: sims[i])) if sims else -1
                prob = float(sims[best_idx]) if best_idx>=0 else 0.0
                top_idxs = [best_idx]
                top_sims = [prob]
            outs.append({"claim": c, "prob_supported": prob, "supported": prob>0.65, "top_evidence_idxs": top_idxs, "top_evidence_sims": top_sims})
        return outs
