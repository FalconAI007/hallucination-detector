import streamlit as st
import os
import re
from dotenv import load_dotenv
load_dotenv()
from retriever import retrieve, rerank_candidates, LAST_MATCH
from llm_client import ask_llm
from verifier import verify_claims
from gnn_loader import predict_with_gnn, load_gnn

st.set_page_config(page_title="Hallucination Detector", layout="wide")
st.title("Hallucination Detector")

# Default question chosen from your dataset so retrieval won't be empty
default_q = 'The director of the romantic comedy "Big Stone Gap" is based in what New York city?'
question = st.text_area("Enter a question", value=default_q)
top_k = st.number_input("Top-k retrieved", min_value=1, max_value=20, value=int(os.getenv("TOP_K", 5)))
use_rerank = st.checkbox("Rerank retrieved by embedding similarity", value=True)
use_gnn = st.checkbox("Use local GNN verifier if available", value=True)
run = st.button("Run")


def parse_claims_from_llm(llm_text):
    lines = [l.strip() for l in llm_text.splitlines() if l.strip()]
    if not lines:
        cand = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)', llm_text.replace("\n"," ")) if s.strip()]
        out = []
        for s in cand:
            if len(s.split()) > 3 and not re.match(r'^(Therefore|So|Hence|In conclusion)\b', s, flags=re.I):
                t = s.rstrip()
                if not t.endswith("."):
                    t = t + "."
                out.append({"claim": t, "annotated_idx": None})
        return out
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_num = re.match(r'^\s*(\d+)[\.\)]\s*(.+)', line)
        if m_num:
            claim_text = m_num.group(2).strip()
            annotated_idx = None
            j = i + 1
            while j < len(lines) and j <= i + 3:
                m_ev = re.match(r'^(?:EVIDENCE|Evidence)\s*[:\-]\s*(none|\d+)', lines[j], flags=re.I)
                if m_ev:
                    v = m_ev.group(1)
                    if v.lower() != "none":
                        try:
                            annotated_idx = int(v)
                        except Exception:
                            annotated_idx = None
                    j += 1
                    break
                m_inline = re.search(r'(?:EVIDENCE|Evidence)\s*[:\-]\s*(none|\d+)', claim_text, flags=re.I)
                if m_inline:
                    v = m_inline.group(1)
                    claim_text = re.sub(r'(?:EVIDENCE|Evidence)\s*[:\-]\s*(none|\d+)', '', claim_text, flags=re.I).strip()
                    if v.lower() != "none":
                        try:
                            annotated_idx = int(v)
                        except Exception:
                            annotated_idx = None
                    break
                j += 1
            i = j
            if claim_text and not re.match(r'^(Therefore|So|Hence|Thus|In conclusion)\b', claim_text, flags=re.I):
                if not claim_text.endswith("."):
                    claim_text = claim_text + "."
                results.append({"claim": claim_text, "annotated_idx": annotated_idx})
            continue
        m_pair = re.match(r'^(.+?)\.\s*(?:EVIDENCE|Evidence)\s*[:\-]\s*(none|\d+)\s*$', line, flags=re.I)
        if m_pair:
            c = m_pair.group(1).strip()
            v = m_pair.group(2)
            annotated_idx = None if v.lower() == "none" else int(v)
            if not c.endswith("."):
                c = c + "."
            results.append({"claim": c, "annotated_idx": annotated_idx})
            i += 1
            continue
        if len(line.split()) > 3 and not re.match(r'^(Therefore|So|Hence|Thus|In conclusion)\b', line, flags=re.I):
            m_ev_inline = re.search(r'(?:EVIDENCE|Evidence)\s*[:\-]\s*(none|\d+)', line, flags=re.I)
            annotated_idx = None
            if m_ev_inline:
                v = m_ev_inline.group(1)
                if v.lower() != "none":
                    try:
                        annotated_idx = int(v)
                    except Exception:
                        annotated_idx = None
                line = re.sub(r'(?:EVIDENCE|Evidence)\s*[:\-]\s*(none|\d+)', '', line, flags=re.I).strip()
            txt = line
            if not txt.endswith("."):
                txt = txt + "."
            results.append({"claim": txt, "annotated_idx": annotated_idx})
        i += 1
    return results


def short_snip(s, max_chars=400):
    return s if len(s) <= max_chars else s[:max_chars].rsplit(" ",1)[0] + "..."


if run:
    status = st.empty()
    progress = st.empty()

    # ------------------ Retrieval ------------------
    status.info("Retrieving evidence")
    try:
        candidates = retrieve(question, top_k=max(50, top_k))
        if use_rerank:
            try:
                candidates = rerank_candidates(question, candidates, top_k=top_k)
            except Exception:
                candidates = candidates[:top_k]
        retrieved = candidates[:top_k]
    except Exception as e:
        retrieved = []
        status.error("Retriever error: " + str(e))

    st.subheader(f"Raw retrieved (first {top_k})")
    st.write(retrieved[:top_k])
    st.markdown(f"**Retriever match debug**: {getattr(LAST_MATCH, 'value', None)}")

    # ------------------ LLM ------------------
    status.info("Getting LLM answer")
    try:
        llm_answer = ask_llm(question, retrieved)
    except Exception as e:
        llm_answer = f"LLM call failed: {e}"

    st.subheader("Raw LLM answer")
    st.write(llm_answer)

    # ------------------ Claim parsing ------------------
    status.info("Extracting claims")
    parsed = parse_claims_from_llm(llm_answer)
    claims = [p["claim"] for p in parsed]
    annotated_evidence_list = [p.get("annotated_idx") for p in parsed]

    # drop conclusions like "Therefore..."
    filtered, filtered_ann = [], []
    for p, ann in zip(parsed, annotated_evidence_list):
        txt = p.get("claim","").strip()
        if re.match(r'^(Therefore|Hence|So|Thus|In conclusion|Therefore,)', txt, flags=re.I):
            continue
        filtered.append(p["claim"])
        filtered_ann.append(ann)
    claims, annotated_evidence_list = filtered, filtered_ann

    # ------------------ Verification ------------------
    status.info("Running verifier")
    verif = []
    for i, claim in enumerate(claims):
        ann = annotated_evidence_list[i] if i < len(annotated_evidence_list) else None
        if ann is not None:
            try:
                idx = int(ann)
                if 0 <= idx < len(retrieved):
                    snippet = retrieved[idx]
                    res = verify_claims([claim], [snippet], sim_threshold=float(os.getenv("SIM_THRESHOLD",0.65)))
                    if res:
                        res0 = res[0]
                        res0["claim"] = claim
                        res0["annotated_evidence_idx"] = idx
                        verif.append(res0)
                        continue
            except Exception:
                pass
        try:
            res = verify_claims([claim], retrieved, sim_threshold=float(os.getenv("SIM_THRESHOLD",0.65)))
            if res:
                r0 = res[0]
                r0["claim"] = claim
                verif.append(r0)
            else:
                verif.append({"claim": claim, "best_snippet": None, "best_sentence": None, "sim": 0.0,
                              "prob_supported": 0.0, "supported": False, "top_evidence_idxs": []})
        except Exception as e:
            verif.append({"claim": claim, "best_snippet": None, "best_sentence": None, "sim": 0.0,
                          "prob_supported": 0.0, "supported": False, "top_evidence_idxs": [], "error": str(e)})

    status.success("Done")

    # ------------------ Scoring ------------------
    annotated_verif = [v for v in verif if v.get("annotated_evidence_idx") is not None]
    if annotated_verif:
        score = sum([float(v.get("prob_supported",0.0)) for v in annotated_verif]) / max(1, len(annotated_verif))
    else:
        score = sum([float(v.get("prob_supported",0.0)) for v in verif]) / max(1, len(verif)) if verif else 0.0
    final_status = "Grounded" if score >= 0.8 else "Partially Grounded" if score >= 0.4 else "Hallucinated"
    st.markdown(f"## Status: {final_status} — Score: {score:.2f}")

    # ------------------ UI sections ------------------
    st.subheader("LLM Answer (claims extracted)")
    st.info(llm_answer)

    st.subheader("Retrieved Evidence")
    if not retrieved:
        st.write("No evidence found")
    else:
        for i, r in enumerate(retrieved):
            st.write(f"[{i}] **{r.get('source')}** — {short_snip(r.get('snippet',''))}")

    st.subheader("Verification Highlights")
    if not verif:
        st.write("No verification output")
    else:
        for v in verif:
            col1, col2 = st.columns([0.06, 1])
            color = "green" if v.get("supported", False) else "red"
            col1.markdown(
                f"<div style='width:14px;height:14px;border-radius:7px;background:{color};margin-top:6px;'></div>",
                unsafe_allow_html=True)
            score_val = v.get("sim", v.get("prob_supported", 0.0))
            try:
                score_txt = f"{float(score_val):.3f}"
            except Exception:
                score_txt = str(score_val)
            claim_text = v.get("claim", "")
            lines = [f"Claim: {claim_text}", f"Score: {score_txt}"]
            if v.get("best_sentence"):
                lines.append(f"Best sentence: {v['best_sentence']}")
            elif v.get("best_snippet"):
                bsn = v["best_snippet"]
                lines.append(f"Best evidence: {bsn.get('source','')} — {short_snip(bsn.get('snippet',''))}")
            if v.get("top_evidence_idxs"):
                lines.append(f"Top evidence idxs: {v.get('top_evidence_idxs')}")
            if v.get("annotated_evidence_idx") is not None:
                lines.append(f"Annotated evidence idx: {v.get('annotated_evidence_idx')}")
            col2.write("\n\n".join(lines))

    progress.empty()
    status.empty()

st.caption("If GNN or OpenAI key is missing, fallback verifier is used.")

