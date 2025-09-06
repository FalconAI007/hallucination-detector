import os
from dotenv import load_dotenv
load_dotenv()
def _make_prompt(question, retrieved):
    context = "\n\n".join([f"{i}: {r.get('source','')} — {r.get('snippet','')}" for i, r in enumerate(retrieved[:10])])
    if context:
        prompt = (
            "You are given evidence below. Answer the question using only the evidence. "
            "For each factual claim you make, append a line 'EVIDENCE: <idx>' where <idx> is the 0-based index "
            "of the retrieved snippet that supports the claim. If no retrieved snippet supports the claim, write 'EVIDENCE: none'.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer concisely and list claims followed by EVIDENCE lines."
        )
    else:
        prompt = (
            "No evidence available. Answer concisely. If you are unsure, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
    return prompt
def ask_llm(question, retrieved):
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not key:
        return "LLM not configured (OPENAI_API_KEY missing)."
    prompt = _make_prompt(question, retrieved)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        try:
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=512, temperature=0.0)
            try:
                choice = resp.choices[0].message.content
            except Exception:
                try:
                    choice = resp["choices"][0]["message"]["content"]
                except Exception:
                    choice = str(resp)
            return choice.strip()
        except Exception as e_new_call:
            new_err = str(e_new_call)
    except Exception as e_new_import:
        new_err = str(e_new_import)
    try:
        import openai
        openai.api_key = key
        try:
            resp = openai.ChatCompletion.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0.0)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e_legacy_call:
            return f"LLM error: new_client_error={new_err!s} legacy_call_error={e_legacy_call!s}"
    except Exception as e_legacy_import:
        return f"LLM error: new_client_error={new_err!s} legacy_import_error={e_legacy_import!s}"


