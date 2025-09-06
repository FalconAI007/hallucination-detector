# Hallucination Detector for LLMs

A **Streamlit-based prototype** that detects hallucinations in Large Language Model (LLM) answers.

The system:
1. Retrieves relevant evidence (from dataset/jsonl files).
2. Generates an answer using an LLM (OpenAI or fallback).
3. Extracts factual claims from the answer.
4. Verifies each claim against retrieved evidence using embeddings or an optional Graph Neural Network (GNN).
5. Flags answers as:
   - ✅ **Grounded**
   - ⚠️ **Partially Grounded**
   - ❌ **Hallucinated**

---

## 🚀 Demo Workflow
- **Input:** A natural language question  
- **Process:** Retrieval → LLM Answer → Claim Extraction → Verification  
- **Output:** Answer with status + per-claim highlights  

---

## 🛠 Setup

### 1. Clone the repo

git clone https://github.com/FalconAI007/hallucination-detector.git
cd hallucination-detector/streamlit_app


2. Create a virtual environment

python -m venv .venv
.venv\Scripts\activate   # On Windows


3. Install dependencies

pip install -r requirements.txt


4. Configure environment

Create a .env file in streamlit_app/:

OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
TOP_K=5


5. Run the app

streamlit run app.py

Open http://localhost:8501 in your browser.