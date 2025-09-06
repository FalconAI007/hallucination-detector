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

hallucination-detector/
│
├── streamlit_app/          # main app code
│   ├── app.py
│   ├── retriever.py
│   ├── verifier.py
│   ├── llm_client.py
│   ├── gnn_loader.py
│   ├── requirements.txt
│   ├── .env (ignored by git)
│   ├── README.md
│   ├── data/               # datasets & retrieval files
│   │   ├── retrieval_results.json
│   │   ├── hotpot_clean.jsonl
│   │   └── README.md
│   ├── models/             # trained GNN model weights
│   │   └── README.md     
│   └── .streamlit/         # Streamlit config
│       └── config.toml
│
└── .gitignore



## 🛠 Setup

### 1. Clone the repository
```bash
git clone https://github.com/FalconAI007/hallucination-detector.git
cd hallucination-detector/streamlit_app
````

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a file named `.env` inside `streamlit_app/` and add:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
TOP_K=5
```

### 5. Prepare folders

Make sure these folders exist (some may already be created):

```
streamlit_app/data/    -> for datasets like retrieval_results.json, hotpot_clean.jsonl
streamlit_app/models/  -> for GNN weights or other trained models
```

### 6. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.
