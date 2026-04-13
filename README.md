# 💊 Pharmacovigilance RAG System

Production-grade Retrieval-Augmented Generation system for pharmacovigilance queries, built with Hugging Face BGE embeddings + FAISS.

---

## 📁 Project Structure

```
project/
├── app/
│   ├── streamlit_app.py       # Streamlit UI
│   └── chatbot.py             # LLM integration stub (future)
│
├── src/
│   ├── loaders/
│   │   ├── pdf_loader.py      # PyMuPDF-based PDF text extraction
│   │   └── json_loader.py     # FAERS JSON loader
│   ├── processing/
│   │   ├── chunker.py         # Token-aware overlapping chunker
│   │   └── cleaner.py         # Unicode / whitespace normalisation
│   ├── embeddings/
│   │   └── hf_embedder.py     # HF Inference API client (BGE, batched + retry)
│   ├── vectorstore/
│   │   └── faiss_store.py     # IndexFlatIP build / save / load (singleton)
│   ├── retrieval/
│   │   └── retriever.py       # Query pipeline + drug-intent boosting
│   └── utils/
│       ├── config.py          # All paths & hyperparams
│       └── logger.py          # Structured logger
│
├── data/
│   ├── raw/pdfs/              # Drop PDF books here
│   └── processed/
│       └── faers_rag.json     # Pre-processed FAERS records
│
├── vectorstore/faiss_index/   # Auto-created by build_index.py
│   ├── index.faiss
│   ├── metadata.pkl
│   └── texts.pkl
│
├── scripts/
│   └── build_index.py         # One-time index builder
│
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your HF API key

```bash
cp .env.example .env
# Edit .env and set HF_API_KEY=hf_your_token_here
```

### 3. Add data

- Place PDF books in `data/raw/pdfs/`
- Place processed FAERS JSON at `data/processed/faers_rag.json`

**FAERS JSON schema** (list of objects):
```json
[
  {
    "text": "Patient reported headache after taking aspirin.",
    "drug": "ASPIRIN",
    "reaction": "headache"
  }
]
```

### 4. Build the index (once)

```bash
python scripts/build_index.py
```

### 5. Run the UI

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Architecture

```
Query
  │
  ▼
detect_drug()          ← keyword intent detection
  │
  ▼
embed_query()          ← BGE prefix + HF API + L2 norm
  │
  ▼
VectorStore.search()   ← FAISS IndexFlatIP (cosine sim)
  │
  ▼
re-score (optional)    ← FAERS boost when drug detected
  │
  ▼
Top-K results          → Streamlit UI / LLM context (future)
```

### Embedding model

**BAAI/bge-large-en-v1.5** via Hugging Face Inference API

- Documents: raw text, L2-normalised
- Queries: prefixed with *"Represent this sentence for searching relevant passages: …"*, L2-normalised
- Similarity: inner product (= cosine after normalisation)

---

## ⚙️ Configuration

All settings live in `src/utils/config.py`. Key values:

| Setting | Default | Description |
|---|---|---|
| `PDF_CHUNK_SIZE` | 400 | Approx tokens per chunk |
| `PDF_CHUNK_OVERLAP` | 80 | Overlap between chunks |
| `API_BATCH_SIZE` | 32 | Documents per HF API call |
| `API_MAX_RETRIES` | 5 | Retries on 429/503 |
| `DEFAULT_TOP_K` | 5 | Retrieved results |
| `_FAERS_BOOST` | 0.05 | Score bonus for drug-matched FAERS |

---

## 🔮 Future Extensions

The codebase is structured for clean extension:

### Add LLM answer generation
Edit `app/chatbot.py` → implement `_generate()` and `_init_backend()` for your chosen LLM provider (OpenAI, Bedrock, local GGUF).

### Add re-ranking
Insert a cross-encoder re-ranker between `faiss_store.search()` and the final return in `retriever.py`.

### Add chat history
`Chatbot.history` is already populated; feed it into the LLM prompt.

### Replace keyword drug detection
Swap `detect_drug()` in `retriever.py` with a proper NER model (scispaCy, medspaCy, or a HF NER endpoint).

### Hybrid search
Add BM25 (via `rank_bm25`) alongside FAISS and fuse scores (Reciprocal Rank Fusion).

---

## 🧪 Test Queries

- `"What is pharmacovigilance?"`
- `"Side effects of aspirin"` ← triggers drug-intent boost
- `"Adverse effects of ibuprofen"` ← triggers drug-intent boost
- `"How are adverse drug reactions reported?"`
- `"Signal detection methods in drug safety"`