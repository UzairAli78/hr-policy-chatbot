# 🤖 hr-policy-chatbot

**PolicAI** — an intelligent HR Policy Assistant powered by a full RAG (Retrieval-Augmented Generation) pipeline. Employees can ask natural language questions about company HR policies and receive accurate, grounded answers sourced directly from uploaded policy documents. Built with Flask, ChromaDB, Sentence Transformers, and Groq (Llama 3.1).

---

## 📁 Project Structure

```
hr-policy-chatbot/
├── app.py                   # Flask backend — REST API & route handlers
├── rag_pipeline.py          # Full RAG pipeline (embed → retrieve → LLM → guardrail)
├── ingest.py                # Document ingestion pipeline (chunking + embedding → ChromaDB)
├── config.py                # Central configuration (models, paths, RAG settings)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed to version control)
├── .python-version          # Python 3.11.0
├── .gitignore               # Excludes venv, __pycache__, .env, chroma_db
├── templates/
│   └── index.html           # PolicAI chat UI (dark-themed, single-page)
├── data/                    # Drop HR documents here (.txt, .pdf, .docx)
│   └── sample_hr_policy.txt # Sample: Apex Solutions HR Handbook v3.0
└── chroma_db/               # Persistent ChromaDB vector store (auto-generated)
```

---

## ✨ Features

- **RAG pipeline** — questions are answered using only retrieved HR document chunks, never hallucinated facts
- **Multi-format ingestion** — supports `.txt`, `.pdf`, and `.docx` HR documents
- **Sentence-level chunking** — documents split into ~500-token overlapping chunks for precise retrieval
- **Cosine similarity search** — ChromaDB with configurable similarity threshold (default: 0.30)
- **Groq LLM** — `llama-3.1-8b-instant` for fast, free-tier-friendly responses
- **Hallucination guardrail** — detects and replaces off-policy answers with an HR referral message
- **Conversation memory** — retains last 6 turns of context per session
- **Live document upload** — new HR docs can be ingested through the UI without restarting the server
- **Dark-themed chat UI** — teal/gold accented design using Syne + DM Sans fonts

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-org/hr-policy-chatbot.git
cd hr-policy-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Requires **Python 3.11.0** (see `.python-version`).

---

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

### 3. Add HR Documents

Place your HR policy files (`.txt`, `.pdf`, or `.docx`) in the `./data/` directory. A sample policy is already included:

```
data/
└── sample_hr_policy.txt    # Apex Solutions HR Handbook — 16-section policy document
```

---

### 4. Ingest Documents

Run ingestion once to populate the vector store before starting the server:

```bash
RUN_INGEST=true python app.py
```

Or ingest manually via the Python shell:

```python
from ingest import DocumentIngester
ingester = DocumentIngester()
ingester.ingest_directory("./data")
```

---

### 5. Start the Server

```bash
python app.py
```

Open **http://localhost:5000** in your browser. The PolicAI chat interface will be ready.

For production deployment:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/api/chat` | Send a question, receive an AI answer |
| `POST` | `/api/upload` | Upload and ingest a new HR document |
| `GET` | `/api/stats` | Returns vector store statistics |
| `POST` | `/api/clear` | Clears conversation history |

### `POST /api/chat`

**Request:**
```json
{ "message": "How many days of annual leave do I get?" }
```

**Response:**
```json
{
  "answer": "According to the Leave Policy, employees are entitled to...",
  "sources": ["sample_hr_policy.txt"],
  "context_used": true
}
```

### `POST /api/upload`

Send a `multipart/form-data` request with a `file` field (`.txt`, `.pdf`, or `.docx`, max 16 MB).

**Response:**
```json
{
  "message": "✔ Successfully ingested 'new_policy.pdf' — 42 new chunks added.",
  "chunks_added": 42,
  "filename": "new_policy.pdf"
}
```

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `hr_policies` | ChromaDB collection name |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq LLM model |
| `MAX_TOKENS` | `1024` | Max tokens per LLM response |
| `TEMPERATURE` | `0.1` | Low temperature = factual, grounded output |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_CUTOFF` | `0.30` | Minimum cosine similarity to include a chunk |
| `CHUNK_SIZE` | `500` | Approximate tokens per chunk |
| `CHUNK_OVERLAP` | `3` | Sentences of overlap between chunks |
| `MAX_UPLOAD_MB` | `16` | Maximum file upload size |
| `MAX_HISTORY_TURNS` | `6` | Conversation turns kept in memory |

---

## 🏗️ How It Works

### Ingestion Pipeline (`ingest.py`)

1. **Read** — extracts text from `.txt` (plain read), `.pdf` (PyPDF2), or `.docx` (python-docx)
2. **Chunk** — tokenises into sentences using NLTK `punkt_tab`, groups into ~500-token chunks with 3-sentence overlap
3. **Embed** — encodes chunks using `sentence-transformers/all-MiniLM-L6-v2`
4. **Store** — upserts embeddings + metadata into a persistent ChromaDB collection; duplicate IDs are skipped automatically

### RAG Pipeline (`rag_pipeline.py`)

1. **Preprocess** — normalises whitespace in the user query
2. **Embed query** — same embedding model as ingestion for a consistent vector space
3. **Retrieve** — cosine similarity search in ChromaDB; filters results below `SIMILARITY_CUTOFF`
4. **Prompt build** — injects retrieved chunks into a strict system prompt that instructs the LLM to answer only from provided context
5. **LLM call** — sends system prompt + rolling conversation history to Groq
6. **Guardrail** — if no context was found, or if hallucination phrases are detected in the response, the answer is replaced with an HR referral message
7. **History update** — appends the turn to the rolling conversation buffer (capped at `MAX_HISTORY_TURNS`)

---

## ⚠️ Notes & Limitations

- **`GROQ_API_KEY` is required** — without it, all `/api/chat` calls will return an error
- The `chroma_db/` directory is excluded in `.gitignore`; vector data does not persist across fresh clones without re-running ingestion
- Auto-ingestion on startup is **disabled by default** — set `RUN_INGEST=true` to trigger it once, then remove the flag to avoid re-processing on every restart
- PyTorch is pinned to the CPU build (`torch==2.2.2+cpu`) — GPU acceleration requires a manual `torch` reinstall targeting your CUDA version
- The similarity threshold of `0.30` is intentionally permissive; tighten it in `config.py` if off-topic answers slip through

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `flask` + `flask-cors` | REST API server |
| `groq` | Groq LLM client (Llama 3.1) |
| `chromadb` | Local persistent vector store |
| `sentence-transformers` | Text embeddings (`all-MiniLM-L6-v2`) |
| `nltk` | Sentence tokenisation for chunking |
| `PyPDF2` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `python-dotenv` | `.env` file loading |
| `gunicorn` | Production WSGI server |
