import os

# ── Embedding ────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_PATH        = "./chroma_db"
COLLECTION_NAME    = "hr_policies"

# ── Groq LLM ─────────────────────────────────────────────────
GROQ_MODEL         = "llama-3.1-8b-instant"   # fast & free-tier friendly
MAX_TOKENS         = 1024
TEMPERATURE        = 0.1                        # low = factual, grounded

# ── RAG ──────────────────────────────────────────────────────
TOP_K              = 5                          # top chunks to retrieve
SIMILARITY_CUTOFF  = 0.30                       # cosine similarity threshold
CHUNK_SIZE         = 500                        # approx tokens per chunk
CHUNK_OVERLAP      = 3                          # overlap sentences

# ── Files ────────────────────────────────────────────────────
DATA_DIR           = "./data"
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_UPLOAD_MB      = 16

# ── Conversation ─────────────────────────────────────────────
MAX_HISTORY_TURNS  = 6                          # last N turns kept in memory