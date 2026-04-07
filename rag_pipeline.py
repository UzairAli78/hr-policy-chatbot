"""
rag_pipeline.py
Full RAG pipeline:
  1. Embed query  (HuggingFace sentence-transformers)
  2. Similarity search  (ChromaDB cosine)
  3. Prompt builder  (system + context + query)
  4. LLM generation  (Groq – llama-3.1-8b-instant)
  5. Guardrail check  (no-hallucination policy)
"""

import os
import re
import logging
from typing import List, Dict, Optional

import nltk
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import config

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── NLTK bootstrap ──────────────────────────────────────────
# NOTE: NLTK 3.8+ uses 'punkt_tab' instead of the old 'punkt'.
# We catch both LookupError and OSError because nltk.data.find() raises
# OSError (not LookupError) when the directory exists but inner files are
# missing — which happens when only the legacy punkt package was downloaded.
for resource, package in [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
]:
    try:
        nltk.data.find(resource)
    except (LookupError, OSError):
        nltk.download(package, quiet=True)


# ── SYSTEM PROMPT ───────────────────────────────────────────
SYSTEM_TEMPLATE = """You are PolicAI, an intelligent HR Policy Assistant for a company.
Your job is to answer employee questions accurately, using ONLY the HR policy excerpts provided below.

━━━ STRICT RULES ━━━
1. Base your answer EXCLUSIVELY on the Context provided. Never invent facts.
2. If the Context does not contain enough information to answer the question, respond with the exact phrase:
   "I'm sorry, I don't have that information in our current HR documentation. Please contact the HR department directly for clarification."
3. Cite the source document name at the end of your answer.
4. Be professional, warm, and concise.
5. If the question is ambiguous, ask the employee to clarify.
6. Never guess, assume, or use outside knowledge about policies.

━━━ HR POLICY CONTEXT ━━━
{context}
━━━ END OF CONTEXT ━━━
"""


class HRChatbot:
    """End-to-end HR Policy Chatbot with RAG pipeline."""

    def __init__(self):
        logger.info("Loading embedding model …")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        logger.info("Connecting to ChromaDB …")
        self.chroma = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.chroma.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Initialising Groq client …")
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set – chat will return an error.")
        self.groq = Groq(api_key=api_key)

        self.history: List[Dict[str, str]] = []   # conversation memory
        logger.info("HRChatbot ready ✔")

    # ── NLP preprocessing ───────────────────────────────────
    def _preprocess(self, text: str) -> str:
        """Normalise whitespace and trailing punctuation."""
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Step 1 & 2 : Embed + Retrieve ───────────────────────
    def retrieve(self, query: str) -> Dict:
        """
        Embed the query and perform cosine similarity search in ChromaDB.
        Returns a dict with 'chunks' (text list) and 'sources' (file list).
        """
        if self.collection.count() == 0:
            return {"chunks": [], "sources": []}

        embedding = self.embedder.encode(query).tolist()
        k = min(config.TOP_K, self.collection.count())

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        raw_docs  = results["documents"][0]  if results["documents"]  else []
        raw_meta  = results["metadatas"][0]  if results["metadatas"]  else []
        raw_dists = results["distances"][0]  if results["distances"]  else []

        chunks, sources = [], []
        for doc, meta, dist in zip(raw_docs, raw_meta, raw_dists):
            similarity = 1.0 - dist          # cosine distance → similarity
            if similarity >= config.SIMILARITY_CUTOFF:
                chunks.append(doc)
                src = meta.get("source", "HR Policy Document")
                if src not in sources:
                    sources.append(src)

        logger.info("Retrieved %d relevant chunks (threshold=%.2f)", len(chunks), config.SIMILARITY_CUTOFF)
        return {"chunks": chunks, "sources": sources}

    # ── Step 3 : Build prompt ────────────────────────────────
    def _build_system(self, chunks: List[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return SYSTEM_TEMPLATE.format(context=context)

    # ── Step 5 : Guardrail ───────────────────────────────────
    def _guardrail(self, answer: str, chunks: List[str]) -> str:
        """
        If no context was found but the model still produced a long answer,
        that answer is likely hallucinated – replace with an apology.
        """
        if not chunks:
            return (
                "I'm sorry, I don't have specific information about that in our current HR "
                "documentation. For accurate guidance, please reach out to the HR department "
                "directly or refer to the official company intranet."
            )
        # Detect common hallucination phrases when no grounding context exists
        hallucination_flags = [
            "as of my knowledge", "based on general", "typically companies",
            "in most organisations", "generally speaking",
        ]
        lower = answer.lower()
        if any(flag in lower for flag in hallucination_flags):
            return (
                "I'm sorry, I wasn't able to find a definitive answer in our HR documents. "
                "Please contact the HR department for accurate information on this topic."
            )
        return answer

    # ── Main chat entry point ────────────────────────────────
    def chat(self, user_message: str) -> Dict:
        """
        Full RAG pipeline:
        preprocess → retrieve → prompt → LLM → guardrail → respond
        """
        # 1. NLP preprocess
        query = self._preprocess(user_message)

        # 2. Retrieve context
        context_data = self.retrieve(query)
        chunks  = context_data["chunks"]
        sources = context_data["sources"]

        # 3. No context → early apology (guardrail fires immediately)
        if not chunks:
            apology = self._guardrail("", [])
            return {"answer": apology, "sources": [], "context_used": False}

        # 4. Build system prompt
        system_msg = self._build_system(chunks)

        # 5. Assemble messages (include rolling conversation history)
        history_slice = self.history[-(config.MAX_HISTORY_TURNS * 2):]
        messages = [*history_slice, {"role": "user", "content": query}]

        # 6. Call Groq LLM
        try:
            resp = self.groq.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "system", "content": system_msg}, *messages],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=0.9,
            )
            answer = resp.choices[0].message.content.strip()

        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            return {
                "answer": (
                    "I'm sorry, I encountered a technical issue while processing your request. "
                    "Please try again in a moment or contact the HR department directly."
                ),
                "sources": [],
                "context_used": False,
                "error": str(exc),
            }

        # 7. Guardrail check
        answer = self._guardrail(answer, chunks)

        # 8. Update rolling history
        self.history.append({"role": "user",      "content": query})
        self.history.append({"role": "assistant",  "content": answer})
        if len(self.history) > config.MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(config.MAX_HISTORY_TURNS * 2):]

        return {"answer": answer, "sources": sources, "context_used": True}

    # ── Utilities ────────────────────────────────────────────
    def stats(self) -> Dict:
        return {
            "total_chunks":    self.collection.count(),
            "collection_name": config.COLLECTION_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model":       config.GROQ_MODEL,
        }

    def clear_history(self):
        self.history = []
        logger.info("Conversation history cleared.")