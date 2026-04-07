"""
ingest.py
Ingestion pipeline:
  HR docs (PDF / DOCX / TXT)
  → Chunker (~500 tokens with overlap)
  → HuggingFace Embedding model
  → ChromaDB vector store
"""

import os
import re
import glob
import logging
from pathlib import Path
from typing import List, Dict

import nltk
import chromadb
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── NLTK bootstrap ──────────────────────────────────────────
# ── NLTK bootstrap ──────────────────────────────────────────
# NOTE: NLTK 3.8+ uses 'punkt_tab' instead of the old 'punkt'.
# We catch both LookupError and OSError because nltk.data.find() raises
# OSError (not LookupError) when the directory exists but inner files are
# missing — which happens when only the legacy punkt package was downloaded.
for resource, package in [
    ("tokenizers/punkt_tab", "punkt_tab"),
]:
    try:
        nltk.data.find(resource)
    except (LookupError, OSError):
        nltk.download(package, quiet=True)


class DocumentIngester:
    """Reads HR documents, chunks them, embeds them, stores in ChromaDB."""

    def __init__(self):
        logger.info("Loading embedding model for ingestion …")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        self.chroma = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.chroma.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("DocumentIngester ready ✔")

    # ── File readers ─────────────────────────────────────────
    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()

    def _read_pdf(self, path: str) -> str:
        try:
            import PyPDF2
            text = []
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text)
        except Exception as exc:
            logger.error("PDF read error (%s): %s", path, exc)
            return ""

    def _read_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as exc:
            logger.error("DOCX read error (%s): %s", path, exc)
            return ""

    # ── Chunker ──────────────────────────────────────────────
    def _chunk(self, text: str, source_name: str) -> List[Dict]:
        """
        Split text into overlapping sentence-based chunks of ~CHUNK_SIZE tokens.
        Approximate token count: len(text) / 4.
        """
        # Normalise whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[Dict] = []
        current: List[str] = []
        current_tokens = 0
        chunk_idx = 0
        stem = Path(source_name).stem

        for sent in sentences:
            est_tokens = max(1, len(sent) // 4)

            if current_tokens + est_tokens > config.CHUNK_SIZE and current:
                body = " ".join(current).strip()
                if len(body) > 60:
                    chunks.append({
                        "id":    f"{stem}__chunk_{chunk_idx}",
                        "text":  body,
                        "source": source_name,
                        "chunk_index": chunk_idx,
                    })
                    chunk_idx += 1

                # Keep overlap (last N sentences)
                overlap = current[-config.CHUNK_OVERLAP:] if len(current) >= config.CHUNK_OVERLAP else current[:]
                current = overlap + [sent]
                current_tokens = sum(max(1, len(s) // 4) for s in current)
            else:
                current.append(sent)
                current_tokens += est_tokens

        # Flush remaining
        if current:
            body = " ".join(current).strip()
            if len(body) > 60:
                chunks.append({
                    "id":    f"{stem}__chunk_{chunk_idx}",
                    "text":  body,
                    "source": source_name,
                    "chunk_index": chunk_idx,
                })

        logger.info("'%s' → %d chunks", source_name, len(chunks))
        return chunks

    # ── Ingest single file ───────────────────────────────────
    def ingest_file(self, filepath: str) -> int:
        ext = Path(filepath).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", ext)
            return 0

        readers = {".txt": self._read_txt, ".pdf": self._read_pdf, ".docx": self._read_docx}
        text = readers[ext](filepath)
        if not text.strip():
            logger.warning("No text extracted from %s", filepath)
            return 0

        source_name = Path(filepath).name
        chunks = self._chunk(text, source_name)
        if not chunks:
            return 0

        # Avoid duplicate IDs
        try:
            existing = set(self.collection.get()["ids"])
        except Exception:
            existing = set()

        new_chunks = [c for c in chunks if c["id"] not in existing]
        if not new_chunks:
            logger.info("All chunks from '%s' already indexed.", source_name)
            return 0

        texts      = [c["text"]   for c in new_chunks]
        ids        = [c["id"]     for c in new_chunks]
        metadatas  = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in new_chunks]
        embeddings = self.embedder.encode(texts).tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Added %d new chunks from '%s' to ChromaDB.", len(new_chunks), source_name)
        return len(new_chunks)

    # ── Ingest directory ─────────────────────────────────────
    def ingest_directory(self, directory: str) -> Dict[str, int]:
        os.makedirs(directory, exist_ok=True)
        results: Dict[str, int] = {}
        for ext in config.ALLOWED_EXTENSIONS:
            for fp in glob.glob(os.path.join(directory, f"*{ext}")):
                results[fp] = self.ingest_file(fp)
        return results

    def stats(self) -> Dict:
        return {"total_chunks": self.collection.count()}