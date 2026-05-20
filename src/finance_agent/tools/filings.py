"""RAG retriever over a single SEC filing, persisted as a local FAISS index."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_agent.config import Settings, get_settings
from finance_agent.llm import get_embeddings

logger = logging.getLogger(__name__)


class FilingsRetriever:
    """Build, persist, and query a FAISS index over an SEC filing.

    The index is built once per filing and cached on disk under
    ``settings.faiss_index_path``. Subsequent runs load instantly.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings or get_embeddings()
        self._store: FAISS | None = None

    @property
    def index_dir(self) -> Path:
        return Path(self._settings.faiss_index_path)

    @property
    def source_path(self) -> Path:
        return Path(self._settings.filings_text_path)

    def _ensure_loaded(self) -> FAISS:
        if self._store is not None:
            return self._store

        index_file = self.index_dir / "index.faiss"
        if index_file.exists():
            logger.info("Loading FAISS index from %s", self.index_dir)
            self._store = FAISS.load_local(
                str(self.index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("No FAISS index at %s — building from %s", self.index_dir, self.source_path)
            self._store = self.build()
        return self._store

    def build(self) -> FAISS:
        """Read the filing, chunk it, embed it, persist the index, return the store."""
        if not self.source_path.exists():
            raise FileNotFoundError(f"Filings text not found at {self.source_path}")

        text = self.source_path.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": self.source_path.name}],
        )
        logger.info("Embedding %d chunks from %s", len(chunks), self.source_path.name)

        store = FAISS.from_documents(chunks, self._embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_dir))
        logger.info("Saved FAISS index to %s", self.index_dir)

        self._store = store
        return store

    def query(self, question: str, k: int | None = None) -> list[Document]:
        """Return the top-k most relevant chunks for ``question``."""
        store = self._ensure_loaded()
        return store.similarity_search(question, k=k or self._settings.retrieval_k)
