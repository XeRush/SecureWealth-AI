"""
RAG Pipeline for SecureWealth Twin AI.

Implements retrieval-augmented generation using FAISS vector store and an
optional LLM (OpenAI) for explanation generation only — never for decisions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS

from app.models.response import DocumentSource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FALLBACK_MESSAGE = (
    "No relevant knowledge was found for your query. "
    "Please consult a certified financial advisor."
)

_SIMILARITY_THRESHOLD = 0.5  # minimum similarity score to include a document


# ---------------------------------------------------------------------------
# SourceTracker
# ---------------------------------------------------------------------------

class SourceTracker:
    """Collects DocumentSource objects for the `sources` response array."""

    def __init__(self) -> None:
        self._sources: List[DocumentSource] = []

    def add(self, source: DocumentSource) -> None:
        self._sources.append(source)

    def get_sources(self) -> List[DocumentSource]:
        return list(self._sources)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Accepts a KnowledgeBaseIngester (or its FAISS store) and provides:
    - retrieve(query, top_k) — semantic search with similarity filtering
    - generate_answer(query, context_docs, retrieved_texts) — LLM explanation
    - answer(query) — combined retrieve + generate
    - ping() — health check
    """

    def __init__(self, vectorstore: Optional[FAISS] = None) -> None:
        """
        Args:
            vectorstore: A populated FAISS vectorstore. If None, the pipeline
                         will always return the fallback message.
        """
        self._vectorstore = vectorstore
        self._llm = self._build_llm()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llm():
        """Build LLM if OpenAI API key is configured; otherwise return None."""
        try:
            from app.config import settings
            from langchain_openai import ChatOpenAI

            if settings.openai_api_key:
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.2,
                    api_key=settings.openai_api_key,
                )
        except Exception:
            pass
        return None

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert FAISS L2 distance to a similarity score in [0, 1]."""
        return 1.0 / (1.0 + distance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[DocumentSource], List[str]]:
        """
        Retrieve the most relevant documents for *query*.

        Returns:
            A tuple of (List[DocumentSource], List[str]) where the second
            element contains the raw text of each retrieved chunk (used by
            generate_answer).  Documents with similarity < 0.5 are filtered out.
        """
        if self._vectorstore is None:
            return [], []

        try:
            raw_results = self._vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception:
            return [], []

        sources: List[DocumentSource] = []
        texts: List[str] = []

        for doc, distance in raw_results:
            similarity = self._distance_to_similarity(distance)
            if similarity < _SIMILARITY_THRESHOLD:
                continue
            meta = doc.metadata or {}
            sources.append(
                DocumentSource(
                    title=meta.get("title", "Unknown"),
                    origin=meta.get("origin", "Unknown"),
                    similarity_score=round(similarity, 4),
                )
            )
            texts.append(doc.page_content)

        return sources, texts

    def generate_answer(
        self,
        query: str,
        context_docs: List[DocumentSource],
        retrieved_texts: List[str],
    ) -> str:
        """
        Generate a natural-language answer grounded in *retrieved_texts*.

        The LLM is used ONLY for explanation — never for financial decisions.
        If no context documents are available, returns the fallback message.
        If no LLM is configured, returns a simple concatenation of retrieved texts.
        """
        if not context_docs:
            return FALLBACK_MESSAGE

        if self._llm is None:
            # Graceful degradation: concatenate retrieved chunks
            return " ".join(retrieved_texts)

        context_block = "\n\n".join(
            f"[{doc.title} — {doc.origin}]\n{text}"
            for doc, text in zip(context_docs, retrieved_texts)
        )
        prompt = (
            "You are a helpful financial knowledge assistant. "
            "Using ONLY the context below, answer the user's question. "
            "Do not make up information not present in the context.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        try:
            response = self._llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            # LLM call failed — fall back to concatenated texts
            return " ".join(retrieved_texts)

    def answer(self, query: str) -> dict:
        """
        Full RAG pipeline: retrieve then generate.

        Returns:
            {"answer": str, "sources": List[DocumentSource]}
        """
        tracker = SourceTracker()
        sources, texts = self.retrieve(query)

        for source in sources:
            tracker.add(source)

        answer_text = self.generate_answer(query, sources, texts)

        return {
            "answer": answer_text,
            "sources": tracker.get_sources(),
        }

    def ping(self) -> bool:
        """Return True if the vector store is populated and queryable."""
        if self._vectorstore is None:
            return False
        try:
            return self._vectorstore.index.ntotal > 0
        except AttributeError:
            return False
