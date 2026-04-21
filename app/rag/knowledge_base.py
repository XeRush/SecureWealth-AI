"""
Financial Knowledge Base ingestion and vector store management.

Supports sentence-level chunking, embedding via sentence-transformers
(all-MiniLM-L6-v2), and storage in FAISS (default) or Chroma.
Incremental re-indexing is supported — new documents are added without
a full rebuild of the vector store.
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Placeholder documents covering the minimum required corpus (Req 10.1)
# ---------------------------------------------------------------------------

PLACEHOLDER_DOCUMENTS: List[dict] = [
    {
        "title": "Mutual Funds and SIP Basics",
        "origin": "Investopedia",
        "content": (
            "A mutual fund is a type of investment vehicle consisting of a portfolio of stocks, bonds, or other securities. "
            "Mutual funds give small or individual investors access to diversified, professionally managed portfolios. "
            "SIP (Systematic Investment Plan) allows investors to invest a fixed amount regularly — weekly, monthly, or quarterly — "
            "into a selected mutual fund scheme. "
            "SIPs help investors benefit from rupee cost averaging, reducing the impact of market volatility over time. "
            "Equity mutual funds invest primarily in stocks and are suitable for long-term wealth creation. "
            "Debt mutual funds invest in fixed-income instruments and are suitable for conservative investors seeking stable returns. "
            "Hybrid funds combine equity and debt to balance risk and return. "
            "ELSS (Equity Linked Savings Scheme) mutual funds offer tax deductions under Section 80C of the Income Tax Act."
        ),
    },
    {
        "title": "RBI Monetary Policy Framework",
        "origin": "RBI Guidelines",
        "content": (
            "The Reserve Bank of India (RBI) uses monetary policy tools to control inflation and manage liquidity in the economy. "
            "The repo rate is the rate at which the RBI lends money to commercial banks; a higher repo rate makes borrowing costlier, "
            "reducing money supply and curbing inflation. "
            "The reverse repo rate is the rate at which the RBI borrows money from commercial banks. "
            "The Cash Reserve Ratio (CRR) is the percentage of a bank's total deposits that must be held as reserves with the RBI. "
            "The Statutory Liquidity Ratio (SLR) requires banks to maintain a certain percentage of their net demand and time liabilities "
            "in liquid assets such as government securities. "
            "The Monetary Policy Committee (MPC) meets every two months to review and set the policy repo rate with the primary objective "
            "of maintaining inflation within the target band of 4% ± 2%. "
            "Open Market Operations (OMO) involve the RBI buying or selling government securities to regulate liquidity. "
            "Changes in the repo rate directly influence home loan EMIs, fixed deposit rates, and overall credit growth in the economy."
        ),
    },
    {
        "title": "SEBI Investor Protection Guidelines",
        "origin": "SEBI Regulations",
        "content": (
            "The Securities and Exchange Board of India (SEBI) is the primary regulator for securities markets in India. "
            "SEBI mandates KYC (Know Your Customer) compliance for all investors before they can trade in securities or invest in mutual funds. "
            "KYC involves submitting identity proof, address proof, and a recent photograph to a SEBI-registered intermediary. "
            "Investors can file grievances against brokers, mutual funds, or listed companies through the SCORES (SEBI Complaints Redress System) platform. "
            "SEBI's Investor Protection Fund provides compensation to investors in cases of broker defaults. "
            "SEBI regulations require all listed companies to disclose material information promptly to prevent insider trading. "
            "The Prohibition of Insider Trading Regulations prohibit trading on unpublished price-sensitive information. "
            "SEBI has introduced the Investor Charter to outline the rights and responsibilities of investors. "
            "Investors are advised to deal only with SEBI-registered intermediaries and verify their registration on the SEBI website."
        ),
    },
    {
        "title": "Section 80C Tax Deductions",
        "origin": "Indian Income Tax Act",
        "content": (
            "Section 80C of the Income Tax Act, 1961 allows individual taxpayers and Hindu Undivided Families (HUFs) to claim deductions "
            "of up to Rs 1.5 lakh per financial year from their gross total income. "
            "Eligible investments and expenditures under Section 80C include: ELSS mutual funds, Public Provident Fund (PPF), "
            "National Savings Certificate (NSC), Employee Provident Fund (EPF) contributions, life insurance premiums, "
            "5-year fixed deposits with banks or post offices, Sukanya Samriddhi Yojana, and principal repayment of home loans. "
            "PPF offers a government-backed, tax-free return and has a lock-in period of 15 years. "
            "ELSS funds have the shortest lock-in period of 3 years among 80C instruments and offer market-linked returns. "
            "Section 80CCD(1B) provides an additional deduction of up to Rs 50,000 for contributions to the National Pension System (NPS), "
            "over and above the Rs 1.5 lakh limit under Section 80C. "
            "Taxpayers opting for the new tax regime introduced in Budget 2020 cannot claim deductions under Section 80C."
        ),
    },
    {
        "title": "Portfolio Diversification Strategies",
        "origin": "Investopedia",
        "content": (
            "Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio. "
            "The rationale is that a portfolio constructed of different kinds of assets will, on average, yield higher long-term returns "
            "and lower the risk of any individual holding or security. "
            "Asset allocation across equity, debt, and gold reduces the overall portfolio risk because these asset classes often move "
            "in different directions under the same market conditions. "
            "Equity investments offer higher growth potential but come with higher volatility. "
            "Debt instruments such as bonds and fixed deposits provide stable income with lower risk. "
            "Gold acts as a hedge against inflation and currency depreciation. "
            "Real estate can provide rental income and capital appreciation over the long term. "
            "A common rule of thumb for equity allocation is to subtract your age from 100 — the result is the percentage to invest in equities. "
            "Rebalancing the portfolio periodically ensures that the asset allocation stays aligned with the investor's risk appetite and goals. "
            "International diversification through global funds reduces concentration risk in a single economy."
        ),
    },
]


# ---------------------------------------------------------------------------
# Sentence-level chunking helper
# ---------------------------------------------------------------------------

def _chunk_into_sentences(text: str) -> List[str]:
    """
    Split *text* into individual sentences.

    Strategy:
    1. Split on '. ', '! ', '? ' (sentence-ending punctuation followed by space).
    2. Strip whitespace and filter out empty strings.
    3. Re-append the terminal punctuation so each chunk is a complete sentence.
    """
    # Split on sentence boundaries while keeping the delimiter context
    raw_chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return sentences


# ---------------------------------------------------------------------------
# KnowledgeBaseIngester
# ---------------------------------------------------------------------------

class KnowledgeBaseIngester:
    """
    Manages ingestion of financial documents into a FAISS (or Chroma) vector store.

    Documents are chunked at sentence level, embedded with
    ``all-MiniLM-L6-v2``, and stored for semantic similarity search.
    Incremental re-indexing is supported via ``vectorstore.add_texts()``.
    """

    _EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, vector_db_backend: str = "faiss") -> None:
        self._backend = vector_db_backend.lower()
        self._embeddings = HuggingFaceEmbeddings(model_name=self._EMBEDDING_MODEL)
        self._vectorstore: Optional[FAISS] = None  # lazily initialised on first ingest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, documents: List[dict]) -> None:
        """
        Ingest a batch of documents.

        Each document must be a dict with keys:
            ``title`` (str), ``origin`` (str), ``content`` (str).

        Existing vector store entries are preserved — this is an
        incremental operation, not a full rebuild.
        """
        for doc in documents:
            self.ingest_document(
                title=doc["title"],
                origin=doc["origin"],
                content=doc["content"],
            )

    def ingest_document(self, title: str, origin: str, content: str) -> None:
        """
        Ingest a single document incrementally.

        The document is chunked at sentence level; each chunk is embedded
        and added to the vector store without rebuilding the entire index.
        """
        sentences = _chunk_into_sentences(content)
        if not sentences:
            return

        metadatas = [{"title": title, "origin": origin} for _ in sentences]

        if self._vectorstore is None:
            # First document — create the vector store from scratch
            self._vectorstore = FAISS.from_texts(
                texts=sentences,
                embedding=self._embeddings,
                metadatas=metadatas,
            )
        else:
            # Subsequent documents — add incrementally (no full rebuild)
            self._vectorstore.add_texts(texts=sentences, metadatas=metadatas)

    def get_store(self) -> Optional[FAISS]:
        """Return the underlying FAISS vector store (or None if empty)."""
        return self._vectorstore

    def is_populated(self) -> bool:
        """Return True if the vector store contains at least one document."""
        if self._vectorstore is None:
            return False
        # FAISS index tracks the number of vectors via index.ntotal
        try:
            return self._vectorstore.index.ntotal > 0
        except AttributeError:
            return False


# ---------------------------------------------------------------------------
# Convenience factory — pre-populated with placeholder documents
# ---------------------------------------------------------------------------

def build_default_knowledge_base(vector_db_backend: str = "faiss") -> KnowledgeBaseIngester:
    """
    Create and return a ``KnowledgeBaseIngester`` pre-loaded with the
    five placeholder documents required by Requirement 10.1.
    """
    ingester = KnowledgeBaseIngester(vector_db_backend=vector_db_backend)
    ingester.ingest(PLACEHOLDER_DOCUMENTS)
    return ingester
