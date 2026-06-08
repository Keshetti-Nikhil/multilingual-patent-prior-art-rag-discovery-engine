"""Typed specs and results for the prior-art retrieval bench."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Retriever(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class Language(StrEnum):
    EN = "en"
    DE = "de"
    JA = "ja"
    ZH = "zh"
    FR = "fr"


class DataSpec(BaseModel):
    """Generator spec for a synthetic multilingual patent corpus."""

    n_docs: int = Field(default=184_000, ge=100)
    chunks_per_doc: int = Field(default=5, ge=1, le=12)
    n_topics: int = Field(default=240, ge=2)
    n_queries: int = Field(default=900, ge=10)
    vocab_size: int = Field(default=12_000, ge=200)
    sig_vocab_size: int = Field(default=20_000, ge=200)
    dim: int = Field(default=48, ge=8, le=256)
    tokens_per_chunk: int = Field(default=22, ge=6, le=128)
    cross_lingual_frac: float = Field(default=0.45, ge=0.0, le=1.0)
    index_noise: float = Field(default=0.72, ge=0.0)
    seed: int = Field(default=20240117)

    @property
    def n_paragraphs(self) -> int:
        return self.n_docs * self.chunks_per_doc

    @property
    def n_langs(self) -> int:
        return len(Language)


class RetrievalResult(BaseModel):
    """Metrics for one retriever, sliced by query type."""

    retriever: Retriever
    recall_at_k: dict[int, float]
    ndcg_at_10: float
    mrr: float
    hit_at_1: float
    context_precision: float
    faithfulness: float
    mrr_monolingual: float
    mrr_cross_lingual: float
    latency_ms_per_query: float
    n_queries: int
