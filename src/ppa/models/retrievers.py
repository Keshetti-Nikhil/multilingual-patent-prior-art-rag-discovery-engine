"""The four retrievers: BM25, dense, hybrid (RRF), and hybrid + rerank.

All retrievers return, per query, a ranked array of *document* ids. Relevance
in this bench is at the document level (the gold prior-art document), so
paragraph scores are aggregated to their parent document by taking the best
paragraph score.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from ppa.data.generate import Corpus, QuerySet
from ppa.types import Retriever

_K1 = 1.5
_B = 0.75
_TOPN = 100  # ranked docs returned per query


def _doc_index(doc_id: np.ndarray, n_docs: int) -> list[np.ndarray]:
    order = np.argsort(doc_id, kind="stable")
    sorted_doc = doc_id[order]
    bounds = np.searchsorted(sorted_doc, np.arange(n_docs + 1))
    return [order[bounds[d] : bounds[d + 1]] for d in range(n_docs)]


def _rank_docs(par_scores: np.ndarray, doc_id: np.ndarray, topn: int) -> np.ndarray:
    k = min(par_scores.size, max(topn * 6, 400))
    cand = np.argpartition(-par_scores, k - 1)[:k]
    cand = cand[np.argsort(-par_scores[cand])]
    cdocs = doc_id[cand]
    seen: dict[int, None] = {}
    out: list[int] = []
    for d in cdocs.tolist():
        if d not in seen:
            seen[d] = None
            out.append(d)
            if len(out) >= topn:
                break
    return np.asarray(out, dtype=np.int32)


class RetrieverEngine:
    """Holds the precomputed BM25 weight matrix and exposes the four rankers."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        tf = corpus.tf
        n_par, vocab = tf.shape
        self.n_docs = int(corpus.doc_id.max()) + 1
        # BM25: precompute the length-weighted, idf-scaled term weights.
        df = np.diff(tf.tocsc().indptr).astype(np.float64)
        idf = np.log((n_par - df + 0.5) / (df + 0.5) + 1.0)
        coo = tf.tocoo()
        dl = corpus.doc_lengths
        avgdl = dl.mean()
        denom = coo.data + _K1 * (1 - _B + _B * dl[coo.row] / avgdl)
        wdata = idf[coo.col] * (coo.data * (_K1 + 1)) / denom
        self._w = csc_matrix((wdata, (coo.row, coo.col)), shape=(n_par, vocab))
        self._doc_par = _doc_index(corpus.doc_id, self.n_docs)

    def _bm25_par_scores(self, qrow: np.ndarray) -> np.ndarray:
        terms = qrow
        if terms.size == 0:
            return np.zeros(self._w.shape[0], dtype=np.float64)
        return np.asarray(self._w[:, terms].sum(axis=1)).ravel()

    def _dense_par_scores(self, qemb: np.ndarray, clean: bool = False) -> np.ndarray:
        emb = self.corpus.clean_embeddings if clean else self.corpus.embeddings
        out: np.ndarray = emb @ qemb
        return out

    def rank(self, retriever: Retriever, qs: QuerySet, qi: int) -> np.ndarray:
        qterms = qs.tf.indices[qs.tf.indptr[qi] : qs.tf.indptr[qi + 1]]
        qemb = qs.embeddings[qi]
        if retriever is Retriever.BM25:
            return _rank_docs(self._bm25_par_scores(qterms), self.corpus.doc_id, _TOPN)
        if retriever is Retriever.DENSE:
            return _rank_docs(self._dense_par_scores(qemb), self.corpus.doc_id, _TOPN)

        bm = _rank_docs(self._bm25_par_scores(qterms), self.corpus.doc_id, _TOPN)
        de = _rank_docs(self._dense_par_scores(qemb), self.corpus.doc_id, _TOPN)
        fused = _rrf(bm, de, _TOPN)
        if retriever is Retriever.HYBRID:
            return fused
        # rerank: rescore top fused docs with the higher-fidelity (clean) encoder
        head = fused[:50]
        clean = self.corpus.clean_embeddings
        rer = np.array([clean[self._doc_par[d]] @ qemb for d in head], dtype=object)
        best = np.array([float(s.max()) if s.size else -1e9 for s in rer])
        reordered = head[np.argsort(-best)]
        return np.concatenate([reordered, fused[50:]])[:_TOPN]


def _rrf(a: np.ndarray, b: np.ndarray, topn: int, k: int = 60) -> np.ndarray:
    score: dict[int, float] = {}
    for rank, d in enumerate(a.tolist()):
        score[d] = score.get(d, 0.0) + 1.0 / (k + rank + 1)
    for rank, d in enumerate(b.tolist()):
        score[d] = score.get(d, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(score, key=lambda d: -score[d])
    return np.asarray(ordered[:topn], dtype=np.int32)
