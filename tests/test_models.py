import numpy as np

from ppa.data.generate import generate
from ppa.models.retrievers import RetrieverEngine
from ppa.types import DataSpec, Retriever

SPEC = DataSpec(n_docs=2500, n_queries=150, n_topics=50, seed=11)


def test_rank_returns_doc_ids() -> None:
    corpus, qs = generate(SPEC)
    engine = RetrieverEngine(corpus)
    ranked = engine.rank(Retriever.HYBRID, qs, 0)
    assert ranked.ndim == 1
    assert ranked.max() < engine.n_docs


def test_dense_bridges_cross_lingual() -> None:
    corpus, qs = generate(SPEC)
    engine = RetrieverEngine(corpus)
    x_idx = np.where(qs.is_cross_lingual)[0][:40]
    bm = np.mean([engine.rank(Retriever.BM25, qs, i)[0] == qs.gold_doc[i] for i in x_idx])
    de = np.mean([engine.rank(Retriever.DENSE, qs, i)[0] == qs.gold_doc[i] for i in x_idx])
    # dense should pinpoint cross-lingual gold docs far better than lexical BM25
    assert de > bm


def test_all_retrievers_run() -> None:
    corpus, qs = generate(SPEC)
    engine = RetrieverEngine(corpus)
    for r in Retriever:
        ranked = engine.rank(r, qs, 3)
        assert ranked.size > 0
