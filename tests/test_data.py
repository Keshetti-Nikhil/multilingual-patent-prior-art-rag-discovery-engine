import numpy as np
import pytest

from ppa.data.generate import generate
from ppa.types import DataSpec

SPEC = DataSpec(n_docs=2000, n_queries=120, n_topics=40, seed=7)


@pytest.mark.parametrize("seed", [7, 23])
def test_deterministic(seed: int) -> None:
    a_c, a_q = generate(DataSpec(n_docs=1500, n_queries=80, n_topics=30, seed=seed))
    b_c, b_q = generate(DataSpec(n_docs=1500, n_queries=80, n_topics=30, seed=seed))
    assert np.array_equal(a_c.doc_id, b_c.doc_id)
    assert np.allclose(a_c.embeddings, b_c.embeddings)
    assert np.array_equal(a_q.gold_doc, b_q.gold_doc)


def test_shapes() -> None:
    corpus, qs = generate(SPEC)
    assert corpus.tf.shape[0] == SPEC.n_paragraphs
    assert corpus.embeddings.shape == (SPEC.n_paragraphs, SPEC.dim)
    assert qs.gold_doc.size == SPEC.n_queries


def test_embeddings_normalised() -> None:
    corpus, _ = generate(SPEC)
    norms = np.linalg.norm(corpus.embeddings[:500], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_cross_lingual_has_no_token_overlap() -> None:
    corpus, qs = generate(SPEC)
    # build doc->tokens once
    overlaps = []
    for qi in np.where(qs.is_cross_lingual)[0][:20]:
        gold = int(qs.gold_doc[qi])
        pars = np.where(corpus.doc_id == gold)[0]
        doc_tokens: set[int] = set()
        for p in pars:
            doc_tokens.update(corpus.tf.indices[corpus.tf.indptr[p] : corpus.tf.indptr[p + 1]])
        qtok = set(qs.tf.indices[qs.tf.indptr[qi] : qs.tf.indptr[qi + 1]])
        overlaps.append(len(qtok & doc_tokens))
    assert max(overlaps) == 0  # cross-lingual queries share no surface tokens with the gold doc
