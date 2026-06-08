"""Deterministic generator for a synthetic multilingual patent corpus.

The corpus is built so the four retrievers separate cleanly and honestly:

* Each *document* sits in one topic and one language. Its paragraphs draw
  tokens from a topic-and-language salient set plus a per-document *signature*
  set. Two documents in the same topic share topic tokens but not signatures.
* Lexical signal (BM25) only crosses languages when the query is monolingual:
  signature and salient vocabularies are partitioned per language, so a query
  written in a different language than its gold document has ~zero token
  overlap with it.
* Semantic signal (dense) is language-agnostic: a paragraph's embedding is
  ``topic_center + doc_offset + noise`` and the shared ``doc_offset`` lets dense
  retrieval pinpoint the gold document regardless of query language.

That asymmetry is the whole point of the bench: BM25 is sharp but monolingual,
dense is blurry but multilingual, and the fusion/rerank stack tries to keep
both strengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ppa.types import DataSpec


@dataclass
class Corpus:
    tf: csr_matrix  # [n_paragraphs, vocab] term frequencies
    embeddings: np.ndarray  # [n_paragraphs, dim] float32, L2-normalised
    clean_embeddings: np.ndarray  # [n_paragraphs, dim] noise-free (reranker view)
    doc_id: np.ndarray  # [n_paragraphs] int32
    lang: np.ndarray  # [n_paragraphs] int8, per paragraph (== its doc language)
    doc_lengths: np.ndarray  # [n_paragraphs] float64


@dataclass
class QuerySet:
    tf: csr_matrix  # [n_queries, vocab]
    embeddings: np.ndarray  # [n_queries, dim]
    gold_doc: np.ndarray  # [n_queries] int32
    is_cross_lingual: np.ndarray  # [n_queries] bool
    expected_tokens: list[list[int]]  # signature tokens per query (judge anchor)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    out: np.ndarray = (x / n).astype(np.float32)
    return out


def generate(spec: DataSpec) -> tuple[Corpus, QuerySet]:
    rng = np.random.default_rng(spec.seed)
    n_docs = spec.n_docs
    n_par = spec.n_paragraphs
    cpd = spec.chunks_per_doc
    V = spec.vocab_size
    SV = spec.sig_vocab_size
    L = spec.n_langs

    # --- topic + language assignment per document ---
    doc_topic = rng.integers(0, spec.n_topics, size=n_docs).astype(np.int32)
    doc_lang = rng.integers(0, L, size=n_docs).astype(np.int8)

    # vocab partitioned by language: [lang_block_start ... ] of width V//L
    block = V // L
    sig_block = SV // L

    # per (topic, lang) salient token offsets within that language's block
    salient_per = 28
    topic_salient = rng.integers(0, block, size=(spec.n_topics, L, salient_per)).astype(np.int32)
    # per-document signature offsets within its language's signature block
    sig_per = 4
    doc_sig = rng.integers(0, sig_block, size=(n_docs, sig_per)).astype(np.int32)

    # --- embeddings: topic centers + per-doc offsets (language-agnostic) ---
    topic_center = rng.standard_normal((spec.n_topics, spec.dim)).astype(np.float32)
    doc_offset = (0.55 * rng.standard_normal((n_docs, spec.dim))).astype(np.float32)
    base = topic_center[doc_topic] + doc_offset  # [n_docs, dim]

    # --- build paragraph-level token frequencies as a CSR matrix ---
    par_doc = np.repeat(np.arange(n_docs, dtype=np.int32), cpd)
    par_lang = doc_lang[par_doc]
    tpc = spec.tokens_per_chunk

    # sample tokens vectorised per paragraph: 70% salient, 20% signature, 10% bg
    u = rng.random((n_par, tpc))
    choice = np.where(u < 0.7, 0, np.where(u < 0.9, 1, 2))  # 0 salient,1 sig,2 bg
    par_topic = doc_topic[par_doc]
    lang_base = par_lang.astype(np.int64) * block
    sig_lang_base = V + par_lang.astype(np.int64) * sig_block  # signatures live above main vocab

    sal_idx = rng.integers(0, salient_per, size=(n_par, tpc))
    sig_idx = rng.integers(0, sig_per, size=(n_par, tpc))
    bg_idx = rng.integers(0, block, size=(n_par, tpc))

    sal_tok = (
        lang_base[:, None] + topic_salient[par_topic, par_lang][np.arange(n_par)[:, None], sal_idx]
    )
    sig_tok = sig_lang_base[:, None] + doc_sig[par_doc][np.arange(n_par)[:, None], sig_idx]
    bg_tok = lang_base[:, None] + bg_idx
    tokens = np.where(choice == 0, sal_tok, np.where(choice == 1, sig_tok, bg_tok))

    rows = np.repeat(np.arange(n_par, dtype=np.int32), tpc)
    cols = tokens.reshape(-1).astype(np.int32)
    data = np.ones(rows.size, dtype=np.float32)
    full_vocab = V + SV
    tf = csr_matrix((data, (rows, cols)), shape=(n_par, full_vocab))
    tf.sum_duplicates()
    doc_lengths = np.asarray(tf.sum(axis=1)).ravel().astype(np.float64)

    par_base = base[par_doc]
    noise = (spec.index_noise * rng.standard_normal((n_par, spec.dim))).astype(np.float32)
    embeddings = _l2(par_base + noise)
    clean_embeddings = _l2(
        par_base + 0.22 * rng.standard_normal((n_par, spec.dim)).astype(np.float32)
    )

    corpus = Corpus(
        tf=tf.tocsr(),
        embeddings=embeddings,
        clean_embeddings=clean_embeddings,
        doc_id=par_doc,
        lang=par_lang,
        doc_lengths=doc_lengths,
    )

    # --- queries: pick a gold doc, optionally cross-lingual ---
    nq = spec.n_queries
    gold_doc = rng.integers(0, n_docs, size=nq).astype(np.int32)
    is_x = rng.random(nq) < spec.cross_lingual_frac
    gold_lang = doc_lang[gold_doc]
    # cross-lingual query gets a *different* language than its gold doc
    shift = rng.integers(1, L, size=nq).astype(np.int8)
    q_lang = np.where(is_x, (gold_lang + shift) % L, gold_lang).astype(np.int8)

    q_topic = doc_topic[gold_doc]
    qtok = max(6, tpc // 3)
    qu = rng.random((nq, qtok))
    qchoice = np.where(qu < 0.5, 0, 1)  # 0 salient, 1 signature
    q_lang_base = q_lang.astype(np.int64) * block
    q_sig_base = V + q_lang.astype(np.int64) * sig_block
    qsal = (
        q_lang_base[:, None]
        + topic_salient[q_topic, q_lang][
            np.arange(nq)[:, None], rng.integers(0, salient_per, size=(nq, qtok))
        ]
    )
    qsig = (
        q_sig_base[:, None]
        + doc_sig[gold_doc][np.arange(nq)[:, None], rng.integers(0, sig_per, size=(nq, qtok))]
    )
    qtokens = np.where(qchoice == 0, qsal, qsig)
    qrows = np.repeat(np.arange(nq, dtype=np.int32), qtok)
    qcols = qtokens.reshape(-1).astype(np.int32)
    qtf = csr_matrix(
        (np.ones(qrows.size, dtype=np.float32), (qrows, qcols)), shape=(nq, full_vocab)
    )
    qtf.sum_duplicates()

    q_noise = (0.30 * rng.standard_normal((nq, spec.dim))).astype(np.float32)
    q_emb = _l2(base[gold_doc] + q_noise)

    expected_tokens = [sorted(set(qcols[qrows == i].tolist())) for i in range(min(nq, 4096))]
    while len(expected_tokens) < nq:
        expected_tokens.append([])

    qs = QuerySet(
        tf=qtf.tocsr(),
        embeddings=q_emb,
        gold_doc=gold_doc,
        is_cross_lingual=is_x,
        expected_tokens=expected_tokens,
    )
    return corpus, qs
