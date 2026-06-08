"""End-to-end orchestration: generate, retrieve, score, chart, summarise."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ppa.bench import metrics as M
from ppa.data.generate import QuerySet, generate
from ppa.llm.llm_client import LLMClient, MockLLMClient
from ppa.models.retrievers import RetrieverEngine
from ppa.types import DataSpec, RetrievalResult, Retriever

_KS = (1, 5, 10, 20, 50)


def _doc_tokens(engine: RetrieverEngine, doc: int) -> list[int]:
    pars = engine._doc_par[doc]
    tf = engine.corpus.tf
    toks: set[int] = set()
    for p in pars[:3].tolist():
        toks.update(tf.indices[tf.indptr[p] : tf.indptr[p + 1]].tolist())
    return sorted(toks)


def evaluate(
    engine: RetrieverEngine, qs: QuerySet, retriever: Retriever, llm: LLMClient
) -> RetrievalResult:
    nq = qs.gold_doc.size
    recall = {k: 0.0 for k in _KS}
    ndcg = mrr = hit1 = cprec = faith = 0.0
    mrr_mono = mrr_x = 0.0
    n_mono = int((~qs.is_cross_lingual).sum())
    n_x = int(qs.is_cross_lingual.sum())

    t0 = time.perf_counter()
    for qi in range(nq):
        ranked = engine.rank(retriever, qs, qi)
        gold = int(qs.gold_doc[qi])
        for k in _KS:
            recall[k] += M.recall_at_k(ranked, gold, k)
        ndcg += M.ndcg_at_k(ranked, gold, 10)
        rr = M.reciprocal_rank(ranked, gold)
        mrr += rr
        hit1 += M.hit_at_k(ranked, gold, 1)
        cprec += M.context_precision_at_k(ranked, gold, 10)
        if qs.is_cross_lingual[qi]:
            mrr_x += rr
        else:
            mrr_mono += rr
        top_doc = int(ranked[0]) if ranked.size else gold
        faith += llm.judge_faithfulness(qs.expected_tokens[qi], _doc_tokens(engine, top_doc))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return RetrievalResult(
        retriever=retriever,
        recall_at_k={k: recall[k] / nq for k in _KS},
        ndcg_at_10=ndcg / nq,
        mrr=mrr / nq,
        hit_at_1=hit1 / nq,
        context_precision=cprec / nq,
        faithfulness=faith / nq,
        mrr_monolingual=mrr_mono / max(1, n_mono),
        mrr_cross_lingual=mrr_x / max(1, n_x),
        latency_ms_per_query=elapsed_ms / nq,
        n_queries=nq,
    )


def run(out_dir: Path, spec: DataSpec, llm: LLMClient | None = None) -> dict[str, object]:
    llm = llm or MockLLMClient()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, qs = generate(spec)
    engine = RetrieverEngine(corpus)

    results = [evaluate(engine, qs, r, llm) for r in Retriever]
    summary = {
        "spec": spec.model_dump(),
        "n_paragraphs": spec.n_paragraphs,
        "n_cross_lingual_queries": int(qs.is_cross_lingual.sum()),
        "results": [r.model_dump() for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    fig_dir = Path("results/figures")
    try:
        from ppa.viz.charts import render_all

        render_all(results, qs, fig_dir)
    except Exception as exc:  # charts are best-effort; never fail the bench on them
        summary["chart_error"] = str(exc)
    return summary
