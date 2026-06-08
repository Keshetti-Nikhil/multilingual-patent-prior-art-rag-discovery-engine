import json

from ppa.llm.llm_client import MockLLMClient
from ppa.runner import run
from ppa.types import DataSpec, Retriever


def test_end_to_end_smoke(tmp_path) -> None:
    spec = DataSpec(n_docs=2500, n_queries=150, n_topics=50, seed=5)
    run(tmp_path / "out", spec, llm=MockLLMClient())
    assert (tmp_path / "out" / "summary.json").exists()
    data = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert len(data["results"]) == len(Retriever)
    for r in data["results"]:
        assert 0.0 <= r["mrr"] <= 1.0
        assert 0.0 <= r["faithfulness"] <= 1.0


def test_rerank_not_worse_than_dense_overall(tmp_path) -> None:
    spec = DataSpec(n_docs=3000, n_queries=200, n_topics=60, seed=9)
    summary = run(tmp_path / "out", spec, llm=MockLLMClient())
    by = {r["retriever"]: r for r in summary["results"]}
    # the rerank stack should land at or above plain hybrid on MRR
    assert by[Retriever.HYBRID_RERANK.value]["mrr"] >= by[Retriever.HYBRID.value]["mrr"] - 0.02
