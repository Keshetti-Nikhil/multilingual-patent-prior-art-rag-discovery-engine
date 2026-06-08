import numpy as np

from ppa.bench import metrics as M


def test_reciprocal_rank_top() -> None:
    ranked = np.array([5, 2, 9])
    assert M.reciprocal_rank(ranked, 5) == 1.0
    assert M.reciprocal_rank(ranked, 9) == 1.0 / 3.0
    assert M.reciprocal_rank(ranked, 99) == 0.0


def test_ndcg_positions() -> None:
    ranked = np.array([1, 2, 3, 4])
    assert M.ndcg_at_k(ranked, 1, 10) == 1.0
    assert M.ndcg_at_k(ranked, 4, 10) < M.ndcg_at_k(ranked, 1, 10)
    assert M.ndcg_at_k(ranked, 99, 10) == 0.0


def test_hit_and_recall() -> None:
    ranked = np.array([7, 3, 1])
    assert M.hit_at_k(ranked, 3, 5) == 1.0
    assert M.hit_at_k(ranked, 3, 1) == 0.0
    assert M.recall_at_k(ranked, 7, 1) == 1.0


def test_context_precision() -> None:
    ranked = np.array([7, 3, 1])
    assert M.context_precision_at_k(ranked, 7, 10) == 0.1
    assert M.context_precision_at_k(ranked, 99, 10) == 0.0
