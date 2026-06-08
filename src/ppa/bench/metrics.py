"""Pure retrieval metric functions, all operating on ranked document ids."""

from __future__ import annotations

import numpy as np


def hit_at_k(ranked: np.ndarray, gold: int, k: int) -> float:
    return 1.0 if gold in ranked[:k].tolist() else 0.0


def recall_at_k(ranked: np.ndarray, gold: int, k: int) -> float:
    # single relevant document, so recall@k == hit@k here
    return hit_at_k(ranked, gold, k)


def reciprocal_rank(ranked: np.ndarray, gold: int) -> float:
    hits = np.where(ranked == gold)[0]
    return 1.0 / (hits[0] + 1.0) if hits.size else 0.0


def ndcg_at_k(ranked: np.ndarray, gold: int, k: int) -> float:
    hits = np.where(ranked[:k] == gold)[0]
    if not hits.size:
        return 0.0
    return float(1.0 / np.log2(hits[0] + 2.0))  # ideal DCG == 1 for a single relevant doc


def context_precision_at_k(ranked: np.ndarray, gold: int, k: int) -> float:
    """Fraction of the top-k that is relevant (1/k if the gold doc is present)."""
    return (1.0 / k) if gold in ranked[:k].tolist() else 0.0
