"""Six chart builders. Self-contained palette so the repo needs no external deps.

Palette "Ledger Slate" — editorial, used only by this project.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from ppa.data.generate import QuerySet
from ppa.types import RetrievalResult

_COLORS = ("#0F3460", "#16C79A", "#FFB400", "#E94560", "#533483", "#A8DADC")
_BG, _GRID, _ACCENT = "#F4F1E8", "#D7CEB2", "#E94560"


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": _BG,
            "axes.facecolor": _BG,
            "savefig.facecolor": _BG,
            "axes.edgecolor": _GRID,
            "axes.grid": True,
            "grid.color": _GRID,
            "grid.alpha": 0.55,
            "axes.prop_cycle": cycler(color=list(_COLORS)),
            "font.size": 11,
            "axes.titlesize": 13,
        }
    )


def render_all(results: list[RetrievalResult], qs: QuerySet, out: Path) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    names = [r.retriever.value for r in results]

    # 1. recall@k curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = sorted(next(iter(results)).recall_at_k)
    for r in results:
        ax.plot(ks, [r.recall_at_k[k] for k in ks], marker="o", label=r.retriever.value)
    ax.set(xlabel="k", ylabel="Recall@k", title="Recall@k by retriever", xscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart_1.png", dpi=130)
    plt.close(fig)

    # 2. NDCG@10 bars
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, [r.ndcg_at_10 for r in results], color=_COLORS[: len(results)])
    ax.set(ylabel="NDCG@10", title="Ranking quality (NDCG@10)")
    fig.tight_layout()
    fig.savefig(out / "chart_2.png", dpi=130)
    plt.close(fig)

    # 3. MRR split: monolingual vs cross-lingual
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(results))
    ax.bar(
        x - 0.2, [r.mrr_monolingual for r in results], 0.4, label="monolingual", color=_COLORS[0]
    )
    ax.bar(
        x + 0.2, [r.mrr_cross_lingual for r in results], 0.4, label="cross-lingual", color=_ACCENT
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set(ylabel="MRR", title="MRR by query type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart_3.png", dpi=130)
    plt.close(fig)

    # 4. latency vs MRR scatter
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(results):
        ax.scatter(r.latency_ms_per_query, r.mrr, s=160, color=_COLORS[i], label=r.retriever.value)
        ax.annotate(r.retriever.value, (r.latency_ms_per_query, r.mrr), fontsize=8)
    ax.set(xlabel="latency (ms/query)", ylabel="MRR", title="Quality vs latency")
    fig.tight_layout()
    fig.savefig(out / "chart_4.png", dpi=130)
    plt.close(fig)

    # 5. judge faithfulness bars
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, [r.faithfulness for r in results], color=_COLORS[: len(results)])
    ax.set(ylabel="faithfulness", title="LLM-judge faithfulness of top-1 context")
    fig.tight_layout()
    fig.savefig(out / "chart_5.png", dpi=130)
    plt.close(fig)

    # 6. query-type distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_x = int(qs.is_cross_lingual.sum())
    n_m = int(qs.gold_doc.size - n_x)
    ax.pie(
        [n_m, n_x],
        labels=["monolingual", "cross-lingual"],
        autopct="%1.0f%%",
        colors=[_COLORS[0], _ACCENT],
    )
    ax.set_title("Evaluation query mix")
    fig.tight_layout()
    fig.savefig(out / "chart_6.png", dpi=130)
    plt.close(fig)
