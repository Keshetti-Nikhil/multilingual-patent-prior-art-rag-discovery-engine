"""Typer CLI: ``ppa info`` and ``ppa bench``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from ppa.llm.llm_client import LLMClient, MockLLMClient, RealAnthropicClient
from ppa.runner import run
from ppa.types import DataSpec

app = typer.Typer(add_completion=False, help="Multilingual patent prior-art retrieval bench.")
console = Console()


@app.command()
def info() -> None:
    """Print the default data spec."""
    spec = DataSpec()
    console.print(f"[bold]Paragraphs:[/bold] {spec.n_paragraphs:,}  Topics: {spec.n_topics}")
    console.print(f"Queries: {spec.n_queries:,}  Cross-lingual frac: {spec.cross_lingual_frac}")


@app.command()
def bench(
    out_dir: Path = typer.Option(Path("runs/latest")),
    n_docs: int = typer.Option(184_000),
    n_queries: int = typer.Option(900),
    cross_lingual_frac: float = typer.Option(0.45),
    seed: int = typer.Option(20240117),
    real_llm: bool = typer.Option(False, help="Use the real Anthropic judge."),
) -> None:
    """Run the retrieval benchmark and write summary.json + figures."""
    spec = DataSpec(
        n_docs=n_docs, n_queries=n_queries, cross_lingual_frac=cross_lingual_frac, seed=seed
    )
    llm: LLMClient = RealAnthropicClient() if real_llm else MockLLMClient()
    summary = run(out_dir, spec, llm=llm)

    table = Table(title=f"Prior-art retrieval — {spec.n_paragraphs:,} paragraphs")
    for col in ("retriever", "MRR", "NDCG@10", "Hit@1", "Recall@50", "mono", "x-ling", "faith"):
        table.add_column(col)
    results = cast("list[dict[str, Any]]", summary["results"])
    for r in results:
        table.add_row(
            r["retriever"],
            f"{r['mrr']:.3f}",
            f"{r['ndcg_at_10']:.3f}",
            f"{r['hit_at_1']:.3f}",
            f"{r['recall_at_k']['50']:.3f}"
            if "50" in r["recall_at_k"]
            else f"{r['recall_at_k'][50]:.3f}",
            f"{r['mrr_monolingual']:.3f}",
            f"{r['mrr_cross_lingual']:.3f}",
            f"{r['faithfulness']:.3f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
