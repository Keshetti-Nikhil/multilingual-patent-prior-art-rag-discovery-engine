"""LLM client protocol with an offline mock and a real Anthropic implementation.

The mock judge blends a deterministic hash with a lexical grounding anchor so
faithfulness scores form a meaningful ranking across retrievers even with no
API key, while the real client adds genuine signal when ``--real-llm`` is set.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol


class LLMClient(Protocol):
    def judge_faithfulness(self, query_tokens: list[int], retrieved_tokens: list[int]) -> float: ...


class MockLLMClient:
    """Deterministic offline judge. Anchored on token-overlap grounding."""

    def judge_faithfulness(self, query_tokens: list[int], retrieved_tokens: list[int]) -> float:
        key = ",".join(map(str, query_tokens)) + "|" + ",".join(map(str, retrieved_tokens[:32]))
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        hash_score = ((h % 1000) / 999.0) * 0.4 + 0.55  # [0.55, 0.95]
        if query_tokens:
            anchor = len(set(query_tokens) & set(retrieved_tokens)) / len(set(query_tokens))
        else:
            anchor = 0.0
        return float(0.35 * hash_score + 0.65 * anchor)


class RealAnthropicClient:
    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5") -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def judge_faithfulness(self, query_tokens: list[int], retrieved_tokens: list[int]) -> float:
        prompt = (
            "Rate 0-10 how well the retrieved context supports answering the query. "
            f"Query terms: {query_tokens[:20]}. Context terms: {retrieved_tokens[:40]}. "
            "Reply with a single number."
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str = msg.content[0].text
        m = re.search(r"\d+\.?\d*", text)
        if not m:
            return 0.5
        v = float(m.group(0))
        return max(0.0, min(1.0, v / 10.0 if v > 1 else v))
