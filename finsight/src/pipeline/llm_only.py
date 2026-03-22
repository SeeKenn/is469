"""
llm_only.py — Variant 0: LLM-Only (No Retrieval) Baseline
Pipeline: User Query → Generate (no retrieval at all)

The hallucination baseline — demonstrates what happens when the LLM answers
financial questions purely from its training data, with no grounding in the
actual SEC filings. This variant establishes the minimum performance floor
and quantifies the value that RAG components add.

Expected behaviour:
  - High answer relevance (LLMs are fluent)
  - Low faithfulness (answers are ungrounded)
  - No context precision/recall (no retrieval)
"""

import time
from typing import Dict

from src.generation.generator import Generator
from src.utils.config_loader import load_config, load_prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prompt that allows ungrounded generation (no context provided)
_V0_SYSTEM_PROMPT = """\
You are FinSight, a financial analyst assistant specialising in Microsoft Corporation (NASDAQ: MSFT).
Answer the user's question about Microsoft's financial performance, business segments, and strategy
to the best of your knowledge.

IMPORTANT: You do NOT have access to any specific SEC filings or documents for this query.
Answer based on your general knowledge. If you are unsure about a specific number or date,
say so explicitly rather than guessing.

Do not provide investment advice, stock recommendations, or price targets."""


class LLMOnlyPipeline:
    """
    V0 — LLM-only baseline (no retrieval).
    Purpose: Measure the hallucination floor and establish the value of RAG.
    """

    VARIANT_NAME = "v0_llm_only"
    DESCRIPTION = "LLM-only (no retrieval) — hallucination baseline"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._generator = None

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.cfg)
        return self._generator

    def ask(self, question: str) -> Dict:
        """
        Answer a question using only the LLM — no retrieval, no context.

        Returns the same result dict shape as other pipelines for consistency.
        """
        t0 = time.time()

        # Call the LLM backend directly (bypass Generator.generate which
        # expects chunks). We use the backend's .chat() method directly.
        user_prompt = (
            f"QUESTION: {question}\n\n"
            f"Provide a factual answer about Microsoft Corporation. "
            f"If you are not certain about specific numbers or dates, "
            f"state that explicitly."
        )

        raw = self.generator._backend.chat(_V0_SYSTEM_PROMPT, user_prompt)

        total_latency = (time.time() - t0) * 1000

        result = {
            "answer": raw["answer"],
            "citations": [],                # no citations possible
            "retrieved_chunks": [],         # no retrieval
            "context_used": "",             # no context
            "latency_ms": round(total_latency, 2),
            "generation_latency_ms": raw.get("latency_ms", 0),
            "variant": self.VARIANT_NAME,
            "model": raw.get("model", ""),
            "insufficient_evidence": False,
            "input_tokens": raw.get("input_tokens", 0),
            "output_tokens": raw.get("output_tokens", 0),
            "total_tokens": raw.get("total_tokens", 0),
            "error": raw.get("error"),
        }

        logger.info(
            f"[V0 LLM-Only] Q: '{question[:60]}...' | "
            f"Latency: {total_latency:.0f}ms | "
            f"Tokens: {result['total_tokens']}"
        )
        return result
