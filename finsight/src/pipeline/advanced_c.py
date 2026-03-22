"""
advanced_c.py — Variant 4: Query Rewriting + Hybrid Retrieval + Reranker
Pipeline: LLM Query Rewrite → BM25 + Dense → RRF → Reranker → Generator

Builds on V3 by adding an LLM-based query rewriting step before retrieval.
The rewriter decomposes ambiguous or underspecified queries into precise
sub-queries, improving retrieval quality for complex questions.

Expected behaviour:
  - Significant improvement on ambiguous queries
  - Moderate improvement on multi-hop queries
  - Minimal overhead on simple factual queries
  - Extra LLM call adds latency
"""

import time
from typing import Dict, List

from src.retrieval.hybrid_retriever import HybridRetriever, _detect_fiscal_periods
from src.retrieval.reranker import Reranker
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Query rewriting prompt ────────────────────────────────────────────────────

_REWRITE_SYSTEM = """\
You are a query rewriting assistant for a financial document QA system that \
searches Microsoft Corporation SEC filings (10-K and 10-Q reports).

Your job is to rewrite the user's query into a better search query that will \
retrieve the most relevant passages from these financial documents.

Rules:
1. Keep the rewritten query concise (1-2 sentences).
2. Make implicit fiscal periods explicit (e.g., "last year" → "FY2024").
3. Expand abbreviations (e.g., "COGS" → "cost of goods sold / cost of revenue").
4. Add document-type hints when helpful (e.g., "10-K annual report" or "10-Q quarterly report").
5. For comparison queries, break into the specific periods being compared.
6. Preserve the original intent — do NOT change what is being asked.
7. If the query is already clear and specific, return it unchanged.

Respond with ONLY the rewritten query, nothing else."""

_REWRITE_USER = "Original query: {question}\n\nRewritten query:"


class AdvancedCPipeline:
    """
    V4 — Query rewriting + hybrid retrieval + reranking.
    Best for: ambiguous queries, underspecified questions, complex comparisons.
    """

    VARIANT_NAME = "v4_advanced_c"
    DESCRIPTION = "LLM query rewriting + hybrid retrieval + reranking"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._retriever = None
        self._reranker = None
        self._generator = None

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.cfg)
        return self._retriever

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.cfg)
        return self._reranker

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.cfg)
        return self._generator

    def _rewrite_query(self, question: str) -> Dict:
        """
        Use the LLM to rewrite the query for better retrieval.

        Returns:
            Dict with 'rewritten_query', 'original_query', 'rewrite_latency_ms'
        """
        t0 = time.time()
        user_prompt = _REWRITE_USER.format(question=question)

        raw = self.generator._backend.chat(_REWRITE_SYSTEM, user_prompt)
        rewrite_latency = (time.time() - t0) * 1000

        rewritten = raw.get("answer", "").strip()

        # Fallback: if the rewrite is empty, nonsensical, or way too long, use original
        if not rewritten or len(rewritten) > len(question) * 3:
            rewritten = question

        logger.info(
            f"[V4 Rewrite] '{question[:50]}...' → '{rewritten[:50]}...' "
            f"({rewrite_latency:.0f}ms)"
        )

        return {
            "rewritten_query": rewritten,
            "original_query": question,
            "rewrite_latency_ms": round(rewrite_latency, 2),
            "rewrite_tokens": raw.get("total_tokens", 0),
        }

    def ask(self, question: str) -> Dict:
        """
        Answer a question using query rewriting + hybrid retrieval + reranking.
        """
        t0 = time.time()
        rerank_top_k = self.cfg["retrieval"]["rerank_top_k"]
        final_context_k = self.cfg["retrieval"]["final_context_k"]

        # Step 1: Rewrite the query
        rewrite_result = self._rewrite_query(question)
        search_query = rewrite_result["rewritten_query"]

        # Step 2: Hybrid retrieval using rewritten query
        t_retrieval = time.time()
        candidates = self.retriever.retrieve(search_query)
        retrieval_latency = (time.time() - t_retrieval) * 1000

        # Step 3: Rerank — use ORIGINAL question for semantic relevance scoring
        # (rewritten query is optimised for retrieval, not for relevance judgment)
        t_rerank = time.time()
        reranked = self.reranker.rerank(question, candidates, top_k=rerank_top_k)
        reranking_latency = (time.time() - t_rerank) * 1000

        # Step 4: Fiscal period lock (same as V3)
        period_slots = int(self.cfg.get("retrieval", {}).get("period_guaranteed_slots", 3))
        detected = _detect_fiscal_periods(question)

        if detected and period_slots > 0:
            period_chunks = [
                c for c in reranked
                if c["metadata"].get("fiscal_period", "") in detected
            ]
            guaranteed = period_chunks[:period_slots]
            guaranteed_ids = {c["metadata"].get("chunk_id") for c in guaranteed}
            remaining = [c for c in reranked if c["metadata"].get("chunk_id") not in guaranteed_ids]
            context_chunks = (guaranteed + remaining)[:final_context_k]
        else:
            context_chunks = reranked[:final_context_k]

        # Step 5: Generate — use original question (not rewritten)
        t_gen = time.time()
        gen_result = self.generator.generate(question, context_chunks)
        generation_latency = (time.time() - t_gen) * 1000

        # Step 6: Citations
        citations = format_citations(gen_result["answer"], context_chunks)

        total_latency = (time.time() - t0) * 1000
        top_rerank_score = reranked[0]["rerank_score"] if reranked else 0.0

        found_by_both = sum(1 for c in candidates if c.get("found_by") == "both")
        found_by_dense = sum(1 for c in candidates if c.get("found_by") == "dense")
        found_by_sparse = sum(1 for c in candidates if c.get("found_by") == "sparse")

        result = {
            "answer": gen_result["answer"],
            "citations": citations,
            "retrieved_chunks": context_chunks,
            "all_candidates": candidates,
            "context_used": gen_result.get("context_used", ""),
            "latency_ms": round(total_latency, 2),
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "reranking_latency_ms": round(reranking_latency, 2),
            "rewrite_latency_ms": round(rewrite_result["rewrite_latency_ms"], 2),
            "generation_latency_ms": round(generation_latency, 2),
            "variant": self.VARIANT_NAME,
            "model": gen_result.get("model", ""),
            "insufficient_evidence": gen_result.get("insufficient_evidence", False),
            "input_tokens": gen_result.get("input_tokens", 0) + rewrite_result.get("rewrite_tokens", 0),
            "output_tokens": gen_result.get("output_tokens", 0),
            "total_tokens": gen_result.get("total_tokens", 0) + rewrite_result.get("rewrite_tokens", 0),
            "top_rerank_score": top_rerank_score,
            "n_candidates": len(candidates),
            "n_reranked": len(reranked),
            "fusion_stats": {
                "found_by_both": found_by_both,
                "found_by_dense_only": found_by_dense,
                "found_by_sparse_only": found_by_sparse,
            },
            "query_rewrite": {
                "original": rewrite_result["original_query"],
                "rewritten": rewrite_result["rewritten_query"],
                "rewrite_latency_ms": rewrite_result["rewrite_latency_ms"],
            },
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V4 Adv-C] Q: '{question[:50]}...' | "
            f"Rewrite: '{search_query[:50]}...' | "
            f"Candidates: {len(candidates)} → Reranked: {len(reranked)} | "
            f"Latency: {total_latency:.0f}ms (rewrite: {rewrite_result['rewrite_latency_ms']:.0f}ms)"
        )
        return result
