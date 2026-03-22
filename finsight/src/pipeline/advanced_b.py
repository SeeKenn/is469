"""
advanced_b.py — Variant 3: Hybrid Retrieval + Cross-Encoder Reranker
Pipeline: BM25(top-20) + Dense(top-20) → RRF Merge → Reranker → Generator (top-5)

Best of both worlds:
- BM25 handles keyword-rich queries (company names, exact metrics, tickers)
- Dense handles semantic/conceptual queries (risks, strategy, outlook)
- RRF fusion deduplicates and ranks without score normalisation
- Reranker then refines the merged candidate set
"""

import time
from typing import Dict

from src.retrieval.hybrid_retriever import HybridRetriever, _detect_fiscal_periods
from src.retrieval.reranker import Reranker
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedBPipeline:
    """
    V3 — Hybrid retrieval (BM25 + dense + RRF) with cross-encoder reranking.
    Best for: cross-document synthesis, trend queries, risk factor questions,
              questions mixing exact financial terms with conceptual context.
    """

    VARIANT_NAME = "v3_advanced_b"
    DESCRIPTION = "Hybrid retrieval (BM25 + dense + RRF) + cross-encoder reranking"

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

    def ask(self, question: str) -> Dict:
        """
        Answer a question using hybrid retrieval + reranking.
        """
        t0 = time.time()
        rerank_top_k = self.cfg["retrieval"]["rerank_top_k"]
        final_context_k = self.cfg["retrieval"]["final_context_k"]

        # Step 1: Hybrid retrieval — BM25 + dense + period-boost, fused by RRF
        t_retrieval = time.time()
        candidates = self.retriever.retrieve(question)
        retrieval_latency = (time.time() - t_retrieval) * 1000

        # Step 2: Rerank the merged candidates
        t_rerank = time.time()
        reranked = self.reranker.rerank(question, candidates, top_k=rerank_top_k)
        reranking_latency = (time.time() - t_rerank) * 1000

        # Step 3: Apply fiscal-period lock then slice to final_context_k.
        #   For queries mentioning a specific year/quarter, guarantee that the
        #   best-scored chunks FROM THAT PERIOD appear in the final context.
        #   `rerank_top_k` is intentionally set larger than `final_context_k` so
        #   period chunks ranked just outside the natural top-8 by cross-encoder
        #   (e.g. numeric reconciliation tables) still have a chance here.
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
            context_chunks = (guaranteed + remaining)[: final_context_k]
            logger.debug(
                f"Period lock for {detected}: {len(guaranteed)}/{period_slots} guaranteed slots "
                f"({len(period_chunks)} period chunks in reranked-{len(reranked)})"
            )
        else:
            context_chunks = reranked[:final_context_k]

        # Step 4: Generate
        t_gen = time.time()
        gen_result = self.generator.generate(question, context_chunks)
        generation_latency = (time.time() - t_gen) * 1000

        # Step 5: Citations
        citations = format_citations(gen_result["answer"], context_chunks)

        total_latency = (time.time() - t0) * 1000
        top_rerank_score = reranked[0]["rerank_score"] if reranked else 0.0

        # Count how many candidates came from each source
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
            "generation_latency_ms": round(generation_latency, 2),
            "variant": self.VARIANT_NAME,
            "model": gen_result.get("model", ""),
            "insufficient_evidence": gen_result.get("insufficient_evidence", False),
            "input_tokens": gen_result.get("input_tokens", 0),
            "output_tokens": gen_result.get("output_tokens", 0),
            "total_tokens": gen_result.get("total_tokens", 0),
            "top_rerank_score": top_rerank_score,
            "n_candidates": len(candidates),
            "n_reranked": len(reranked),
            "fusion_stats": {
                "found_by_both": found_by_both,
                "found_by_dense_only": found_by_dense,
                "found_by_sparse_only": found_by_sparse,
            },
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V3 Adv-B] Q: '{question[:60]}...' | "
            f"Candidates: {len(candidates)} (both={found_by_both}, d={found_by_dense}, s={found_by_sparse}) "
            f"→ Reranked: {len(reranked)} | "
            f"Top rerank: {top_rerank_score:.4f} | "
            f"Latency: {total_latency:.0f}ms"
        )
        return result
