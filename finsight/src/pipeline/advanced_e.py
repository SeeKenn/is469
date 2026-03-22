"""
advanced_e.py — Variant 6: Hybrid Retrieval + Reranker + Context Compression
Pipeline: BM25 + Dense → RRF → Reranker → Context Compression → Generator

Builds on V3 by adding a context compression step after reranking.
Retrieved chunks are filtered/summarised to remove irrelevant text before
being passed to the LLM, improving the signal-to-noise ratio.

Expected behaviour:
  - Improved faithfulness (less noise → fewer hallucinations)
  - Improved performance on multi-hop queries (cleaner context)
  - Moderate improvement on ambiguous queries
  - Additional processing adds some latency
"""

import time
from typing import Dict, List

from src.retrieval.hybrid_retriever import HybridRetriever, _detect_fiscal_periods
from src.retrieval.reranker import Reranker
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.generation.context_manager import ContextManager, ContextOptimizer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedEPipeline:
    """
    V6 — Hybrid retrieval + reranking + context compression.
    Best for: multi-hop reasoning, noisy contexts, long-context questions.
    """

    VARIANT_NAME = "v6_advanced_e"
    DESCRIPTION = "Hybrid retrieval + reranking + context compression"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._retriever = None
        self._reranker = None
        self._generator = None
        self._context_manager = None
        self._context_optimizer = None

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

    @property
    def context_manager(self) -> ContextManager:
        if self._context_manager is None:
            model_name = self.cfg["generation"].get("model", "qwen2.5-14b")
            self._context_manager = ContextManager(
                model_name=model_name,
                reserved_for_output=self.cfg["generation"].get("max_tokens", 512),
                reserved_for_prompt=800,
            )
        return self._context_manager

    @property
    def context_optimizer(self) -> ContextOptimizer:
        if self._context_optimizer is None:
            self._context_optimizer = ContextOptimizer(self.context_manager)
        return self._context_optimizer

    def _compress_chunks(self, chunks: List[Dict], question: str) -> List[Dict]:
        """
        Compress retrieved chunks by extracting the most relevant sentences.

        For each chunk, keeps only the sentences most likely to contain
        financial information relevant to the question. This reduces noise
        from boilerplate text, legal disclaimers, and irrelevant context.

        Returns:
            List of compressed chunk dicts (same structure, shorter text)
        """
        compressed = []
        total_original_chars = 0
        total_compressed_chars = 0

        for chunk in chunks:
            original_text = chunk.get("text", "")
            total_original_chars += len(original_text)

            # Use ContextOptimizer to extract key financial sentences
            compressed_text = self.context_optimizer.extract_key_sentences(
                original_text,
                max_sentences=8,  # Keep top 8 sentences per chunk
            )
            total_compressed_chars += len(compressed_text)

            compressed_chunk = {
                **chunk,
                "text": compressed_text,
                "original_text_length": len(original_text),
                "compressed": len(compressed_text) < len(original_text),
            }
            compressed.append(compressed_chunk)

        compression_ratio = (
            total_compressed_chars / total_original_chars
            if total_original_chars > 0 else 1.0
        )
        logger.debug(
            f"Context compression: {total_original_chars} → {total_compressed_chars} chars "
            f"({compression_ratio:.1%} retained)"
        )

        return compressed

    def ask(self, question: str) -> Dict:
        """
        Answer a question using hybrid retrieval + reranking + context compression.
        """
        t0 = time.time()
        rerank_top_k = self.cfg["retrieval"]["rerank_top_k"]
        final_context_k = self.cfg["retrieval"]["final_context_k"]

        # Step 1: Hybrid retrieval
        candidates = self.retriever.retrieve(question)

        # Step 2: Rerank
        reranked = self.reranker.rerank(question, candidates, top_k=rerank_top_k)

        # Step 3: Fiscal period lock (same as V3)
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
            pre_compression_chunks = (guaranteed + remaining)[:final_context_k]
        else:
            pre_compression_chunks = reranked[:final_context_k]

        # Step 4: Context compression — extract key sentences from each chunk
        t_compress = time.time()
        context_chunks = self._compress_chunks(pre_compression_chunks, question)
        compression_latency = (time.time() - t_compress) * 1000

        # Step 5: Generate
        gen_result = self.generator.generate(question, context_chunks)

        # Step 6: Citations
        citations = format_citations(gen_result["answer"], context_chunks)

        total_latency = (time.time() - t0) * 1000
        top_rerank_score = reranked[0]["rerank_score"] if reranked else 0.0

        found_by_both = sum(1 for c in candidates if c.get("found_by") == "both")
        found_by_dense = sum(1 for c in candidates if c.get("found_by") == "dense")
        found_by_sparse = sum(1 for c in candidates if c.get("found_by") == "sparse")

        # Compute compression stats
        n_compressed = sum(1 for c in context_chunks if c.get("compressed"))
        orig_chars = sum(c.get("original_text_length", len(c.get("text", ""))) for c in context_chunks)
        final_chars = sum(len(c.get("text", "")) for c in context_chunks)

        result = {
            "answer": gen_result["answer"],
            "citations": citations,
            "retrieved_chunks": context_chunks,
            "all_candidates": candidates,
            "context_used": gen_result.get("context_used", ""),
            "latency_ms": round(total_latency, 2),
            "generation_latency_ms": gen_result.get("latency_ms", 0),
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
            "compression_stats": {
                "chunks_compressed": n_compressed,
                "total_chunks": len(context_chunks),
                "original_chars": orig_chars,
                "compressed_chars": final_chars,
                "compression_ratio": round(final_chars / orig_chars, 3) if orig_chars else 1.0,
                "compression_latency_ms": round(compression_latency, 2),
            },
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V6 Adv-E] Q: '{question[:50]}...' | "
            f"Candidates: {len(candidates)} → Reranked: {len(reranked)} | "
            f"Compression: {orig_chars}→{final_chars} chars "
            f"({final_chars/orig_chars:.0%} retained, {compression_latency:.0f}ms) | "
            f"Latency: {total_latency:.0f}ms"
        )
        return result
