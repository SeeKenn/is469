"""
advanced_d.py — Variant 5: Metadata-Aware Filtered Dense Retrieval
Pipeline: Metadata Filter → Dense Retrieval → Generate

Uses structured metadata filtering (fiscal year, quarter, document type) to
narrow the search space BEFORE embedding retrieval. This eliminates temporal
confusion at the retrieval level — the dense retriever never sees chunks from
the wrong fiscal period.

Expected behaviour:
  - Strong improvement on temporal queries (e.g., "Q2 FY2025 revenue")
  - Strong improvement on structured queries (e.g., "latest 10-K risk factors")
  - May underperform on open-ended queries where the period is not specified
  - Requires well-tagged metadata in chunks
"""

import time
from typing import Dict, List, Optional

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.query_processor import FiscalPeriodExtractor
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedDPipeline:
    """
    V5 — Metadata-aware filtered dense retrieval.
    Best for: temporal queries, period-specific lookups, structured questions.
    """

    VARIANT_NAME = "v5_advanced_d"
    DESCRIPTION = "Metadata-filtered dense retrieval (fiscal period / doc type)"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._retriever = None
        self._generator = None
        self._period_extractor = FiscalPeriodExtractor()

    @property
    def retriever(self) -> DenseRetriever:
        if self._retriever is None:
            self._retriever = DenseRetriever(self.cfg)
        return self._retriever

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.cfg)
        return self._generator

    def _build_metadata_filter(self, question: str) -> Dict:
        """
        Extract structured metadata filters from the question.

        Returns dict with:
            - fiscal_filter: ChromaDB where clause (or None)
            - relaxed_filter: Fallback filter (or None)
            - fiscal_info: Extracted period info
            - filter_applied: Description of what was filtered
        """
        fiscal_info = self._period_extractor.extract(question)
        strict_filter = self._period_extractor.to_metadata_filter(fiscal_info)
        relaxed_filter = self._period_extractor.to_relaxed_filter(fiscal_info)

        # Build human-readable description of the filter
        parts = []
        if fiscal_info.get("fiscal_year"):
            parts.append(fiscal_info["fiscal_year"])
        if fiscal_info.get("quarter"):
            parts.append(fiscal_info["quarter"])
        if fiscal_info.get("doc_type"):
            parts.append(fiscal_info["doc_type"])

        filter_desc = " + ".join(parts) if parts else "none"

        return {
            "fiscal_filter": strict_filter,
            "relaxed_filter": relaxed_filter,
            "fiscal_info": fiscal_info,
            "filter_applied": filter_desc,
        }

    def ask(self, question: str) -> Dict:
        """
        Answer a question using metadata-filtered dense retrieval.
        """
        t0 = time.time()
        top_k = self.cfg["retrieval"]["final_context_k"]
        dense_top_k = self.cfg["retrieval"].get("dense_top_k", 20)

        # Step 1: Extract metadata filters from the question
        t_filter = time.time()
        filter_info = self._build_metadata_filter(question)
        filter_latency = (time.time() - t_filter) * 1000
        fiscal_filter = filter_info["fiscal_filter"]
        relaxed_filter = filter_info["relaxed_filter"]

        # Step 2: Retrieve with strict metadata filter
        t_retrieval = time.time()
        retrieved = []
        filter_strategy = "none"

        if fiscal_filter:
            retrieved = self.retriever.retrieve(
                question,
                top_k=dense_top_k,
                fiscal_filter=fiscal_filter,
                use_fiscal_filtering=False,  # we're providing our own filter
            )
            filter_strategy = "strict"

            # Fallback: if strict filter returns too few results, try relaxed
            if len(retrieved) < 3 and relaxed_filter and relaxed_filter != fiscal_filter:
                logger.info(
                    f"[V5] Strict filter returned {len(retrieved)} results, "
                    f"falling back to relaxed filter"
                )
                retrieved = self.retriever.retrieve(
                    question,
                    top_k=dense_top_k,
                    fiscal_filter=relaxed_filter,
                    use_fiscal_filtering=False,
                )
                filter_strategy = "relaxed"

            # Last resort: if relaxed filter also too few, retrieve unfiltered
            if len(retrieved) < 3:
                logger.info(
                    f"[V5] Relaxed filter returned {len(retrieved)} results, "
                    f"falling back to unfiltered retrieval"
                )
                retrieved = self.retriever.retrieve(
                    question,
                    top_k=dense_top_k,
                    use_fiscal_filtering=True,  # use the retriever's built-in filtering
                )
                filter_strategy = "unfiltered_fallback"

        else:
            # No period detected — standard unfiltered dense retrieval
            retrieved = self.retriever.retrieve(
                question,
                top_k=dense_top_k,
                use_fiscal_filtering=True,
            )
            filter_strategy = "auto"

        retrieval_latency = (time.time() - t_retrieval) * 1000

        # Slice to final context size
        context_chunks = retrieved[:top_k]

        # Step 3: Generate
        t_gen = time.time()
        gen_result = self.generator.generate(question, context_chunks)
        generation_latency = (time.time() - t_gen) * 1000

        # Step 4: Citations
        citations = format_citations(gen_result["answer"], context_chunks)

        total_latency = (time.time() - t0) * 1000

        result = {
            "answer": gen_result["answer"],
            "citations": citations,
            "retrieved_chunks": context_chunks,
            "context_used": gen_result.get("context_used", ""),
            "latency_ms": round(total_latency, 2),
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "reranking_latency_ms": 0,
            "filter_latency_ms": round(filter_latency, 2),
            "generation_latency_ms": round(generation_latency, 2),
            "variant": self.VARIANT_NAME,
            "model": gen_result.get("model", ""),
            "insufficient_evidence": gen_result.get("insufficient_evidence", False),
            "input_tokens": gen_result.get("input_tokens", 0),
            "output_tokens": gen_result.get("output_tokens", 0),
            "total_tokens": gen_result.get("total_tokens", 0),
            "metadata_filter": {
                "filter_applied": filter_info["filter_applied"],
                "filter_strategy": filter_strategy,
                "fiscal_info": filter_info["fiscal_info"],
                "n_results_after_filter": len(retrieved),
            },
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V5 Adv-D] Q: '{question[:50]}...' | "
            f"Filter: {filter_info['filter_applied']} ({filter_strategy}) | "
            f"Retrieved: {len(retrieved)} → Context: {len(context_chunks)} | "
            f"Latency: {total_latency:.0f}ms"
        )
        return result
