# FinSight: RAG-based Financial Filings QA System
## Final Project Report

**Course:** IS469 - Generative AI with Large Language Models
**Project:** Domain-Specific RAG System for SEC Filings Analysis
**Company Focus:** Microsoft Corporation (NASDAQ: MSFT)

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [System Architecture](#3-system-architecture)
4. [Methodology](#4-methodology)
5. [Experimental Results](#5-experimental-results)
6. [Qualitative Error Analysis](#6-qualitative-error-analysis)
7. [Discussion & Insights](#7-discussion--insights)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [Risks and Guardrails](#9-risks-and-guardrails)
10. [Conclusion](#10-conclusion)
11. [Appendix](#11-appendix)

---

## 1. Introduction

Financial filings such as SEC 10-K and 10-Q reports are critical sources of information for investors, analysts, and researchers. However, these documents are lengthy (often 100+ pages), dense with technical language, and require significant domain knowledge to navigate efficiently. Traditional keyword search is inadequate for extracting nuanced financial insights that span multiple sections or require temporal reasoning.

This project presents **FinSight**, a Retrieval-Augmented Generation (RAG) system designed specifically for question-answering over Microsoft Corporation's official SEC filings. We implement and rigorously compare **seven pipeline variants (V0–V6)** to understand how individual RAG components affect performance across different query types.

### Key Contributions
1. **End-to-end RAG implementation** with seven configurable pipelines covering a full component ablation: no retrieval (V0), dense-only (V1), dense + reranking (V2), hybrid+reranking (V3), query rewriting (V4), metadata filtering (V5), and context compression (V6)
2. **Controlled experimental design** isolating the contribution of each RAG component to performance across four query categories
3. **Comprehensive evaluation** using RAGAS metrics, numerical accuracy, and qualitative error analysis
4. **Actionable insights** on which RAG components benefit which query types

---

## 2. Problem Statement & Objectives

### Problem Statement
Extracting specific financial insights from SEC filings requires navigating hundreds of pages of complex financial documents, understanding Microsoft's fiscal calendar (July–June fiscal year), cross-referencing information across multiple filing periods, and distinguishing between similar-sounding metrics across different time periods.

### Objectives
1. Develop a domain-specific QA system that provides accurate, citation-backed answers
2. Compare seven RAG pipeline variants to isolate the contribution of each component
3. Analyse failure modes to understand when and why different approaches succeed or fail
4. Demonstrate reproducible evaluation methodology for domain-specific RAG systems

### Study Hypothesis
Different RAG components provide selective benefits depending on query type, rather than uniformly improving performance across all categories. No single pipeline is optimal for all query types.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                     USER QUERY                          │
                              └─────────────────────────┬───────────────────────────────┘
                                                        │
                              ┌─────────────────────────▼───────────────────────────────┐
                              │         QUERY PROCESSING (V4: LLM Query Rewrite)        │
                              │         (Fiscal period detection, normalisation)        │
                              └─────────────────────────┬───────────────────────────────┘
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────────┐
                    │                                   │                                   │
          ┌─────────▼─────────┐               ┌────────▼────────┐                           │
          │   DENSE RETRIEVER │               │  SPARSE (BM25)  │             V0: Skip      │
          │  (ChromaDB +      │               │   RETRIEVER     │             retrieval     │
          │   MPNet embed)    │               │   (V3,V4,V6)    │             entirely      │
          │  V5: pre-filtered │               │                 │                           │
          └─────────┬─────────┘               └────────┬────────┘                           │
                    │                                  │                                    │
                    │  V1: Direct to Generator         │                                    │
                    │  ────────────────────────────────────────────────────────────────────►│
                    │                                  │                                    │
                    │  V2,V3,V4,V6: via Reranker       │                                    │
                    └───────────────┬──────────────────┘                                    │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │    RRF FUSION (V3, V4, V6 only)   │                                   │
                    │      Score = Σ(1/(k + rank_i))    │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │     CROSS-ENCODER RERANKER        │                                   │
                    │     (ms-marco-MiniLM-L-6-v2)      │                                   │
                    │     (V2, V3, V4, V6 only)         │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │   CONTEXT COMPRESSION (V6 only)   │                                   │
                    │   Filters irrelevant chunk text   │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐◄──────────────────────────────────┘
                    │         TOP-K CONTEXT             │
                    │     (Final retrieved chunks)      │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │           LLM GENERATOR           │
                    │    (gpt-4o-mini via OpenAI API)  │
                    │   + Citation formatting + Guards  │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │     ANSWER WITH CITATIONS         │
                    │        [Doc-1], [Doc-2]...        │
                    └───────────────────────────────────┘
```

---

### 3.2 Seven Pipeline Variants

Each variant introduces a single additional component to isolate its impact:

| Variant | Pipeline Flow | Key Addition | Purpose |
|---------|--------------|--------------|---------|
| **V0 LLM-only** | Generate only | No retrieval | Hallucination baseline |
| **V1 Baseline** | Dense → Generate | ChromaDB dense retrieval (all-mpnet-base-v2), top-k=5 | Establishes retrieval floor |
| **V2 Advanced A** | Dense → Rerank → Generate | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2), top-k=5 | Improve ranking precision |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion | Lexical + semantic combination |
| **V4 Advanced C** | Query Rewrite → BM25 + Dense → RRF → Rerank → Generate | LLM-based query rewriting | Handles ambiguous queries |
| **V5 Advanced D** | Metadata Filter → Dense → Generate | Fiscal period metadata pre-filtering | Low-latency precision boost |
| **V6 Advanced E** | BM25 + Dense → RRF → Rerank → Compress → Generate | Context compression | Reduce noise for multi-hop |

---

### 3.3 Model Configuration
| Component | Model | Details |
|-----------|-------|---------|
| Embeddings | sentence-transformers/all-mpnet-base-v2 | 768-dim, normalised |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder, max_len=512 |
| Generator | gpt-4o-mini | via OpenAI-compatible API, temp=0.0, max_tokens=512 |
| RAGAS Judge | gpt-4o-mini | Same configured judge model, max_tokens=1024 |

---

### 3.4 Project Structure

```
finsight/
├── config/
│   ├── settings.yaml          # Main configuration (models, thresholds, paths)
│   ├── chunking.yaml          # Chunking experiment configurations
│   └── prompts.yaml           # All prompt templates
├── data/
│   ├── raw/                   # Optional raw SEC filing PDFs for full rebuilds
│   ├── processed/             # Chunked + tagged JSON per document
│   └── metadata/              # Metadata schema
├── indexes/
│   ├── chroma/                # ChromaDB vector store
│   └── bm25/                  # BM25 index pickle files
├── src/
│   ├── ingestion/             # PDF parsing, text cleaning
│   ├── chunking/              # Fixed-size chunking, metadata tagging
│   ├── indexing/              # ChromaDB + BM25 index builders
│   ├── retrieval/             # Dense, sparse, hybrid retrievers + reranker
│   ├── generation/            # LLM generator + citation formatter
│   ├── pipeline/              # V0–V6 end-to-end pipeline implementations
│   └── utils/                 # Config loader, logger utilities
├── evaluation/
│   ├── eval_dataset.json      # 20-question benchmark (4 categories)
│   ├── benchmark.csv          # Benchmark with extended metadata
│   ├── run_evaluation.py      # RAGAS + numerical accuracy evaluation runner
│   ├── ablation_study.py      # 4-step component ablation
│   ├── category_analysis.py   # Per-category breakdown and error analysis
│   └── results/               # JSON result files per evaluation run
├── app/
│   └── streamlit_app.py       # Interactive Streamlit UI
└── scripts/
    ├── ingest_all.py          # Full ingestion pipeline
    ├── build_index.py         # Index construction
    ├── run_query.py           # CLI query tool
    └── smoke_test.py          # Sanity check script
```

---

## 4. Methodology
### 4.1 Dataset

We indexed **9 Microsoft SEC filings** spanning FY2022 to Q2 FY2026:

| Document Type | Count | Fiscal Periods Covered |
|--------------|-------|------------------------|
| 10-K (Annual) | 4 | FY2022, FY2023, FY2024, FY2025 |
| 10-Q (Quarterly) | 5 | Q1–Q3 FY2025, Q1–Q2 FY2026 |

**Important note:** Microsoft's fiscal year runs July 1–June 30. FY2024 = July 2023–June 2024.

---

### 4.2 Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500–800 tokens | Balance between context and embedding quality |
| Overlap | 10–20% | Preserve cross-boundary context |
| Metadata | doc_type, fiscal_period, section | Enable filtered retrieval (V5) |

---

### 4.3 Evaluation Framework

#### Benchmark Dataset

- **20 questions** across 4 query categories (5 per category)
- Ground truth answers sourced directly from SEC filing text
- Questions intended to require actual retrieval, though a few remained partially answerable from LLM training knowledge alone

| Category | Questions | What It Tests |
|----------|-----------|---------------|
| **Factual Retrieval** | q001–q005 | Single-document, single-concept lookups (margins, R&D, income) |
| **Temporal Reasoning** | q006–q010 | Fiscal period-specific queries and sequential QoQ comparisons |
| **Multi-Hop Reasoning** | q011–q015 | Answers requiring synthesis across multiple filings or sections |
| **Comparative Analysis** | q016–q020 | Cross-period and cross-segment performance comparisons |

#### Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| **Faithfulness** | RAGAS | Are all claims in the answer supported by retrieved context? |
| **Answer Relevancy** | RAGAS | Does the answer address the question asked? |
| **Context Recall** | RAGAS | Did retrieval surface the relevant information? |
| **Context Precision** | RAGAS | What fraction of retrieved context was actually useful? |
| **Numerical Accuracy** | Custom | Does the answer contain the key numbers from the ground truth? |

---

## 5. Experimental Results

*Source note: Unless otherwise stated, all values in Sections 5.1-5.4 are taken directly from `evaluation/results/eval_results.json` (rounded for display). Section 5.5 uses `evaluation/results/ablation_results.json`. In the tables below, `0.000` means the saved run recorded an actual zero score, while `n/a` means the saved JSON stores `NaN` for that slice, so the metric is unavailable rather than zero.*

### 5.1 Aggregate RAGAS Performance — All Seven Variants

| Metric | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|--------|----|----|----|----|----|----|----|
| **Faithfulness** | 0.098 | 0.738 | 0.843 | 0.786 | 0.760 | 0.804 | 0.700 |
| **Answer Relevancy** | 0.360 | 0.483 | 0.478 | 0.711 | 0.784 | 0.483 | 0.738 |
| **Context Recall** | 0.000 | 0.592 | 0.367 | 0.250 | 0.463 | 0.517 | 0.383 |
| **Context Precision** | 0.000 | 0.607 | 0.741 | 0.706 | 0.686 | 0.602 | 0.574 |
| **Numerical Accuracy** | 0.250 | 0.350 | 0.450 | 0.600 | 0.500 | 0.300 | 0.550 |
| **Avg Latency (s)** | 3.94 | 3.46 | 4.50 | 7.71 | 8.51 | 3.63 | 10.52 |

*Note: V0 has no retrieved context, so its context recall and context precision remain 0.000 throughout. Its non-zero faithfulness score (0.098) reflects the saved RAGAS output on ungrounded answers rather than evidence-backed retrieval. Numerical Accuracy is computed as the fraction of questions where key figures from the ground truth appear verbatim in the generated answer.*

#### Key Observations:

1. **V0 (LLM-only) establishes the ungrounded baseline** — its near-zero faithfulness (0.098) and zero context metrics confirm that retrieval is necessary for verifiable answers, even though numerical accuracy still reaches 0.250 from parametric memory alone.

2. **Answer relevancy is highest for retrieval-breadth variants** — V4 leads at 0.784, followed by V6 at 0.738 and V3 at 0.711, while V2’s reranking-only pipeline remains flat at 0.478.

3. **Context recall peaks in V1 and V5** — V1 achieves the highest recall (0.592), followed by V5 (0.517), showing that dense retrieval and metadata filtering preserve broader coverage than the heavier hybrid pipelines.

4. **V2 delivers the cleanest grounded context** — it achieves the highest context precision (0.741) and the highest faithfulness (0.843), confirming the value of reranking for answer grounding.

5. **V5 offers the strongest low-latency grounded profile** — at 3.63s average latency with 0.804 faithfulness, it is the most practical fast deployment option, though it remains brittle on cross-period tasks.

6. **Numerical accuracy is strongest in the multi-stage retrieval pipelines** — V3 leads at 0.600 and V6 follows at 0.550, suggesting that harder numerical questions benefit from richer evidence sets.

7. **Latency rises steadily with pipeline complexity** — V1/V5 are the fastest grounded systems, V2 adds a modest reranking cost, and V6 is the slowest at 10.52s.

---

### 5.2 Category-based RAGAS Performance

*Note: The category tables below are taken directly from the `aggregate.category_ragas` fields in `evaluation/results/eval_results.json`. A `0.00` score indicates the saved run recorded a complete failure on that dimension. An `n/a` cell indicates the saved JSON contains `NaN` for that metric/category slice, so the score should be treated as unavailable rather than zero.*

---

### 5.2.1 Factual Retrieval RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | 0.000 | 0.378 | 0.000 | 0.000 |
| V1 | 0.733 | 0.988 | 0.667 | 0.593 |
| V2 | 0.667 | 0.988 | 0.533 | n/a |
| V3 | 0.633 | 0.984 | 0.400 | 0.497 |
| V4 | 0.700 | 0.988 | 0.600 | 0.482 |
| V5 | 0.700 | 0.988 | 0.633 | 0.549 |
| V6 | 0.714 | 0.788 | 0.600 | 0.510 |

#### Key Observations
1. **Dense retrieval remains sufficient for simple factual lookups** — V1, V4, and V5 all reach 0.988 answer relevancy, indicating that most factual questions can be answered once the correct chunk is surfaced.

2. **Reranking does not improve factual end-task performance in the saved run** — V2 matches V1 on answer relevancy but trails it on recall, and the saved JSON records factual context precision as `n/a` for this slice.

3. **Hybrid retrieval mainly helps factual numerical matching** — V3 lifts numerical accuracy to 60%, suggesting lexical anchoring helps when exact figures matter.

4. **Context compression hurts simple factual answer relevancy** — V6 drops to 0.788 relevancy despite recovering many of the key figures numerically.

---

### 5.2.2 Temporal Reasoning RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | 0.187 | 0.000 | 0.000 | 0.000 |
| V1 | 1.000 | 0.387 | 0.400 | 0.482 |
| V2 | 1.000 | 0.387 | 0.400 | n/a |
| V3 | 0.861 | 0.585 | 0.300 | 0.686 |
| V4 | 0.561 | 0.555 | 0.200 | 0.676 |
| V5 | 1.000 | 0.387 | 0.400 | 0.491 |
| V6 | 0.762 | 0.551 | 0.600 | 0.404 |

#### Key Observations
1. **Hybrid retrieval is strongest on temporal answer relevancy** — V3 achieves the best temporal relevancy (0.585), outperforming the dense-only variants at 0.387.

2. **Perfect faithfulness does not guarantee correct temporal coverage** — V1, V2, and V5 all score 1.000 in faithfulness while still producing low-relevancy answers because they often ground only part of the requested comparison.

3. **Query rewriting improves temporal relevance but trades off grounding** — V4 raises temporal relevancy to 0.555 but falls to 0.561 in faithfulness.

4. **More retrieved context is not enough by itself** — V6 reaches the highest temporal recall (0.600) but still trails V3 on final answer relevancy.



---

### 5.2.3 Multi-Hop Reasoning RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | 0.000 | 0.521 | 0.000 | 0.000 |
| V1 | n/a | 0.357 | 0.733 | 0.630 |
| V2 | 0.843 | 0.347 | 0.133 | 0.807 |
| V3 | 0.817 | 0.547 | 0.000 | 0.792 |
| V4 | n/a | 0.725 | 0.600 | 0.742 |
| V5 | n/a | 0.357 | 0.667 | 0.645 |
| V6 | 0.730 | 0.729 | 0.133 | 0.628 |

#### Key Observations
1. **V4 and V6 lead multi-hop answer relevancy** — at 0.725 and 0.729 respectively, they substantially outperform the dense-only and reranking-only baselines.

2. **High recall alone does not guarantee synthesis quality** — V1 and V5 retrieve broad context (0.733 and 0.667 recall) but still remain at 0.357 relevancy.

3. **Reranking can over-filter multi-hop evidence** — V2 reaches 0.807 precision but collapses to 0.133 recall and 0.347 relevancy.

4. **Hybrid retrieval helps, but not enough on its own** — V3 improves over V1 on multi-hop relevancy, yet still trails the rewriting and compression variants.

---

### 5.2.4 Comparative Analysis RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | n/a | 0.541 | 0.000 | 0.000 |
| V1 | 0.706 | 0.199 | 0.567 | 0.722 |
| V2 | 0.863 | 0.189 | 0.400 | 0.791 |
| V3 | n/a | 0.726 | 0.300 | 0.848 |
| V4 | n/a | 0.867 | 0.450 | 0.843 |
| V5 | 0.920 | 0.199 | 0.367 | 0.724 |
| V6 | n/a | 0.884 | 0.200 | 0.755 |

#### Key Observations

1. **Comparative queries clearly favour V6 and V4** — V6 achieves the highest comparative relevancy (0.884), followed by V4 (0.867), far ahead of the dense-only baselines.

2. **Grounding alone is not enough for comparisons** — V1 and V5 remain highly faithful but weakly relevant at 0.199 because they often fail to align the required periods side by side.

3. **Hybrid retrieval provides a strong intermediate baseline** — V3 reaches 0.726 relevancy, confirming that lexical anchoring helps comparative retrieval even before query rewriting or compression.

4. **Precision is not the deciding factor on comparative tasks** — V3-V6 all maintain high context precision, but the strongest results still depend on coverage and evidence alignment.

---

### 5.3 Numerical Accuracy by Category

| Category | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|----------|----|----|----|----|----|----|----|
| Factual Retrieval | 40% | 20% | 40% | 60% | 40% | 0% | 60% |
| Temporal Reasoning | 20% | 80% | 60% | 80% | 80% | 80% | 60% |
| Multi-Hop Reasoning | 20% | 0% | 40% | 40% | 0% | 0% | 40% |
| Comparative Analysis | 20% | 40% | 40% | 60% | 80% | 40% | 60% |

*Numerical Accuracy = fraction of questions where at least one key figure from the ground-truth answer (e.g. "$245.1 billion", "16%", "69.8%") appears verbatim in the generated answer. Computed by `evaluation/metrics.py::compute_numeric_match()`.*

Note: Faithfulness, Context Recall, and Context Precision are undefined for V0 (no retrieved context). V0 faithfulness of 0.098 reflects the saved RAGAS output on questions where the LLM produced partially verifiable claims from training memory. Numerical Accuracy = fraction of questions where key figures from the ground truth appear verbatim in the generated answer.

#### Key Observations

1. **Factual retrieval benefits most from hybrid retrieval and compression** — V3 and V6 achieve the highest factual numerical accuracy at 60%, while V5 drops to 0%.

2. **Temporal reasoning is broadly tractable once the right filing is retrieved** — V1, V3, V4, and V5 all reach 80% temporal numerical accuracy.

3. **Multi-hop reasoning remains the hardest category** — no variant exceeds 40%, showing that retrieval improvements alone do not solve multi-step synthesis.

4. **Comparative analysis benefits most from query rewriting** — V4 leads at 80%, with V3 and V6 next at 60%.

#### Key summary

Numerical accuracy highlights that retrieval coverage and query structuring (V3, V4, V6) are critical for complex queries, while specialised methods like metadata filtering (V5) and dense-only retrieval (V1) fail outside narrow use cases — particularly for multi-hop and comparative reasoning.

A notable anomaly persists: V0 (no retrieval) achieves 40% numerical accuracy on factual retrieval — higher than V1 (20%) and matching V2 and V4. This occurs because several factual questions concern Microsoft metrics that fall within the model's training data (pre-FY2024), allowing V0 to answer correctly from memory. This underscores the importance of evaluating retrieval systems on questions that are genuinely unanswerable without the indexed documents.

---

### 5.4 Latency vs. Accuracy Trade-off

| Variant | Avg Latency (s) | Faithfulness | Trade-off Summary |
|---------|-----------------|--------------|-------------------|
| V0 | 3.94 | 0.098 | Fast, completely ungrounded |
| V1 | 3.46 | 0.738 | Fastest RAG baseline; moderate grounding |
| V2 | 4.50 | 0.843 | Higher faithfulness with modest latency increase (reranking) |
| V3 | 7.71 | 0.786 | Higher latency from hybrid retrieval; moderate faithfulness |
| V4 | 8.51 | 0.760 | Query rewriting + hybrid retrieval; highest answer relevancy |
| V5 | 3.63 | 0.804 | Metadata filtering; strong low-latency grounding without reranking |
| V6 | 10.52 | 0.700 | Highest latency; compression overhead with reduced faithfulness |

#### Key observations

1. **V2 offers the strongest faithfulness-latency trade-off** — it reaches the highest faithfulness (0.843) while adding only about one second over V1.

2. **V5 is the strongest low-latency deployment candidate** — at 3.63s with 0.804 faithfulness, metadata filtering provides cheap grounding but remains brittle on cross-period tasks.

3. **V4 improves answer quality more than efficiency** — query rewriting helps push answer relevancy to 0.784, but the full pipeline still costs 8.51s on average.

4. **V3 and V6 show the cost of broader retrieval pipelines** — both are materially slower than V1/V2, and neither surpasses V2 on faithfulness.

5. **V1 remains the fastest grounded baseline, while V0 is fast but unusably ungrounded** — retrieval is still necessary for trustworthy answers.

#### Key Summary 

The most effective improvements come from precision-enhancing components (reranking, query rewriting), while recall-expanding components (hybrid retrieval, compression) introduce significant latency without consistent gains in faithfulness.

---

### 5.5 Ablation Study: Four-Configuration Component Analysis

The saved ablation run compares four retrieval configurations on a single benchmark question (`q001`) to isolate each component's effect:

| Method | Retrieval Strategy | Avg Latency (s) | Answer Quality | Key Observation |
|--------|--------------------|-----------------|----------------|-----------------|
| Dense-only | Dense (embedding-based) | 9.73 | High (correct + derived) | Strong semantic retrieval but inefficient and verbose|
| Sparse-only | BM25 (lexical) | 2.47 | Low (missing key figures) | Fast but fails on semantic matching |
| Hybrid (no rerank) | BM25 + Dense (RRF) | 7.23 | High (concise + correct) | Combines lexical + semantic strengths effectively |
| Hybrid + Rerank | Hybrid + Cross-encoder | 11.35 | High (slightly less precise numerically) | Improved ranking but diminishing returns vs cost |

### Key observations

1. Dense retrieval ensures completeness but introduces inefficiency
  * The dense-only setup successfully retrieves semantically relevant chunks, enabling correct numerical derivations (e.g., gross margin calculation). However, its high latency (9.73s) and verbose reasoning indicate over-retrieval, where the model compensates for imperfect ranking by generating longer answers.
    * This suggests that dense retrieval alone lacks precision in ordering the most relevant chunks early.

2. Sparse retrieval is fast but insufficient for semantic queries
  * BM25 achieves the lowest latency (2.47s), but fails to retrieve the exact figures required to answer the question. Instead, it surfaces loosely related financial commentary (e.g., margin trends), demonstrating that lexical matching alone cannot handle paraphrased or numerically grounded queries. This confirms that sparse retrieval lacks semantic coverage for financial QA tasks.

3. Hybrid retrieval (without reranking) provides the best balance of accuracy and efficiency
  * The hybrid approach successfully retrieves both the exact financial table (via lexical signals) and supporting semantic context (via embeddings). This results in a concise and correct answer with moderate latency (7.23s). The absence of reranking does not significantly harm answer quality, indicating that Reciprocal Rank Fusion (RRF) is already effective at prioritizing relevant documents.

4. Reranking improves ordering but shows diminishing returns in end-task performance
  * Adding cross-encoder reranking increases latency significantly (11.35s) while yielding only marginal gains in retrieval precision. In this case, it slightly degrades numerical accuracy (rounding to 70% instead of ~69.8%), suggesting that better-ranked context does not always translate to better generation. This highlights a key trade-off: reranking optimizes retrieval metrics more than final answer correctness.

5. Lexical signals are critical for numerical extraction tasks
  * Both hybrid variants outperform dense-only retrieval in producing concise, correct answers. This is because financial figures (e.g., “171,008”, “245,122”) are better matched through exact token overlap rather than semantic similarity. The failure of sparse-only retrieval, however, shows that lexical matching must be complemented by semantic retrieval to ensure coverage.

6. Overall: Hybrid retrieval without reranking is the most cost-effective configuration
  * Considering both latency and answer quality, the hybrid (no rerank) setup achieves the best trade-off. It captures the benefits of multi-retriever fusion while avoiding the computational overhead of cross-encoder reranking, making it a strong baseline for production systems.

```bash
# Run the full 4-configuration ablation study
python evaluation/ablation_study.py

# Quick test (5 questions)
python evaluation/ablation_study.py --limit 5
``` 

---

## 6. Qualitative Error Analysis

### 6.1 Error Taxonomy (Five Failure Types)

Following the framework defined in the outline (§6.4):

| Failure Type | Description |
|--------------|-------------|
| **Retrieval Failure** | Relevant chunk not retrieved at all |
| **Ranking Failure** | Relevant chunk retrieved but ranked too low to appear in top-k |
| **Chunking Failure** | Information split across chunk boundaries, breaking coherence |
| **Query Understanding Failure** | Query ambiguity or under-specification prevents correct retrieval |
| **Generation Failure** | LLM misinterprets context or generates unsupported claims |

### 6.2 Component–Failure Mapping

The earlier qualitative matrix overstated some failure types. The table below is **derived from** `evaluation/results/eval_results.json` by running the heuristic classifier in `evaluation/category_analysis.py`; the saved materialised summary lives in [evaluation/results/category_report.json](evaluation/results/category_report.json). Under that derived analysis, the per-variant mapping is:

| Failure Type | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| Retrieval Failure | 0 | 2 | 6 | 2 | 2 | 3 | 3 |
| Query Understanding Failure | 0 | 5 | 1 | 0 | 0 | 4 | 0 |
| Generation Failure | 11 | 0 | 1 | 0 | 1 | 0 | 0 |
| Ranking Failure | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Chunking Failure | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

This makes the component pattern clearer, but it should still be read as **heuristic failure labeling rather than definitive root-cause attribution**. All 11 V0 failures were classified as generation failures; V1 and V5 are dominated by query-understanding failures; and V2, V3, V4, and V6 are dominated by retrieval failures in the saved classifier output.

Two caveats matter. First, **ranking failure** and **chunking failure** appear as zero not because those issues are impossible, but because the current heuristic in [evaluation/category_analysis.py](evaluation/category_analysis.py) rarely surfaces them as separate labels; most such cases are absorbed into retrieval or generation failure. Second, the table should be read as a mapping of *observed dominant failure behaviour in this experiment*, not as a universal statement about what each component can or cannot do.

From a systems perspective, the main takeaway is that adding retrieval components reduced pure hallucination sharply after V0, but the new bottleneck became **evidence selection and evidence alignment**, not raw answer generation. That shift is exactly what the case studies show: the stronger pipelines usually fail by missing or misassembling evidence, not by inventing unsupported claims from scratch.

### 6.3 Case Studies

All answers below are abridged excerpts from `evaluation/results/eval_results.json`, with truncation only for readability.

---

#### Case 1: Knowledge-Cutoff Failure Without Retrieval — V0 vs V1 (q001)

**Question:** "What was Microsoft's gross profit and gross margin percentage for FY2024?"

**Ground truth:** Gross profit $171.0B on revenue of $245.1B → gross margin ≈ 69.8%.

**V0 (LLM-Only) actual answer:**
> "As of my last knowledge update in October 2023, I do not have the specific figures for Microsoft's gross profit and gross margin percentage for FY2024, as that fiscal year had not concluded and the financial results had not been publicly reported at that time. For the most accurate and up-to-date information, I recommend checking Microsoft's official financial statements or their investor relations website."

**V1 (Baseline) actual answer:**
> "For FY2024, Microsoft's total revenue was $245,122 million, and the total cost of revenue was $74,114 million. Gross Profit = $245,122M − $74,114M = $171,008 million. Gross Margin Percentage ≈ 69.7% [Doc-3]."

**Assessment:** V0 does not hallucinate a number here; instead, it refuses because its parametric knowledge predates the FY2024 filing. V1 retrieves the income statement directly and computes the answer correctly. This is still a clear demonstration of why retrieval is non-negotiable for post-cutoff financial questions.

**Failure type:** Generation failure in the saved classifier output; operationally, this case is a stale-knowledge refusal that is fixed entirely by basic dense retrieval (V1).

---

#### Case 2: Temporal Retrieval Failure — V1 vs V3 (q006)

**Question:** "Did Azure's constant-currency growth rate accelerate or decelerate from Q1 FY2025 to Q2 FY2025, and by how much?"

**Ground truth:** Decelerated from 33% (Q1 FY2025) to 31% (Q2 FY2025) — a 2 pp decrease.

**V1 (Baseline) actual answer:**
> "The provided context does not contain data specifically for Q2 FY2025. The context includes Q1 FY2025 data which shows that Azure and other cloud services revenue grew 33% … There is no information available regarding Azure's growth rate for Q2 FY2025."

**V3 (Hybrid) actual answer:**
> "Q1 FY2025: Azure and other cloud services revenue grew by **33%** [Doc-3]. Q2 FY2025: Azure and other cloud services revenue grew by **31%** [Doc-1]. This indicates that Azure's growth rate **decelerated** from Q1 FY2025 to Q2 FY2025 by **2 percentage points**."

**Assessment:** V1 correctly refuses to hallucinate but retrieves only the Q1 filing — the Q2 quarterly report is semantically indistinguishable from Q1 to the dense encoder. V3's BM25 component anchors on the exact strings "Q1 FY2025" and "Q2 FY2025", pulling both filings into context and producing a perfect answer.

**Failure type:** Retrieval failure (fiscal period discrimination) in V1. Fixed by hybrid retrieval in V3.

---

#### Case 3: Wrong-Evidence Retrieval Regression — V3 vs V1 (q011)

**Question:** "What was Microsoft's operating margin in FY2024 compared to FY2022, and what drove the change?"

**Ground truth:** FY2024 ≈ 44.6% vs FY2022 ≈ 42.1% — a 2.5 pp improvement driven by mix shift toward cloud services.

**V1 (Baseline) actual answer:**
> "In FY2024, operating income was $109,433M on revenue of $245,122M → Operating Margin ≈ 44.7%. In FY2022, operating income was $83,383M on revenue of $198,270M → Operating Margin ≈ 42.0%. The change is approximately 2.7 percentage points [Doc-4][Doc-9]."

**V3 (Hybrid) actual answer:**
> "In FY2024, Microsoft's operating income increased by $20.9 billion or 24% … The operating margin for FY2024 can be inferred from the increase in operating income, but the **exact operating margin percentage is not explicitly stated** in the provided context … a direct comparison cannot be made."

**Assessment:** V3 retrieves growth-rate percentage chunks ("operating income increased 24%") which are semantically close to "operating margin" but computationally insufficient — the generator cannot derive the margin ratio from a growth-rate statement. V1's simpler dense retrieval surfaces the absolute income statement rows, which the generator correctly uses to compute ratios. This is a regression: a more sophisticated pipeline performs worse because RRF fusion promoted the wrong chunk type.

**Failure type:** Retrieval failure in the saved heuristic classifier output. Qualitatively, this is a wrong-evidence-type regression: V3 retrieved percentage-change text instead of the absolute figures needed for margin calculation, and the answer failed downstream as a result.

---

#### Case 4: Comparative Analysis Remained Brittle — V1 vs V3 vs V4 (q016)

**Question:** "How did the cumulative revenue growth of Microsoft's three business segments compare over the FY2022 to FY2024 period?"

**Ground truth:** Intelligent Cloud +40% ($75.3B → $105.4B); Productivity +22% ($63.4B → $77.7B); More Personal Computing +4% ($59.6B → $62.0B).

**V1 (Baseline) actual answer:**
> "The provided context only contains data for FY2022 and does not include any information for FY2023 or FY2024. Therefore, I cannot provide a comparison of cumulative revenue growth … The provided documents do not contain sufficient information to answer this question."

**V3 (Hybrid) actual answer:**
> "FY2022 showed a higher growth rate of 18% compared to FY2024's growth rate of 12%. This indicates that while all segments continued to grow, the rate of growth slowed from FY2022 to FY2024."

**V4 (Query Rewriting) actual answer:**
> "FY2022 Revenue: Productivity and Business Processes $69,274M, Intelligent Cloud $87,907M, More Personal Computing $54,734M … FY2024 Revenue: Productivity and Business Processes $85,000M, Intelligent Cloud $137.4B, More Personal Computing $57,000M … cumulative growth 31.8%."

**Assessment:** This question is harder than simple multi-document retrieval because it requires the system to align the same segment table across two fiscal endpoints and then compute per-segment cumulative growth. V1 fails first at retrieval coverage, surfacing only one endpoint and therefore refusing. V3 retrieves more evidence but answers the wrong question: it compares overall company growth rates in FY2022 and FY2024 instead of segment-level cumulative growth from FY2022 to FY2024. V4 is worse in a different way: it produces a confident structured answer, but mixes incompatible figures, treats Microsoft Cloud totals as if they were segment revenue, and invents intermediate values such as "$85,000M" and "$57,000M." In other words, more aggressive retrieval expansion did not solve the task; it increased the opportunity for structured hallucination once evidence alignment broke down.

**Failure type:** Manual qualitative interpretation rather than the saved heuristic classifier output: V1 shows a retrieval-coverage failure; V3 answers the wrong comparison despite broader evidence; and V4 generates a confident but unsupported synthesis on noisy multi-document evidence. This case explains why comparative analysis remained brittle even after adding hybrid retrieval and query rewriting.

---

### 6.4 Error Distribution

Using the automatic classifier in [evaluation/category_analysis.py](evaluation/category_analysis.py) on `evaluation/results/eval_results.json`, the derived analysis assigns **30 total failures across V1–V6**. The distribution is more concentrated than the earlier qualitative estimate suggested:

| Failure Type | Count | Share of Classified Failures | Main Concentration |
|--------------|-------|------------------------------|--------------------|
| **Retrieval Failure** | 18 | 60.0% | Temporal reasoning (8), comparative analysis (5), multi-hop reasoning (4) |
| **Query Understanding Failure** | 10 | 33.3% | Multi-hop reasoning (6), comparative analysis (4) |
| **Generation Failure** | 2 | 6.7% | Multi-hop reasoning only |
| **Ranking Failure** | 0 | 0.0% | Not surfaced separately by the current heuristic |
| **Chunking Failure** | 0 | 0.0% | Not surfaced separately by the current heuristic |

Two patterns stand out. First, **retrieval failure is the dominant bottleneck**, accounting for three-fifths of all classified errors. This is especially visible in temporal questions, where systems often retrieved only one fiscal period and then refused rather than hallucinating, as in q006. Second, **query-understanding failure is concentrated in multi-hop and comparative questions**, where the model must interpret an underspecified request, align multiple years or segments, and retrieve the right evidence for each sub-part of the comparison.

The near-zero counts for ranking and chunking should not be interpreted as proof that those problems never occurred. Rather, the current rule-based classifier in [evaluation/category_analysis.py](evaluation/category_analysis.py) tended to absorb those cases into retrieval or generation failure unless there was a very clear observable signal. In other words, the automatic counts are most reliable for identifying the dominant failure families, not for perfectly separating every low-level cause.

At the variant level, the weakest pipelines were **V2** and **V5** for retrieval-related misses, while **V1** and **V5** accounted for most query-understanding failures. By contrast, **V3**, **V4**, and **V6** reduced query-understanding failures substantially, but they did not eliminate retrieval misses entirely. This matches the case studies above: better retrieval breadth helps, but complex comparative questions still fail when evidence must be aligned and reasoned over consistently.

---

## 7. Discussion & Insights

### 7.1 Component × Query Type Sensitivity

| Component | Factual | Temporal | Multi-Hop | Comparative |
|-----------|---------|----------|-----------|-------------|
| Dense Retrieval (V1) | ✓✓✓ | ✗ | ✗ | ✗ |
| + Reranking (V2) | ✓ | ✗ | ✗ | ✗ |
| + Hybrid Retrieval (V3) | ✓ | ✓✓✓ | ✓ | ✓✓ |
| + Query Rewriting (V4) | ✗ | ✓✓ | ✓✓ | ✓✓✓ |
| + Metadata Filtering (V5) | ✓ | ✓✓ | ✗ | ✗ |
| + Context Compression (V6) | ✗ | ✓ | ✓✓✓ | ✓✓✓ |

✓✓✓ = primary strength, ✓✓ = strong improvement, ✓ = moderate improvement, ✗ = no improvement or degradation

These labels are based on the saved category-level outcomes in Section 5, prioritising answer relevancy and numerical accuracy over intuition about what a component "should" help.

**Core Finding:** No single pipeline variant dominates across all query types, confirming the study hypothesis. Optimal performance requires adaptive pipeline selection based on query characteristics.

---

### 7.2 Component-Level Insights

#### Insight 1: Hybrid Retrieval Is Essential for Financial Queries
Financial queries contain specific lexical markers — "FY2024", "Q1 FY2025", "$245.1 billion", "10-K" — that pure dense retrieval struggles to anchor on. BM25's exact term matching complements semantic search:

- **Dense-only (V1)** achieves 0.483 answer relevancy — it struggles with fiscal period discrimination and exact figure matching
- **Hybrid (V3)** reaches 0.711 answer relevancy, a roughly 47% improvement, by combining semantic context understanding with BM25 lexical anchoring

The effect is strongest in comparative analysis: V3 achieves 0.726 comparative relevancy vs V1's 0.199, a 3.6× improvement driven by BM25 surfacing both fiscal periods simultaneously.

#### Insight 2: Reranking Delivers the Highest Faithfulness and Precision
V2 adds about 1.04 seconds over V1 (4.50s vs 3.46s) while improving context precision from 0.607 to 0.741 and faithfulness from 0.738 to 0.843. V2 achieves the highest faithfulness of all seven variants, meaning its answers are the most tightly grounded in the retrieved context.

The cross-encoder re-scores each query–chunk pair directly, filtering the noisiest candidates without requiring the full overhead of hybrid retrieval.

#### Insight 3: Metadata Filtering (V5) Achieves the Best Latency Efficiency
V5 is the second-fastest RAG variant at 3.63s average latency — just behind V1 (3.46s) and far below the hybrid variants — while maintaining 0.804 faithfulness. Pre-filtering the ChromaDB search space by fiscal period metadata before running dense retrieval reduces retrieval noise without introducing reranking cost.

Notably, **V5 intentionally omits the cross-encoder reranker** — the metadata filter is designed as a lower-cost substitute, confirmed by reranking latency = 0ms across all 20 evaluated questions.

The trade-off: V5 requires consistent metadata tagging at ingest time. Its performance collapses on multi-hop queries spanning two fiscal years — reflected in 0% multi-hop numerical accuracy — because pre-filtering by one period excludes the other.

#### Insight 4: Query Rewriting (V4) Strengthens Comparative Coverage
V4 raises aggregate answer relevancy to 0.784 and delivers 80% comparative numerical accuracy, one of the joint-highest category-specific numerical accuracy scores in the study. Its context recall (0.463) is solid but not the highest overall, and the added query-rewrite + hybrid pipeline raises latency to 8.51s.

V4 achieves 80% comparative numerical accuracy — tied with the best temporal results in the saved run — confirming its particular strength for multi-period synthesis questions.

#### Insight 5: Context Compression (V6) Targets Multi-Hop and Long-Context Failures
V6 applies sentence-level extraction after reranking to distil financially relevant content before generation. It reaches 0.729 multi-hop relevancy, 0.884 comparative relevancy, and 55% aggregate numerical accuracy, but it is also the slowest variant at 10.52s. Compression is most beneficial when relevant evidence is scattered across many chunks and the generator needs a cleaner final context.

---

### 7.3 Query Type Conclusion Summary

| Variant | Primary Benefit (Query-Type Specific) | Best Query Type | Justification (Based on Pipeline) |
|---------|--------------------------------------|-----------------|-----------------------------------|
| V0 | Baseline for simple factual recall without grounding | Factual (limited) | No retrieval → relies on parametric knowledge; can partially answer common factual queries but fails on structured or time-specific questions |
| V1 | Reliable single-document grounding for direct lookups | Factual | Dense retrieval surfaces semantically relevant chunks, which is sufficient for single-hop factual queries located within one document |
| V2 | High-faithfulness grounding for precision-focused queries | Factual / Grounding | Reranking improves context precision and faithfulness most strongly on queries where the main challenge is selecting the cleanest evidence, not assembling many pieces across periods |
| V3 | Strong cross-period retrieval for temporal and comparative queries | Temporal, Comparative | BM25 captures exact fiscal terms (e.g., “Q1 FY2025”) while dense retrieval captures semantics; RRF fusion ensures both periods are retrieved together |
| V4 | Structured retrieval for comparative and multi-document queries | Comparative, Multi-Hop | Query rewriting expands ambiguous questions into clearer retrieval targets, which is especially helpful for cross-period comparisons and multi-document synthesis |
| V5 | Fast and precise retrieval for single-period temporal queries | Temporal, Factual | Metadata filtering restricts search to a specific fiscal period, improving precision and speed when the query targets a known timeframe |
| V6 | Noise-reduced context for complex multi-hop and comparative reasoning | Multi-Hop, Comparative | Compression removes irrelevant text after retrieval, helping the model focus on key facts needed to synthesise answers across multiple sources |

Each variant’s benefit is most pronounced when its pipeline modification directly addresses the dominant challenge of the query type — whether it is locating the correct document (V3, V5), structuring the query (V4), selecting precise evidence (V2), or filtering noise during synthesis (V6).

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

| Limitation | Impact | Severity |
|------------|--------|----------|
| Fixed-size chunking | Financial tables may split across boundaries | High |
| No table-aware parsing | Numeric data extraction is unreliable for dense tables | High |
| Single company scope | Findings not validated on other filings | Low |
| Metadata dependency in V5 | Requires consistent metadata tagging at ingest | Medium |
| Small benchmark (20 questions) | Limits statistical significance of category comparisons | Medium |
| LLM variability | Results vary slightly across runs despite fixed seed | Low |

### 8.2 Future Work

1. **Table-Aware Chunking:** Implement table detection during PDF parsing to keep financial statement rows intact within chunks

2. **Semantic Chunking:** Section-aware chunking that respects document structure (MD&A, Risk Factors, Financial Statements) for better multi-hop retrieval

3. **Adaptive Pipeline Routing:** Classify query type at runtime and dynamically select the best-performing variant (e.g., route temporal queries to V5, multi-hop to V6)

4. **Expanded Benchmark:** Increase to 50+ questions with real-user queries collected from financial analysts to reduce synthetic benchmark bias

5. **Fine-tuned Embeddings:** Fine-tune the embedding model on financial terminology to improve discrimination between fiscal periods and segment names

6. **Retrieval-level metrics:** Implement chunk-level annotation to measure metrics like MRR and top-k hit rate

---

## 9. Risks and Guardrails

### 9.1 Risk Identification

| Risk | Description | Severity |
|------|-------------|----------|
| **Hallucination** | LLM generates plausible-sounding but incorrect financial figures not found in retrieved context | Critical |
| **Stale data** | System answers questions using outdated filings when newer filings have been published | High |
| **Retrieval miss** | Relevant chunks not retrieved, causing the LLM to answer from parametric memory | High |
| **Numerical precision errors** | LLM rounds or paraphrases figures (e.g., "$245 billion" instead of "$245.1 billion") | Medium |
| **Out-of-scope queries** | User asks about non-Microsoft companies or topics not covered in indexed filings | Medium |
| **Adversarial queries** | Deliberately ambiguous queries designed to extract unsupported comparisons | Low |

### 9.2 Implemented Mitigations

1. Prompt-level insufficient-evidence handling (implemented in `config/prompts.yaml` and `src/generation/generator.py`)
* The system prompt explicitly instructs the generator not to guess and to return an insufficient-evidence answer when the retrieved excerpts do not support the requested claim.
  * This behavior is visible in evaluation cases such as V1 on q016, where the model refuses rather than fabricating a cross-period comparison.

2. Citation requirement and citation formatting (implemented in `config/prompts.yaml`, `src/generation/generator.py`, and `src/generation/citation_formatter.py`)
* The prompt requires inline `[Doc-N]` citations for factual claims, and the citation formatter maps those references back to chunk metadata for inspection.
  * This makes grounding auditable even when an answer is only partially correct.

3. Scope filtering (implemented in `src/generation/generator.py` with topic controls from `config/settings.yaml`)
* Questions that do not match the allowed Microsoft-focused topic list are rejected during answer generation, reducing the risk of unsupported answers on out-of-scope prompts.

4. Temperature = 0 (configured in `config/settings.yaml`)
* The generator runs at temperature 0.0 to minimise stochastic variation in numerical outputs. 
  * This is especially important for financial figures where rounding behaviour at higher temperatures introduces numerical inconsistency.

### 9.3 Residual Risks and Mitigations Not Yet Implemented

| Residual Risk | Proposed Mitigation |
|---------------|---------------------|
| Stale filings | Automated ingestion pipeline to detect and index new EDGAR filings on release |
| Numerical precision | Post-processing step to verify that quoted figures match source chunk text verbatim |
| Adversarial prompts | Input sanitisation and query intent classification before retrieval |
| Index drift | Periodic re-embedding when the embedding model is updated |

---

## 10. Conclusion

### Study Hypothesis

This project demonstrates that **RAG components provide selective, query-type-dependent benefits** — confirming the study hypothesis that no single pipeline is universally optimal.

### Overall Results

| Finding | Evidence |
|---------|----------|
| RAG vs. no RAG: critical for grounding | V0 faithfulness = 0.098 → V1 faithfulness = 0.738 |
| Query rewriting achieves the highest overall answer relevancy | V4 = 0.784 vs V1 = 0.483 and V3 = 0.711 |
| Reranking delivers the strongest grounding quality | V2 has the highest faithfulness (0.843) and context precision (0.741) |
| Hybrid retrieval achieves the best aggregate numerical accuracy | V3 = 0.600, ahead of V6 = 0.550 and V4 = 0.500 |
| Dense-only retrieval remains the fastest RAG baseline | V1 = 3.46s, slightly faster than V5 = 3.63s |
| Comparative and multi-hop questions remain the most demanding | V1 scores only 0.199 on comparative relevancy and 0.357 on multi-hop relevancy, while advanced variants are required to recover performance |

### Recommendations for Financial RAG Systems

1. Use hybrid retrieval (dense + BM25) for fiscal-period-specific and cross-period comparison queries, where lexical anchors such as fiscal years and quarter labels matter.
2. Apply cross-encoder reranking for highest context precision (V2, 0.741).
3. Implement metadata pre-filtering as a low-latency precision improvement for narrowly scoped temporal queries with a clearly identifiable fiscal period.
4. Use query rewriting (V4) or context compression (V6) for comparative and multi-hop questions.
5. Maintain an LLM-only baseline to quantify the value added by each retrieval component, and ensure benchmark questions are genuinely unanswerable from model training data alone.

FinSight demonstrates that thoughtful, component-level RAG design — combined with query-type-aware evaluation — produces both better systems and clearer research insights than aggregate benchmarking alone.

---

## 11. Appendix

### A. Hyperparameters

```yaml
embeddings:
  model: sentence-transformers/all-mpnet-base-v2
  batch_size: 32
  normalize: true

retrieval:
  baseline_top_k: 5
  dense_top_k: 25
  sparse_top_k: 25
  rerank_top_k: 20
  final_context_k: 10
  rrf_k: 60
  weak_evidence_threshold: -3.0

reranker:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  max_length: 512

generation:
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 512
  timeout_seconds: 120
```

### B. Benchmark Dataset Categories

| Category | Question IDs | What It Tests |
|----------|-------------|---------------|
| Factual Retrieval | q001–q005 | Gross margin, R&D expense, net income, operating income, revenue — diverse metrics across different filing years |
| Temporal Reasoning | q006–q010 | Sequential quarter comparisons, YoY delta, QoQ acceleration/deceleration |
| Multi-Hop Reasoning | q011–q015 | Operating margin FY22 vs FY24, segment growth ranking, slowest growth year, revenue share shift, PBP growth acceleration |
| Comparative Analysis | q016–q020 | Cumulative segment growth, margin trends, CAGR, Azure vs company divergence |

### C. Reproducing Results

```bash
# 1. Fast path: rebuild indexes from the committed processed artefacts if needed
python scripts/build_index.py

# 1b. Full raw-to-index rebuild (requires placing the 9 PDFs in data/raw/ first)
python scripts/ingest_all.py --force
python scripts/build_index.py --reset

# 2. Full evaluation — all 7 variants with RAGAS
python evaluation/run_evaluation.py

# 3. Skip RAGAS for fast Q&A verification
python evaluation/run_evaluation.py --skip-ragas

# 4. Single variant
python evaluation/run_evaluation.py --variants v3_advanced_b

# 5. Ablation study
python evaluation/ablation_study.py

# 6. Category and error analysis
python evaluation/category_analysis.py

```

### D. Reproducibility Controls

- **Random seed:** 42
- **Python version:** 3.11
- **Runtime configuration:** core parameters are set via `config/settings.yaml`, `config/prompts.yaml`, and `config/chunking.yaml`, though a small number of helper defaults remain hardcoded in code
- **Key dependencies:** see `requirements.txt`
- **ChromaDB SQLite compatibility:** handled automatically by `chromadb_compat.py`

---

*Report generated: April 2026*
*FinSight — IS469 Final Project*
