# FinSight: RAG-based Financial Filings QA System
## 1. Problem Motivations & Objectives
### Problem Statement
Financial filings (e.g., SEC 10-K and 10-Q reports) are lengthy, complex, and difficult for users to navigate efficiently. Extracting specific financial insights—such as revenue trends, segment performance, or risk disclosures—requires significant manual effort and domain knowledge.

### Objective
To develop FinSight, a domain-specific question-answering (QA) system over Microsoft SEC filings using Retrieval-Augmented Generation (RAG) techniques. The system aims to:
- Provide accurate, citation-backed answers to financial questions
- Compare multiple RAG architectures to evaluate performance trade-offs under varying query types

### Study Hypothesis
A systematic study of how different RAG components behave under different query types


* Different RAG components provide selective benefits depending on query types, rather than uniformly improving performance

## 2. Project Scope
This project implements & evaluates 1 base LLM and 6 RAG variants.
| Variant | Pipeline | Description |
|---------|----------|-------------|
| **V0 LLM-only** | User Query → Generate | No retrieval (LLM-only baseline); tests hallucination and lack of grounding |
| **V1 Baseline** | Dense → Generate | Fixed-size chunking + embedding retrieval; basic RAG baseline |
| **V2 Advanced A** | Dense → Rerank → Generate | Dense retrieval + cross-encoder reranking; improves precision at top-k |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion + reranking; improves recall and overall robustness |
| **V4 Advanced C** | Query Rewrite → BM25 + Dense → RRF → Rerank → Generate | LLM-based query rewriting to improve retrieval for ambiguous or underspecified queries |
| **V5 Advanced D** | Metadata Filter → Dense → Generate | Metadata-aware filtering (e.g., year, document type) before retrieval; improves precision for structured queries |
| **V6 Advanced E** | BM25 + Dense → RRF → Rerank → Context Compression → Generate | Context filtering/compression to remove irrelevant information before generation; improves faithfulness and reduces noise |

Each variant introduces a single additional capability to isolate its impact on retrieval quality, answer accuracy, and robustness across query types.


## 3. System Architecture
### Project Structure
```
finsight/
├── config/
│   ├── settings.yaml        # Main configuration
│   ├── chunking.yaml        # Chunking experiment configs
│   └── prompts.yaml         # All prompt templates
├── data/
│   ├── raw/                 # Microsoft SEC filing PDFs (not committed to git)
│   ├── processed/           # Chunked + tagged JSON per document
│   └── metadata/            # Metadata schema
├── indexes/
│   ├── chroma/              # ChromaDB vector store (not committed)
│   └── bm25/                # BM25 index pickle files (not committed)
├── src/
│   ├── ingestion/           # PDF parsing, cleaning
│   ├── chunking/            # Chunking strategies, metadata tagging
│   ├── indexing/            # ChromaDB + BM25 index builders
│   ├── retrieval/           # Dense, sparse, hybrid retrievers + reranker
│   ├── generation/          # LLM generator + citation formatter
│   ├── pipeline/            # V1/V2/V3 end-to-end pipelines
│   └── utils/               # Config loader, logger
├── evaluation/
│   ├── eval_dataset.json    # 20-question benchmark (4 categories)
│   ├── run_evaluation.py    # RAGAS evaluation runner
│   └── results/             # JSON result files per run
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── scripts/
│   ├── ingest_all.py        # Full ingestion pipeline
│   ├── build_index.py       # Index builder
│   ├── run_query.py         # CLI query tool
│   └── smoke_test.py        # Quick sanity check
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_chunking_experiment.ipynb
    └── 03_retrieval_debug.ipynb
```

### Pipeline Overview
```
User Query
   ↓
Query Processing
   ↓
Retriever
   ├── Dense (V1, V2)
   └── Hybrid BM25 + Dense (V3)
   ↓
Fusion (RRF, V3 only)
   ↓
Reranker (V2, V3)
   ↓
Top-k Retrieved Chunks
   ↓
LLM Generator
   ↓
Answer + Citations
```

### Data Flow
1. Ingestion: Parse SEC filing PDFs into structured texts
2. Chunking: Split documents into fixed-sized segments with metadata
3. Indexing:
    * Dense embeddings -> vector store (ChromaDB)
    * Sparse Indexing -> BM25
4. Retrieval: Dense / Sparse / Hybrid retrieval
5. Generation: LLM produces answers using retrieved content
6. Output: Answers with source citations

## 4. Dataset
### Training Data
The following Microsoft SEC filings are indexed (already downloaded to `data/raw/`):

| Filename | Document | Period |
|----------|----------|--------|
| `msft_10k_fy2022.pdf` | Annual Report (10-K) | FY2022 (ended Jun 30, 2022) |
| `msft_10k_fy2023.pdf` | Annual Report (10-K) | FY2023 (ended Jun 30, 2023) |
| `msft_10k_fy2024.pdf` | Annual Report (10-K) | FY2024 (ended Jun 30, 2024) |
| `msft_10k_fy2025.pdf` | Annual Report (10-K) | FY2025 (ended Jun 30, 2025) |
| `msft_10q_q1_fy2025.pdf` | Quarterly Report (10-Q) | Q1 FY2025 (Sep 2024) |
| `msft_10q_q2_fy2025.pdf` | Quarterly Report (10-Q) | Q2 FY2025 (Dec 2024) |
| `msft_10q_q3_fy2025.pdf` | Quarterly Report (10-Q) | Q3 FY2025 (Mar 2025) |
| `msft_10q_q1_fy2026.pdf` | Quarterly Report (10-Q) | Q1 FY2026 (Sep 2025) |
| `msft_10q_q2_fy2026.pdf` | Quarterly Report (10-Q) | Q2 FY2026 (Dec 2025) |

Source: [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=789019) / [Microsoft Investor Relations](https://investor.microsoft.com/sec-filings)

**Note:** Microsoft fiscal year ends June 30. Q1=Jul–Sep, Q2=Oct–Dec, Q3=Jan–Mar, Q4=Apr–Jun.

Characteristics:
* Long-form financial documents
* Structured but noisy text (tables, footnotes)
* Requires numerical reasoning, and cross-section referencing


### Testing Data
????

## **5. Methodology (Detailed Design & Rationale)**

### **5.1 Study Design Overview**
Framed as a controlled experiment to evaluate how different RAG components affect performance across varying query types. Each variant introduces one additional capability to isolate its effects on:
* Retrival quality
* Answer accuracy
* Robustness across query types

| Component          | Variants Tested              |
| ------------------ | ---------------------------- |
| Retrieval Strategy | Dense, Hybrid (BM25 + Dense) |
| Query Processing   | Raw, Query Rewriting         |
| Context Selection  | Top-k, Reranked              |
| Context Quality    | Raw, Compressed              |

### **5.2 Document Processing & Chunking Strategy**
The ingestion pipeline converts SEC filing PDFs into structured text using a parsing module that removes boilerplate artifacts (headers, footers, page numbers) and preserves section-level structure where possible.

**a. Fixed-size Chunking (Baseline)**
* Chunk-size: 500–800 tokens with 10-20% overlap to preserve context continuity
* Metadata tagging:
  * Document type (10-K / 10-Q)
  * Fiscal year / quarter
  * Section labels (if identifiable, e.g., “Risk Factors”, “MD&A”)
* Rationale:
   * Fixed-size chunking ensures consistent embedding quality
   * Retrieval efficiency for large-scale indexing
   * Overlap reduces boundary fragmentation (important for financial narratives)

**b. Semantic Chunking (Experimental)**
* Split documents based on logical sections and semantic boundaries
* Rationale:
   * Preserves contextual coherence
   * Better supports multi-hop reasoning

**Trade-offs**
| Approach   | Strength           | Weakness                    |
|------------|--------------------|-----------------------------|
| Fixed-size | Efficient, uniform | May break context           |
| Semantic   | Coherent reasoning | Less consistent chunk sizes |
* May split tables or financial statements incorrectly
* May reduce coherence for multi-paragraph reasoning

---

### **5.3 Retrieval Strategies**

#### **a. Dense Retrieval (V1, V2, V5)**
* Uses embedding model to encode query and document chunks
* Retrieves top-k semantically similar chunks via vector similarity
* Strengths:
   * Captures semantic meaning (e.g., “cloud revenue” ≈ “Azure growth”)
* Weaknesses:
   * May miss exact keyword matches (e.g., specific financial terms or figures)

---

#### **b. Sparse Retrieval (BM25) (V3, V4, V5)**
* Lexical matching based on term frequency and inverse document frequency
* Strengths:
   * Strong for keyword-heavy or precise queries (e.g., “FY2024 revenue”)
* Weaknesses:
   * Cannot capture semantic relationships

---

#### **c. Hybrid Retrieval with RRF (V3, V4, V6)**
* Combines Dense + BM25 results
* Uses **Reciprocal Rank Fusion (RRF)** to merge rankings
   * Score = Σ (1 / (k + rank_i))
* Rationale:
   * Improves recall by combining semantic + lexical signals
   * Reduces risk of missing relevant chunks
   * Better performacne on multi-hop queries and ambiguous queries
* Trade-off:
   * Introduces noise which requires reranking

---

#### **d. Metadata-Aware Retrieval (V5)**
* Applies structured filtering before retrieval:
   * Fiscal year
   * Section
   * Document type
* Rationale:
   * Reduce irrelevant search space
   * Improves precision for temporal and structured queries

---

### **5.4 Query Processing (V4)**
**Purpose**
* Improve retrieval for:
   * Ambiguous queries
   * Underspecified queries

**Expected Behaviour**
| Query Type | Effect                  |
|------------|-------------------------|
| Factual    | Minimal improvement     |
| Ambiguous  | Significant improvement |
| Multi-hop  | Moderate improvement    |

---

### **5.5 Reranking Module (V2, V3, V4, V6)**
* Cross-encoder model evaluates query–chunk pairs
* Produces relevance scores for final ranking

**Purpose:**
* Improve **precision at top-k**
* Filter noisy results from hybrid retrieval
* Analyse trade-offs: increased latency and additional compute cost

**Expected Behaviour**
| Query Type | Effect   |
|------------|----------|
| Factual    | Moderate |
| Multi-hop  | High     |
| Ambiguous  | High     |

---

### **5.6 Context Processing (V6)**
**Context Compression**
* Filters or summarises retrieved chunks before generation
* Rationale:
   * Financial filings contain redundant and noisy text
   * Reducing context improves signal-to-noise ratio

**Expected Behaviour**
| Query Type | Effect   |
|------------|----------|
| Factual    | Minimal  |
| Multi-hop  | High     |
| Ambiguous  | Moderate |

---

### **5.7 Generation Module**
* LLM generates answers conditioned on retrieved context

**Key Design Choices:**
* Low temperature → reduce hallucination
* Structured prompts → enforce answer format
* Context-limited answering → ensure grounding

**Output Requirements:**
* Direct answer
* Supporting citations (document + section (if any))

---

### **5.8 Guardrails & Safety Mechanisms**

**Implemented Controls:**
* Out-of-scope detection (reject unrelated queries)
* Mandatory citation enforcement
* Context-limited answering (no external knowledge)
* Rationale:
   * Financial QA requires high factual reliability
   * Prevent hallucinated financial figures
   * Strong traceability

---

## **6. Evaluation Plan (Comprehensive Framework)**
### **6.1 Evaludation Design Overview**
The evaluation is structured to systematically analyse how different RAG components (retrieval, reranking, query processing, and context handling) behave across different query types.

Rather than reporting only aggregate performance, results are analysed along two axes:
* Pipeline Variants (V0–V6) → component-level differences
* Query Categories → task-level differences

--- 

### **6.2 Quantitative Evaluation**
Evaluation will be conducted using **RAGAS metrics**, including:
* **Faithfulness**: Alignment of generated answer with retrieved context
* **Answer Relevance**: Relevance of answer to the query
* **Context Precision**: Quality (proportion that are relevant) of retrieved chunks
* **Context Recall**: Coverage of all relevant information

**Additional Metrics**
* Exact Match / Numerical Accuracy (%) → correctness of financial values
* Top-k Hit Rate (%) → whether relevant chunk appears in top-k
* MPR (Mean Reciprocal Rank) → ranking effectiveness

**Experimental Setup:**
* 20 benchmark questions across 4 categories
* Same dataset across all variants
* Controlled parameters
   * top-k
   * temperature
   * prompt template

---

### **6.3 Category-Based Evaluation (Core Study)**
| Category                 | Description                           |
|--------------------------|---------------------------------------|
| **Factual Retrieval**    | Direct lookup (e.g., revenue, income) |
| **Temporal Reasoning**   | Time-based comparisons                |
| **Multi-hop Reasoning**  | Cross-section synthesis               |
| **Comparative Analysis** | Multi-period comparisons              |

**Evaluation Objective**
For each category:
* Identify which RAG component contributes most
* Analyse performance differences across variants

**Insight Mapping**
| Component           | Expected Strength             |
|---------------------|-------------------------------|
| Dense retrieval     | Semantic queries              |
| Hybrid retrieval    | Keyword + ambiguous queries   |
| Reranking           | Multi-hop queries             |
| Query rewriting     | Ambiguous queries             |
| Metadata filtering  | Temporal / structured queries |
| Context compression | Multi-hop / noisy contexts    |

---

### **6.4 Qualitative Evaluation**
A structured **error analysis framework** is used to diagnose system behaviour.

**Failure Categories**
1. **Retrieval Failure:** → Relevant chunk not retrieved
2. **Ranking Failure:** → Relevant chunk retrieved but ranked too low
3. **Chunking Failure:** → Information split across chunks
4. **Query Understanding Failure** (NEW) → Query 
5. **Generation Failure:** → LLM misinterprets context or hallucinates

| Failure Type       | V0 | V1 | V2 | V3 |
|--------------------|----|----|----|----|
| Retrieval Failure  | %  | %  | %  | %  |
| Ranking Failure    | %  | %  | %  | %  |
| Generation Failure | %  | %  | %  | %  |

---

#### **Analysis Method**
For each failure:
* Compare expected vs generated answer
* Inspect retrieved chunks
* Identify root cause
* Propose fix

---

### **6.3 Category-Based Evaluation**

Questions will be grouped into:
* **Factual Retrieval** (e.g., revenue figures)
* **Temporal Reasoning** (e.g., quarter-over-quarter growth)
* **Multi-hop Reasoning** (cross-section synthesis)
* **Comparative Analysis** (e.g., year-over-year changes)

**Goal:**
* Identify which pipeline performs best per category
* Understand strengths & downsides of each retrieval strategy

---

### **6.4 Ablation Study (Component-Level Analysis)**
To isolate impact of components:

| Experiment         | Description               |
|--------------------|---------------------------|
| Dense only         | Baseline retrieval        |
| BM25 only          | Sparse retrieval          |
| Hybrid (no rerank) | Fusion without refinement |
| Hybrid + rerank    | Full pipeline             |

**Metrics Compared:**
* RAGAS scores
* Retrieval precision
* Answer accuracy

---

### **6.5 Latency & Efficiency Analysis**
**Measure:**
* End-to-end response time per query
* Retrieval vs reranking vs generation time

**Purpose:** Evaluate trade-offs between:
  * Accuracy
  * Speed
  * Computational cost

---

## **7. Expected Insights & Hypotheses**
The study aims to validate:

1. **Hybrid retrieval improves recall**
   * Especially for complex or ambiguous queries

2. **Reranking improves precision**
   * Reduces noise introduced by hybrid retrieval

3. **Trade-off exists between latency and performance**
   * Advanced pipelines yield better accuracy but slower responses

4. **Chunking strategy impacts answer quality**
   * Poor chunking leads to incomplete or incorrect answers

---

## **8. Risks, Limitations & Mitigation**

### **8.1 Key Risks**
1. **Hallucinated financial values**
2. **Incorrect retrieval leading to misleading answers**
3. **Loss of context due to chunk boundaries**
4. **Ambiguity in financial language**

---

### **8.2 Mitigation Strategies**
* Enforce citation-based responses
* Restrict knowledge to verified SEC filings
* Use reranking to improve retrieval accuracy
* Apply conservative generation settings

---

### **8.3 System Limitations**
* Fixed chunking may break semantic structure
* Limited ability to interpret tables or structured data
* Restricted to single-company dataset
* Limited multi-document reasoning capability

---

## **9. Reproducibility & Experimental Control**

* Fixed random seed (42) ensures consistent results
* All parameters configurable via YAML
* Full pipeline reproducible via scripts:
  * ingestion
  * indexing
  * querying
  * evaluation
* Version-pinned dependencies ensure environment stability

---

## **10. Deliverables & Success Criteria**

### **Deliverables**
* Functional QA system (CLI + Streamlit UI)
* Code repository with documentation
* Evaluation results (quantitative + qualitative)
* Final report with analysis and insights

---

### **Success Criteria**
* Demonstrated improvement of advanced RAG over baseline
* Clear explanation of trade-offs
* Robust evaluation with both quantitative and qualitative evidence
* Reproducible and well-documented system

**Examples of Quantitative Results from Project Study**
1. RAG Performance Metrics

| Metric            | Unit        | V0 (LLM-only) | V1 (Basic) | V2 (Rerank) | V3 (Hybrid) | Interpretation                          |
|-------------------|-------------|---------------|-------------|-------------|-------------| --------------------------------------- |
| Faithfulness      | Score (0–1) | 0.58          | 0.72        | 0.81        | **0.86**    | Higher = less hallucination             |
| Answer Relevance  | Score (0–1) | 0.62          | 0.75        | 0.83        | **0.88**    | Better alignment to query               |
| Context Precision | Score (0–1) | —             | 0.68        | 0.79        | **0.85**    | Better chunk selection (RAG only)       |
| Context Recall    | Score (0–1) | —             | 0.70        | 0.76        | **0.91**    | Hybrid improves recall significantly    |


2. Latency & Efficiency Metrics

| Metric            | Unit        | V0   | V1   | V2   | V3   |
|-------------------|-------------|------|------|------|------|
| Avg Response Time | seconds (s) | 1.0s | 1.8s | 2.9s | 3.6s |
| Retrieval Time    | seconds (s) | —    | 0.6s | 0.6s | 1.2s |
| Reranking Time    | seconds (s) | —    | —    | 1.1s | 1.3s |
| Generation Time   | seconds (s) | 1.0s | 1.2s | 1.2s | 1.1s |


3. Accuracy Success Metrics

| Metric              | Unit | V0  | V1  | V2  | V3      |
|---------------------|------|-----|-----|-----|---------|
| Correct Answer Rate | %    | 48% | 65% | 78% | **85%** |


4. Retrieval Quality metrics

| Metric                     | Unit        | V1   | V2   | V3       |
| -------------------------- | ----------- | ---- | ---- | -------- |
| Top-3 Hit Rate             | %           | 60%  | 72%  | **88%**  |
| MRR (Mean Reciprocal Rank) | Score (0–1) | 0.55 | 0.69 | **0.81** |


5. Category-based Performance
| Category             | Unit      | V0  | V1  | V2  | V3      |
| -------------------- | --------- | --- | --- | --- | ------- |
| Factual Questions    | % correct | 65% | 80% | 88% | **90%** |
| Temporal Reasoning   | % correct | 40% | 60% | 72% | **85%** |
| Multi-hop Reasoning  | % correct | 30% | 50% | 68% | **82%** |
| Comparative Analysis | % correct | 35% | 55% | 70% | **84%** |


6. Error / Failure Metrics
| Failure Type       | Unit | V0  | V1  | V2  | V3     |
| ------------------ | ---- | --- | --- | --- | ------ |
| Retrieval Failures | %    | —   | 25% | 15% | **8%** |
| Hallucination Rate | %    | 35% | 18% | 10% | **6%** |
| Chunking Errors    | %    | —   | 12% | 12% | 11%    |


7. Ablation Study Metrics

| Setup              | Faithfulness | Recall   | Latency (s) |
| ------------------ | ------------ | -------- | ----------- |
| LLM-only (V0)      | 0.58         | —        | 1.0         |
| Dense only         | 0.72         | 0.70     | 1.8         |
| BM25 only          | 0.68         | 0.75     | 1.6         |
| Hybrid (no rerank) | 0.80         | 0.90     | 2.5         |
| Hybrid + rerank    | **0.86**     | **0.91** | 3.6         |


### Mapping to Overall Success Criteria
| Success Criterion       | Metric Used                        | Unit             |
| ----------------------- | ---------------------------------- | ---------------- |
| Improved answer quality | Faithfulness, relevance            | 0–1 score        |
| Better retrieval        | Recall, hit rate                   | %                |
| Trade-offs explained    | Latency, cost                      | seconds, USD/SGD |
| Robust evaluation       | Accuracy, failure rate             | %                |
| Insight generation      | Category breakdown, V0 vs RAG gap  | %                |

---

## **12. Conclusion**
This project aims to bridge **practical system development** and **research-driven evaluation** in the domain of financial QA. By systematically comparing multiple RAG architectures and analyzing their behavior, FinSight will provide meaningful insights into the effectiveness of advanced retrieval strategies in real-world applications.

---
