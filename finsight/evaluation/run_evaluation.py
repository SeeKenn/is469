"""
run_evaluation.py
RAGAS-based evaluation pipeline for FinSight.

Runs every question from eval_dataset.json through both "baseline" and "advanced"
retrieval modes, computes RAGAS metrics (faithfulness, answer_relevancy,
context_recall, context_precision), and saves a side-by-side comparison.

Uses the same qwen2.5-14b via vLLM (http://localhost:8000/v1) as the RAGAS judge LLM.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --limit 5
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_eval_dataset(path: str) -> list:
    """Load the evaluation dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} questions from {path}")
    return dataset


def build_pipeline(mode: str, cfg: dict):
    """Build the appropriate pipeline for the given mode."""
    if mode == "baseline":
        from src.pipeline.baseline import BaselinePipeline
        from src.generation.citation_formatter import format_citations

        class _BaselineMode:
            VARIANT_NAME = "baseline_mode"

            def __init__(self, cfg):
                self._pipeline = BaselinePipeline(cfg)
                self._top_k = cfg["retrieval"].get("baseline_top_k", 5)

            def ask(self, question: str) -> dict:
                t0 = time.time()
                retrieved = self._pipeline.retriever.retrieve(question, top_k=self._top_k)
                gen_result = self._pipeline.generator.generate(question, retrieved)
                citations = format_citations(gen_result["answer"], retrieved)
                total_latency = (time.time() - t0) * 1000
                return {
                    "answer": gen_result["answer"],
                    "citations": citations,
                    "retrieved_chunks": retrieved,
                    "context_used": gen_result.get("context_used", ""),
                    "latency_ms": round(total_latency, 2),
                    "variant": "baseline_mode",
                    "model": gen_result.get("model", ""),
                    "insufficient_evidence": gen_result.get("insufficient_evidence", False),
                    "error": gen_result.get("error"),
                }

        return _BaselineMode(cfg)
    else:
        from src.pipeline.advanced_b import AdvancedBPipeline
        return AdvancedBPipeline(cfg)


def run_questions(pipeline, dataset: list) -> list:
    """Run all questions through a pipeline, collecting results."""
    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        logger.info(f"  [{i}/{len(dataset)}] {item['id']}: {question[:60]}...")

        t0 = time.time()
        try:
            result = pipeline.ask(question)
            latency_s = time.time() - t0

            contexts = [
                c.get("text", "") for c in result.get("retrieved_chunks", [])
            ]

            results.append({
                "id": item["id"],
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": ground_truth,
                "category": item.get("category", ""),
                "source_doc": item.get("source_doc", ""),
                "latency_seconds": round(latency_s, 3),
                "error": result.get("error"),
            })
            logger.info(f"    -> {latency_s:.1f}s | answered")
        except Exception as e:
            latency_s = time.time() - t0
            logger.error(f"    -> ERROR: {e}")
            results.append({
                "id": item["id"],
                "question": question,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "ground_truth": ground_truth,
                "category": item.get("category", ""),
                "source_doc": item.get("source_doc", ""),
                "latency_seconds": round(latency_s, 3),
                "error": str(e),
            })

        time.sleep(0.3)

    return results


def compute_ragas_metrics(results: list, cfg: dict) -> dict:
    """Compute RAGAS metrics using the local vLLM endpoint as the judge LLM."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        logger.error(
            f"Missing dependency for RAGAS evaluation: {e}. "
            f"Install with: pip install ragas langchain langchain-openai datasets"
        )
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    gen_cfg = cfg["generation"]
    base_url = gen_cfg.get("base_url", "http://localhost:8000/v1")
    api_key = gen_cfg.get("api_key", "dummy")
    model = gen_cfg.get("model", "qwen2.5-14b")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0.0,
            max_tokens=512,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=cfg["embeddings"]["model"],
            openai_api_base=base_url,
            openai_api_key=api_key,
        )
    )

    valid_results = [r for r in results if not r.get("error")]
    if not valid_results:
        logger.warning("No valid results to evaluate with RAGAS")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    eval_data = {
        "question": [r["question"] for r in valid_results],
        "answer": [r["answer"] for r in valid_results],
        "contexts": [r["contexts"] for r in valid_results],
        "ground_truth": [r["ground_truth"] for r in valid_results],
    }
    dataset = Dataset.from_dict(eval_data)

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        scores = {
            "faithfulness": round(float(eval_result.get("faithfulness", 0.0)), 4),
            "answer_relevancy": round(float(eval_result.get("answer_relevancy", 0.0)), 4),
            "context_recall": round(float(eval_result.get("context_recall", 0.0)), 4),
            "context_precision": round(float(eval_result.get("context_precision", 0.0)), 4),
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        logger.info("Falling back to manual metric computation...")
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    return scores


def print_comparison_table(all_results: dict):
    """Print a clean side-by-side comparison table."""
    modes = list(all_results.keys())
    metrics_keys = ["faithfulness", "answer_relevancy", "context_recall",
                    "context_precision", "avg_latency_seconds"]

    header = f"{'Metric':<25}"
    for mode in modes:
        header += f" | {mode:>12}"
    print("\n" + "=" * (25 + 15 * len(modes)))
    print("  RAGAS EVALUATION RESULTS — FinSight")
    print("=" * (25 + 15 * len(modes)))
    print(header)
    print("-" * (25 + 15 * len(modes)))

    for metric in metrics_keys:
        row = f"{metric:<25}"
        for mode in modes:
            val = all_results[mode]["aggregate"].get(metric, 0.0)
            row += f" | {val:>12.4f}"
        print(row)

    print("-" * (25 + 15 * len(modes)))
    row = f"{'n_questions':<25}"
    for mode in modes:
        n = len(all_results[mode]["per_question"])
        row += f" | {n:>12}"
    print(row)
    print("=" * (25 + 15 * len(modes)))


def main():
    parser = argparse.ArgumentParser(description="FinSight RAGAS evaluation pipeline")
    parser.add_argument(
        "--dataset",
        default="evaluation/eval_dataset.json",
        help="Path to the evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (for quick testing)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "advanced"],
        choices=["baseline", "advanced"],
        help="Modes to evaluate (default: both)",
    )
    args = parser.parse_args()

    cfg = load_config()

    dataset_path = PROJECT_ROOT / args.dataset
    if not dataset_path.exists():
        logger.error(f"Evaluation dataset not found: {dataset_path}")
        sys.exit(1)

    dataset = load_eval_dataset(str(dataset_path))
    if args.limit:
        dataset = dataset[: args.limit]
        logger.info(f"Limited to {len(dataset)} questions")

    all_results = {}

    for mode in args.modes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating mode: {mode}")
        logger.info(f"{'=' * 60}")

        pipeline = build_pipeline(mode, cfg)
        per_question = run_questions(pipeline, dataset)

        logger.info(f"Computing RAGAS metrics for {mode}...")
        ragas_scores = compute_ragas_metrics(per_question, cfg)

        latencies = [r["latency_seconds"] for r in per_question if not r.get("error")]
        avg_latency = sum(latencies) / max(len(latencies), 1)

        aggregate = {
            **ragas_scores,
            "avg_latency_seconds": round(avg_latency, 4),
        }

        all_results[mode] = {
            "per_question": per_question,
            "aggregate": aggregate,
        }

        logger.info(f"Mode {mode} complete: {json.dumps(aggregate, indent=2)}")

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
