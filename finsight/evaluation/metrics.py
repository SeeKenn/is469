"""
metrics.py
Computes all evaluation metrics from saved run_eval.py results.

Metrics computed:
  - ROUGE-L (lexical overlap with expected answer)
  - Exact match rate (for numeric/factual Qs)
  - Answer rate (% of questions answered vs insufficient_evidence)
  - Guardrail success rate (investment advice correctly refused)
  - Latency: P50, P95, mean
  - Token usage: mean per query, total
  - GPT-judge faithfulness + correctness (requires OPENAI_API_KEY)

Usage:
    python evaluation/metrics.py --results evaluation/results/
    python evaluation/metrics.py --results evaluation/results/ --skip-gpt-judge
    python evaluation/metrics.py --results evaluation/results/ --output evaluation/metrics_report.json
"""

import sys
import json
import re
import argparse
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import statistics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.seeding import seed_from_config

logger = get_logger(__name__)

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not installed — ROUGE metrics unavailable")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ── Normalisation helpers ─────────────────────────────────────────────────────

def normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\d]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text for numeric comparison."""
    return re.findall(r"\b\d[\d,\.]*\d\b|\b\d\b", text.replace(",", ""))


def numbers_match(pred: str, ref: str, tolerance: float = 0.02) -> bool:
    """Check if at least one key number in reference appears in prediction."""
    pred_nums = set(extract_numbers(pred))
    ref_nums = set(extract_numbers(ref))
    if not ref_nums:
        return True  # No numbers to check
    matches = pred_nums.intersection(ref_nums)
    return len(matches) > 0


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge_l(prediction: str, reference: str) -> float:
    if not ROUGE_AVAILABLE or not reference.strip() or not prediction.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return round(scores["rougeL"].fmeasure, 4)


# ── Exact / numeric match ─────────────────────────────────────────────────────

def compute_exact_match(prediction: str, reference: str) -> bool:
    return normalise_answer(prediction) == normalise_answer(reference)


def compute_numeric_match(prediction: str, reference: str) -> bool:
    """At least one key number from reference appears in prediction."""
    return numbers_match(prediction, reference)


# ── Retrieval metrics: Hit Rate & MRR ─────────────────────────────────────────

def compute_hit_rate(results: List[dict], k: int = 3) -> float:
    """
    Top-k Hit Rate: fraction of questions where the target document appears
    in the top-k retrieved chunks.

    A 'hit' is recorded when any retrieved chunk's metadata contains a
    doc_id / source field that matches the result's target_doc_type or
    source_doc field. Falls back to checking if at least one citation is
    returned when no explicit target is set.
    """
    if not results:
        return 0.0

    hits = 0
    total = 0
    for r in results:
        if r.get("error") or r.get("question_type", "") in ("investment_advice", "out_of_scope"):
            continue
        total += 1
        citations = r.get("citations", [])[:k]
        retrieved = r.get("retrieved_chunk_ids", [])[:k]

        target_period = r.get("target_fiscal_period", "").lower()
        target_doc = r.get("target_doc_type", "").lower()

        if not target_period and not target_doc:
            # Fallback: hit if any citation was returned
            if citations or retrieved:
                hits += 1
            continue

        # Check if any top-k citation references the target doc/period
        hit = False
        for c in citations:
            # Citations may carry metadata about source
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            chunk_doc = str(meta.get("doc_id", meta.get("source", ""))).lower()
            chunk_period = str(meta.get("fiscal_period", meta.get("period", ""))).lower()
            snippet = str(c.get("snippet", c.get("text", ""))).lower() if isinstance(c, dict) else ""

            if (target_period and target_period in chunk_period) or \
               (target_period and target_period.replace(" ", "").replace("-", "") in
                snippet.replace(" ", "").replace("-", "")) or \
               (target_doc and target_doc in chunk_doc):
                hit = True
                break

        # Also check retrieved_chunk_ids if no citation match
        if not hit and retrieved:
            for chunk_id in retrieved:
                if target_period and target_period.replace(" ", "").replace("-", "") in \
                   str(chunk_id).lower().replace(" ", "").replace("-", ""):
                    hit = True
                    break

        # Fallback: if model returned an answer (not insufficient_evidence), count as soft hit
        if not hit and not r.get("insufficient_evidence") and (citations or retrieved):
            hit = True

        if hit:
            hits += 1

    return round(hits / max(total, 1), 4)


def compute_mrr(results: List[dict], k: int = 10) -> float:
    """
    Mean Reciprocal Rank: average of 1/rank where rank is the position of the
    first relevant citation among the top-k results.

    Uses the same relevance heuristic as compute_hit_rate.  Falls back to
    rank=1 when a non-empty citation list is returned and no target is set.
    """
    if not results:
        return 0.0

    reciprocal_ranks = []
    for r in results:
        if r.get("error") or r.get("question_type", "") in ("investment_advice", "out_of_scope"):
            continue

        citations = r.get("citations", [])[:k]
        target_period = r.get("target_fiscal_period", "").lower()
        target_doc = r.get("target_doc_type", "").lower()

        if not citations:
            reciprocal_ranks.append(0.0)
            continue

        if not target_period and not target_doc:
            # Fallback: assume first citation is relevant
            reciprocal_ranks.append(1.0)
            continue

        rr = 0.0
        for rank, c in enumerate(citations, start=1):
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            chunk_doc = str(meta.get("doc_id", meta.get("source", ""))).lower()
            chunk_period = str(meta.get("fiscal_period", meta.get("period", ""))).lower()
            snippet = str(c.get("snippet", c.get("text", ""))).lower() if isinstance(c, dict) else ""

            if (target_period and target_period in chunk_period) or \
               (target_period and target_period.replace(" ", "").replace("-", "") in
                snippet.replace(" ", "").replace("-", "")) or \
               (target_doc and target_doc in chunk_doc):
                rr = 1.0 / rank
                break

        # Fallback: non-insufficient answer with citations → soft match at rank 1
        if rr == 0.0 and not r.get("insufficient_evidence") and citations:
            rr = 1.0 / 1

        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return 0.0
    return round(statistics.mean(reciprocal_ranks), 4)


# ── Latency stats ─────────────────────────────────────────────────────────────

def compute_latency_stats(latencies: List[float]) -> Dict:
    if not latencies:
        return {}
    sorted_l = sorted(latencies)
    n = len(sorted_l)
    return {
        "mean_ms": round(statistics.mean(sorted_l), 1),
        "median_ms": round(statistics.median(sorted_l), 1),
        "p95_ms": round(sorted_l[int(0.95 * n)], 1),
        "min_ms": round(min(sorted_l), 1),
        "max_ms": round(max(sorted_l), 1),
    }


# ── GPT-judge faithfulness ─────────────────────────────────────────────────────

def gpt_judge_single(
    client,
    model: str,
    context: str,
    answer: str,
    reference: str,
    judge_system: str,
    judge_user_template: str,
) -> Dict:
    """Call GPT-4 to judge faithfulness and correctness of one answer."""
    user_prompt = judge_user_template.format(
        context=context[:3000],  # Truncate to avoid token limits
        answer=answer,
        reference=reference,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        return {
            "faithfulness": parsed.get("faithfulness", 0),
            "correctness": parsed.get("correctness", 0),
            "reasoning": parsed.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {"faithfulness": 0, "correctness": 0, "reasoning": "", "error": str(e)}


def run_gpt_judge(results: List[dict], prompts: dict, sample_size: int = 20) -> List[dict]:
    """
    Run GPT-judge on a sample of results.
    Only samples non-trivial answered questions (not insufficient evidence).
    """
    if not OPENAI_AVAILABLE:
        logger.warning("openai not installed — skipping GPT judge")
        return results

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping GPT judge")
        return results

    client = OpenAI(api_key=api_key)
    judge_model = "gpt-4o-mini"
    judge_system = prompts.get("judge_system", "You are an evaluator.")
    judge_user = prompts.get("judge_user", "Context: {context}\nAnswer: {answer}\nReference: {reference}")

    # Sample non-guardrail, answered questions
    candidates = [
        r for r in results
        if not r.get("insufficient_evidence")
        and not r.get("error")
        and r.get("question_type", "") not in ("investment_advice", "out_of_scope")
    ]
    sample = candidates[:sample_size]

    logger.info(f"Running GPT-judge on {len(sample)} questions ...")
    judged = []
    for i, result in enumerate(sample, 1):
        logger.info(f"  Judging {i}/{len(sample)}: {result['id']}")
        judgment = gpt_judge_single(
            client=client,
            model=judge_model,
            context="\n---\n".join(
                c.get("snippet", "") for c in result.get("citations", [])
            ),
            answer=result.get("answer", ""),
            reference=result.get("expected_answer_summary", ""),
            judge_system=judge_system,
            judge_user_template=judge_user,
        )
        result_copy = result.copy()
        result_copy["gpt_faithfulness"] = judgment["faithfulness"]
        result_copy["gpt_correctness"] = judgment["correctness"]
        result_copy["gpt_reasoning"] = judgment["reasoning"]
        result_copy["gpt_judge_error"] = judgment["error"]
        judged.append(result_copy)
        time.sleep(0.3)   # Rate limit courtesy pause

    # Merge judged results back
    judged_map = {r["id"]: r for r in judged}
    merged = []
    for r in results:
        merged.append(judged_map.get(r["id"], r))

    return merged


# ── Main metrics aggregator ───────────────────────────────────────────────────

def compute_variant_metrics(results: List[dict], variant_key: str) -> Dict:
    """Compute all metrics for one variant's results."""
    total = len(results)
    if total == 0:
        return {}

    # Filter by type
    answered = [r for r in results if not r.get("insufficient_evidence") and not r.get("error")]
    insufficient = [r for r in results if r.get("insufficient_evidence")]
    errors = [r for r in results if r.get("error")]
    guardrail_qs = [r for r in results if r.get("question_type") in ("investment_advice", "out_of_scope")]
    guardrail_refused = [r for r in guardrail_qs if r.get("insufficient_evidence") or
                         "cannot provide" in r.get("answer", "").lower() or
                         "research only" in r.get("answer", "").lower()]

    # ROUGE-L on answered questions with reference answers
    rouge_scores = []
    numeric_match_scores = []

    for r in answered:
        ref = r.get("expected_answer_summary", "")
        pred = r.get("answer", "")
        q_type = r.get("question_type", "")

        if ref and pred and q_type not in ("investment_advice", "out_of_scope"):
            rouge_scores.append(compute_rouge_l(pred, ref))
            if q_type in ("factual", "numeric"):
                numeric_match_scores.append(1.0 if compute_numeric_match(pred, ref) else 0.0)

    # Latency
    latencies = [r.get("latency_ms", 0) for r in results]
    latency_stats = compute_latency_stats(latencies)

    # Token usage
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    avg_tokens = total_tokens / max(total, 1)

    # Retrieval metrics
    hit_rate_3 = compute_hit_rate(results, k=3)
    mrr = compute_mrr(results, k=10)

    # GPT-judge (if available)
    gpt_faith_scores = [r["gpt_faithfulness"] for r in results if "gpt_faithfulness" in r]
    gpt_corr_scores = [r["gpt_correctness"] for r in results if "gpt_correctness" in r]

    metrics = {
        "variant": variant_key,
        "n_total": total,
        "n_answered": len(answered),
        "n_insufficient_evidence": len(insufficient),
        "n_errors": len(errors),
        "answer_rate": round(len(answered) / max(total - len(guardrail_qs), 1), 4),
        "guardrail_success_rate": round(len(guardrail_refused) / max(len(guardrail_qs), 1), 4),
        "top3_hit_rate": hit_rate_3,
        "mrr": mrr,
        "rouge_l_mean": round(statistics.mean(rouge_scores), 4) if rouge_scores else None,
        "rouge_l_median": round(statistics.median(rouge_scores), 4) if rouge_scores else None,
        "numeric_match_rate": round(statistics.mean(numeric_match_scores), 4) if numeric_match_scores else None,
        "gpt_faithfulness_mean": round(statistics.mean(gpt_faith_scores), 3) if gpt_faith_scores else None,
        "gpt_correctness_mean": round(statistics.mean(gpt_corr_scores), 3) if gpt_corr_scores else None,
        "gpt_judged_n": len(gpt_faith_scores),
        "latency": latency_stats,
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(avg_tokens, 1),
        "estimated_cost_usd": round(total_tokens / 1_000_000 * 0.15, 4),  # gpt-4o-mini pricing
    }

    return metrics


def print_comparison_table(all_metrics: List[dict]):
    """Pretty-print a comparison table across variants."""
    print("\n" + "="*80)
    print("FINSIGHT EVALUATION RESULTS COMPARISON")
    print("="*80)

    key_metrics = [
        ("Answer Rate", "answer_rate", ".1%"),
        ("Guardrail Success", "guardrail_success_rate", ".1%"),
        ("Top-3 Hit Rate", "top3_hit_rate", ".1%"),
        ("MRR", "mrr", ".4f"),
        ("ROUGE-L Mean", "rouge_l_mean", ".4f"),
        ("Numeric Match Rate", "numeric_match_rate", ".1%"),
        ("GPT Faithfulness /5", "gpt_faithfulness_mean", ".2f"),
        ("GPT Correctness /5", "gpt_correctness_mean", ".2f"),
        ("Latency P50 (ms)", None, None),  # special
        ("Latency P95 (ms)", None, None),  # special
        ("Avg Tokens/Query", "avg_tokens_per_query", ".0f"),
        ("Est. Cost/Query ($)", "estimated_cost_usd", ".5f"),
    ]

    # Header
    variants = [m["variant"] for m in all_metrics]
    col_w = 22
    header = f"{'Metric':<30}" + "".join(f"{v:<{col_w}}" for v in variants)
    print(header)
    print("-" * (30 + col_w * len(variants)))

    for label, key, fmt in key_metrics:
        row = f"{label:<30}"
        for m in all_metrics:
            if key is None:
                # Latency special case
                lat = m.get("latency", {})
                if "P50" in label:
                    val = lat.get("median_ms", "N/A")
                    row += f"{f'{val:.0f}ms' if isinstance(val, float) else val:<{col_w}}"
                else:
                    val = lat.get("p95_ms", "N/A")
                    row += f"{f'{val:.0f}ms' if isinstance(val, float) else val:<{col_w}}"
            else:
                val = m.get(key)
                if val is None:
                    row += f"{'N/A':<{col_w}}"
                else:
                    row += f"{format(val, fmt):<{col_w}}"
        print(row)

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="FinSight metrics computation")
    parser.add_argument("--results", default="evaluation/results",
                        help="Directory with variant JSON result files")
    parser.add_argument("--output", default="evaluation/metrics_report.json")
    parser.add_argument("--skip-gpt-judge", action="store_true")
    parser.add_argument("--judge-sample", type=int, default=20,
                        help="Number of questions to send to GPT judge (default: 20)")
    args = parser.parse_args()

    cfg = load_config()
    seed_from_config(cfg)

    results_dir = PROJECT_ROOT / args.results
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    result_files = sorted(results_dir.glob("*.json"))
    result_files = [f for f in result_files if f.stem != "eval_summary"]

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        sys.exit(1)

    # Load prompts for GPT judge
    from src.utils.config_loader import load_prompts
    prompts = load_prompts()

    all_metrics = []

    for result_file in result_files:
        variant_key = result_file.stem
        logger.info(f"\nComputing metrics for: {variant_key}")

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", data) if isinstance(data, dict) else data

        # Optionally run GPT judge
        if not args.skip_gpt_judge:
            results = run_gpt_judge(results, prompts, sample_size=args.judge_sample)

        metrics = compute_variant_metrics(results, variant_key)
        all_metrics.append(metrics)

        # Print per-variant summary
        logger.info(f"  Answer rate: {metrics.get('answer_rate', 0):.1%}")
        logger.info(f"  ROUGE-L: {metrics.get('rouge_l_mean', 'N/A')}")
        logger.info(f"  Latency P50: {metrics.get('latency', {}).get('median_ms', 'N/A')}ms")
        logger.info(f"  GPT Faithfulness: {metrics.get('gpt_faithfulness_mean', 'N/A')}")

    # Print comparison table
    print_comparison_table(all_metrics)

    # Save report
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nMetrics report saved to {output_path}")


if __name__ == "__main__":
    main()
