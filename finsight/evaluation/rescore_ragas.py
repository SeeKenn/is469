"""Re-compute RAGAS metrics on existing evaluation results."""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from evaluation.run_evaluation import compute_ragas_metrics, print_comparison_table

logger = get_logger(__name__)

def main():
    cfg = load_config()
    results_path = PROJECT_ROOT / "evaluation/results/eval_results.json"

    with open(results_path) as f:
        all_results = json.load(f)

    for mode in all_results:
        logger.info(f"Re-scoring RAGAS metrics for {mode}...")
        per_question = all_results[mode]["per_question"]
        ragas_scores = compute_ragas_metrics(per_question, cfg)

        latencies = [r["latency_seconds"] for r in per_question if not r.get("error")]
        avg_latency = sum(latencies) / max(len(latencies), 1)

        all_results[mode]["aggregate"] = {
            **ragas_scores,
            "avg_latency_seconds": round(avg_latency, 4),
        }
        logger.info(f"  {mode}: {json.dumps(ragas_scores, indent=2)}")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated results saved to {results_path}")

    print_comparison_table(all_results)

if __name__ == "__main__":
    main()
