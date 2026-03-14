"""
Score context_precision for all modes, truncating each context chunk to
fit within qwen2.5-14b's 8192-token context window.

RAGAS context_precision sends ALL context chunks in one prompt, so long
chunks (12 × ~500 tokens) overflow the 8192-token limit.  Truncating to
~400 chars per chunk keeps the total well under the limit while still
capturing the key information needed to judge relevance.
"""
import sys, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── tuneable ──────────────────────────────────────────────────────────────
CHUNK_CHAR_LIMIT = 300   # chars per context chunk fed to the RAGAS judge
MAX_TOKENS_OUT   = 512   # output tokens for the RAGAS judge LLM
# ─────────────────────────────────────────────────────────────────────────


def score_context_precision(per_question: list, cfg: dict) -> float:
    """Return mean context_precision across valid results, truncating contexts."""
    from ragas import evaluate, RunConfig
    from ragas.metrics import context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    gen_cfg = cfg["generation"]
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=gen_cfg.get("model", "qwen2.5-14b"),
            openai_api_base=gen_cfg.get("base_url", "http://localhost:8000/v1"),
            openai_api_key=gen_cfg.get("api_key", "dummy"),
            temperature=0.0,
            max_tokens=MAX_TOKENS_OUT,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=cfg["embeddings"]["model"])
    )

    valid = [r for r in per_question if not r.get("error")]
    if not valid:
        logger.warning("No valid results — returning 0.0")
        return 0.0

    dataset = Dataset.from_dict({
        "question":     [r["question"]   for r in valid],
        "answer":       [r["answer"]     for r in valid],
        # Truncate each chunk so the full prompt fits in 8192 tokens
        "contexts":     [[c[:CHUNK_CHAR_LIMIT] for c in r["contexts"]] for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    })

    # max_workers=1 + longer timeout prevents overloading the vLLM server
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180),
    )

    raw = result["context_precision"]
    if isinstance(raw, (list, np.ndarray)):
        clean = [v for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return round(float(np.mean(clean)) if clean else 0.0, 4)
    return round(float(raw), 4)


def main():
    cfg  = load_config()
    path = PROJECT_ROOT / "evaluation/results/eval_results.json"

    with open(path) as f:
        data = json.load(f)

    for mode in data:
        per_q = data[mode].get("per_question", [])
        if not per_q:
            existing = data[mode].get("aggregate", {}).get("context_precision")
            logger.info(f"Skipping '{mode}' — no per_question data "
                        f"(keeping existing context_precision={existing})")
            continue

        logger.info(f"Scoring context_precision for '{mode}' "
                    f"(chunks truncated to {CHUNK_CHAR_LIMIT} chars, max_workers=1) ...")
        score = score_context_precision(per_q, cfg)
        logger.info(f"  {mode}: context_precision = {score}")
        data[mode].setdefault("aggregate", {})["context_precision"] = score

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {path}")

    # ── pretty print final table ──────────────────────────────────────────
    modes   = list(data.keys())
    metrics = ["faithfulness", "answer_relevancy", "context_recall",
               "context_precision", "avg_latency_seconds"]
    col_w   = 14

    sep = "=" * (26 + (col_w + 3) * len(modes))
    print(f"\n{sep}")
    print("  RAGAS EVALUATION RESULTS — FinSight (qwen2.5-14b)")
    print(sep)
    header = f"{'Metric':<26}| " + " | ".join(f"{m:>{col_w}}" for m in modes)
    print(header)
    print("-" * len(sep))
    for k in metrics:
        row = f"{k:<26}| "
        row += " | ".join(
            f"{data[m]['aggregate'].get(k, 0.0):>{col_w}.4f}" for m in modes
        )
        print(row)
    print("-" * len(sep))
    print(f"{'n_questions':<26}| " +
          " | ".join(f"{'20':>{col_w}}" for _ in modes))
    print(sep)


if __name__ == "__main__":
    main()
