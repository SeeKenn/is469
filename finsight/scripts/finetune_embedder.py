"""
finetune_embedder.py
Fine-tunes the sentence-transformer embedding model on FinSight financial data.

Goal: Teach the model that "Q2 FY2025" and "Q1 FY2025" are semantically DIFFERENT
      even though they share most tokens — eliminating the temporal confusion that
      caused q007-style failures.

Training data: data/finetune/embedder_train.json
  Each example: {"query", "positive", "hard_negative"}

Loss: TripletLoss with hard negatives
  - Pulls (query, positive) together
  - Pushes (query, hard_negative) apart — especially wrong-period chunks

Output: models/finsight-embedder/

Usage:
    python scripts/finetune_embedder.py
    python scripts/finetune_embedder.py --epochs 5 --batch-size 16
"""

import sys
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  (patches sqlite3 if needed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--warmup",     type=int,   default=100)
    parser.add_argument("--base-model", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "models" / "finsight-embedder"))
    parser.add_argument("--train-data", type=str,
                        default=str(PROJECT_ROOT / "data" / "finetune" / "embedder_train.json"))
    parser.add_argument("--val-data",   type=str,
                        default=str(PROJECT_ROOT / "data" / "finetune" / "embedder_val.json"))
    args = parser.parse_args()

    # ── Imports (heavy, import after arg parse so --help is fast) ─────────────
    from sentence_transformers import SentenceTransformer, losses, evaluation
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from datasets import Dataset as HFDataset
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")

    # ── Load training data ────────────────────────────────────────────────────
    print(f"\nLoading training data from {args.train_data} …")
    with open(args.train_data) as f:
        train_raw = json.load(f)

    print(f"Loading validation data from {args.val_data} …")
    with open(args.val_data) as f:
        val_raw = json.load(f)

    print(f"  Train: {len(train_raw):,} triplets")
    print(f"  Val:   {len(val_raw):,} triplets")

    # ── Build HuggingFace Datasets (new ST v3 API) ────────────────────────────
    # TripletLoss expects columns: "anchor", "positive", "negative"
    MAX_CHARS = 1000   # truncate long passages to keep GPU memory sane

    def make_hf_dataset(data):
        return HFDataset.from_dict({
            "anchor":   [item["query"] for item in data],
            "positive": [item["positive"][:MAX_CHARS] for item in data],
            "negative": [item["hard_negative"][:MAX_CHARS] for item in data],
        })

    train_dataset = make_hf_dataset(train_raw)
    val_dataset   = make_hf_dataset(val_raw)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model …")
    model = SentenceTransformer(args.base_model, device=device)

    # ── Loss: TripletLoss pushes hard negatives (wrong fiscal period) away ────
    train_loss = losses.TripletLoss(model=model)

    # ── Evaluator: triplet accuracy on val set ────────────────────────────────
    evaluator = evaluation.TripletEvaluator(
        anchors=train_dataset["anchor"][:128],    # use subset for speed
        positives=train_dataset["positive"][:128],
        negatives=train_dataset["negative"][:128],
        name="finsight-val",
        show_progress_bar=False,
    )

    # ── Training arguments (new ST v3 API) ────────────────────────────────────
    output_path = args.output_dir
    Path(output_path).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_raw) // args.batch_size
    total_steps     = steps_per_epoch * args.epochs
    print(f"\nTraining for {args.epochs} epochs × {steps_per_epoch} steps = {total_steps} total steps")
    print(f"Warmup steps: {args.warmup}")
    print(f"Learning rate: {args.lr}")
    print()

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_finsight-val_cosine_accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",        # no wandb/tensorboard
        fp16=torch.cuda.is_available(),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # ── Save best model ───────────────────────────────────────────────────────
    model.save_pretrained(output_path)
    print(f"Fine-tuned model saved to: {output_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nRunning final evaluation on validation set …")
    val_evaluator = evaluation.TripletEvaluator(
        anchors=val_dataset["anchor"],
        positives=val_dataset["positive"],
        negatives=val_dataset["negative"],
        name="finsight-val-final",
        show_progress_bar=False,
    )
    score = val_evaluator(model)
    print(f"Final triplet accuracy: {score:.4f}")

    # ── Save training metadata ────────────────────────────────────────────────
    meta = {
        "base_model":       args.base_model,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "train_examples":   len(train_raw),
        "val_examples":     len(val_raw),
        "final_val_score":  score,
        "training_seconds": elapsed,
        "output_path":      output_path
    }
    with open(Path(output_path) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved.")

    print(f"""
=== Embedding fine-tuning complete ===
Model saved : {output_path}
Val accuracy: {score:.4f}  (higher = model better separates wrong-period chunks)

Next step:
  python scripts/finetune_reranker.py
  → then update config/settings.yaml: embeddings.model = {output_path}
""")


if __name__ == "__main__":
    main()
