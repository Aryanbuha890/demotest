import argparse
import os
from typing import List

from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from torch.utils.data import DataLoader
import csv
import json
from datetime import datetime


def read_pairs_with_scores(path: str) -> List[InputExample]:
    examples: List[InputExample] = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both (text_a, text_b, score) and (job_description, resume_text, label)
                a = (row.get("text_a") or row.get("job_description") or "").strip()
                b = (row.get("text_b") or row.get("resume_text") or "").strip()
                s_val = row.get("score") if row.get("score") is not None else row.get("label")
                s = float(s_val) if s_val is not None and str(s_val) != "" else 0.0
                examples.append(InputExample(texts=[a, b], label=s))
    elif ext in (".jsonl", ".json"):  # assume JSONL if many lines, JSON array if single
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                for row in data:
                    a = (row.get("text_a") or row.get("job_description") or "").strip()
                    b = (row.get("text_b") or row.get("resume_text") or "").strip()
                    s_val = row.get("score") if row.get("score") is not None else row.get("label")
                    s = float(s_val) if s_val is not None and str(s_val) != "" else 0.0
                    examples.append(InputExample(texts=[a, b], label=s))
            else:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    a = (row.get("text_a") or row.get("job_description") or "").strip()
                    b = (row.get("text_b") or row.get("resume_text") or "").strip()
                    s_val = row.get("score") if row.get("score") is not None else row.get("label")
                    s = float(s_val) if s_val is not None and str(s_val) != "" else 0.0
                    examples.append(InputExample(texts=[a, b], label=s))
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use .csv, .jsonl, or .json")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Sentence-Transformers model on text pair similarity.")
    parser.add_argument("--train_path", required=True, help="Path to training file (.csv/.jsonl/.json) with columns text_a,text_b,score in [0,1]")
    parser.add_argument("--val_path", default=None, help="Optional validation file with same schema")
    parser.add_argument("--output_dir", required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2", help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=256)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    print(f"Reading training data from {args.train_path}")
    train_examples = read_pairs_with_scores(args.train_path)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluators = []
    if args.val_path:
        print(f"Reading validation data from {args.val_path}")
        val_examples = read_pairs_with_scores(args.val_path)
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="val")
        evaluators.append(evaluator)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        evaluator=evaluators[0] if evaluators else None,
        evaluation_steps=100 if evaluators else 0,
    )

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_path = os.path.join(args.output_dir, f"final-{timestamp}")
    model.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
