import argparse
import os
from typing import List, Tuple

import numpy as np

# Reuse our embedding service
from resume_aii.app.services.embeddings import get_text_embedding, reload_model, get_model_info


def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # assumes vectors are already L2 normalized; but be safe
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def embed_texts(texts: List[str]) -> np.ndarray:
    # get_text_embedding returns list[float] for single text; batch for simplicity
    embs = [np.array(get_text_embedding(t), dtype=np.float32) for t in texts]
    return np.vstack(embs)


def main():
    parser = argparse.ArgumentParser(description='Rank .txt files by similarity to a query (text or file).')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--query', type=str, help='Query text (e.g., job description).')
    group.add_argument('--query_file', type=str, help='Path to a .txt file containing the query text.')
    parser.add_argument('--candidates', nargs='+', required=True, help='Paths to .txt files to rank.')
    parser.add_argument('--model', type=str, default=None, help='Optional: path/name of fine-tuned model to load.')
    args = parser.parse_args()

    if args.model:
        reload_model(args.model)

    # Read query
    if args.query_file:
        query_text = read_text_file(args.query_file)
    else:
        query_text = args.query

    # Read candidate files
    candidate_paths: List[str] = args.candidates
    candidate_texts: List[str] = [read_text_file(p) for p in candidate_paths]

    # Embed
    query_vec = np.array(get_text_embedding(query_text), dtype=np.float32)
    cand_vecs = embed_texts(candidate_texts)

    # Rank
    scores: List[Tuple[str, float]] = []
    for path, vec in zip(candidate_paths, cand_vecs):
        score = cosine_similarity(query_vec, vec)
        scores.append((path, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    model_info = get_model_info()
    print(f"Model: {model_info['name']}")
    print(f"Query: {'(file) ' + args.query_file if args.query_file else args.query}")
    print("\nRanked candidates:")
    for i, (path, score) in enumerate(scores, start=1):
        print(f"{i:2d}. {path}  |  score={score:.4f}")


if __name__ == '__main__':
    main()
