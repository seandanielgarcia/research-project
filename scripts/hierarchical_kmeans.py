#!/usr/bin/env python3
import os, json, argparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def load_clusters(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    return clusters

def encode_texts(texts, model):
    if not texts:
        return np.array([])
    return model.encode(texts)

def hierarchical_kmeans(data, model, depth=2, k=3, min_size=3):
    """
    data: dict of {cluster_name: [texts]}
    depth: levels of recursion (>=1)
    k: number of clusters per level
    min_size: skip clustering if fewer than this many texts
    """
    hierarchy = {}

    for cluster_name, texts in data.items():
        if not texts or len(texts) < min_size:
            hierarchy[cluster_name] = texts
            continue

        emb = encode_texts(texts, model)
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
        subclusters = {}
        for t, lab in zip(texts, labels):
            subclusters.setdefault(f"{cluster_name}-Sub{lab+1}", []).append(t)

        if depth > 1:
            subclusters = hierarchical_kmeans(subclusters, model, depth-1, k, min_size)
        hierarchy[cluster_name] = subclusters

    return hierarchy

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def flatten_structure(structure, level=0):
    if isinstance(structure, dict):
        total = 0
        for v in structure.values():
            total += flatten_structure(v, level+1)
        return total
    elif isinstance(structure, list):
        return len(structure)
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to existing kmeans_*.json")
    ap.add_argument("--out", default="hierarchical_results.json", help="Output JSON file")
    ap.add_argument("--depth", type=int, default=2, help="Recursion depth (levels)")
    ap.add_argument("--k", type=int, default=3, help="Subclusters per level")
    ap.add_argument("--min-size", type=int, default=3, help="Min docs per cluster to recluster")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    clusters = load_clusters(args.input)

    print(f"Running hierarchical KMeans on {len(clusters)} top-level clusters ...")
    hierarchy = hierarchical_kmeans(clusters, model, depth=args.depth, k=args.k, min_size=args.min_size)
    save_json(hierarchy, args.out)

    total_docs = flatten_structure(hierarchy)
    print(f"Saved hierarchical clusters → {args.out}")
    print(f"Total documents processed: {total_docs}")

if __name__ == "__main__":
    main()
