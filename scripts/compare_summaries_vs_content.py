#!/usr/bin/env python3
"""
Compare clustering results between summaries and full content using Rand Index.
Runs k=16 clustering on both and computes similarity.
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score


def load_data(csv_path, reports_only=False):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    summary_col = cols.get("summary", None)
    content_col = cols.get(
        "full content",
        cols.get("content", cols.get("title & content", None))
    )
    report_col = cols.get("is report", None)
    post_id_col = cols.get("post id", None)

    if summary_col is None or content_col is None:
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    data = df
    if reports_only and report_col in df.columns:
        data = df[df[report_col].astype(str).str.lower().isin(["true","yes","y","1"])]

    summaries, contents, post_ids = [], [], {}
    idx = 0

    for _, row in data.iterrows():
        summary = str(row[summary_col]).strip()
        content = str(row[content_col]).strip()
        
        if (summary and summary.lower() not in ["nan","no content"] and
            content and content.lower() not in ["nan","no content"]):
            pid = str(row[post_id_col]) if post_id_col else f"post_{idx}"
            summaries.append(summary)
            contents.append(content)
            post_ids[idx] = pid
            idx += 1

    return summaries, contents, post_ids



def get_embeddings(sentences, model):
    return model.encode(sentences, show_progress_bar=True)


# ------------------------------

def run_kmeans(embeddings, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km



def labels_to_clusters(labels, post_ids):
    clusters = {}
    for idx, label in enumerate(labels):
        pid = post_ids[idx]
        cluster_name = f"Cluster {label+1}"
        clusters.setdefault(cluster_name, []).append(pid)
    return clusters



def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Compare clustering on summaries vs full content")
    ap.add_argument("--input", required=True, help="Path to CSV file")
    ap.add_argument("--k", type=int, default=16, help="Number of clusters")
    ap.add_argument("--out", default="results/clustering/compare_summaries_content_k16", help="Output directory")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    ap.add_argument("--reports-only", action="store_true",
                    help="If set, cluster only posts marked as reports.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SentenceTransformer(args.model)

    print("Loading data...")
    summaries, contents, post_ids = load_data(args.input, reports_only=args.reports_only)
    
    if not summaries:
        print("No valid data found!")
        return
    
    print(f"Loaded {len(summaries)} posts with both summaries and full content")

    print("\nGenerating embeddings for summaries...")
    summary_emb = get_embeddings(summaries, model)
    
    print("Generating embeddings for full content...")
    content_emb = get_embeddings(contents, model)

    print(f"\nRunning KMeans (k={args.k}) on summaries...")
    summary_labels, summary_km = run_kmeans(summary_emb, args.k, args.random_state)
    summary_clusters = labels_to_clusters(summary_labels, post_ids)
    
    print(f"Running KMeans (k={args.k}) on full content...")
    content_labels, content_km = run_kmeans(content_emb, args.k, args.random_state)
    content_clusters = labels_to_clusters(content_labels, post_ids)

    print("\nComputing Rand Index...")
    rand_index = rand_score(summary_labels, content_labels)
    print(f"Rand Index (not adjusted): {rand_index:.4f}")

    save_json(summary_clusters, os.path.join(args.out, "summary_clusters.json"))
    save_json(content_clusters, os.path.join(args.out, "content_clusters.json"))
    
    metrics = {
        "rand_index": float(rand_index),
        "k": args.k,
        "num_posts": len(summaries),
        "random_state": args.random_state
    }
    save_json(metrics, os.path.join(args.out, "comparison_metrics.json"))

    print(f"\nResults saved to {args.out}/")
    print(f"Rand Index: {rand_index:.4f}")

if __name__ == "__main__":
    main()

