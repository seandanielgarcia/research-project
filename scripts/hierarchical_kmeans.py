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

def hierarchical_kmeans(data, model, depth=2, k=3, min_size=3, csv_path=None, post_id_map=None):
    """
    data: dict of {cluster_name: [post_ids]}
    depth: levels of recursion (>=1)
    k: number of clusters per level
    min_size: skip clustering if fewer than this many texts
    csv_path: path to CSV with post content (optional)
    post_id_map: mapping from post_id to text content (optional)
    """
    hierarchy = {}

    for cluster_name, post_ids in data.items():
        if not post_ids or len(post_ids) < min_size:
            hierarchy[cluster_name] = post_ids
            continue

        # If we have post content mapping, encode texts; otherwise just recluster IDs
        if post_id_map:
            texts = [post_id_map.get(pid, "") for pid in post_ids]
            emb = encode_texts(texts, model)
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
        else:
            # No content available, just split by index
            labels = np.array([i % k for i in range(len(post_ids))])
        
        subclusters = {}
        for pid, lab in zip(post_ids, labels):
            subclusters.setdefault(f"{cluster_name}-Sub{lab+1}", []).append(pid)

        if depth > 1:
            subclusters = hierarchical_kmeans(subclusters, model, depth-1, k, min_size, csv_path, post_id_map)
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

def load_post_content_mapping(csv_path):
    """Load post content from CSV for re-embedding during hierarchical clustering."""
    import pandas as pd
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cols = {c.lower().strip(): c for c in df.columns}
        post_id_col = cols.get("post id", None)
        text_col = cols.get("full content", cols.get("title & content", None))
        
        if post_id_col and text_col:
            post_id_map = {}
            for _, row in df.iterrows():
                post_id = str(row[post_id_col]) if pd.notna(row[post_id_col]) else None
                content = str(row[text_col]) if pd.notna(row[text_col]) else ""
                if post_id:
                    post_id_map[post_id] = content.strip()
            return post_id_map
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to existing kmeans_*.json")
    ap.add_argument("--out", default="hierarchical_results.json", help="Output JSON file")
    ap.add_argument("--depth", type=int, default=2, help="Recursion depth (levels)")
    ap.add_argument("--k", type=int, default=3, help="Subclusters per level")
    ap.add_argument("--min-size", type=int, default=3, help="Min docs per cluster to recluster")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--csv", help="Path to CSV file with post content for better clustering")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    clusters = load_clusters(args.input)
    
    post_id_map = load_post_content_mapping(args.csv)

    print(f"Running hierarchical KMeans on {len(clusters)} top-level clusters ...")
    hierarchy = hierarchical_kmeans(clusters, model, depth=args.depth, k=args.k, min_size=args.min_size, csv_path=args.csv, post_id_map=post_id_map)
    save_json(hierarchy, args.out)

    total_docs = flatten_structure(hierarchy)
    print(f"Saved hierarchical clusters → {args.out}")
    print(f"Total documents processed: {total_docs}")

if __name__ == "__main__":
    main()
