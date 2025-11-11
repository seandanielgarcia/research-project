#!/usr/bin/env python3
import os, json, argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

def load_clusters(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def flatten_structure(structure):
    if isinstance(structure, dict):
        return sum(flatten_structure(v) for v in structure.values())
    elif isinstance(structure, list):
        return len(structure)
    return 0

def load_post_content_mapping(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    id_col = cols.get("post id")
    text_col = cols.get("full content") or cols.get("title & content")
    if not id_col or not text_col:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        pid = str(row[id_col]).strip() if pd.notna(row[id_col]) else None
        content = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
        if pid:
            mapping[pid] = content
    return mapping

def encode_texts(texts, model):
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

def hierarchical_kmeans(data, model, depth=1, k=2, min_size=3, post_id_map=None, embeddings_cache=None, seed=42):
    np.random.seed(seed)
    hierarchy = {}
    for cluster_name, post_ids in data.items():
        if not post_ids or len(post_ids) < min_size:
            hierarchy[cluster_name] = {"_leaf": post_ids}
            continue

        if post_id_map:
            texts = [post_id_map.get(pid, "") for pid in post_ids]
            valid_mask = [bool(t.strip()) for t in texts]
            valid_ids = [pid for pid, ok in zip(post_ids, valid_mask) if ok]
            if not valid_ids:
                hierarchy[cluster_name] = {"_leaf": post_ids}
                continue
            emb = np.array([
                embeddings_cache[pid] if pid in embeddings_cache
                else model.encode(post_id_map[pid], show_progress_bar=False, normalize_embeddings=True)
                for pid in valid_ids
            ])
            for pid, vec in zip(valid_ids, emb):
                embeddings_cache[pid] = vec
        else:
            hierarchy[cluster_name] = {"_leaf": post_ids}
            continue

        k_eff = min(k, len(emb))
        if k_eff < 2:
            hierarchy[cluster_name] = {"_leaf": post_ids}
            continue

        km = KMeans(n_clusters=k_eff, random_state=seed, n_init="auto")
        labels = km.fit_predict(emb)

        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            hierarchy[cluster_name] = {"_leaf": post_ids}
            continue

        subclusters = {}
        for pid, lab in zip(valid_ids, labels):
            subclusters.setdefault(f"{cluster_name}-Sub{lab+1}", {"_leaf": []})["_leaf"].append(pid)

        if depth > 1:
            subclusters = hierarchical_kmeans(
                {k_: v["_leaf"] for k_, v in subclusters.items()},
                model, depth-1, k, min_size, post_id_map, embeddings_cache, seed
            )
        hierarchy[cluster_name] = subclusters
    return hierarchy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="hierarchical_results.json")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--csv")
    args = ap.parse_args()

    print(f"Loading model '{args.model}' ...")
    model = SentenceTransformer(args.model)
    clusters = load_clusters(args.input)
    post_id_map = load_post_content_mapping(args.csv)
    embeddings_cache = {}

    print(f"Running hierarchical K-Means on {len(clusters)} top-level clusters...")
    hierarchy = hierarchical_kmeans(
        clusters, model, depth=args.depth, k=args.k,
        min_size=args.min_size, post_id_map=post_id_map,
        embeddings_cache=embeddings_cache
    )

    save_json(hierarchy, args.out)
    print(f"\nSaved results → {args.out}")
    print(f"Total documents processed: {flatten_structure(hierarchy)}")

if __name__ == "__main__":
    main()
