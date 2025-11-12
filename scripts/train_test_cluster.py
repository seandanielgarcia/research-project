import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def load_sentences(csv_path, reports_only=False):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("full content", cols.get("title & content", None))
    report_col = cols.get("is report", "Is Report")
    post_id_col = cols.get("post id", None)

    if text_col is None or text_col not in df.columns:
        print("No 'Full Content' or 'Title & Content' column found. Nothing to cluster.")
        return [], {}

    data = df
    if reports_only and report_col in df.columns:
        data = df[df[report_col].astype(str).str.strip().str.lower().isin(
            ["yes", "y", "true", "1"]
        )]

    sentences = []
    post_ids = {}
    idx = 0
    for _, row in data.iterrows():
        content = str(row[text_col]) if pd.notna(row[text_col]) else ""
        content_clean = content.strip().lower()
        if content_clean and content_clean != "nan" and content_clean != "no content":
            post_id = str(row[post_id_col]) if post_id_col and pd.notna(row[post_id_col]) else f"post_{idx}"
            sentences.append(content.strip())
            post_ids[idx] = post_id
            idx += 1
    return sentences, post_ids

def get_embeddings(sentences, model, embed_type="sentence"):
    if embed_type == "word":
        embs = []
        for s in sentences:
            tokens = s.split()
            if not tokens:
                embs.append(np.zeros(model.get_sentence_embedding_dimension()))
                continue
            token_embs = model.encode(tokens)
            avg_emb = np.mean(token_embs, axis=0)
            embs.append(avg_emb)
        return np.array(embs)
    else:
        return model.encode(sentences)

def train_and_predict(train_sentences, train_post_ids, test_sentences, test_post_ids, k, model, embed_type):
    print(f"Embedding {len(train_sentences)} training posts...")
    train_emb = get_embeddings(train_sentences, model, embed_type)
    print(f"Training KMeans with k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_labels = km.fit_predict(train_emb)
    train_loss = km.inertia_

    train_clusters = {}
    for _, (post_id, label) in enumerate(zip(train_post_ids.values(), train_labels)):
        cluster_name = f"Cluster {label+1}"
        train_clusters.setdefault(cluster_name, []).append(post_id)

    print(f"Embedding {len(test_sentences)} test posts...")
    test_emb = get_embeddings(test_sentences, model, embed_type)
    print(f"Predicting clusters for test data...")
    test_labels = km.predict(test_emb)

    dists = np.linalg.norm(test_emb[:, None] - km.cluster_centers_, axis=2)
    test_loss = float(np.mean(np.min(np.square(dists), axis=1)))

    test_clusters = {}
    for _, (post_id, label) in enumerate(zip(test_post_ids.values(), test_labels)):
        cluster_name = f"Cluster {label+1}"
        test_clusters.setdefault(cluster_name, []).append(post_id)

    metrics = {
        "train_loss": float(train_loss),
        "test_loss": float(test_loss)
    }
    return train_clusters, test_clusters, metrics

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser(description="Train KMeans on half the data and predict on the other half")
    ap.add_argument("--input", required=True, help="Input CSV file to split 50/50")
    ap.add_argument("--k", type=int, default=15, help="K for KMeans")
    ap.add_argument("--out", default="results/clustering/train_test", help="Output directory")
    ap.add_argument("--reports-only", action="store_true", help="Use only report posts")
    ap.add_argument("--embed-type", choices=["sentence", "word"], default="sentence",
                    help="Choose 'sentence' or 'word' embeddings")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SentenceTransformer(args.model)

    print(f"Loading data from {args.input}...")
    all_sentences, all_post_ids = load_sentences(args.input, reports_only=args.reports_only)
    if not all_sentences:
        print("No valid content found.")
        return

    print(f"Loaded {len(all_sentences)} posts")
    indices = np.arange(len(all_sentences))
    rng = np.random.RandomState(args.random_state)
    rng.shuffle(indices)
    split_idx = len(indices) // 2
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_sentences = [all_sentences[i] for i in train_indices]
    train_post_ids = {new_idx: all_post_ids[old_idx] for new_idx, old_idx in enumerate(train_indices)}
    test_sentences = [all_sentences[i] for i in test_indices]
    test_post_ids = {new_idx: all_post_ids[old_idx] for new_idx, old_idx in enumerate(test_indices)}

    print(f"Split into {len(train_sentences)} training posts and {len(test_sentences)} test posts")
    train_clusters, test_clusters, metrics = train_and_predict(
        train_sentences, train_post_ids,
        test_sentences, test_post_ids,
        args.k, model, args.embed_type
    )

    train_out = os.path.join(args.out, f"train_kmeans_k{args.k}_{args.embed_type}.json")
    test_out = os.path.join(args.out, f"test_kmeans_k{args.k}_{args.embed_type}.json")
    metrics_out = os.path.join(args.out, f"metrics_k{args.k}_{args.embed_type}.json")

    save_json(train_clusters, train_out)
    save_json(test_clusters, test_out)
    save_json(metrics, metrics_out)

    print(f"\nTraining clusters: {len(train_clusters)} clusters, {sum(len(v) for v in train_clusters.values())} posts")
    print(f"Test clusters: {len(test_clusters)} clusters, {sum(len(v) for v in test_clusters.values())} posts")
    print(f"Train loss: {metrics['train_loss']:.4f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Saved to:")
    print(f"  Training: {train_out}")
    print(f"  Test: {test_out}")
    print(f"  Metrics: {metrics_out}")

if __name__ == "__main__":
    main()
