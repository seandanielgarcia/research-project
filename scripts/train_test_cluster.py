#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ------------------------------
# Load sentences + ids
# ------------------------------
def load_sentences(csv_path, reports_only=False):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    text_col = cols.get(
        "content",
        cols.get("full content", cols.get("title & content", None))
    )
    report_col = cols.get("is report", None)
    post_id_col = cols.get("post id", None)

    if text_col is None or text_col not in df.columns:
        return [], {}

    data = df
    if reports_only and report_col in df.columns:
        data = df[df[report_col].astype(str).str.lower().isin(["true","yes","y","1"])]

    sentences, post_ids = [], {}
    idx = 0

    for _, row in data.iterrows():
        content = str(row[text_col]).strip()
        if content and content.lower() not in ["nan","no content"]:
            pid = str(row[post_id_col]) if post_id_col else f"post_{idx}"
            sentences.append(content)
            post_ids[idx] = pid
            idx += 1

    return sentences, post_ids


# ------------------------------
# Embeddings
# ------------------------------
def get_embeddings(sentences, model):
    return model.encode(sentences)


# ------------------------------
# Train KMeans on train split
# ------------------------------
def run_train(train_emb, train_post_ids, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(train_emb)
    train_loss = float(km.inertia_)

    clusters = {}
    for pid, label in zip(train_post_ids.values(), labels):
        clusters.setdefault(f"Cluster {label+1}", []).append(pid)

    return km, clusters, train_loss



def compute_val_loss_with_train(km, val_emb):
    return abs(km.score(val_emb))


# ------------------------------
# Independent KMeans on val
# ------------------------------
def run_val_kmeans(val_emb, val_post_ids, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(val_emb)

    loss = abs(km.score(val_emb))
    avg_loss = loss / len(val_emb)

    clusters = {}
    for pid, label in zip(val_post_ids.values(), labels):
        clusters.setdefault(f"Cluster {label+1}", []).append(pid)

    return clusters, loss, avg_loss


# ------------------------------
# Save JSON
# ------------------------------
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="results/standard")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--reports-only", action="store_true",
                    help="If set, cluster only posts marked as reports.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SentenceTransformer(args.model)

    sentences, post_ids = load_sentences(args.input, reports_only=args.reports_only)
    if not sentences:
        return

    idx = np.arange(len(sentences))
    np.random.RandomState(args.random_state).shuffle(idx)
    split = len(idx) // 2

    train_idx = idx[:split]
    val_idx = idx[split:]

    train_sentences = [sentences[i] for i in train_idx]
    val_sentences = [sentences[i] for i in val_idx]

    train_ids = {i: post_ids[j] for i, j in enumerate(train_idx)}
    val_ids   = {i: post_ids[j] for i, j in enumerate(val_idx)}

    train_emb = get_embeddings(train_sentences, model)
    val_emb   = get_embeddings(val_sentences, model)

    km, train_clusters, train_loss = run_train(train_emb, train_ids, args.k)

    val_loss_with_train = compute_val_loss_with_train(km, val_emb)

    val_clusters, val_loss_indep, val_loss_avg = run_val_kmeans(val_emb, val_ids, args.k)

    save_json(train_clusters, os.path.join(args.out, "train_clusters.json"))
    save_json(val_clusters, os.path.join(args.out, "val_clusters.json"))
    save_json({
        "train_loss": train_loss,
        "val_loss_using_train_centroids": val_loss_with_train,
        "val_loss_independent": val_loss_indep,
        "val_loss_independent_avg": val_loss_avg
    }, os.path.join(args.out, "metrics.json"))

    print("Done standard KMeans.")

if __name__ == "__main__":
    main()
