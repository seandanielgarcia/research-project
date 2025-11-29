#!/usr/bin/env python3
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

    text_col = cols.get(
        "content",
        cols.get("full content", cols.get("title & content", None))
    )
    report_col = cols.get("is report", None)
    post_id_col = cols.get("post id", None)
    score_col = cols.get("score", None)

    if text_col is None or text_col not in df.columns:
        return [], {}, []

    data = df
    if reports_only and report_col in df.columns:
        data = df[df[report_col].astype(str).str.lower().isin(["true","yes","y","1"])]

    sentences, post_ids, scores = [], {}, []
    idx = 0

    for _, row in data.iterrows():
        content = str(row[text_col]).strip()
        if content and content.lower() not in ["nan","no content"]:
            pid = str(row[post_id_col]) if post_id_col else f"post_{idx}"
            sc = float(row[score_col]) if score_col else 1.0
            sentences.append(content)
            post_ids[idx] = pid
            scores.append(sc)
            idx += 1
    return sentences, post_ids, scores


def get_embeddings(sentences, model):
    return model.encode(sentences)


def run_train_weighted(train_emb, train_post_ids, train_scores, k):
    w = np.log1p(np.array(train_scores))
    w = w / w.sum()

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(train_emb, sample_weight=w)

    loss = float(km.inertia_)

    clusters = {}
    cluster_score_stats = {}
    for pid, label, score in zip(train_post_ids.values(), labels, train_scores):
        cluster_name = f"Cluster {label+1}"
        clusters.setdefault(cluster_name, []).append(pid)
        stats = cluster_score_stats.setdefault(
            cluster_name, {"total_score": 0.0, "count": 0, "avg_score": 0.0}
        )
        stats["total_score"] += float(score)
        stats["count"] += 1

    for stats in cluster_score_stats.values():
        if stats["count"] > 0:
            stats["avg_score"] = stats["total_score"] / stats["count"]
        else:
            stats["avg_score"] = 0.0

    return km, clusters, loss, cluster_score_stats


def run_val_independent(val_emb, val_post_ids, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(val_emb)

    loss = abs(km.score(val_emb))
    avg = loss / len(val_emb)

    clusters = {}
    for pid, label in zip(val_post_ids.values(), labels):
        clusters.setdefault(f"Cluster {label+1}", []).append(pid)

    return clusters, loss, avg


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="results/weighted")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--reports-only", action="store_true",
                    help="If set, run clustering only on rows marked as reports.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SentenceTransformer(args.model)

    sentences, post_ids, scores = load_sentences(args.input, reports_only=args.reports_only)
    if not sentences:
        return

    idx = np.arange(len(sentences))
    np.random.RandomState(args.random_state).shuffle(idx)
    split = len(idx) // 2

    train_idx = idx[:split]
    val_idx   = idx[split:]

    train_sentences = [sentences[i] for i in train_idx]
    val_sentences   = [sentences[i] for i in val_idx]

    train_ids = {i: post_ids[j] for i,j in enumerate(train_idx)}
    val_ids   = {i: post_ids[j] for i,j in enumerate(val_idx)}

    train_scores = [scores[i] for i in train_idx]

    train_emb = get_embeddings(train_sentences, model)
    val_emb   = get_embeddings(val_sentences, model)

    km, train_clusters, train_loss, train_score_stats = run_train_weighted(
        train_emb, train_ids, train_scores, args.k
    )

    val_loss_with_train = abs(km.score(val_emb))

    val_clusters, val_loss_indep, val_loss_avg = run_val_independent(val_emb, val_ids, args.k)

    save_json(train_clusters, os.path.join(args.out, "train_clusters.json"))
    save_json(val_clusters, os.path.join(args.out, "val_clusters.json"))
    save_json({
        "train_loss": train_loss,
        "val_loss_using_train_centroids": val_loss_with_train,
        "val_loss_independent": val_loss_indep,
        "val_loss_independent_avg": val_loss_avg
    }, os.path.join(args.out, "metrics.json"))

    save_json(train_score_stats, os.path.join(args.out, "train_cluster_score_stats.json"))

    print("Average Score per Cluster:")
    for cluster in sorted(train_score_stats.keys(), key=lambda c: int(c.split()[1])):
        avg = train_score_stats[cluster]["avg_score"]
        count = train_score_stats[cluster]["count"]
        print(f"  {cluster}: avg_score={avg:.2f} (n={count})")

    print("Done weighted KMeans.")

if __name__ == "__main__":
    main()
