#!/usr/bin/env python3
import os, json, argparse, pandas as pd, matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from bertopic import BERTopic

def load_sentences(csv_path, reports_only=False):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("title & content", "Title & Content")
    report_col = cols.get("is report", "Is Report")

    data = df
    if reports_only and report_col in df.columns:
        data = df[df[report_col].astype(str).str.strip().str.lower().isin(
            ["yes", "y", "true", "1"]
        )]

    sentences = data[text_col].dropna().astype(str).tolist()
    return sentences

def run_kmeans(sentences, k, model):
    emb = model.encode(sentences)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
    clusters = {}
    for s, lab in zip(sentences, labels):
        clusters.setdefault(f"Cluster {lab+1}", []).append(s)
    return clusters

def elbow_method(sentences, max_k, model, outdir):
    emb = model.encode(sentences)
    inertias = []
    K = range(1, max_k+1)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42).fit(emb)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(K, inertias, "bx-")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.savefig(os.path.join(outdir, "elbow.png"))
    plt.close()

def run_bertopic(sentences, model):
    emb = model.encode(sentences)
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(sentences, emb)
    clusters = {}
    for s, t in zip(sentences, topics):
        clusters.setdefault(f"Topic {t}", []).append(s)
    return clusters

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="posts.csv")
    ap.add_argument("--k", type=int, default=6, help="K for KMeans")
    ap.add_argument("--out", default="results", help="output dir")
    ap.add_argument("--reports-only", action="store_true", help="Use only report posts")
    ap.add_argument("--elbow", type=int, default=0, help="Run elbow method up to max k")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    posts = load_sentences(args.input, reports_only=args.reports_only)
    if not posts:
        print("No posts found to cluster.")
        return

    # Elbow method if requested
    if args.elbow > 0:
        elbow_method(posts, args.elbow, model, args.out)

    # KMeans
    k_clusters = run_kmeans(posts, args.k, model)
    save_json(k_clusters, os.path.join(args.out, "kmeans.json"))

    # BERTopic
    b_clusters = run_bertopic(posts, model)
    save_json(b_clusters, os.path.join(args.out, "bertopic.json"))

    print("Done:", args.out)

if __name__ == "__main__":
    main()
