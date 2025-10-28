#!/usr/bin/env python3
import os, json, argparse, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from bertopic import BERTopic
import numpy as np

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

def run_kmeans(sentences, post_ids, k, model, embed_type):
    emb = get_embeddings(sentences, model, embed_type)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
    clusters = {}
    for idx, (s, lab) in enumerate(zip(sentences, labels)):
        post_id = post_ids.get(idx, f"post_{idx}")
        clusters.setdefault(f"Cluster {lab+1}", []).append(post_id)
    return clusters

def run_bertopic(sentences, post_ids, model, embed_type):
    emb = get_embeddings(sentences, model, embed_type)
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(sentences, emb)
    clusters = {}
    for idx, (s, t) in enumerate(zip(sentences, topics)):
        post_id = post_ids.get(idx, f"post_{idx}")
        clusters.setdefault(f"Topic {t}", []).append(post_id)
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
    ap.add_argument("--embed-type", choices=["sentence", "word"], default="sentence",
                    help="Choose 'sentence' or 'word' embeddings")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences, post_ids = load_sentences(args.input, reports_only=args.reports_only)
    if not sentences:
        print("No valid content found to cluster.")
        return

    k_clusters = run_kmeans(sentences, post_ids, args.k, model, args.embed_type)
    save_json(k_clusters, os.path.join(args.out, f"kmeans_{args.embed_type}.json"))

    b_clusters = run_bertopic(sentences, post_ids, model, args.embed_type)
    save_json(b_clusters, os.path.join(args.out, f"bertopic_{args.embed_type}.json"))

    print(f"Done: {args.out} using {args.embed_type}-level embeddings")

if __name__ == "__main__":
    main()
