#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def load_sentences(csv_path, reports_only=False):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    text_col = cols.get(
        "summary",
        cols.get("content",
        cols.get("full content", cols.get("title & content", None)))
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


def get_embeddings(sentences, model, embedding_model=None, openai_client=None):
    print(f"Generating embeddings for {len(sentences)} sentences...")
    
    if embedding_model and openai_client:
        # Use OpenAI embeddings
        embeddings = []
        batch_size = 100
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            response = openai_client.embeddings.create(
                model=embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"  Processed {min(i+batch_size, len(sentences))}/{len(sentences)} sentences...")
        embeddings = np.array(embeddings)
    else:
        # Use SentenceTransformer
        embeddings = model.encode(sentences, show_progress_bar=True)
    
    print("Embeddings complete.")
    return embeddings


def run_weighted_kmeans(emb, post_ids, scores, k, random_state=42):
    w = np.log1p(np.array(scores))
    w = w / w.sum()

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(emb, sample_weight=w)

    clusters = {}
    for idx, label in enumerate(labels):
        pid = post_ids[idx]
        cluster_name = f"Cluster {label+1}"
        clusters.setdefault(cluster_name, []).append(pid)

    return km, clusters


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_post_dates(csv_path):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    
    post_id_col = cols.get("post id", None)
    timestamp_col = cols.get("timestamp", None)
    
    if post_id_col is None or timestamp_col is None:
        return {}
    
    post_dates = {}
    for _, row in df.iterrows():
        pid = str(row[post_id_col])
        ts_str = str(row[timestamp_col])
        
        if pd.notna(ts_str) and ts_str.lower() not in ["nan", "no content", ""]:
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                post_dates[pid] = dt
            except Exception:
                continue
    
    return post_dates


def get_cluster_dates(clusters, post_dates):
    cluster_dates = {}
    
    for cluster_name, post_ids in clusters.items():
        dates = []
        for pid in post_ids:
            if pid in post_dates:
                dates.append(post_dates[pid])
        
        if dates:
            cluster_dates[cluster_name] = dates
    
    return cluster_dates


def create_box_plot(cluster_dates, output_path, title="Cluster Date Distributions"):
    if not cluster_dates:
        return
    
    cluster_names = []
    date_arrays = []
    all_dates = []
    
    sorted_clusters = sorted(cluster_dates.items(), key=lambda x: x[0])
    
    for cluster_name, dates in sorted_clusters:
        if len(dates) > 0:
            cluster_names.append(cluster_name)
            date_arrays.append(dates)
            all_dates.extend(dates)
    
    if not date_arrays:
        return
    
    first_date = min(all_dates)
    last_date = max(all_dates)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(cluster_names) * 0.4)))
    
    date_numeric = []
    for dates in date_arrays:
        date_numeric.append([mdates.date2num(d) for d in dates])
    
    bp = ax.boxplot(date_numeric, labels=cluster_names, vert=True, patch_artist=True)
    
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylim(mdates.date2num(first_date), mdates.date2num(last_date))
    ax.set_yticks([mdates.date2num(first_date), mdates.date2num(last_date)])
    ax.set_yticklabels([first_date.strftime('%Y-%m-%d'), last_date.strftime('%Y-%m-%d')])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Date', fontsize=12)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="results/weighted_mpnet")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                    help="SentenceTransformer model or OpenAI embedding model (e.g., text-embedding-ada-002)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--reports-only", action="store_true",
                    help="If set, run clustering only on rows marked as reports.")
    ap.add_argument("--api-key", help="OpenAI API key (required if using OpenAI embeddings)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # Check if using OpenAI embeddings
    use_openai = args.model.startswith("text-embedding-") or args.model.startswith("text-")
    if use_openai:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise SystemExit("OpenAI API key required. Set --api-key or export OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)
        model = None
        embedding_model = args.model
    else:
        model = SentenceTransformer(args.model)
        openai_client = None
        embedding_model = None

    sentences, post_ids, scores = load_sentences(args.input, reports_only=args.reports_only)
    if not sentences:
        return

    emb = get_embeddings(sentences, model, embedding_model=embedding_model, openai_client=openai_client)

    km, clusters = run_weighted_kmeans(emb, post_ids, scores, args.k, args.random_state)

    save_json(clusters, os.path.join(args.out, "clusters.json"))

    print("\nCreating date distribution plots...")
    post_dates = load_post_dates(args.input)
    
    if post_dates:
        cluster_dates = get_cluster_dates(clusters, post_dates)
        
        if cluster_dates:
            plot_path = os.path.join(args.out, "clusters_date_boxplot.png")
            create_box_plot(cluster_dates, plot_path, title="Cluster Date Distributions")
            print(f"Saved clusters date plot to {plot_path}")
    else:
        print("No timestamp data found, skipping date plots")

    print("Done weighted KMeans.")

if __name__ == "__main__":
    main()
