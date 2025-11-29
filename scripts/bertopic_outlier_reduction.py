#!/usr/bin/env python3
"""
BERTopic with outlier reduction.
Implements multiple outlier reduction strategies to reassign outlier documents to topics.
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

def load_data(csv_path, text_field="summary", reports_only=False):
    """Load data from CSV, supporting both summaries and full content."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    # Support multiple text field options
    if text_field == "summary":
        text_col = cols.get("summary", None)
    elif text_field == "content":
        text_col = cols.get(
            "full content",
            cols.get("content", cols.get("title & content", None))
        )
    else:
        text_col = cols.get(text_field.lower(), None)

    report_col = cols.get("is report", None)
    post_id_col = cols.get("post id", None)

    if text_col is None or text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_field}. Found: {list(df.columns)}")

    data = df
    if reports_only and report_col in df.columns:
        data = df[df[report_col].astype(str).str.lower().isin(["true","yes","y","1"])]

    texts, post_ids = [], {}
    idx = 0

    for _, row in data.iterrows():
        text = str(row[text_col]).strip()
        if text and text.lower() not in ["nan","no content"]:
            pid = str(row[post_id_col]) if post_id_col else f"post_{idx}"
            texts.append(text)
            post_ids[idx] = pid
            idx += 1

    return texts, post_ids



# Run BERTopic with outlier reduction
def run_bertopic_with_outlier_reduction(
    texts,
    post_ids,
    embeddings,
    strategy="distributions",
    calculate_probabilities=False,
    threshold=0,
    reduce_nr_topics=None,
    n_neighbors=15,
    n_components=5,
    min_cluster_size=10
):
    """
    Run BERTopic with outlier reduction.
    
    Strategies:
    - "distributions": Use topic distributions (default)
    - "c-tf-idf": Uses c-TF-IDF representations
    - "embeddings": Uses document embeddings
    - "probabilities": Uses HDBSCAN probabilities (requires calculate_probabilities=True)
    
    Args:
        reduce_nr_topics: If set, reduce topics to this number after outlier reduction
    """

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        prediction_data=True
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=calculate_probabilities,
        verbose=True
    )
  
    print(f"\nFitting BERTopic model...")
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    # Count initial outliers (topic -1)
    initial_outliers = sum(1 for t in topics if t == -1)
    initial_num_topics = len(set(t for t in topics if t != -1))
    print(f"Initial topics: {initial_num_topics}")
    print(f"Initial outliers: {initial_outliers} ({100*initial_outliers/len(topics):.1f}%)")
    
   
    print(f"\nReducing outliers using strategy: {strategy}...")
    reduce_kwargs = {
        "documents": texts,
        "topics": topics,
        "strategy": strategy,
        "threshold": threshold
    }
    
    if strategy == "probabilities":
        if probs is None:
            raise ValueError("Probabilities strategy requires calculate_probabilities=True")
        reduce_kwargs["probabilities"] = probs
    elif strategy == "embeddings":
        reduce_kwargs["embeddings"] = embeddings
    
    new_topics = topic_model.reduce_outliers(**reduce_kwargs)
    

    remaining_outliers = sum(1 for t in new_topics if t == -1)
    print(f"Remaining outliers: {remaining_outliers} ({100*remaining_outliers/len(new_topics):.1f}%)")
    print(f"Outliers reduced: {initial_outliers - remaining_outliers} ({100*(initial_outliers-remaining_outliers)/len(topics):.1f}%)")
    
   
    if reduce_nr_topics is not None:
        print(f"\nReducing number of topics to {reduce_nr_topics}...")
        topic_model.reduce_topics(texts, nr_topics=reduce_nr_topics)
        new_topics = topic_model.topics_
        if probs is not None:
            probs = topic_model.probabilities_
    
    topic_model.update_topics(texts, topics=new_topics)
    
    # Convert to cluster format
    clusters = {}
    for idx, topic in enumerate(new_topics):
        pid = post_ids[idx]
        cluster_name = f"Topic {topic}" if topic != -1 else "Outlier"
        clusters.setdefault(cluster_name, []).append(pid)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    return clusters, new_topics, topics, topic_model, topic_info



def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="BERTopic with outlier reduction")
    ap.add_argument("--input", required=True, help="Path to CSV file")
    ap.add_argument("--out", default="results/clustering/bertopic_outlier_reduction", help="Output directory")
    ap.add_argument("--text-field", choices=["summary", "content"], default="summary",
                    help="Which text field to use (summary or content)")
    ap.add_argument("--strategy", choices=["c-tf-idf", "embeddings", "distributions", "probabilities"],
                    default="distributions", help="Outlier reduction strategy (default: distributions)")
    ap.add_argument("--calculate-probabilities", action="store_true",
                    help="Calculate probabilities (required for 'probabilities' strategy)")
    ap.add_argument("--threshold", type=float, default=0,
                    help="Threshold for assigning topics to outlier documents")
    ap.add_argument("--reduce-nr-topics", type=int, default=None,
                    help="Reduce number of topics to this value after outlier reduction")
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="UMAP n_neighbors parameter")
    ap.add_argument("--n-components", type=int, default=5,
                    help="UMAP n_components parameter")
    ap.add_argument("--min-cluster-size", type=int, default=10,
                    help="HDBSCAN min_cluster_size parameter")
    ap.add_argument("--reports-only", action="store_true",
                    help="If set, cluster only posts marked as reports.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    texts, post_ids = load_data(args.input, text_field=args.text_field, reports_only=args.reports_only)
    
    if not texts:
        print("No valid data found!")
        return
    
    print(f"Loaded {len(texts)} posts")

    # Create embeddings
    print(f"\nGenerating embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # Run BERTopic with outlier reduction
    clusters, new_topics, original_topics, topic_model, topic_info = run_bertopic_with_outlier_reduction(
        texts,
        post_ids,
        embeddings,
        strategy=args.strategy,
        calculate_probabilities=args.calculate_probabilities or (args.strategy == "probabilities"),
        threshold=args.threshold,
        reduce_nr_topics=args.reduce_nr_topics,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_cluster_size=args.min_cluster_size
    )

    # Save results
    save_json(clusters, os.path.join(args.out, f"bertopic_clusters_{args.strategy}.json"))
    
    # Save topic info as CSV
    topic_info.to_csv(os.path.join(args.out, f"topic_info_{args.strategy}.csv"), index=False)
    
    # Save metrics
    initial_outliers = sum(1 for t in original_topics if t == -1)
    remaining_outliers = sum(1 for t in new_topics if t == -1)
    num_topics = len(set(t for t in new_topics if t != -1))
    
    initial_num_topics = len(set(t for t in original_topics if t != -1))
    
    metrics = {
        "num_posts": len(texts),
        "num_topics": num_topics,
        "initial_num_topics": initial_num_topics,
        "strategy": args.strategy,
        "threshold": args.threshold,
        "initial_outliers": int(initial_outliers),
        "remaining_outliers": int(remaining_outliers),
        "outliers_reduced": int(initial_outliers - remaining_outliers),
        "initial_outlier_percentage": float(100 * initial_outliers / len(texts)),
        "remaining_outlier_percentage": float(100 * remaining_outliers / len(texts)),
        "reduction_percentage": float(100 * (initial_outliers - remaining_outliers) / len(texts)),
        "text_field": args.text_field,
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "min_cluster_size": args.min_cluster_size,
        "reduce_nr_topics": args.reduce_nr_topics if args.reduce_nr_topics else None
    }
    save_json(metrics, os.path.join(args.out, f"metrics_{args.strategy}.json"))
    
    print(f"\nResults saved to {args.out}/")
    print(f"Number of topics: {num_topics}")
    print(f"Outlier reduction: {initial_outliers} -> {remaining_outliers}")

if __name__ == "__main__":
    main()

