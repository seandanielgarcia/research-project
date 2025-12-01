#!/usr/bin/env python3
"""
BERTopic with outlier reduction.
Implements multiple outlier reduction strategies to reassign outlier documents to topics.
"""
import os
import json
import argparse
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    GEMINI_AVAILABLE = True
    load_dotenv()
except ImportError:
    GEMINI_AVAILABLE = False

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


def load_existing_labels(labels_path: str) -> Dict[str, str]:
    """Load existing labels if file exists."""
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def make_label_prompt(cluster_name: str, examples: List[str], max_examples: int) -> str:
    """Create prompt for label generation."""
    trimmed = [s for s in examples if s and str(s).strip()]
    if max_examples > 0:
        trimmed = trimmed[:max_examples]
    
    joined_examples = "\n\n".join(f"Example {i+1}: {ex}" for i, ex in enumerate(trimmed))
    return (
        "You are labeling clusters of Reddit posts about AI behavior. "
        "Your task is to produce a **distinct, descriptive label (4–8 words)** that captures what makes this cluster "
        "unique compared to other possible clusters.\n\n"
        "Guidelines:\n"
        "- The label should be general enough to unite all posts in the cluster not just one of the examples\n"
        "- Avoid generic phrases like 'AI discussion' or 'general opinions about AI'.\n"
        "- Emphasize what specifically unites these posts — e.g. topic, attitude, focus, or controversy.\n"
        "- Avoid punctuation except basic hyphens.\n"
        "4 to 6 words is best\n"
        "- Return only the label.\n\n"
        f"Cluster name (internal): {cluster_name}\n\n"
        f"Examples:\n{joined_examples}"
    )


def call_gemini_for_label(prompt: str, model, max_retries: int = 3, retry_delay: float = 2.0) -> str:
    """Call Gemini API with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            if not hasattr(resp, "text") or not resp.text:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return "(empty response)"
            return resp.text.strip().splitlines()[0].strip("\" ")
        except Exception as e:
            error_msg = f"(error: {e.__class__.__name__})"
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
                continue
            return error_msg
    return "(error: max retries exceeded)"


def generate_cluster_labels(
    clusters: Dict[str, List[str]],
    csv_path: str,
    model,
    sample_per_cluster: int = -1,
    sleep_s: float = 0.5,
    existing_labels: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Generate labels for clusters using Gemini API."""
    labels = existing_labels.copy() if existing_labels else {}
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    summary_col = cols.get("summary", None)
    post_id_col = cols.get("post id", None)
    
    if summary_col is None:
        print("Warning: 'Summary' column not found, cannot generate labels")
        return labels
    
    # Create post_id to summary mapping
    post_to_summary = {}
    for _, row in df.iterrows():
        pid = str(row[post_id_col]) if post_id_col else None
        if pid:
            summary = str(row[summary_col]).strip()
            if summary and summary.lower() not in ["nan", "no content"]:
                post_to_summary[pid] = summary
    
    # Generate labels for clusters
    clusters_to_label = []
    for cluster_name in clusters.keys():
        # Skip if already has a valid label (not an error)
        if cluster_name in labels:
            existing_label = labels[cluster_name]
            if not existing_label.startswith("(error:") and not existing_label.startswith("(empty"):
                continue
        clusters_to_label.append(cluster_name)
    
    if not clusters_to_label:
        print("All clusters already have labels")
        return labels
    
    print(f"\nGenerating labels for {len(clusters_to_label)} clusters...")
    for cluster_name in clusters_to_label:
        post_ids = clusters[cluster_name]
        examples = [post_to_summary.get(pid, "") for pid in post_ids if pid in post_to_summary]
        examples = [ex for ex in examples if ex]
        
        if not examples:
            labels[cluster_name] = "(no content)"
            continue
        
        prompt = make_label_prompt(cluster_name, examples, sample_per_cluster)
        label = call_gemini_for_label(prompt, model)
        labels[cluster_name] = label
        print(f"  {cluster_name}: {label}")
        
        if sleep_s > 0:
            time.sleep(sleep_s)
    
    return labels


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
    ap.add_argument("--generate-labels", action="store_true",
                    help="Generate cluster labels using Gemini API")
    ap.add_argument("--label-model", default="gemini-1.5-pro", help="Gemini model for label generation")
    ap.add_argument("--label-sample", type=int, default=-1, help="Examples per cluster for labeling (-1 for all)")
    ap.add_argument("--label-sleep", type=float, default=0.5, help="Sleep seconds between label API calls")
    ap.add_argument("--api-key", help="Google API key (fallback: GOOGLE_API_KEY env)")
    ap.add_argument("--finish-labels", action="store_true",
                    help="Finish incomplete labels (retry failed ones)")
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
    
    # Generate labels if requested
    if args.generate_labels or args.finish_labels:
        if not GEMINI_AVAILABLE:
            print("\nWarning: Gemini API not available. Install google-generativeai and python-dotenv to generate labels.")
        else:
            api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("gemini_key")
            if not api_key:
                print("\nWarning: No API key found. Set --api-key or export GOOGLE_API_KEY")
            else:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(args.label_model)
                
                labels_path = os.path.join(args.out, "cluster_labels.json")
                existing_labels = {}
                if args.finish_labels:
                    existing_labels = load_existing_labels(labels_path)
                    print(f"\nLoaded {len(existing_labels)} existing labels")
                
                labels = generate_cluster_labels(
                    clusters,
                    args.input,
                    gemini_model,
                    sample_per_cluster=args.label_sample,
                    sleep_s=args.label_sleep,
                    existing_labels=existing_labels
                )
                
                save_json(labels, labels_path)
                print(f"\n✅ Saved {len(labels)} cluster labels → {labels_path}")

if __name__ == "__main__":
    main()

