import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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

def cluster_test_data(test_sentences, test_post_ids, k, model, embed_type):
    """Cluster the test dataset independently with k=16"""
    print(f"Embedding {len(test_sentences)} test posts...")
    test_emb = get_embeddings(test_sentences, model, embed_type)
    print(f"Clustering test data with k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    test_labels = km.fit_predict(test_emb)
    
    test_clusters = {}
    for idx, (post_id, label) in enumerate(zip(test_post_ids.values(), test_labels)):
        cluster_name = f"Cluster {label+1}"
        test_clusters.setdefault(cluster_name, []).append(post_id)
    
    return test_clusters, test_labels

def load_existing_test_clusters(test_json_path):
    """Load the test clusters from train_test_cluster.py output"""
    with open(test_json_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    return clusters

def create_label_mapping(clusters):
    """Create a mapping from post_id to cluster label"""
    post_id_to_label = {}
    for cluster_name, post_ids in clusters.items():
        # Handle both "Cluster 1" and "Cluster 1 - Label" formats
        import re
        match = re.search(r'Cluster (\d+)', cluster_name)
        if match:
            cluster_num = int(match.group(1)) - 1  # Convert "Cluster 1" to 0
            for post_id in post_ids:
                post_id_to_label[post_id] = cluster_num
    return post_id_to_label

def compare_clusters(clusters1, clusters2, test_post_ids):
    """Compare two clusterings using post IDs"""
    mapping1 = create_label_mapping(clusters1)
    mapping2 = create_label_mapping(clusters2)
    
    # Get labels for all test post IDs
    labels1 = []
    labels2 = []
    for post_id in test_post_ids.values():
        if post_id in mapping1 and post_id in mapping2:
            labels1.append(mapping1[post_id])
            labels2.append(mapping2[post_id])
    
    if len(labels1) == 0:
        print("No overlapping post IDs found!")
        return None, None
    
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    return ari, nmi

def print_cluster_sizes(clusters, name):
    """Print cluster size statistics"""
    sizes = [len(post_ids) for post_ids in clusters.values()]
    print(f"\n{name} cluster sizes:")
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Total posts: {sum(sizes)}")
    print(f"  Min size: {min(sizes)}")
    print(f"  Max size: {max(sizes)}")
    print(f"  Mean size: {np.mean(sizes):.1f}")
    print(f"  Median size: {np.median(sizes):.1f}")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser(description="Cluster only the test dataset and compare with train/test predictions")
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--k", type=int, default=16, help="K for KMeans")
    ap.add_argument("--out", default="results/clustering/test_only", help="Output directory")
    ap.add_argument("--reports-only", action="store_true", help="Use only report posts")
    ap.add_argument("--embed-type", choices=["sentence", "word"], default="sentence",
                    help="Choose 'sentence' or 'word' embeddings")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--compare-with", help="Path to test clusters JSON from train_test_cluster.py to compare")
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
    test_indices = indices[split_idx:]

    test_sentences = [all_sentences[i] for i in test_indices]
    test_post_ids = {new_idx: all_post_ids[old_idx] for new_idx, old_idx in enumerate(test_indices)}

    print(f"Using {len(test_sentences)} test posts (second half)")
    
    # Cluster test data independently
    test_clusters, test_labels = cluster_test_data(
        test_sentences, test_post_ids,
        args.k, model, args.embed_type
    )

    test_out = os.path.join(args.out, f"test_only_kmeans_k{args.k}_{args.embed_type}.json")
    save_json(test_clusters, test_out)

    print_cluster_sizes(test_clusters, "Test-only clustering")
    print(f"\nSaved test-only clusters to: {test_out}")

    # Compare with existing test clusters if provided
    if args.compare_with and os.path.exists(args.compare_with):
        print(f"\nComparing with clusters from: {args.compare_with}")
        existing_clusters = load_existing_test_clusters(args.compare_with)
        print_cluster_sizes(existing_clusters, "Train/test prediction clusters")
        
        ari, nmi = compare_clusters(test_clusters, existing_clusters, test_post_ids)
        if ari is not None:
            print(f"\nComparison metrics:")
            print(f"  Adjusted Rand Index (ARI): {ari:.4f} (1.0 = identical, 0.0 = random)")
            print(f"  Normalized Mutual Info (NMI): {nmi:.4f} (1.0 = identical, 0.0 = independent)")
            
            comparison_out = os.path.join(args.out, f"comparison_k{args.k}_{args.embed_type}.json")
            save_json({
                "adjusted_rand_index": float(ari),
                "normalized_mutual_info": float(nmi),
                "test_only_clusters": len(test_clusters),
                "predicted_clusters": len(existing_clusters),
                "overlapping_posts": len([pid for pid in test_post_ids.values() 
                                         if pid in create_label_mapping(test_clusters) 
                                         and pid in create_label_mapping(existing_clusters)])
            }, comparison_out)
            print(f"  Saved comparison metrics to: {comparison_out}")

if __name__ == "__main__":
    main()

