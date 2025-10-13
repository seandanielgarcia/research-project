#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import argparse

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

def elbow_method(sentences, max_k, model, output_dir):
    print(f"Generating embeddings for {len(sentences)} sentences...")
    emb = model.encode(sentences)
    
    print(f"Running KMeans for k=1 to k={max_k}...")
    inertias = []
    K = range(1, max_k+1)
    
    for k in K:
        print(f"  Processing k={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
        inertias.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, "bx-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method for Optimal k", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(K)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "elbow.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Elbow curve saved to: {output_path}")
    
    for i, (k, inertia) in enumerate(zip(K, inertias)):
        if i > 0:
            improvement = inertias[i-1] - inertia
            print(f"k={k}: Inertia={inertia:.2f}, Improvement={improvement:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Generate elbow curve for clustering")
    parser.add_argument("--input", required=True, help="Path to CSV file with posts")
    parser.add_argument("--max-k", type=int, default=15, help="Maximum k to test (default: 15)")
    parser.add_argument("--output", default="results/elbow", help="Output directory (default: results/elbow)")
    parser.add_argument("--reports-only", action="store_true", help="Use only report posts")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    
    args = parser.parse_args()
    
    print(f"Loading sentences from: {args.input}")
    sentences = load_sentences(args.input, reports_only=args.reports_only)
    
    if not sentences:
        print("No posts found to cluster.")
        return
    
    print(f"Loaded {len(sentences)} sentences")
    
    print(f"Loading sentence transformer model: {args.model}")
    model = SentenceTransformer(args.model)
    
    elbow_method(sentences, args.max_k, model, args.output)

if __name__ == "__main__":
    main()
