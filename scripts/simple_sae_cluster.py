#!/usr/bin/env python3
"""
Simple SAE-based clustering for Reddit data.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn

class SimpleSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k_active):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.k_active = k_active
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        top_k_values, top_k_indices = torch.topk(encoded, self.k_active, dim=1)
        sparse_encoded = torch.zeros_like(encoded)
        sparse_encoded.scatter_(1, top_k_indices, top_k_values)
        
        decoded = self.decoder(sparse_encoded)
        return decoded, sparse_encoded
    
    def get_features(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            top_k_values, top_k_indices = torch.topk(encoded, self.k_active, dim=1)
            sparse_encoded = torch.zeros_like(encoded)
            sparse_encoded.scatter_(1, top_k_indices, top_k_values)
            return sparse_encoded.numpy()

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("title & content", "Title & Content")
    sentences = df[text_col].dropna().astype(str).tolist()
    return sentences

def get_embeddings(sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

def train_sae(embeddings, hidden_dim=64, k_active=8, epochs=50):
    model = SimpleSAE(embeddings.shape[1], hidden_dim, k_active)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, _ = model(torch.tensor(embeddings, dtype=torch.float32))
        loss = nn.functional.mse_loss(reconstructed, torch.tensor(embeddings, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def cluster_on_sae_features(embeddings, k_clusters=8, hidden_dim=64, k_active=8):
    print(f"Training SAE with {hidden_dim} features, {k_active} active")
    sae = train_sae(embeddings, hidden_dim, k_active)
    
    print("Getting SAE features")
    features = sae.get_features(torch.tensor(embeddings, dtype=torch.float32))
    
    print(f"Clustering {features.shape[0]} examples into {k_clusters} clusters")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    return cluster_labels, features, sae

def save_results(sentences, cluster_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    clusters = {}
    for sentence, label in zip(sentences, cluster_labels):
        cluster_name = f"SAE_Cluster_{label + 1}"
        clusters.setdefault(cluster_name, []).append(sentence)
    
    with open(os.path.join(output_dir, "sae_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--k", type=int, default=8, help="Number of clusters")
    parser.add_argument("--hidden", type=int, default=64, help="SAE hidden dimension")
    parser.add_argument("--active", type=int, default=8, help="Number of active SAE features")
    parser.add_argument("--out", default="results/clustering/sae", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    sentences = load_data(args.input)
    print(f"Loaded {len(sentences)} sentences")
    
    print("Generating embeddings")
    embeddings = get_embeddings(sentences)
    
    print("Running SAE clustering")
    cluster_labels, features, sae = cluster_on_sae_features(
        embeddings, 
        k_clusters=args.k,
        hidden_dim=args.hidden,
        k_active=args.active
    )
    
    silhouette_avg = silhouette_score(features, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
 
    for i in range(args.k):
        count = np.sum(cluster_labels == i)
        print(f"Cluster {i+1}: {count} posts")
    
    save_results(sentences, cluster_labels, args.out)

if __name__ == "__main__":
    main()
