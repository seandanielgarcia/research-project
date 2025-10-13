#!/usr/bin/env python3
"""
SAE clustering with LLM interpretation using Gemini API.
Based on the approach you provided.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import google.generativeai as genai

def load_env_file(env_path=".env"):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    os.environ[key] = value

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

def load_data(csv_path):
    """Load Reddit posts from CSV."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("title & content", "Title & Content")
    posts = df[text_col].dropna().astype(str).tolist()
    return posts

def generate_embeddings(posts, model_name="all-MiniLM-L6-v2"):
    """Generate sentence embeddings."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Generating embeddings...")
    embeddings = model.encode(posts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings

def train_sae(embeddings, hidden_dim=128, epochs=50, lr=1e-3, lambda_sparse=1e-3):
    """Train the Sparse Autoencoder."""
    input_dim = embeddings.shape[1]
    sae = SparseAutoencoder(input_dim, hidden_dim)
    
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X = torch.tensor(embeddings, dtype=torch.float32)
    
    print(f"Training SAE with {hidden_dim} features...")
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        x_hat, z = sae(X)
        mse_loss = criterion(x_hat, X)
        sparsity_loss = torch.mean(torch.abs(z))
        loss = mse_loss + lambda_sparse * sparsity_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return sae

def extract_features(sae, embeddings):
    """Extract latent features from SAE."""
    X = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        latent = sae.encoder(X).detach().numpy()
    return latent

def find_top_activating_posts(latent, posts, top_n=5):
    """Find top-activating posts for each feature."""
    top_posts = {}
    for i in range(latent.shape[1]):
        top_idxs = latent[:, i].argsort()[-top_n:][::-1]
        top_posts[i] = [posts[j] for j in top_idxs]
    return top_posts

def setup_gemini(api_key):
    """Setup Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def generate_hypothesis_with_gemini(gemini_model, feature_id, examples):
    """Generate hypothesis using Gemini API."""
    prompt = f"""Analyze these Reddit posts and identify the common theme or pattern. What concept or topic do they all relate to?

Posts:
{chr(10).join(f"- {ex}" for ex in examples)}

Provide a concise hypothesis (1-2 sentences) describing what this feature represents:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        hypothesis = response.text.strip()
        return hypothesis
    except Exception as e:
        print(f"Error generating hypothesis for feature {feature_id}: {e}")
        return f"Feature {feature_id}: Error generating hypothesis"

def assign_posts_to_features(latent, posts):
    """Assign each post to its most active feature."""
    post_assignments = latent.argmax(axis=1)
    
    #  clusters based on feature assignments
    clusters = {}
    for i, (post, feature_id) in enumerate(zip(posts, post_assignments)):
        cluster_name = f"Feature_Cluster_{feature_id}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(post)
    
    return clusters, post_assignments

def save_results(clusters, hypotheses, post_assignments, output_dir):
    """Save clustering results."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "sae_llm_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "sae_llm_hypotheses.json"), "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2, ensure_ascii=False)
    
    # Save assignments
    assignments_df = pd.DataFrame({
        "post_index": range(len(post_assignments)),
        "assigned_feature": post_assignments
    })
    assignments_df.to_csv(os.path.join(output_dir, "post_feature_assignments.csv"), index=False)
    
    print(f"Results saved to {output_dir}")

def main():
    # Load environment variables from .env file
    load_env_file()
    
    parser = argparse.ArgumentParser(description="SAE clustering with LLM interpretation")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--gemini-key", help="Gemini API key (or use GEMINI_KEY env var)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="SAE hidden dimension")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--top-n", type=int, default=5, help="Top N posts per feature")
    parser.add_argument("--out", default="results/clustering/sae_llm", help="Output directory")
    parser.add_argument("--lambda-sparse", type=float, default=1e-3, help="Sparsity penalty")
    
    args = parser.parse_args()
    
    gemini_key = args.gemini_key or os.getenv("gemini_key")
    if not gemini_key:
        raise ValueError("Gemini API key required. Provide via --gemini-key or set gemini_key environment variable")
    
    print(f"Loading data from {args.input}")
    posts = load_data(args.input)
    print(f"Loaded {len(posts)} posts")
    
    embeddings = generate_embeddings(posts)
    
    sae = train_sae(embeddings, hidden_dim=args.hidden_dim, epochs=args.epochs, 
                   lambda_sparse=args.lambda_sparse)
    
    print("Extracting latent features...")
    latent = extract_features(sae, embeddings)
    
    print(f"Finding top {args.top_n} posts per feature...")
    top_posts = find_top_activating_posts(latent, posts, top_n=args.top_n)
    
    print("Setting up Gemini API...")
    gemini_model = setup_gemini(gemini_key)
    
    print("Generating hypotheses with Gemini...")
    hypotheses = {}
    for feature_id, examples in tqdm(top_posts.items()):
        hypothesis = generate_hypothesis_with_gemini(gemini_model, feature_id, examples)
        hypotheses[feature_id] = hypothesis
        print(f"Feature {feature_id}: {hypothesis}")
    
    print("Assigning posts to features...")
    clusters, post_assignments = assign_posts_to_features(latent, posts)
    
    print("\n=== Cluster Sizes ===")
    for cluster_name, cluster_posts in clusters.items():
        print(f"{cluster_name}: {len(cluster_posts)} posts")
    
    save_results(clusters, hypotheses, post_assignments, args.out)
    
    print(f"\nSAE clustering with LLM interpretation completed!")
    print(f"Results saved to {args.out}")

if __name__ == "__main__":
    main()
