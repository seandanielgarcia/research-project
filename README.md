# Reddit Scraper

A tool for scraping Reddit posts about LLM Harms, categorizing them, and analyzing patterns using various clustering techniques.

# Overview

This project scrapes Reddit posts related to AI systems (like ChatGPT, Claude, etc.) and uses clustering algorithms to identify patterns and categorize incidents such as hallucinations, misinformation, sycophantic behavior, and misalignment.

## Data

- Source: Reddit posts from various AI-related subreddits
- Location: `data/combined_posts.csv`
- Contains post IDs, summaries, titles, content, metadata

## Scripts

## Data Collection

**scrape_and_clean.py**
- Scrapes Reddit posts about AI issues
- Uses Gemini API to determine if post is a "report" of problematic behavior
- Outputs cleaned data to CSV

example usage
```bash
python scripts/scrape_and_clean.py --output data/output.csv
```

### Clustering

**sentence_cluster.py**
- Clusters posts using KMeans and BERTopic
- Saves post IDs per cluster (not summaries) for efficiency
- Supports sentence or word-level embeddings

example usage
```bash 
python scripts/sentence_cluster.py --input data/combined_posts.csv --k 6 --out results/clustering/my_clusters/
```

**hierarchical_kmeans.py**
- Performs hierarchical clustering on existing clusters
- Re-clusters large clusters into sub-clusters
- Can work with IDs for faster processing


example usage
```bash
python scripts/hierarchical_kmeans.py --input results/clustering/kmeans.json --csv data/combined_posts.csv --out results/hierarchical.json
```

### Analysis

**sae_with_llm.py**
- Uses Sparse Autoencoders (SAE) for clustering
- LLM-based interpretation of clusters
- Requires Gemini API key

example usage
```bash
python scripts/sae_with_llm.py --input data/combined_posts.csv --k 32 --out results/sae_clusters/
```

## Utilities

**utils.py** provides helper functions for exploring clusters:

example usage
```python
from utils import get_cluster_summaries, get_all_clusters

clusters = get_all_clusters('results/clustering/kmeans.json')

summaries = get_cluster_summaries('Cluster 1', 'results/clustering/kmeans.json', 'data/combined_posts.csv')
```

Available functions:
- `get_cluster_summaries()` - Get summary texts for a cluster
- `get_cluster_full_content()` - Get full content for a cluster
- `get_all_clusters()` - List all cluster names
- `get_cluster_sizes()` - Get number of posts per cluster
- `search_cluster_content()` - Search for terms within a cluster

## Output Format

Clustering results are stored as JSON with post IDs:

```json
{
  "Cluster 1": ["post_id_1", "post_id_2", ...],
  "Cluster 2": ["post_id_3", "post_id_4", ...]
}
```

Use `utils.py` to retrieve actual post content using these IDs.

## Requirements

See `requirements.txt` for dependencies. Key packages:
- pandas
- sentence-transformers
- sklearn
- bertopic
- google-generativeai (for LLM features)
- praw (for Reddit API)

## Setup

1. Create `.env` file with credentials:
```
client_id=your_reddit_client_id
client_secret=your_reddit_client_secret
user_agent=your_user_agent
username=your_username
password=your_password
gemini_key=your_gemini_api_key
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Results

Stored in `results/clustering/`:
- Each subdirectory contains clustering results
- JSON files use post IDs (new format)
- Use utils.py to retrieve and explore content

