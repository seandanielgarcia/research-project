# Scripts

Scripts for data collection, clustering, and analysis.

## Data Collection

**scrape_and_clean.py**
- Scrape Reddit posts from AI-related subreddits
- Use Gemini API to classify posts as "reports" of problematic AI behavior
- Output cleaned CSV with metadata

## Clustering

**sentence_cluster.py**
- Cluster posts using KMeans or BERTopic
- Support sentence-level or word-level embeddings
- Save post IDs (not text summaries) for efficiency

Arguments:
- `--input`: Path to CSV file
- `--k`: Number of clusters (default: 6)
- `--out`: Output directory
- `--reports-only`: Only use report posts
- `--embed-type`: "sentence" or "word"

**hierarchical_kmeans.py**
- Re-cluster existing clusters hierarchically
- Split large clusters into sub-clusters
- Can provide CSV for content-aware re-clustering

Arguments:
- `--input`: Path to existing kmeans JSON
- `--out`: Output JSON file
- `--depth`: Recursion depth (default: 2)
- `--k`: Subclusters per level (default: 3)
- `--csv`: Optional CSV for content-based clustering

**sae_with_llm.py**
- Use Sparse Autoencoders for clustering
- LLM-based interpretation of learned features
- Requires Gemini API key
- Outputs clusters with labeled hypotheses

Arguments:
- `--input`: Path to CSV file
- `--k`: Number of clusters
- `--out`: Output directory

## Utilities

**utils.py**
Helper functions for exploring clustering results:
- Load and inspect cluster data
- Retrieve post content by ID
- Search within clusters
- Get statistics about clusters

**dump_extraction.py**
Extract and clean data from scraped files.

**append_csvs.py**
Combine multiple CSV files.

**elbow_curve.py**
Generate elbow curves to determine optimal number of clusters.

