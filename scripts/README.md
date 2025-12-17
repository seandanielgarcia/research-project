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

**weighted_kmeans.py**
- Weighted kmeans clustering using post scores
- Supports OpenAI embeddings or sentence-transformers
- Creates date distribution boxplots

Arguments:
- `--input`: Path to CSV file
- `--k`: Number of clusters (default: 16)
- `--out`: Output directory
- `--model`: SentenceTransformer model or OpenAI embedding model
- `--reports-only`: Only use report posts
- `--api-key`: OpenAI API key (if using OpenAI embeddings)

**bertopic_outlier_reduction.py**
- BERTopic clustering with outlier reduction strategies
- Can generate cluster labels using Gemini
- Supports different outlier assignment methods (c-tf-idf, embeddings, distributions, probabilities)

Arguments:
- `--input`: Path to CSV file
- `--out`: Output directory
- `--text-field`: "summary" or "content" (default: summary)
- `--strategy`: Outlier reduction strategy (default: distributions)
- `--reports-only`: Only use report posts
- `--generate-labels`: Generate cluster labels using Gemini
- `--label-model`: Gemini model name (default: gemini-1.5-pro)

**train_test_cluster.py**
- Split data into train/test and cluster separately
- Useful for testing cluster stability

Arguments:
- `--input`: Path to CSV file
- `--k`: Number of clusters (default: 16)
- `--out`: Output directory
- `--model`: SentenceTransformer model name
- `--reports-only`: Only use report posts

**sae_with_llm.py**
- Use Sparse Autoencoders for clustering
- LLM-based interpretation of learned features
- Requires Gemini API key
- Outputs clusters with labeled hypotheses

Arguments:
- `--input`: Path to CSV file
- `--k`: Number of clusters
- `--out`: Output directory

**label_clusters.py**
- Generate cluster labels using OpenAI
- Uses contrastive examples to distinguish clusters
- Outputs labels with two-sentence descriptions

Arguments:
- `--clusters`: Path to clusters JSON
- `--csv`: Path to CSV with post data
- `--out`: Output JSON file (default: cluster_labels.json)
- `--model`: OpenAI model name (default: gpt-5-mini)
- `--sample`: Examples per cluster (default: 12)
- `--api-key`: OpenAI API key

**plot_cluster_timelines.py**
- Plot cluster activity over time as stacked histograms
- Shows daily post counts per cluster
- Can use cluster labels for better legends

Arguments:
- `--clusters`: Path to clusters.json
- `--csv`: CSV file with Post ID and Timestamp columns
- `--labels`: Optional cluster_labels.json for labeled legends
- `--out`: Output PNG path

## Utilities

**utils.py**
Helper functions for exploring clustering results:
- Load and inspect cluster data
- Retrieve post content by ID
- Search within clusters
- Get statistics about clusters

**dump_extraction.py**
- Extract and clean data from scraped files

**elbow_curve.py**
- Generate elbow curves to determine optimal number of clusters
- Plots inertia vs k to help choose cluster count

Arguments:
- `--input`: Path to CSV file
- `--max-k`: Maximum k to test (default: 20)
- `--output`: Output directory for plots

**wordfrequency_plot.py**
- Plot keyword frequency trends over time
- Tracks mentions of specific terms (e.g., "sam", "altman")
- Hardcoded paths, modify script to use your data

