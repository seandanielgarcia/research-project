#!/usr/bin/env python3
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def load_cluster_file(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_id_format(cluster_data: Dict[str, List[str]]) -> bool:
    if not cluster_data:
        return False
    
    first_cluster = next((cluster for cluster in cluster_data.values() if cluster), None)
    if not first_cluster:
        return False
    
    for item in first_cluster[:3]:
        if isinstance(item, str):
            word_count = len(item.split())
            char_count = len(item)
            
            if word_count > 10 or char_count > 100:
                return False
            
            if char_count < 20 and item.replace('_', '').isalnum():
                return True
    
    return False


def load_post_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_post_id_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    cols = {c.lower().strip(): c for c in df.columns}
    post_id_col = cols.get("post id", None)
    
    if post_id_col is None:
        raise ValueError("CSV file must have a 'Post ID' column")
    
    post_map = {}
    for _, row in df.iterrows():
        post_id = str(row[post_id_col])
        post_map[post_id] = row.to_dict()
    
    return post_map


def get_cluster_content(cluster_id: str, json_path: str, csv_path: Optional[str] = None, 
                       field: str = "Summary") -> List[Dict[str, Any]]:
    clusters = load_cluster_file(json_path)
    
    if cluster_id not in clusters:
        raise ValueError(f"Cluster '{cluster_id}' not found in {json_path}")
    
    cluster_items = clusters[cluster_id]
    
    is_ids = is_id_format(clusters)
    
    if not is_ids:
        if field.lower() == "summary":
            return [{"post_id": f"unknown_{i}", field: item} 
                   for i, item in enumerate(cluster_items)]
        else:
            if csv_path is None:
                raise ValueError(f"csv_path required to get field '{field}' from old format JSON")
    
    if csv_path is None:
        raise ValueError("csv_path is required when JSON contains post IDs")
    
    df = load_post_data(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    
    field_lower = field.lower()
    data_col = cols.get(field_lower, field)
    
    if data_col not in df.columns:
        raise ValueError(f"Field '{field}' not found in CSV file")
    
    post_id_col = cols.get("post id", None)
    if post_id_col is None:
        raise ValueError("CSV file must have a 'Post ID' column")
    
    id_to_data = {}
    for _, row in df.iterrows():
        post_id = str(row[post_id_col])
        id_to_data[post_id] = row[data_col] if pd.notna(row[data_col]) else ""
    
    results = []
    for post_id in cluster_items:
        if post_id in id_to_data:
            results.append({
                "post_id": post_id,
                field: id_to_data[post_id]
            })
        else:
            results.append({
                "post_id": post_id,
                field: f"[Post ID {post_id} not found in CSV]"
            })
    
    return results


def get_cluster_summaries(cluster_id: str, json_path: str, csv_path: Optional[str] = None) -> List[str]:
    results = get_cluster_content(cluster_id, json_path, csv_path, field="Summary")
    return [item["Summary"] for item in results]


def get_cluster_full_content(cluster_id: str, json_path: str, csv_path: str) -> List[str]:
    results = get_cluster_content(cluster_id, json_path, csv_path, field="Full Content")
    return [item["Full Content"] for item in results]


def get_all_clusters(json_path: str) -> List[str]:
    clusters = load_cluster_file(json_path)
    return list(clusters.keys())


def get_cluster_sizes(json_path: str) -> Dict[str, int]:
    clusters = load_cluster_file(json_path)
    return {cluster: len(post_ids) for cluster, post_ids in clusters.items()}


def search_cluster_content(cluster_id: str, json_path: str, csv_path: Optional[str],
                          search_term: str, field: str = "Summary") -> List[Dict[str, Any]]:
    results = get_cluster_content(cluster_id, json_path, csv_path, field=field)
    search_term_lower = search_term.lower()
    
    matching = []
    for item in results:
        content = item.get(field, "")
        if search_term_lower in content.lower():
            matching.append(item)
    
    return matching


def list_clusters_with_sizes(json_path: str) -> List[Dict[str, Any]]:
    sizes = get_cluster_sizes(json_path)
    return [{"cluster": k, "size": v} for k, v in sizes.items()]


def clusters_to_dataframe(
    json_path: str,
    csv_path: Optional[str] = None,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    clusters = load_cluster_file(json_path)
    use_fields = fields or ["Summary"]
    rows: List[Dict[str, Any]] = []

    ids_format = is_id_format(clusters)
    id_to_data: Dict[str, Dict[str, Any]] = {}

    if ids_format and csv_path:
        df = load_post_data(csv_path)
        cols = {c.lower().strip(): c for c in df.columns}
        post_id_col = cols.get("post id", None)
        if post_id_col is None:
            raise ValueError("CSV file must have a 'Post ID' column")
        field_cols: Dict[str, str] = {}
        for f in use_fields:
            f_lower = f.lower()
            col = cols.get(f_lower, f)
            if col not in df.columns:
                raise ValueError(f"Field '{f}' not found in CSV file")
            field_cols[f] = col
        for _, r in df.iterrows():
            pid = str(r[post_id_col])
            id_to_data[pid] = {f: (r[field_cols[f]] if pd.notna(r[field_cols[f]]) else "") for f in use_fields}

    for cluster, items in clusters.items():
        for itm in items:
            rec: Dict[str, Any] = {"cluster": cluster}
            if ids_format:
                rec["post_id"] = itm
                if csv_path and itm in id_to_data:
                    rec.update(id_to_data[itm])
            else:
                target = use_fields[0] if use_fields else "Summary"
                rec[target] = itm
            rows.append(rec)

    return pd.DataFrame(rows)


def sample_cluster_items(
    cluster_id: str,
    json_path: str,
    csv_path: Optional[str] = None,
    field: str = "Summary",
    n: int = 20,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    all_items = get_cluster_content(cluster_id, json_path, csv_path, field=field)
    if n <= 0 or len(all_items) <= n:
        return all_items
    rng = np.random.RandomState(random_state)
    idxs = rng.choice(len(all_items), size=n, replace=False)
    return [all_items[int(i)] for i in idxs]


def plot_clusters(json_path: str, output_path: str):
    clusters = load_cluster_file(json_path)
    names, sizes = zip(*sorted(((k, len(v)) for k, v in clusters.items()), key=lambda x: x[1], reverse=True))
    x = np.arange(len(names))
    plt.figure(figsize=(11, 6))
    bars = plt.bar(x, sizes, color="#3b82f6", edgecolor="#1e3a8a", linewidth=0.8, alpha=0.9)
    plt.gca().set_facecolor("#f9fafb")
    plt.title("Cluster Distribution", fontsize=18, weight="bold", color="#111827", pad=20)
    plt.ylabel("Number of Posts", fontsize=12, color="#111827", labelpad=10)
    plt.xticks(x, names, rotation=30, ha="right", fontsize=10, color="#111827")
    plt.yticks(color="#111827")
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width() / 2, size + 0.5, str(size),
                 ha="center", va="bottom", fontsize=10, color="#111827", weight="bold")
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python utils.py <json_path> <csv_path> [cluster_id]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    csv_path = sys.argv[2]
    
    print("Available clusters:")
    clusters = get_all_clusters(json_path)
    sizes = get_cluster_sizes(json_path)
    for cluster in sorted(clusters):
        print(f"  {cluster}: {sizes[cluster]} posts")
    
    if len(sys.argv) >= 4:
        cluster_id = sys.argv[3]
        print(f"\nCluster: {cluster_id}")
        summaries = get_cluster_summaries(cluster_id, json_path, csv_path)
        for i, summary in enumerate(summaries[:5], 1):
            print(f"  {i}. {summary}")
        if len(summaries) > 5:
            print(f"  ... and {len(summaries) - 5} more")
