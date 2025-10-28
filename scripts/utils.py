#!/usr/bin/env python3
import json
import pandas as pd
from typing import Dict, List, Any, Optional


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
