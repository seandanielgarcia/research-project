#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def append_csv_files() -> Optional[Path]:
    """
    Append the legacy CSV files in the old data directory while preserving structure.
    Handles different column structures by standardizing them.

    Returns the path to the combined CSV, or None if no data was combined.
    """

    data_dir = Path("/Users/seangarcia/Downloads/reddit_scraper/old")
    files = [
        "993octoberposts.csv",
        "new_posts_9_30.csv",
        "past3months.csv",
    ]

    standard_columns = [
        "Post ID",
        "Summary",
        "Full Title",
        "Full Content",
        "Timestamp",
        "Score",
        "Comments",
        "URL",
        "Is Report",
    ]

    combined_data: List[pd.DataFrame] = []

    for file in files:
        file_path = data_dir / file
        print(f"Processing {file}...")

        try:
            df = pd.read_csv(file_path)
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {list(df.columns)}")

            if "Title & Content" in df.columns:
                df_standardized = pd.DataFrame()
                df_standardized["Post ID"] = [f"legacy_{i}" for i in range(len(df))]
                df_standardized["Summary"] = df["Title & Content"]
                df_standardized["Full Title"] = df["Title & Content"]
                df_standardized["Full Content"] = df["Title & Content"]
                df_standardized["Timestamp"] = df["Timestamp"]
                df_standardized["Score"] = df["Score"]
                df_standardized["Comments"] = df["Comments"]
                df_standardized["URL"] = df["URL"]
                df_standardized["Is Report"] = df["Is Report"]
            else:
                df_standardized = df.copy()

            for col in standard_columns:
                if col not in df_standardized.columns:
                    df_standardized[col] = ""

            df_standardized = df_standardized[standard_columns]
            df_standardized["Source File"] = file

            combined_data.append(df_standardized)
            print(f"  - Processed {len(df_standardized)} rows")

        except Exception as e:
            print(f"  - Error processing {file}: {e}")
            continue

    if not combined_data:
        print("No data to combine!")
        return None

    final_df = pd.concat(combined_data, ignore_index=True)
    output_path = data_dir / "combined_posts.csv"
    final_df.to_csv(output_path, index=False)

    print(f"\nCombined file saved to: {output_path}")
    print(f"Total rows: {len(final_df)}")
    print(f"Columns: {list(final_df.columns)}")

    print("\nSummary by source file:")
    print(final_df["Source File"].value_counts())

    return output_path


def load_clusters(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r") as f:
        return json.load(f)


def load_label_map(label_file: Optional[str]) -> Optional[Dict[str, Any]]:
    if not label_file:
        return None
    with open(label_file, "r") as f:
        return json.load(f)


def map_cluster_labels(
    clusters: Dict[str, List[str]],
    label_map: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    if not label_map:
        return clusters

    remapped: Dict[str, List[str]] = {}
    for key, items in clusters.items():
        label = label_map.get(key, key)
        if isinstance(label, str):
            remapped[label] = items
        else:
            remapped[key] = items
    return remapped


def load_post_dates(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    post_id_col = cols.get("post id")
    timestamp_col = cols.get("timestamp")
    summary_col = cols.get("summary")

    if post_id_col is None or timestamp_col is None:
        raise ValueError("CSV must contain 'Post ID' and 'Timestamp' columns")

    post_dates: Dict[str, datetime] = {}
    summary_to_post: Dict[str, str] = {}

    for _, row in df.iterrows():
        pid = str(row[post_id_col])
        ts_str = str(row[timestamp_col])

        if pd.notna(ts_str) and ts_str.lower() not in ["nan", "no content", ""]:
            ts = ts_str.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            post_dates[pid] = dt

            if summary_col is not None:
                summary = str(row[summary_col]).strip()
                if summary and summary.lower() not in ["nan", "no content"]:
                    summary_to_post[summary] = pid

    return post_dates, summary_to_post


def get_cluster_dates(
    clusters: Dict[str, List[str]],
    post_dates: Dict[str, datetime],
    summary_to_post: Optional[Dict[str, str]] = None,
):
    cluster_dates: Dict[str, List[datetime]] = {}

    for cluster_name, items in clusters.items():
        dates: List[datetime] = []
        for item in items:
            if item in post_dates:
                dates.append(post_dates[item])
            elif summary_to_post and item in summary_to_post:
                pid = summary_to_post[item]
                if pid in post_dates:
                    dates.append(post_dates[pid])
        if dates:
            cluster_dates[cluster_name] = dates

    return cluster_dates


def create_box_plot(cluster_dates, output_path, title):
    if not cluster_dates:
        return

    cluster_names = []
    date_arrays = []

    for cluster_name, dates in sorted(cluster_dates.items()):
        cluster_names.append(cluster_name)
        date_arrays.append(dates)

    date_numeric = [[mdates.date2num(d) for d in dates] for dates in date_arrays]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_height = max(6, len(cluster_names) * 0.45)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    flierprops = dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.7)
    meanprops = dict(
        marker="D",
        markeredgecolor="black",
        markerfacecolor="#ff7f0e",
        markersize=6,
    )
    medianprops = dict(color="red", linewidth=2)
    boxprops = dict(linewidth=1.5, color="#1f77b4")
    whiskerprops = dict(linewidth=1.5, color="#555555")
    capprops = dict(linewidth=1.5, color="#555555")

    bp = ax.boxplot(
        date_numeric,
        labels=cluster_names,
        vert=True,
        patch_artist=True,
        showmeans=True,
        meanprops=meanprops,
        medianprops=medianprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
        whis=1.5,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")
        patch.set_alpha(0.85)

    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Date")
    ax.set_xlabel("Cluster")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_stats(cluster_dates, output_path):
    rows = []

    for cluster_name, dates in sorted(cluster_dates.items()):
        if not dates:
            continue
        ds = sorted(dates)
        earliest = min(ds)
        latest = max(ds)
        if len(ds) % 2 == 0:
            mid1 = ds[len(ds) // 2 - 1]
            mid2 = ds[len(ds) // 2]
            median_ts = (mdates.date2num(mid1) + mdates.date2num(mid2)) / 2
            median_dt = mdates.num2date(median_ts)
        else:
            median_dt = ds[len(ds) // 2]
        delta = (latest - earliest).days if len(ds) > 1 else 0
        rows.append(
            {
                "Cluster": cluster_name,
                "Count": len(ds),
                "Earliest": earliest.isoformat(),
                "Latest": latest.isoformat(),
                "Median": median_dt.isoformat(),
                "Date Range (days)": delta,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def plot_cluster_date_distributions(
    clusters_path: str,
    csv_path: str,
    out_dir: str,
    title: str = "Cluster Date Distributions",
    prefix: str = "cluster_dates",
    label_file: Optional[str] = None,
):
    """
    High-level helper to generate both the boxplot and summary stats
    for cluster date distributions.
    """
    os.makedirs(out_dir, exist_ok=True)

    clusters = load_clusters(clusters_path)
    label_map = load_label_map(label_file)
    clusters = map_cluster_labels(clusters, label_map)
    post_dates, summary_to_post = load_post_dates(csv_path)
    cluster_dates = get_cluster_dates(clusters, post_dates, summary_to_post)

    plot_path = os.path.join(out_dir, f"{prefix}_boxplot.png")
    stats_path = os.path.join(out_dir, f"{prefix}_stats.csv")

    create_box_plot(cluster_dates, plot_path, title=title)
    create_summary_stats(cluster_dates, stats_path)

    return plot_path, stats_path


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
