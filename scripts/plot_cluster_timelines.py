#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_clusters(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_post_dates(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    post_id_col = cols.get("post id")
    ts_col = cols.get("timestamp")
    if post_id_col is None or ts_col is None:
        raise ValueError("CSV must contain 'Post ID' and 'Timestamp' columns")

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    dates = ts.dt.date

    valid = dates.notna()
    post_ids = df.loc[valid, post_id_col].astype(str)
    dates = dates[valid]

    return pd.Series(dates.values, index=post_ids.values)


def build_cluster_daily_counts(
    clusters: Dict[str, List[str]],
    post_dates: pd.Series,
) -> pd.DataFrame:
    all_dates = set(post_dates.values)
    if not all_dates:
        raise ValueError("No valid dates found in CSV")

    date_index = pd.Index(sorted(all_dates), name="date")

    data: Dict[str, List[int]] = {}

    for cluster_name, ids in clusters.items():
        dates = [post_dates.get(str(pid)) for pid in ids]
        dates = [d for d in dates if pd.notna(d)]

        if not dates:
            data[cluster_name] = [0] * len(date_index)
            continue

        s = pd.Series(1, index=pd.Index(dates, name="date"))
        counts = s.groupby(level=0).sum()
        counts = counts.reindex(date_index, fill_value=0)
        data[cluster_name] = counts.values.tolist()

    df = pd.DataFrame(data, index=date_index)
    return df


def plot_cluster_timelines(
    counts: pd.DataFrame,
    out_path: str,
    title: str = "Cluster activity over time",
    legend_labels: Optional[Dict[str, str]] = None,
):
    if counts.empty:
        raise ValueError("No counts to plot")

    dates = pd.to_datetime(counts.index)
    x = mdates.date2num(dates.to_pydatetime())

    totals = counts.sum(axis=0).sort_values(ascending=False)
    ordered_cols = totals.index.tolist()
    y = counts[ordered_cols].T.values  # shape (n_clusters, n_days)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ordered_cols)))
    stack = ax.stackplot(
        x,
        y,
        colors=colors,
        linewidth=0.5,
        edgecolor="white",
        alpha=0.9,
    )

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate()

    ax.set_ylabel("Number of posts")
    ax.set_xlabel("Date")
    ax.set_title(title)

    if legend_labels:
        display_labels = [legend_labels.get(c, c) for c in ordered_cols]
    else:
        display_labels = ordered_cols

    ax.legend(
        stack,
        display_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Clusters",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot per-cluster timelines as stacked histograms"
    )
    ap.add_argument(
        "--clusters",
        required=True,
        help="Path to clusters.json",
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="CSV file with 'Post ID' and 'Timestamp'",
    )
    ap.add_argument(
        "--labels",
        help="Optional cluster_labels.json to show 'Cluster X: name' in legend",
    )
    ap.add_argument(
        "--out",
        default="results/cluster_timelines.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--title",
        default="Cluster activity over time",
        help="Plot title",
    )
    args = ap.parse_args()

    clusters = load_clusters(args.clusters)
    post_dates = load_post_dates(args.csv)
    counts = build_cluster_daily_counts(clusters, post_dates)

    legend_labels: Optional[Dict[str, str]] = None
    if args.labels:
        raw_labels = load_clusters(args.labels)
        legend_labels = {}
        for cluster_name, label_data in raw_labels.items():
            if isinstance(label_data, dict):
                label = (
                    label_data.get("name")
                    or label_data.get("label")
                    or str(label_data)
                )
            else:
                label = str(label_data)
            legend_labels[cluster_name] = f"{cluster_name}: {label}"

    plot_cluster_timelines(
        counts,
        args.out,
        title=args.title,
        legend_labels=legend_labels,
    )


if __name__ == "__main__":
    main()
