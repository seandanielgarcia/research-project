#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_clusters(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_post_dates(
    csv_path: str,
    min_words: int = 0,
    summary_col: Optional[str] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    post_id_col = cols.get("post id")
    ts_col = cols.get("timestamp")
    if post_id_col is None or ts_col is None:
        raise ValueError("CSV must contain 'Post ID' and 'Timestamp' columns")

    summary_field = None
    if min_words > 0:
        if summary_col:
            summary_field = cols.get(summary_col.lower().strip(), summary_col)
        else:
            for field_name in ["summary", "full content", "content", "text"]:
                if field_name in cols:
                    summary_field = cols[field_name]
                    break
    
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    dates = ts.dt.date

    valid = dates.notna()
    
    stats = {
        "total_posts": len(df),
        "valid_dates": valid.sum(),
        "filtered_by_words": 0,
        "final_count": 0,
    }
    
    if min_words > 0 and summary_field and summary_field in df.columns:
        word_counts = df[summary_field].fillna("").astype(str).str.split().str.len()
        word_valid = word_counts >= min_words
        valid = valid & word_valid
        stats["filtered_by_words"] = (~word_valid & dates.notna()).sum()
        print(f"Filtered out {stats['filtered_by_words']} posts with < {min_words} words")
    
    post_ids = df.loc[valid, post_id_col].astype(str)
    dates = dates[valid]
    stats["final_count"] = len(dates)

    return pd.Series(dates.values, index=post_ids.values), stats


def build_cluster_daily_counts(
    clusters: Dict[str, List[str]],
    post_dates: pd.Series,
) -> pd.DataFrame:
    all_dates = set(post_dates.values)
    if not all_dates:
        raise ValueError("No valid dates found in CSV")

    date_index = pd.Index(sorted(all_dates), name="date")

    data: Dict[str, List[int]] = {}
    missing_posts = []

    for cluster_name, ids in clusters.items():
        dates = [post_dates.get(str(pid)) for pid in ids]
        valid_dates = [d for d in dates if pd.notna(d)]
        
        missing = len(ids) - len(valid_dates)
        if missing > 0:
            missing_posts.append((cluster_name, missing))

        if not valid_dates:
            data[cluster_name] = [0] * len(date_index)
            continue

        s = pd.Series(1, index=pd.Index(valid_dates, name="date"))
        counts = s.groupby(level=0).sum()
        counts = counts.reindex(date_index, fill_value=0)
        data[cluster_name] = counts.values.tolist()

    if missing_posts:
        total_missing = sum(m for _, m in missing_posts)
        print(f"Warning: {total_missing} posts across all clusters have no valid dates")
        if total_missing > 100:
            print("  Top clusters with missing dates:")
            for cluster, count in sorted(missing_posts, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {cluster}: {count} missing")

    df = pd.DataFrame(data, index=date_index)
    return df


def plot_cluster_timelines(
    counts: pd.DataFrame,
    out_path: str,
    title: str = "Cluster activity over time",
    legend_labels: Optional[Dict[str, str]] = None,
    plot_type: str = "stacked",
    show_proportions: bool = False,
):
    if counts.empty:
        raise ValueError("No counts to plot")

    dates = pd.to_datetime(counts.index)
    x = mdates.date2num(dates.to_pydatetime())

    totals = counts.sum(axis=0).sort_values(ascending=False)
    ordered_cols = totals.index.tolist()
    
    counts_subset = counts[ordered_cols]
    
    if show_proportions:
        daily_totals = counts_subset.sum(axis=1).values
        daily_totals = np.where(daily_totals == 0, 1, daily_totals)
        plot_data = (counts_subset.values / daily_totals[:, np.newaxis] * 100)
        y_label = "Percentage of posts"
    else:
        plot_data = counts_subset.values
        y_label = "Number of posts"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ordered_cols)))
    
    if plot_type == "line":
        for i, col in enumerate(ordered_cols):
            ax.plot(
                x,
                plot_data[:, i],
                label=legend_labels.get(col, col) if legend_labels else col,
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )
        ax.set_ylabel(y_label)
    else:
        stack = ax.stackplot(
            x,
            *[plot_data[:, i] for i in range(len(ordered_cols))],
            colors=colors,
            linewidth=0.5,
            edgecolor="white",
            alpha=0.9,
        )
        ax.set_ylabel(y_label)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate()

    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if legend_labels:
        display_labels = [legend_labels.get(c, c) for c in ordered_cols]
    else:
        display_labels = ordered_cols

    if plot_type == "line":
        ax.legend(
            display_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            title="Clusters",
        )
    else:
        ax.legend(
            stack,
            display_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            title="Clusters",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot per-cluster timelines as stacked histograms or line plots"
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
    ap.add_argument(
        "--plot-type",
        choices=["stacked", "line"],
        default="stacked",
        help="Plot type: 'stacked' (default) or 'line'",
    )
    ap.add_argument(
        "--proportions",
        action="store_true",
        help="Show relative proportions instead of absolute counts",
    )
    ap.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Filter out posts with fewer than this many words (default: 0 = no filtering)",
    )
    ap.add_argument(
        "--summary-col",
        help="Column name to use for word count filtering (default: auto-detect)",
    )
    args = ap.parse_args()

    print(f"Loading clusters from {args.clusters}...")
    clusters = load_clusters(args.clusters)
    print(f"Found {len(clusters)} clusters")
    
    print(f"Loading post dates from {args.csv}...")
    post_dates, stats = load_post_dates(
        args.csv,
        min_words=args.min_words,
        summary_col=args.summary_col,
    )
    print(f"Loaded {stats['final_count']} posts with valid dates")
    if args.min_words > 0:
        print(f"  Filtered out {stats['filtered_by_words']} posts with < {args.min_words} words")
    
    print("Building daily counts...")
    counts = build_cluster_daily_counts(clusters, post_dates)
    print(f"Date range: {counts.index.min()} to {counts.index.max()}")
    print(f"Total posts per cluster:")
    for cluster, total in counts.sum(axis=0).sort_values(ascending=False).items():
        print(f"  {cluster}: {total}")

    print(f"Generating {args.plot_type} plot...")
    plot_cluster_timelines(
        counts,
        args.out,
        title=args.title,
        legend_labels=None,
        plot_type=args.plot_type,
        show_proportions=args.proportions,
    )
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
