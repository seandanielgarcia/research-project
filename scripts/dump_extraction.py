import json
import csv
from datetime import datetime, timezone

input_file = "r_ChatGPT_posts.jsonl"
output_file = "reddit_posts.csv"

def safe_str(value):
    """Return a safe string (handles None)"""
    if value is None:
        return ""
    return str(value).strip()

def safe_timestamp(ts):
    """Safely convert timestamp to readable UTC datetime"""
    try:
        if ts is None:
            return ""
        return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

with open(input_file, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        f.seek(0)
        data = [json.loads(line) for line in f if line.strip()]

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["post_id", "title", "created_date", "content", "upvotes", "upvote_ratio", "num_comments", "url"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for post in data:
        writer.writerow({
            "post_id": safe_str(post.get("id")),
            "title": safe_str(post.get("title")),
            "created_date": safe_timestamp(post.get("created_utc")),
            "content": safe_str(post.get("selftext")),
            "upvotes": post.get("ups") if post.get("ups") is not None else 0,
            "upvote_ratio": post.get("upvote_ratio") if post.get("upvote_ratio") is not None else "",
            "num_comments": post.get("num_comments") if post.get("num_comments") is not None else 0,
            "url": safe_str(post.get("url"))
        })

print(f"✅ Extracted {len(data)} posts to {output_file}")
