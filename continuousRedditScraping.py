#!/usr/bin/env python3
import praw
import os
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent=os.getenv("user_agent"),
    username=os.getenv("username"),
    password=os.getenv("password")
)

OUTPUT_FILE = "reddit_api_posts.csv"
CHECKPOINT_FILE = "last_id.txt"
SUBREDDIT = "ChatGPT"
LIMIT = 1000  # per run

def load_seen_ids():
    seen = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(row["id"])
    return seen

def save_checkpoint(post_id):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(post_id)

def collect_posts(seen_ids):
    subreddit = reddit.subreddit(SUBREDDIT)
    posts = []
    newest_id = None

    for submission in subreddit.new(limit=LIMIT):
        if submission.fullname in seen_ids:
            continue  # skip duplicates

        posts.append({
            "id": submission.fullname,
            "title": submission.title,
            "content": submission.selftext or "[no content]",
            "timestamp": datetime.fromtimestamp(submission.created_utc, timezone.utc).isoformat(),
            "score": submission.score,
            "comments": submission.num_comments,
            "url": submission.url,
            "author": str(submission.author) if submission.author else "[deleted]"
        })

        if not newest_id:
            newest_id = submission.fullname  # first item = newest

    if newest_id:
        save_checkpoint(newest_id)
    return posts

def save_to_csv(posts):
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "content", "timestamp", "score", "comments", "url", "author"],
            quoting=csv.QUOTE_ALL
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(posts)

def main():
    seen_ids = load_seen_ids()
    posts = collect_posts(seen_ids)
    if not posts:
        print("No new unique posts found.")
        return
    save_to_csv(posts)
    print(f"Collected {len(posts)} new posts. Last checkpoint saved.")

if __name__ == "__main__":
    main()
