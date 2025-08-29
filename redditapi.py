import os
from dotenv import load_dotenv
import praw
import json
from datetime import datetime, timezone

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent=os.getenv("user_agent"),
    username=os.getenv("username"),
    password=os.getenv("password")
)

subreddit = reddit.subreddit("LocalLLaMA")
results = subreddit.search(
    query="refused",
    sort="new",
    time_filter="month",
    limit=None
)

posts = []
for post in results:
    posts.append({
        "id": post.id,
        "title": post.title,
        "author": str(post.author),
        "created_utc": post.created_utc,
        "url": post.url,
        "score": post.score,
        "num_comments": post.num_comments,
        "selftext": post.selftext
    })

output_path = ""
with open(output_path, "w", encoding="utf-8") as f:
    for post in posts:
        f.write(f"=== Post ID: {post['id']} ===\n")
        f.write(f"Title      : {post['title']}\n")
        f.write(f"Author     : {post['author']}\n")
        f.write(f"Date (UTC) : {datetime.fromtimestamp(post['created_utc'], timezone.utc).isoformat()}Z\n")
        f.write(f"URL        : {post['url']}\n")
        f.write(f"Score      : {post['score']}\n")
        f.write(f"Comments   : {post['num_comments']}\n")
        f.write("\n")
        f.write(post["selftext"] or "[no selftext]")
        f.write("\n\n\n")

print(f"Wrote {len(posts)} posts to {output_path}")
