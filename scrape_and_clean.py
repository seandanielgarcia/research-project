#!/usr/bin/env python3
import os
import csv
import argparse
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import praw
import google.generativeai as genai
import sys

load_dotenv()

# Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent=os.getenv("user_agent"),
    username=os.getenv("username"),
    password=os.getenv("password")
)

# Gemini API
genai.configure(api_key=os.getenv("gemini_key"))
model = genai.GenerativeModel("gemini-2.0-flash")

def is_report(post_content):
    prompt = f"""You are a data extraction specialist. Determine if this Reddit post is a report of a specific incident where ChatGPT behaved problematically.

EXAMPLES OF REPORTS (specific incidents where ChatGPT behaved problematically):
1. "Is anyone else getting this? Every time I try to talk to ChatGPT it starts hallucinating messages that I never sent and answering the most random things. I have never sent anything to do with 'Dancing With Water', I have no idea what that is. Is this happening to anyone else or have I just broken my guy?"
2. "While others have trouble getting it to stop making flashcards, I have trouble getting it to make flashcards! I can only make them in one conversation..."
3. "ChatGPT refuses to blame my grandma for war crimes" - specific refusal incident
4. "ChatGPT condoned and assisted with suicide when I pretended to be an AI system" - specific harmful behavior
5. "ChatGPT is secretly remembering chats" - specific concerning behavior
6. "ChatGPT refuses to give a list of sleep aid supplements, but happy to help getting hammered quick" - specific refusal inconsistency
7. "ChatGPT refused to colorise the teacher/headmaster, even when I told it he was 'beating a rug to clean it'" - specific refusal
8. "ChatGPT no longer thinks about anything?? It won't do anything but blindly agree with me" - specific behavioral change
9. "ChatGPT failing to remember its own reset timings for image generation limits" - specific technical failure
10. "ChatGPT refuses to use Arial Black as the font type because it said 'Arial Black/Bold isn't present in this environment'" - specific technical refusal

EXAMPLES OF NON-REPORTS (general discussions about hallucination/harm but not specific incidents):
1. "GPT-5 finally treats 'Thinking' as the job, not garnish. It plans before it speaks, decomposes problems, runs self-checks" - general model description
2. "The Story of PrimeTalk and Lyra The Prompt Optimizer" - general product description
3. "Soy-kids use Chatgpt WRONG" - general opinion piece
4. "Gf broke up with me due chatgpt" - personal relationship issue, not a ChatGPT problem report
5. "These are the protocols I've embedded in my ChatGPT in the personalize settings" - general configuration sharing
6. "Gemini reports after adversarial testing of my ChatGPT" - general testing discussion
7. "AI models are getting worse at creative writing" - general observation without specific incident
8. "ChatGPT's safety features are too restrictive" - general complaint without specific example
9. "The future of AI is concerning" - general speculation about AI
10. "Why do people rely so much on ChatGPT?" - general question about usage patterns

REDDIT POST:
{post_content}

Respond only "Yes" if this is a report, or "No" if not."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip().lower() == "yes"
    except Exception:
        return False

def summarize_post(title, content, max_chars=250):
    text = f"Reddit Post Title: {title}\nContent: {content}"
    prompt = f"""Summarize this Reddit post in a single plain-English sentence.
Do not include labels, formatting, CSV, or extra commentary.
Keep it under {max_chars} characters.

{text}"""
    try:
        response = model.generate_content(prompt)
        summary = response.text.strip().replace("\n", " ")
        return summary[:max_chars]
    except Exception:
        combined = f"{title} — {content}"
        return combined[:max_chars]

def collect_posts(subreddit_name, target_posts=None, stall_timeout=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    last_progress = time.time()

    for i, post in enumerate(subreddit.new(limit=None), 1):
        posts.append({
            "id": post.id,
            "title": post.title,
            "content": post.selftext or "[no content]",
            "timestamp": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
            "score": post.score,
            "comments": post.num_comments,
            "url": post.url,
        })

        if i % 100 == 0:
            if target_posts:
                print(f"Processed {i}/{target_posts} posts")
            else:
                print(f"Processed {i} posts")
            last_progress = time.time()

        if target_posts and i >= target_posts:
            break

        if time.time() - last_progress > stall_timeout:
            print(f"No new posts for {stall_timeout} seconds. Stopping early at {i} posts.")
            break

    return posts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", default="ChatGPT")
    parser.add_argument("--output", default="reddit_posts.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--stall-timeout", type=int, default=10)
    args = parser.parse_args()

    posts = collect_posts(args.subreddit, args.limit, args.stall_timeout)
    if not posts:
        print("No posts found")
        return

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Post ID", "Summary", "Full Title", "Full Content", "Timestamp", "Score", "Comments", "URL", "Is Report"],
            quoting=csv.QUOTE_ALL
        )

        writer.writeheader()

        report_count = 0
        try:
            for idx, post in enumerate(posts, 1):
                # Summarize + classify
                summary = summarize_post(post['title'], post['content'])
                is_report_flag = "N/A"
                if not args.skip_classification:
                    if is_report(summary):
                        is_report_flag = "Yes"
                        report_count += 1
                    else:
                        is_report_flag = "No"

                writer.writerow({
                    "Post ID": post["id"],
                    "Summary": summary,
                    "Full Title": post["title"],
                    "Full Content": post["content"],
                    "Timestamp": post["timestamp"],
                    "Score": post["score"],
                    "Comments": post["comments"],
                    "URL": post["url"],
                    "Is Report": is_report_flag
                })
                f.flush()

                if idx % 50 == 0:
                    print(f"Written {idx}/{len(posts)} rows")

        except KeyboardInterrupt:
            print("\nInterrupted. Partial results saved.")
            sys.exit(0)

    oldest = min(posts, key=lambda x: x["timestamp"])
    print(f"\nWrote {len(posts)} posts to {args.output}")
    if not args.skip_classification:
        pct = (report_count / len(posts)) * 100
        print(f"Reports: {report_count} ({pct:.1f}%)")
    print(f"Oldest post timestamp: {oldest['timestamp']}")

if __name__ == "__main__":
    main()
