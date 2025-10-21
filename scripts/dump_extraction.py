import os, io, json, csv
import zstandard as zstd
from tqdm import tqdm

INPUT_FOLDER = "./dumps"
OUTPUT_FILE = "chatgpt_posts.csv"
PROGRESS_FILE = "processed_files.txt"
BAD_LOG = "bad_lines.log"

def stream_zst(path):
    with open(path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                yield line

def extract_chatgpt_posts():
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        processed = set(open(PROGRESS_FILE).read().splitlines())

    fieldnames = ["id", "created_utc", "title", "selftext", "num_comments", "score", "url"]

    write_header = not os.path.exists(OUTPUT_FILE)
    out_csv = open(OUTPUT_FILE, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    bad_log = open(BAD_LOG, "a", encoding="utf-8")

    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not fname.endswith(".zst") or fname in processed:
            continue
        path = os.path.join(INPUT_FOLDER, fname)
        print(f"Processing {fname}...")
        count = 0
        for line in tqdm(stream_zst(path)):
            try:
                post = json.loads(line)
                if post.get("subreddit", "").lower() == "chatgpt":
                    writer.writerow({
                        "id": post.get("id"),
                        "created_utc": post.get("created_utc"),
                        "title": post.get("title"),
                        "selftext": post.get("selftext"),
                        "num_comments": post.get("num_comments"),
                        "score": post.get("score"),
                        "url": post.get("url"),
                    })
                    count += 1
            except json.JSONDecodeError:
                bad_log.write(f"Bad JSON in {fname}\n")
            except Exception as e:
                bad_log.write(f"{fname} | {e}\n")
        print(f"→ {count} ChatGPT posts found in {fname}")
        with open(PROGRESS_FILE, "a") as pf:
            pf.write(fname + "\n")
    out_csv.close()
    bad_log.close()
    print(f"✅ Extraction complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_chatgpt_posts()
