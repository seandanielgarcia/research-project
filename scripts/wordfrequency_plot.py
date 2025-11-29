import csv
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

input_file = "/data/reddit/reddit_posts.csv"

# daily counters
counts = Counter()
sam_counts = Counter()
altman_counts = Counter()

total_posts = 0

def has_sam(text):
    return "sam" in text.lower()

def has_altman(text):
    return "altman" in text.lower()

with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_posts += 1

        try:
            date = datetime.strptime(row["created_date"], "%Y-%m-%d %H:%M:%S").date()
        except Exception:
            continue

        title = row.get("title", "") or ""
        content = row.get("content", "") or ""
        text = (title + " " + content).lower()

        # total posts for this day
        counts[date] += 1

        # keyword counts
        if has_sam(text):
            sam_counts[date] += 1

        if has_altman(text):
            altman_counts[date] += 1

# convert daily to weekly
weekly_counts = Counter()
weekly_sam = Counter()
weekly_altman = Counter()
week_first_dates = {}

for date, c in counts.items():
    year, week, _ = date.isocalendar()
    key = (year, week)

    weekly_counts[key] += c
    weekly_sam[key] += sam_counts[date]
    weekly_altman[key] += altman_counts[date]

    if key not in week_first_dates or date < week_first_dates[key]:
        week_first_dates[key] = date

sorted_weeks = sorted(weekly_counts.keys())
week_dates = [week_first_dates[w] for w in sorted_weeks]

# normalized keyword frequency = keyword hits / total posts that week
sam_norm = [
    weekly_sam[w] / weekly_counts[w] if weekly_counts[w] > 0 else 0
    for w in sorted_weeks
]

altman_norm = [
    weekly_altman[w] / weekly_counts[w] if weekly_counts[w] > 0 else 0
    for w in sorted_weeks
]

# print summary counts
print("Total posts:", total_posts)
print("Posts containing 'sam':", sum(weekly_sam.values()))
print("Posts containing 'altman':", sum(weekly_altman.values()))

# plot ONLY normalized metrics
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(12, 5))

plt.plot(week_dates, sam_norm, linewidth=2, label="Sam normalized")
plt.plot(week_dates, altman_norm, linewidth=2, label="Altman normalized")

plt.title("Normalized Sam and Altman Mentions Over Time")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("normalized_keyword_trends.png", dpi=300)

print("Chart saved as normalized_keyword_trends.png")
