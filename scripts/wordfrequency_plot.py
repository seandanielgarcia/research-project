import csv
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

input_file = "/data/reddit/reddit_posts.csv"

counts = Counter()
total_posts = 0
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_posts += 1
        try:
            date = datetime.strptime(row["created_date"], "%Y-%m-%d %H:%M:%S").date()
            counts[date] += 1
        except Exception:
            continue

weekly_counts = Counter()
week_first_dates = {}
for date, count in counts.items():
    year, week, _ = date.isocalendar()
    week_key = (year, week)
    weekly_counts[week_key] += count
    if week_key not in week_first_dates or date < week_first_dates[week_key]:
        week_first_dates[week_key] = date

sorted_weeks = sorted(weekly_counts.keys())
weekly_totals = [weekly_counts[w] for w in sorted_weeks]
week_dates = [week_first_dates[w] for w in sorted_weeks]

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 5))
plt.plot(week_dates, weekly_totals, color="#007acc", linewidth=2)
plt.fill_between(week_dates, weekly_totals, color="#007acc", alpha=0.2)
plt.title("Total Posts Over Time", fontsize=14, weight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Weekly Total", fontsize=12)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("total_posts_trend.png", dpi=300)
posts_with_dates = sum(counts.values())
max_weekly = max(weekly_totals) if weekly_totals else 0
print(f"✅ Total posts: {total_posts}")
print(f"✅ Posts with valid dates: {posts_with_dates}")
print(f"✅ Max weekly total: {max_weekly}")
print("✅ Chart saved as total_posts_trend.png")
