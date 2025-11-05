import csv
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

input_file = "reddit_posts.csv"
keyword = "hallucin"

counts = Counter()
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = (row.get("title", "") + " " + row.get("content", "")).lower()
        if keyword in text:
            try:
                date = datetime.strptime(row["created_date"], "%Y-%m-%d %H:%M:%S").date()
                counts[date] += 1
            except Exception:
                continue

dates = sorted(counts)
values = [counts[d] for d in dates]
smoothed = []
window = 7
for i in range(len(values)):
    start = max(0, i - window // 2)
    end = min(len(values), i + window // 2 + 1)
    smoothed.append(sum(values[start:end]) / (end - start))

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 5))
plt.plot(dates, smoothed, color="#007acc", linewidth=2)
plt.fill_between(dates, smoothed, color="#007acc", alpha=0.2)
plt.title("Mentions of 'hallucinate' Over Time", fontsize=14, weight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("7-Day Avg Count", fontsize=12)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("hallucinate_trend_clean.png", dpi=300)
print("✅ Clean chart saved as hallucinate_trend_clean.png")
