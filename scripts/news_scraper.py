import argparse
import csv
import time
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser


def google_news_rss_url(query: str, days: int) -> str:
    # Google News RSS（繁中/台灣）
    # when:Xd 可以限制最近 X 天
    q = f"{query} when:{days}d"
    q = urllib.parse.quote(q)
    return f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"


def parse_entry_date(entry) -> str:
    # RSS 通常是 RFC2822，例如 "Mon, 25 Mar 2026 10:00:00 GMT"
    if getattr(entry, "published", None):
        dt = parsedate_to_datetime(entry.published)
    elif getattr(entry, "updated", None):
        dt = parsedate_to_datetime(entry.updated)
    else:
        dt = datetime.now(timezone.utc)

    # 只保留日期（方便跟股價對齊）
    return dt.astimezone(timezone.utc).date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="例如：台積電 2330")
    ap.add_argument("--days", type=int, default=30, help="抓最近幾天新聞")
    ap.add_argument("--out", default="data/news.csv")
    ap.add_argument("--sleep", type=float, default=0.5, help="避免太快被限制")
    args = ap.parse_args()

    url = google_news_rss_url(args.query, args.days)
    feed = feedparser.parse(url)

    rows = []
    for e in feed.entries:
        date = parse_entry_date(e)
        title = (getattr(e, "title", "") or "").strip()
        summary = (getattr(e, "summary", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()

        # 清理空值（作業要求）
        if not title:
            continue

        rows.append({
            "date": date,
            "title": title,
            "summary": summary,
            "link": link,
            "source": "GoogleNewsRSS"
        })

        time.sleep(args.sleep)

    # 寫 CSV
    # 欄位至少：日期、標題、內容或摘要（這裡用 summary）:contentReference[oaicite:6]{index=6}
    fieldnames = ["date", "title", "summary", "link", "source"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ wrote {args.out} rows={len(rows)}")
    print(f"RSS URL: {url}")


if __name__ == "__main__":
    main()
