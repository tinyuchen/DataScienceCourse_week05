import urllib.parse
import feedparser
import pandas as pd
from datetime import datetime

def google_news_rss(query: str, days: int = 60, lang="zh-TW", region="TW"):
    q = f'({query}) when:{days}d'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({
        "q": q, "hl": lang, "gl": region, "ceid": f"{region}:{lang.split('-')[0]}"
    })
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        dt = None
        try:
            dt = datetime(*e.published_parsed[:6]).date() if hasattr(e, "published_parsed") else None
        except Exception:
            dt = None
        rows.append({"date": dt, "title": title, "content": summary, "source": "Google News RSS", "link": getattr(e, "link", "")})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["title"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date","title"]).sort_values("date")
    return df

def main():
    # 範例：你可自行改
    query = "台積電 OR TSMC"
    days = 120
    df = google_news_rss(query, days=days)
    df.to_csv("data/news.csv", index=False, encoding="utf-8-sig")
    print("saved data/news.csv", len(df))

if __name__ == "__main__":
    main()
