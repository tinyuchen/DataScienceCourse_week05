import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path


def fetch_stock_csv(symbol: str, start: str, end: str, out_csv: str):
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance download empty for {symbol} ({start}~{end})")
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df["close"] = df["Close"].astype(float)
    stock = df[["date", "close"]].copy()
    stock.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return stock


def daily_summary_rule(sent_sum: float, n: int) -> str:
    if n == 0:
        return "No news collected today."
    if sent_sum >= 2:
        return "今日新聞情緒偏正面，可能帶來上行支撐（僅供參考）。"
    if sent_sum <= -2:
        return "今日新聞情緒偏負面，可能帶來下行壓力（僅供參考）。"
    return "今日新聞情緒偏中性，市場可能以盤整或事件驅動為主（僅供參考）。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="2330.TW", help="台股通常用 .TW，例如 2330.TW")
    ap.add_argument("--news_csv", default="data/news.csv")
    ap.add_argument("--stock_csv", default="data/stock.csv")
    ap.add_argument("--out_png", default="output/result.png")
    args = ap.parse_args()

    Path("output").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    news = pd.read_csv(args.news_csv, encoding="utf-8-sig")
    news = news.dropna(subset=["date", "title"]).copy()

    if "sentiment_value" not in news.columns:
        raise RuntimeError("news.csv missing sentiment_value. Run scripts/sentiment.py first.")

    # 每日情緒總分/平均/新聞數
    daily = (news.groupby("date")
                  .agg(sent_sum=("sentiment_value", "sum"),
                       sent_mean=("sentiment_value", "mean"),
                       news_count=("sentiment_value", "size"))
                  .reset_index())

    # 取得股價資料（多抓 2 天避免最後一天算不到隔日報酬）
    start = pd.to_datetime(daily["date"]).min().date().isoformat()
    end = (pd.to_datetime(daily["date"]).max().date() + pd.Timedelta(days=2)).date().isoformat()

    stock = fetch_stock_csv(args.stock, start, end, args.stock_csv)
    stock["date"] = stock["date"].astype(str)
    stock["close"] = stock["close"].astype(float)
    stock = stock.sort_values("date").reset_index(drop=True)
    stock["ret0"] = stock["close"].pct_change()      # 當日報酬
    stock["ret1"] = stock["ret0"].shift(-1)          # 隔日報酬

    merged = stock.merge(daily, on="date", how="left")
    merged[["sent_sum","sent_mean","news_count"]] = merged[["sent_sum","sent_mean","news_count"]].fillna(0)

    valid = merged.dropna(subset=["ret1"]).copy()
    corr = valid[["sent_sum","ret1"]].corr().iloc[0,1]
    print(f"Corr(sent_sum, next_day_return) = {corr:.4f}")

    merged["daily_summary"] = merged.apply(
        lambda r: daily_summary_rule(r["sent_sum"], int(r["news_count"])),
        axis=1
    )
    merged.to_csv("output/merged_daily.csv", index=False, encoding="utf-8-sig")

    # 產圖（同一張 result.png 放 3 個圖）
    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(pd.to_datetime(merged["date"]), merged["sent_sum"], marker="o", linewidth=1)
    ax1.axhline(0, linestyle="--")
    ax1.set_title("Daily News Sentiment Sum (Positive=+1, Neutral=0, Negative=-1)")
    ax1.set_ylabel("sent_sum")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(pd.to_datetime(merged["date"]), merged["close"])
    ax2.set_title(f"Stock Close Price: {args.stock}")
    ax2.set_ylabel("close")

    ax3 = plt.subplot(3, 1, 3)
    ax3.scatter(valid["sent_sum"], valid["ret1"])
    ax3.set_title(f"Sentiment vs Next-day Return (corr={corr:.3f})")
    ax3.set_xlabel("sent_sum")
    ax3.set_ylabel("next_day_return")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close(fig)

    print(f"✅ wrote {args.stock_csv}")
    print(f"✅ wrote {args.out_png}")
    print("✅ wrote output/merged_daily.csv")


if __name__ == "__main__":
    main()
