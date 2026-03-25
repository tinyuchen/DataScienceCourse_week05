import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    STOCK_ID = "2330.TW"
    START = "2023-01-01"
    END   = "2024-12-31"

    # stock
    df_stock = yf.download(STOCK_ID, start=START, end=END, progress=False).reset_index()
    df_stock = df_stock.rename(columns={"Date":"date", "Close":"close"})
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date
    df_stock = df_stock[["date","close"]].dropna()
    df_stock.to_csv("data/stock.csv", index=False, encoding="utf-8-sig")

    # news
    df_news = pd.read_csv("data/news.csv")
    df_news["date"] = pd.to_datetime(df_news["date"]).dt.date
    daily_sent = df_news.groupby("date").agg(
        daily_sent_sum=("sentiment_score","sum"),
        daily_sent_mean=("sentiment_score","mean"),
        news_count=("sentiment_score","size")
    ).reset_index()

    df_s = df_stock.copy()
    df_s["ret0"] = df_s["close"].pct_change()
    df_s["ret1"] = df_s["ret0"].shift(-1)

    df_m = df_s.merge(daily_sent, on="date", how="left")
    df_m[["daily_sent_sum","daily_sent_mean","news_count"]] = df_m[["daily_sent_sum","daily_sent_mean","news_count"]].fillna(0)

    corr_next = df_m[["daily_sent_sum","ret1"]].dropna().corr().iloc[0,1]

    tmp = df_m.dropna(subset=["ret1"]).copy()
    X = tmp[["daily_sent_sum","news_count"]].values
    y = tmp["ret1"].values
    lr = LinearRegression().fit(X, y)
    r2 = r2_score(y, lr.predict(X))

    # plots
    fig = plt.figure(figsize=(14,10))
    ax1 = plt.subplot(3,1,1); ax1.plot(df_m["date"], df_m["daily_sent_sum"]); ax1.set_title("Daily Sentiment (sum)")
    ax2 = plt.subplot(3,1,2); ax2.plot(df_m["date"], df_m["close"]); ax2.set_title(f"Close Price: {STOCK_ID}")
    ax3 = plt.subplot(3,1,3); ax3.scatter(df_m["daily_sent_sum"], df_m["ret1"], s=10); ax3.set_title(f"Sentiment vs Next-day Return (corr={corr_next:.3f}, R2={r2:.4f})")
    plt.tight_layout()
    plt.savefig("output/result.png", dpi=200)
    print("saved output/result.png")

if __name__ == "__main__":
    main()
