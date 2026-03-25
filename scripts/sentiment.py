import argparse
import pandas as pd


LABEL_MAP = {
    "positive": ("Positive", 1),
    "neutral": ("Neutral", 0),
    "negative": ("Negative", -1),
}


def try_transformers_sentiment(texts, model_name: str):
    """
    優先用 transformers（建議，中文可用）
    預設 model：cardiffnlp/twitter-xlm-roberta-base-sentiment
    """
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=model_name)
    outs = clf(texts, truncation=True)
    # outs: [{"label": "...", "score": ...}, ...]
    labels = []
    values = []
    confs = []
    for o in outs:
        lab = (o.get("label") or "").lower()
        # 有些模型 label 會是 "LABEL_0/1/2"，這裡做簡單兼容
        if lab in ("label_0", "0"):
            lab = "negative"
        elif lab in ("label_1", "1"):
            lab = "neutral"
        elif lab in ("label_2", "2"):
            lab = "positive"

        if lab not in LABEL_MAP:
            lab = "neutral"

        label_txt, val = LABEL_MAP[lab]
        labels.append(label_txt)
        values.append(val)
        confs.append(float(o.get("score", 0.0)))
    return labels, values, confs


def fallback_textblob(texts):
    """
    備援：TextBlob（中文效果較弱，但能跑）
    polarity > 0.1 => Positive
    polarity < -0.1 => Negative
    else Neutral
    """
    from textblob import TextBlob
    labels, values, confs = [], [], []
    for t in texts:
        p = float(TextBlob(t).sentiment.polarity)
        if p > 0.1:
            labels.append("Positive"); values.append(1)
        elif p < -0.1:
            labels.append("Negative"); values.append(-1)
        else:
            labels.append("Neutral"); values.append(0)
        confs.append(abs(p))
    return labels, values, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/news.csv")
    ap.add_argument("--outfile", default="data/news.csv")  # 直接覆寫回去最省事
    ap.add_argument("--model", default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    ap.add_argument("--use_textblob", action="store_true", help="強制用 TextBlob（不建議，除非 torch 裝不起來）")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, encoding="utf-8-sig")
    # 清理空值（作業要求）
    df = df.dropna(subset=["date", "title"]).copy()

    # 用 title + summary 當分析文本（內容或摘要符合要求）:contentReference[oaicite:9]{index=9}
    df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()

    texts = df["text"].tolist()

    if args.use_textblob:
        labels, values, confs = fallback_textblob(texts)
    else:
        try:
            labels, values, confs = try_transformers_sentiment(texts, args.model)
        except Exception as e:
            print(f"⚠️ transformers 失敗，改用 TextBlob（原因：{e}）")
            labels, values, confs = fallback_textblob(texts)

    df["sentiment_label"] = labels               # Positive/Neutral/Negative
    df["sentiment_value"] = values               # +1/0/-1
    df["sentiment_confidence"] = confs

    # 丟掉 text 欄位也可以；我建議保留方便查核
    df.to_csv(args.outfile, index=False, encoding="utf-8-sig")
    print(f"✅ wrote {args.outfile} rows={len(df)}")


if __name__ == "__main__":
    main()
