import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

def label_to_score(label: str) -> int:
    label = label.lower()
    if "positive" in label: return 1
    if "negative" in label: return -1
    return 0

def main():
    df = pd.read_csv("data/news.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)
    id2label = model.config.id2label

    texts = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist()
    preds = clf(texts, batch_size=16)

    labels = []
    for p in preds:
        lab = p["label"]
        if "label_" in lab.lower():
            idx = int(lab.split("_")[-1])
            lab = id2label[idx]
        labels.append(lab)

    df["sentiment_label"] = labels
    df["sentiment_score"] = [label_to_score(l) for l in labels]
    df.to_csv("data/news.csv", index=False, encoding="utf-8-sig")
    print("updated data/news.csv with sentiment columns")

if __name__ == "__main__":
    main()
