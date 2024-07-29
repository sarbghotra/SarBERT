import pandas as pd

labeled_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_roberta.csv"
df = pd.read_csv(labeled_file_path)

label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

df['sentiment_roberta'] = df['sentiment_roberta'].map(label_mapping)
df.to_csv(labeled_file_path, index=False)
print(df.head())