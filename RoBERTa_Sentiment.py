import pandas as pd
from transformers import pipeline
from tqdm import tqdm  

cleaned_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_cleaned_tweets.csv"
df = pd.read_csv(cleaned_file_path)
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def classify_sentiment(text):
    result = pipe(text)
    return result[0]['label']

tqdm.pandas()
df['sentiment_roberta'] = df['cleaned_text'].progress_apply(classify_sentiment)

labeled_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_roberta.csv"
df.to_csv(labeled_file_path, index=False)

print(df[['cleaned_text', 'sentiment_roberta']].head())
print(f"Labeled data saved to {labeled_file_path}")