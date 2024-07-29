import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

cleaned_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_cleaned_tweets.csv"
df = pd.read_csv(cleaned_file_path)
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['compound'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment_vader'] = df['compound'].apply(classify_sentiment)

labeled_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_vader.csv"
df.to_csv(labeled_file_path, index=False)

print(df[['cleaned_text', 'compound', 'sentiment_vader']].head())
print(f"Labeled data saved to {labeled_file_path}")