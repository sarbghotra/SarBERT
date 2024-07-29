import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI tweets.csv"
df = pd.read_csv(file_path)

def clean(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned_text'] = df['Text'].apply(clean)

cleaned_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_cleaned_tweets.csv"
df.to_csv(cleaned_file_path, index=False)

print(df[['Text', 'cleaned_text']].head())
print(f"Cleaned data saved to {cleaned_file_path}")