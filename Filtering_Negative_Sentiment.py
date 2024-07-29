import pandas as pd
import re

labeled_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_vader.csv"
df = pd.read_csv(labeled_file_path)

keywords = [
    "costly", "mistake", "problem", "wrong", "threat", "scary", "scared", "stricter",
    "law", "laws", "legal", "issue", "issues", "bad", "fake", "fear", "automate", 
    "existing", "jobs", "school", "education"
]

negative_tweets = df[df['sentiment_vader'] == 'negative']['Text']

def contains_keywords(text, keywords):
    text = re.sub(r'\s+', ' ', text)  
    text = text.lower()  
    for keyword in keywords:
        if keyword in text:
            return True
    return False

filtered_tweets = negative_tweets[negative_tweets.apply(lambda tweet: contains_keywords(tweet, keywords))]

combined_paragraph = ' '.join(filtered_tweets.tolist())

output_file_path = "C:/Users/coolb/Desktop/SarBERT/Negative_Tweets_Examples.txt"
with open(output_file_path, "w", encoding='utf-8') as file:
    file.write(combined_paragraph)

print(f"Combined paragraph saved to {output_file_path}")