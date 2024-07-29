import pandas as pd

roberta_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_roberta.csv"
df_roberta = pd.read_csv(roberta_file_path)

vader_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_vader.csv"
df_vader = pd.read_csv(vader_file_path)

df_combined = pd.merge(df_roberta, df_vader[['cleaned_text', 'sentiment_vader']], on='cleaned_text')

combined_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_combined.csv"
df_combined.to_csv(combined_file_path, index=False)

print(df_combined.head())
print(f"Combined data saved to {combined_file_path}")