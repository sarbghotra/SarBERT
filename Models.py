import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm  

labeled_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_vader.csv"
df = pd.read_csv(labeled_file_path)

tfidf_vectorizer = TfidfVectorizer(max_features=1000) 
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
y = df['sentiment_vader']
X_train, X_test, y_train, y_test = train_test_split(tfidf_df, y, test_size=0.3, random_state=42)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Linear SVM': LinearSVC(max_iter=1000)  
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}\n")

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['sentiment_vader'])

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length', 
            truncation=True,     
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training BERT"): 
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating BERT"): 
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=3)

MAX_LEN = 160
BATCH_SIZE = 16

train_dataset = TweetDataset(
    texts=df.loc[X_train.index, 'cleaned_text'].tolist(),
    labels=df.loc[X_train.index, 'label_encoded'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = TweetDataset(
    texts=df.loc[X_test.index, 'cleaned_text'].tolist(),
    labels=df.loc[X_test.index, 'label_encoded'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(bert_model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * 4
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(4):
    print(f'Epoch {epoch + 1}/{4}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        bert_model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_dataset)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        bert_model,
        test_data_loader,
        loss_fn,
        device,
        len(test_dataset)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

bert_model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for d in tqdm(test_data_loader, desc="Testing BERT"):  
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = bert_model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        predictions.extend(preds)
        true_labels.extend(labels)

predictions = torch.stack(predictions).cpu()
true_labels = torch.stack(true_labels).cpu()

print("BERT Classification Report:")
print(classification_report(true_labels, predictions, target_names=['negative', 'neutral', 'positive']))
print(f"Accuracy: {accuracy_score(true_labels, predictions)}")
print(f"Precision: {precision_score(true_labels, predictions, average='weighted')}")
print(f"Recall: {recall_score(true_labels, predictions, average='weighted')}")
print(f"F1-Score: {f1_score(true_labels, predictions, average='weighted')}")