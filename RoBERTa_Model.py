import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np 

labeled_file_path = "C:/Users/coolb/Desktop/SarBERT/GenerativeAI_labeled_roberta.csv"
df = pd.read_csv(labeled_file_path)
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['sentiment_roberta'])

class RobertaTweetDataset(Dataset):
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

tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
MAX_LEN = 160
BATCH_SIZE = 16
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label_encoded'], test_size=0.3, random_state=42)

train_dataset_roberta = RobertaTweetDataset(
    texts=X_train.tolist(),
    labels=y_train.tolist(),
    tokenizer=tokenizer_roberta,
    max_len=MAX_LEN
)

test_dataset_roberta = RobertaTweetDataset(
    texts=X_test.tolist(),
    labels=y_test.tolist(),
    tokenizer=tokenizer_roberta,
    max_len=MAX_LEN
)

train_data_loader_roberta = DataLoader(train_dataset_roberta, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader_roberta = DataLoader(test_dataset_roberta, batch_size=BATCH_SIZE, shuffle=False)

def train_roberta_epoch(
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

    for d in tqdm(data_loader, desc="Training Roberta"):  
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

def eval_roberta_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating Roberta"): 
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_roberta = model_roberta.to(device)
loss_fn_roberta = torch.nn.CrossEntropyLoss().to(device)
optimizer_roberta = AdamW(model_roberta.parameters(), lr=2e-5, correct_bias=False)
total_steps_roberta = len(train_data_loader_roberta) * 4
scheduler_roberta = torch.optim.lr_scheduler.StepLR(optimizer_roberta, step_size=3, gamma=0.1)

for epoch in range(4):
    print(f'Epoch {epoch + 1}/{4}')
    print('-' * 10)

    train_acc_roberta, train_loss_roberta = train_roberta_epoch(
        model_roberta,
        train_data_loader_roberta,
        loss_fn_roberta,
        optimizer_roberta,
        device,
        scheduler_roberta,
        len(train_dataset_roberta)
    )

    print(f'Train loss {train_loss_roberta} accuracy {train_acc_roberta}')

    val_acc_roberta, val_loss_roberta = eval_roberta_model(
        model_roberta,
        test_data_loader_roberta,
        loss_fn_roberta,
        device,
        len(test_dataset_roberta)
    )

    print(f'Val   loss {val_loss_roberta} accuracy {val_acc_roberta}')
    print()

model_roberta.eval()
predictions_roberta, true_labels_roberta = [], []

with torch.no_grad():
    for d in tqdm(test_data_loader_roberta, desc="Testing Roberta"):  
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = model_roberta(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        predictions_roberta.extend(preds)
        true_labels_roberta.extend(labels)

predictions_roberta = torch.stack(predictions_roberta).cpu()
true_labels_roberta = torch.stack(true_labels_roberta).cpu()

print("Roberta Classification Report:")
print(classification_report(true_labels_roberta, predictions_roberta, target_names=['negative', 'neutral', 'positive']))
print(f"Accuracy: {accuracy_score(true_labels_roberta, predictions_roberta)}")
print(f"Precision: {precision_score(true_labels_roberta, predictions_roberta, average='weighted')}")
print(f"Recall: {recall_score(true_labels_roberta, predictions_roberta, average='weighted')}")
print(f"F1-Score: {f1_score(true_labels_roberta, predictions_roberta, average='weighted')}")