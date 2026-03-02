# Import necessary libraries
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
random_seed = 110550133


# Load and Preprocess Data
class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, labels_list):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet = self.data[idx]["tweet"]
        if "labels" in self.data[idx]:
            labels = [0] * len(self.labels_list)
            for label in self.data[idx]["labels"]:
                labels[self.labels_list.index(label)] = 1
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = torch.zeros(len(self.labels_list), dtype=torch.float)  # Dummy labels for test set

        inputs = self.tokenizer(tweet, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Load data from JSON
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Set paths for the dataset files
train_file = 'dataset/train.json'
val_file = 'dataset/val.json'
test_file = 'dataset/test.json'

train_data = load_data(train_file)
val_data = load_data(val_file)
test_data = load_data(test_file)

# Define model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=12).to(device)

# Define labels list
labels_list = ["ineffective","unnecessary","pharma","rushed","side-effect","mandatory","country","ingredients","political","none","conspiracy","religious"]

# Create datasets
max_len = 128
train_dataset = TweetDataset(train_data, tokenizer, max_len, labels_list)
val_dataset = TweetDataset(val_data, tokenizer, max_len, labels_list)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    seed=random_seed,
)

# Define metrics
def compute_metrics(pred):
    # Convert predictions to binary by thresholding at 0.5 (or use sigmoid if necessary)
    logits = pred.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()  # Apply sigmoid to get probabilities
    y_pred = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1

    # Ensure labels are in binary matrix form
    y_true = pred.label_ids
    # print('12345',y_true,y_pred)
    # Compute metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = (y_true == y_pred).mean()
    return {"f1": f1, "accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()


# Predict on test data
test_dataset = TweetDataset(test_data, tokenizer, max_len, labels_list)
predictions = trainer.predict(test_dataset)

# Prepare submission file
# submission = pd.DataFrame(predictions.predictions, columns=labels_list)
# submission = submission.applymap(lambda x: 1 if x > 0.5 else 0)
# submission.insert(0, "index", [item["ID"] for item in test_data])  # add tweet IDs

# # Save submission to CSV
# submission.to_csv("dataset/output.csv", index=False)
# print("done")

seperated_files = True

for i in range(1,10):
    submission = pd.DataFrame(predictions.predictions, columns=labels_list)
    submission = submission.applymap(lambda x: 1 if x > (i/10) else 0)
    submission.insert(0, "index", [item["ID"] for item in test_data])  # add tweet IDs

    # Save submission to CSV
    if seperated_files:
        filename = "dataset/"+str(trainer.args.num_train_epochs)+"/output_"+str(i/10)+".csv"
    else:
        filename = "dataset/output.csv"
    submission.to_csv(filename, index=False)
    print("done")

for i in range(1,10):
    submission = pd.DataFrame(predictions.predictions, columns=labels_list)
    submission = submission.applymap(lambda x: 1 if x > (i/100) else 0)
    submission.insert(0, "index", [item["ID"] for item in test_data])  # add tweet IDs

    # Save submission to CSV
    if seperated_files:
        filename = "dataset/"+str(trainer.args.num_train_epochs)+"/output_"+str(i/100)+".csv"
    else:
        filename = "dataset/output.csv"
    submission.to_csv(filename, index=False)
    print("done")