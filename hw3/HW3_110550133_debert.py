import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import f1_score

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_data("HW3_dataset/train.json")
val_data = load_data("HW3_dataset/val.json")
test_data = load_data("HW3_dataset/test.json")

# Prepare Dataset Class
class RelevantSituationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, is_test=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        self.utterances = [item["u"] for item in data]
        # self.responses = [item["r"] for item in data]
        self.situations = [item["s"] for item in data]
        if self.is_test:
            self.gold_indices = None
            self.targets = None
        else:
            self.gold_indices = [item["s.gold.index"] for item in data]
            mlb = MultiLabelBinarizer(classes=list(range(len(self.situations[0]))))
            self.targets = mlb.fit_transform(self.gold_indices)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        # response = self.responses[idx]
        situations = " ".join(self.situations[idx])

        # Concatenate utterance, response, and situations
        text = f"Utterance: {utterance} [SEP] Situations: {situations}"

        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        if self.is_test:
            inputs["labels"] = torch.zeros(len(self.situations[0]), dtype=torch.float)
        else:
            inputs["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return inputs

# Initialize DeBERTa Tokenizer and Model
model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=12,  # Adjust this if the number of situations changes
    problem_type="multi_label_classification",
).to(device)

# Prepare Datasets
train_dataset = RelevantSituationDataset(train_data, tokenizer, is_test=False)
val_dataset = RelevantSituationDataset(val_data, tokenizer, is_test=False)
test_dataset = RelevantSituationDataset(test_data, tokenizer, is_test=True)

# Define Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    
    # print(predictions)
    # print(labels)
    # Compute example-based accuracy
    
    correct = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[i][j] == labels[i][j]:
                correct += 1
    
    accuracy = correct / (len(predictions) * len(predictions[0]))
    # Compute micro F1 score
    f1 = f1_score(labels, predictions, average="micro")
    return {"example_accuracy": accuracy, "f1": f1}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs2",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the Model
trainer.train()

# Generate Predictions
predictions = trainer.predict(test_dataset)
logits = torch.tensor(predictions.predictions).cpu().numpy()

# Process Predictions
def process_predictions(logits, threshold=0.5):
    results = []
    for logit in logits:
        sorted_indices = np.argsort(-logit)  # Sort by importance
        relevant_indices = [idx for idx in sorted_indices if logit[idx] >= threshold]
        results.append(relevant_indices)
        # print(relevant_indices)
    average_relevant_indices = np.mean([len(item) for item in results])
    print(f"Average relevant indices for threshold {threshold}: {average_relevant_indices}")
    return results

# Get Relevant Situations
for t in range(1,10):
    relevant_situations = process_predictions(logits, threshold=t/10)

    # Save Predictions
    for i, item in enumerate(test_data):
        item["s.gold.index"] = [int(idx) for idx in relevant_situations[i]]

    output_file = f"HW3_dataset/test_with_s_gold_index_predict_{t}.json"
    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=2)

    print("Saved predictions to", output_file)
