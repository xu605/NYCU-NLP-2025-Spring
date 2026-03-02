import json
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 1: Load Data
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_data("HW3_dataset/train.json")
val_data = load_data("HW3_dataset/val.json")

# Step 2: Prepare the Dataset
class GoldIndexDataset(Dataset):
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
        situations = " ".join(self.situations[idx])

        text = f"Utterance: {utterance} [SEP] Situations: {situations}"

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
        
        # print(f"Input IDs shape: {inputs['input_ids'].shape}")
        # print(f"Attention Mask shape: {inputs['attention_mask'].shape}")
        # print(f"Labels shape: {inputs['labels'].shape}")
        return inputs

# Step 3: Initialize Tokenizer and Dataset
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = GoldIndexDataset(train_data, tokenizer, is_test=False)
val_dataset = GoldIndexDataset(val_data, tokenizer, is_test=False)

# Step 4: Load GPT-2 Model
model = GPT2ForSequenceClassification.from_pretrained(model_name,num_labels=12,problem_type="multi_label_classification").to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results_gpt2",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,  # Adjust based on GPU memory
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    
    correct = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[i][j] == labels[i][j]:
                correct += 1
    
    accuracy = correct / (len(predictions) * len(predictions[0]))
    # Compute micro F1 score
    f1 = f1_score(labels, predictions, average="micro")
    return {"example_accuracy": accuracy, "f1": f1}

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# for i in range(3):  # Check first 3 samples
#     sample = train_dataset[i]
#     print(f"Input IDs: {sample['input_ids'].shape}")
#     print(f"Labels: {sample['labels'].shape}")
#     print(f"Labels content: {sample['labels']}")


# Step 7: Train the Model
trainer.train()

# Step 8: Predict Relevant Indices for Test Data
test_data = load_data("HW3_dataset/test.json")
test_dataset = GoldIndexDataset(test_data, tokenizer, is_test=True)

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

    output_file = f"HW3_dataset/test_with_s_gold_index_predict_gpt_{t}.json"
    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=2)

    print("Saved predictions to", output_file)
