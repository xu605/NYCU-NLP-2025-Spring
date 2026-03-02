import json
import pandas as pd
# from transformers import DataCollatorWithPadding
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import torch
import numpy as np
import evaluate
import tqdm
import time
import google.generativeai as genai

# Check if CUDA is available (GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# debert for relevant situation

# Load Data
def load_data1(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data1 = load_data1("HW3_dataset/train.json")
val_data1 = load_data1("HW3_dataset/val.json")
test_data1 = load_data1("HW3_dataset/test.json")

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
model_name1 = "microsoft/deberta-v3-base"
tokenizer1 = DebertaV2Tokenizer.from_pretrained(model_name1)
model1 = DebertaV2ForSequenceClassification.from_pretrained(
    model_name1,
    num_labels=12,  # Adjust this if the number of situations changes
    problem_type="multi_label_classification",
).to(device)

# Prepare Datasets
train_dataset1 = RelevantSituationDataset(train_data1, tokenizer1, is_test=False)
val_dataset1 = RelevantSituationDataset(val_data1, tokenizer1, is_test=False)
test_dataset1 = RelevantSituationDataset(test_data1, tokenizer1, is_test=True)

# Define Metrics
def compute_metrics1(eval_pred):
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
training_args1 = TrainingArguments(
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
trainer1 = Trainer(
    model=model1,
    args=training_args1,
    train_dataset=train_dataset1,
    eval_dataset=val_dataset1,
    tokenizer=tokenizer1,
    compute_metrics=compute_metrics1,
)

# Train the Model
trainer1.train()

# Generate Predictions
predictions1 = trainer1.predict(test_dataset1)
logits1 = torch.tensor(predictions1.predictions).cpu().numpy()

# Process Predictions
def process_predictions1(logits, threshold=0.5):
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
    relevant_situations = process_predictions1(logits1, threshold=t/10)

    # Save Predictions
    for i, item in enumerate(test_data1):
        item["s.gold.index"] = [int(idx) for idx in relevant_situations[i]]

    output_file = f"HW3_dataset/test_with_s_gold_index_predict_{t}.json"
    with open(output_file, "w") as f:
        json.dump(test_data1, f, indent=2)

    print("Saved predictions to", output_file)





# use gemini to improve

def load_data2(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

train_data2 = load_data2("HW3_dataset/train.json")
val_data2 = load_data2("HW3_dataset/val.json")
test_data2 = load_data2("HW3_dataset/test_with_s_gold_index_predict_5.json")

genai.configure(api_key="AIzaSyCFzBezes8c_UAeBn_I7vUdLzZsC1TqE9U")
model2 = genai.GenerativeModel(model_name="gemini-1.5-flash")

t=4
valid=1

cnt=0
for sample in tqdm.tqdm(test_data2):
    try:
        utterance = sample['u']
        situation = sample['s']
        response = sample['r']
        gold_index = sample['s.gold.index']
        contents=f'The following are given utterance(u) and situation(s) and response(r). utterance : {utterance}, situation : {situation}, response : {response}, gold.index : {gold_index}. The gold.index in situation represent the relevant situation starting by index 0. The gold.index in situation is estimated and it might be wrong. There is about 1/3 of gold.index is wrong. Evaluate the right gold.index and give me a list of relevant situations(s) each consisting of 6~9 numbers, sorted in descending order of importance. These numbers represent the index in the situation list. Expected output like: [1, 3, 5]. no any other answer.'
        response_text = model2.generate_content(contents)
        response_list=json.loads(response_text.text)
        test_predictions=response_list
    except:
        test_predictions=sample['s.gold.index']
        cnt+=1
    sample["s.gold.index"] = test_predictions
    # print('Predicted:', test_predictions)
    time.sleep(t)
    
        
print('Number of failed predictions:', cnt, 'out of', 792)

with open("HW3_dataset/test_with_s_gold_index_cleaned_new.json", 'w') as f:
    json.dump(test_data2, f, indent=2)





# deberta for response quality

# Step 1: Load the dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

train_data = load_data("HW3_dataset/train.json")
val_data = load_data("HW3_dataset/val.json")
# test_data = load_data("HW3_dataset/test.json")
# test_data = load_data("test_with_s_gold_index_cleaned.json")
# test_data = load_data("HW3_dataset/test_with_s_gold_index_new.json") ###
test_data = load_data("HW3_dataset/test_with_s_gold_index_cleaned_new.json")

# Step 2: Preprocess the data
class ResponseDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        
        self.utterances = [item['u'] for item in data]
        self.responses = [item['r'] for item in data]
        self.relevant_situations = [[item['s'][idx] for idx in item['s.gold.index']] for item in data]
        situations = [" ".join(relevant_situation) for relevant_situation in self.relevant_situations]
        self.texts = [utterance + " [SEP] " + situation + " [SEP] " + response for utterance, situation, response in zip(self.utterances, situations, self.responses)]
        
        self.inputs = self.tokenizer(
            self.texts,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        self.labels = [torch.tensor(item.get("r.label", 0), dtype=torch.long) for item in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = {key: value[index] for key, value in self.inputs.items()}
        input["labels"] = self.labels[index]
        return input

# Initialize tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)

# Create datasets
train_dataset = ResponseDataset(train_data, tokenizer, is_train=True)
val_dataset = ResponseDataset(val_data, tokenizer, is_train=True)
test_dataset = ResponseDataset(test_data, tokenizer, is_train=False)

# print(train_dataset[0])

# Step 3: Define the model
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
metric = evaluate.load("accuracy")


# Step 4: Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class
    accuracy = metric.compute(predictions=predictions, references=labels)
    return accuracy


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


print(len(val_dataset))  # Should print 792
print(len(test_dataset))
# print(train_dataset[0])

# Step 5: Train the model
trainer.train()

# Step 6: Generate predictions for the test set
def predict(trainer, dataset):
    predictions = trainer.predict(dataset)
    logits, labels = predictions.predictions, predictions.label_ids
    return np.argmax(logits, axis=1)


test_predictions = predict(trainer, test_dataset)
print(len(test_predictions))

# Step 7: Save predictions in the required Kaggle format
submission = pd.DataFrame({
    'index': range(len(test_predictions)),
    'response_quality': test_predictions
})
submission.to_csv("submission.csv", index=False)