import json
import pandas as pd
# from transformers import DataCollatorWithPadding
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import evaluate

import google.generativeai as genai

genai.configure(api_key="api_key")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Check if CUDA is available (GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Step 1: Load the dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# Step 2: Preprocess the data
class ResponseDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        
        self.utterances = [item['u'] for item in data]
        self.responses = [item['r'] for item in data]
        # print([idx for idx in item['s.gold.index']] for item in data)
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
    
train_data = load_data("HW3_dataset/train.json")
val_data = load_data("HW3_dataset/val.json")
# test_data = load_data("HW3_dataset/test_with_s_gold_index_new2.json")
test_data = load_data("HW3_dataset/test_with_s_gold_index_new3.json")
# test_data = load_data("test_with_s_gold_index_cleaned.json")
# test_data = load_data("HW3_dataset/test_with_s_gold_index_1.json")
# test_data = load_data("HW3_dataset/test_with_s_gold_index_cleaned_new.json")
# test_data = load_data("HW3_dataset/test_with_s_gold_index_predict_9.json")
# test_data=load_data("HW3_dataset/test_with_s_gold_index_predict_5.json")
# test_data=load_data("HW3_dataset/test_with_s_gold_index_predict_gpt_1.json")
# test_data=load_data("HW3_dataset/test_with_s_gold_index_predict_gpt_5.json")
# test_data=load_data("HW3_dataset/test_with_s_gold_index_predict_gpt_9.json")

# load the model
model = DebertaV2ForSequenceClassification.from_pretrained("./results/checkpoint-4620")
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
trainer=Trainer(model=model)


train_dataset = ResponseDataset(train_data, tokenizer, is_train=True)
val_dataset = ResponseDataset(val_data, tokenizer, is_train=True)
test_dataset = ResponseDataset(test_data, tokenizer, is_train=False)

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
submission.to_csv("submission_8.csv", index=False)


    
# compare 2 csv files
import pandas as pd
submission1 = pd.read_csv("submission_1.csv")
submission2 = pd.read_csv("submission_8.csv")
accuracy=(submission1['response_quality']==submission2['response_quality']).sum()/len(submission1)
print(accuracy)