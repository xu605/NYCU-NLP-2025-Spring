import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score
from collections import Counter
from huggingface_hub import login
from peft import get_peft_model, LoraConfig

# Login to Hugging Face
login("api_key")

# Check if CUDA is available (GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)  # Will print "Using device: cuda" if GPU is available, otherwise "cpu"

random_seed = 110550133

# Step 1: Load the dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

train_data = load_data("HW3_dataset/train.json")
val_data = load_data("HW3_dataset/val.json")
test_data = load_data("HW3_dataset/test.json")
print("Loaded data.")

# Step 2: Preprocess the data
class ResponseDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['u'] + " [SEP] " + " ".join(item['s'])
        labels = item.get('r.label', -1)  # Test data may not have labels
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Move tensor to GPU if available
            'attention_mask': inputs['attention_mask'].squeeze(0),  # Move tensor to GPU if available
            'labels': torch.tensor(labels, dtype=torch.long).to(device) if labels != -1 else torch.tensor(0, dtype=torch.long).to(device)
        }

# Step 3: Model and Tokenizer Initialization
model_name = "meta-llama/Llama-3.1-8b"  # Example: Replace with the correct LLaMA 3 model path
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model (Ensure it loads to GPU if available)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # Automatically load model to GPU if available
)

# Step 4: Apply LoRA (Low-Rank Adaptation) for training
lora_config = LoraConfig(
    r=16,  # Rank for the low-rank adaptation
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Add LoRA to the model
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing for memory optimization
model.gradient_checkpointing_enable()  
print(model)

# Create datasets
print("Creating datasets...")
train_dataset = ResponseDataset(train_data, tokenizer)
val_dataset = ResponseDataset(val_data, tokenizer)
test_dataset = ResponseDataset(test_data, tokenizer)

# Check label distribution
train_labels = [d['r.label'] for d in train_data if 'r.label' in d]
print("Label Distribution:", Counter(train_labels))

# Step 5: Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)  # Convert logits to predictions on the correct device
    accuracy = accuracy_score(labels, predictions)
    return {"eval_accuracy": accuracy}

# Step 6: Training setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Reduce epochs for quicker debugging
    per_device_train_batch_size=2,  # Small batch size for low memory usage
    per_device_eval_batch_size=2,
    warmup_steps=100,  # Lower warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="epoch",
    fp16=True,  # Enable mixed precision for GPU
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch sizes
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Step 7: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("Training model...")
trainer.train()

# Step 8: Generate predictions for the test set
def predict(trainer, dataset):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions  # Raw logits
    predicted_classes = torch.argmax(torch.tensor(logits).to(device), axis=1)
    return predicted_classes.tolist()

test_predictions = predict(trainer, test_dataset)

# Step 9: Save predictions in the required Kaggle format
submission = pd.DataFrame({
    'index': range(1, len(test_predictions) + 1),
    'response_quality': test_predictions
})
submission.to_csv("submission.csv", index=False)
