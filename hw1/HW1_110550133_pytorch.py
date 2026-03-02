
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe
import numpy as np
import pandas as pd
import tqdm
import json
from matplotlib import pyplot as plt


# parameters
epochs = 10
BATCH_SIZE = 200
validation_split = 0.2
accuracy_threshold = 0.75
valid_accuracy_threshold = 0.5599
embedding_dim = 100
glove = GloVe(name='6B', dim=embedding_dim)
hidden_dim = 39
lstm_layers = 2
num_features = 2  # verified_purchase, helpful_vote
output_size = 5  # For 5 ratings (1-5)
random_seed = 110550133
learning_rate = 0.001
np.random.seed(random_seed)

class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def preprocess_text(self, text, max_len=50):
        tokens = text.lower().split()[:max_len]
        indices = [glove.stoi.get(token, 0) for token in tokens]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices,dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data[idx]
        
        title_tensor = self.preprocess_text(review['title'])
        text_tensor = self.preprocess_text(review['text'])
        
        verified_purchase = torch.tensor([1 if review['verified_purchase'] else 0], dtype=torch.float32)
        helpful_vote = torch.tensor([review['helpful_vote']], dtype=torch.float32)
        features_tensor = torch.cat((verified_purchase, helpful_vote), dim=0)  # (2,)
        
        rating=torch.zeros(5)
        if 'rating' in review:
            rating[review['rating']-1]=1
        
        combined_text = torch.cat((title_tensor, text_tensor), dim=0)
        
        return combined_text, features_tensor, rating

train_file_path = 'train.json'
test_file_path = 'test.json'
with open(train_file_path) as f:
    train_data = json.load(f)
    # text_length_histogram = [len(review['text'].split()) for review in train_data]
    # plt.hist(text_length_histogram, bins=2500)
    # plt.xlabel('Text Length')
    # plt.ylabel('Frequency')
    # plt.title('Text Length Histogram')
    # plt.show()
with open(test_file_path) as f:
    test_data = json.load(f)

# Split the training data into training and validation sets
np.random.shuffle(train_data)
split_idx = int(len(train_data) * (1 - validation_split))
train_data, val_data = train_data[:split_idx], train_data[split_idx:]
train_dataset = ReviewDataset(train_data)
valid_dataset = ReviewDataset(val_data)
test_dataset = ReviewDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class LSTM_DNN_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, lstm_layers, num_features, output_size, glove_embeddings):
        super(LSTM_DNN_Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=False)  # GloVe embeddings
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        # self.fc_text = nn.Linear(hidden_dim, 64)  # Output of LSTM (2 layers)
        # self.fc_features = nn.Linear(num_features, 32)  # For verified_purchase and helpful_vote
        # self.fc_combined = nn.Linear(64 + 32, 32)  # Combine LSTM and numerical features
        # self.fc_output = nn.Linear(32, output_size)  # Final layer for the output (rating)
        self.fc_output = nn.Linear(hidden_dim, output_size)  # Final layer for the output (rating)
        self.relu = nn.ReLU()
        
        # 0.5367 for hidden_dim=64, 0.5363 for hidden_dim=128
        # super(LSTM_DNN_Model, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=False)  # GloVe embeddings
        # self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        # self.fc_text = nn.Linear(hidden_dim, 128)  # Output of LSTM (2 layers)
        # self.fc_features = nn.Linear(num_features, 32)  # For verified_purchase and helpful_vote
        # self.fc_combined = nn.Linear(128 + 32, 32)  # Combine LSTM and numerical features
        # self.fc_output = nn.Linear(32, output_size)  # Final layer for the output (rating)
        # self.relu = nn.ReLU()
        
        # 0.5348
        # super(LSTM_DNN_Model, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=False)  # GloVe embeddings
        # self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        # self.fc_text = nn.Linear(hidden_dim, 64)  # Output of LSTM (2 layers)
        # self.fc_features = nn.Linear(num_features, 32)  # For verified_purchase and helpful_vote
        # self.fc_combined = nn.Linear(64 + 32, 32)  # Combine LSTM and numerical features
        # self.fc_output = nn.Linear(32, output_size)  # Final layer for the output (rating)
        # self.relu = nn.ReLU()
    def forward(self, text, features):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        # text_out = self.relu(self.fc_text(lstm_out))
        # features_out = self.relu(self.fc_features(features))
        # combined = torch.cat((text_out, features_out), dim=1)
        # combined_out = self.relu(self.fc_combined(combined))
        # output = self.fc_output(combined_out)
        output = self.fc_output(lstm_out)
        return output
    

vocab_size = len(glove.stoi)
glove_embeddings = torch.zeros((vocab_size, embedding_dim))
for word, idx in glove.stoi.items():
    glove_embeddings[idx] = glove.vectors[idx]

torch.manual_seed(random_seed)
model = LSTM_DNN_Model(embedding_dim, hidden_dim, lstm_layers, num_features, output_size, glove_embeddings)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for text_inputs, features_inputs, labels in tqdm.tqdm(train_loader):
        text_inputs, features_inputs, labels = text_inputs.to(device), features_inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(text_inputs, features_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    accuracy = (outputs.argmax(1) == labels.argmax(1)).float().mean()
    with torch.no_grad():
        val_loss = 0.0
        for text_inputs, features_inputs, labels in valid_loader:
            text_inputs, features_inputs, labels = text_inputs.to(device), features_inputs.to(device), labels.to(device)
            outputs = model(text_inputs, features_inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        val_accuracy = (outputs.argmax(1) == labels.argmax(1)).float().mean()
    if validation_split>0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}, Val Loss: {val_loss / len(valid_loader)}, Val Accuracy: {val_accuracy}')
        if val_accuracy > valid_accuracy_threshold:
            break
    else:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}')
    if accuracy > accuracy_threshold:
        break

print('Model training done')
torch.save(model.state_dict(), 'lstm_dnn_model.pth')

model.eval()
with torch.no_grad():
    y_test = []
    for text_inputs, features_inputs, _ in test_loader:
        text_inputs, features_inputs = text_inputs.to(device), features_inputs.to(device)
        outputs = model(text_inputs, features_inputs)
        _, predicted = torch.max(outputs, 1)
        y_test += predicted.tolist()
    # print(y_test)
    
output = [["index", "rating"]]
for i in range(35000):
    output.append([f'index_{i}', y_test[i] + 1])

import csv
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)

print("Predictions saved to output.csv")
