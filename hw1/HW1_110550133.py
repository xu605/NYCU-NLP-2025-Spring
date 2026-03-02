import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


# parameters
epochs = 10
BATCH_SIZE = 200
validation_split = 0.2
accuracy_threshold = 0.75
valid_accuracy_threshold = 0.5599
embedding_dim = 100
lstm_layers = 2
num_features = 2  # verified_purchase, helpful_vote
output_size = 5  # For 5 ratings (1-5)
random_seed = 110550133
learning_rate = 0.001
np.random.seed(random_seed)

def load_glove_embeddings(filepath, embedding_dim):
    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs
    return embeddings_index

glove_embeddings = load_glove_embeddings('glove.6B.100d.txt', embedding_dim)


def create_embedding_matrix(word_index, embedding_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

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

max_len = 50  # Maximum length for padding
tokenizer = Tokenizer()
train_texts = [d['text'] for d in train_data]

tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
padded_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')

# Create embedding matrix
embedding_matrix = create_embedding_matrix(tokenizer.word_index, glove_embeddings, embedding_dim)

# Define input layers
text_input = Input(shape=(max_len,), name="text_input")
features_input = Input(shape=(2,), name="features_input")  # verified_purchase and helpful_vote

# Define embedding layer using pre-trained GloVe
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)(text_input)

# LSTM Layer
lstm_out = LSTM(64)(embedding_layer)

# Combine LSTM output and numerical features
combined = Concatenate()([lstm_out, features_input])

# Fully Connected Layers (DNN)
# fc1 = Dense(64, activation='relu')(combined)
# fc2 = Dense(32, activation='relu')(fc1)
output = Dense(5, activation='softmax')(combined)

# Build the model
model = Model(inputs=[text_input, features_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Assume you have 'train_data' and 'test_data' loaded

train_labels = [d['rating'] - 1 for d in train_data]  # Ratings should be in range [0-4]
train_labels = np.eye(output_size)[train_labels]  # One-hot encoding

features = np.array([[d['verified_purchase'], d['helpful_vote']] for d in train_data])

# Train the model
model.fit([padded_sequences, features], train_labels, epochs=10, batch_size=200)

# Save the model
model.save("keras_model.h5")

# Test data preprocessing
test_texts = [d['text'] for d in test_data]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')
test_features = np.array([[d['verified_purchase'], d['helpful_vote']] for d in test_data])

# Predict
predictions = model.predict([test_padded, test_features])
predicted_labels = np.argmax(predictions, axis=1)

# Save predictions
output = [["index", "rating"]]
for i, prediction in enumerate(predicted_labels):
    output.append([f'index_{i}', prediction + 1])

import csv
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)

print("Predictions saved to output.csv")