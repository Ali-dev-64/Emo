import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


########### IF YOU WANT TO TRAIN YOUR OWN DATA ###########

 # Example sentences and labels (replace with your dataset)
sentences = ["I am so happy today!", "I feel really sad.", "What an amazing experience!"]
labels = ["joy", "sadness", "joy"]
# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(set(labels)), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Train the model
history = model.fit(train_padded, np.array(y_train),
                    epochs=10,
                    validation_data=(test_padded, np.array(y_test)),
                    verbose=2)
# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_padded, np.array(y_test), verbose=2)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
