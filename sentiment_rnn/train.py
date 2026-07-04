"""
Sentiment Analysis on IMDB Movie Reviews using LSTM
-----------------------------------------------------
Trains an LSTM-based RNN to classify movie reviews as positive/negative.

Dataset: IMDB Movie Reviews (50,000 reviews, built into Keras — auto-downloads,
no manual dataset hunting needed).

Run:
    pip install tensorflow numpy
    python train.py

Output:
    - sentiment_lstm_model.h5   (trained model)
    - tokenizer.pickle          (word index, needed by the Streamlit app)
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Config
# -----------------------------
VOCAB_SIZE = 10000      # only keep the top 10,000 most frequent words
MAX_LEN = 200           # pad/truncate every review to 200 words
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 5

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# Pad sequences so every review is the same length (required for batching)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

# -----------------------------
# 2. Build the word index (needed later by Streamlit app to encode user input)
# -----------------------------
word_index = imdb.get_word_index()
# Keras reserves indices 0-3 for padding/start/unknown/unused, so shift by 3
word_index = {word: (idx + 3) for word, idx in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

with open("tokenizer.pickle", "wb") as f:
    pickle.dump(word_index, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved word index to tokenizer.pickle")

# -----------------------------
# 3. Build the model
# -----------------------------
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # binary classification: positive/negative
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 4. Train
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 5. Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------
# 6. Save model
# -----------------------------
model.save("sentiment_lstm_model.h5")
print("Saved model to sentiment_lstm_model.h5")

# Save config too (so the Streamlit app uses the exact same settings)
config = {"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN}
with open("config.pickle", "wb") as f:
    pickle.dump(config, f)

print("\nDone! Now run: streamlit run app.py")
