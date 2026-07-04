"""
Streamlit App: Movie Review Sentiment Analyzer
-------------------------------------------------
Loads the trained LSTM model and lets the user type a review
to get a live positive/negative prediction with confidence score.

Run:
    streamlit run app.py
"""

import pickle
import re
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Load model + tokenizer (cached so it only loads once)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("sentiment_lstm_model.h5")
    with open("tokenizer.pickle", "rb") as f:
        word_index = pickle.load(f)
    with open("config.pickle", "rb") as f:
        config = pickle.load(f)
    return model, word_index, config

model, word_index, config = load_artifacts()
VOCAB_SIZE = config["vocab_size"]
MAX_LEN = config["max_len"]


def preprocess_text(text):
    """Convert raw review text into the padded integer sequence the model expects."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # strip punctuation
    tokens = text.split()

    sequence = []
    for word in tokens:
        idx = word_index.get(word, 2)  # 2 = <UNK>
        # Keras IMDB dataset only kept top VOCAB_SIZE words; map rare words to <UNK>
        if idx >= VOCAB_SIZE:
            idx = 2
        sequence.append(idx)

    padded = pad_sequences([sequence], maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


def predict_sentiment(text):
    padded = preprocess_text(text)
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if prob >= 0.5 else "Negative 😞"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, float(prob), float(confidence)


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown(
    "Type or paste a movie review below. An **LSTM-based RNN**, trained on the "
    "IMDB dataset (50,000 reviews), will predict whether it's **positive** or **negative**."
)

st.divider()

example_reviews = {
    "-- Select an example --": "",
    "Positive example": "This movie was absolutely fantastic! The acting was superb, "
                         "the story kept me engaged from start to finish, and the cinematography was stunning. "
                         "One of the best films I've seen this year.",
    "Negative example": "What a waste of time. The plot made no sense, the acting was wooden, "
                         "and I almost fell asleep halfway through. I would not recommend this to anyone.",
}

choice = st.selectbox("Try an example or write your own:", list(example_reviews.keys()))
default_text = example_reviews[choice]

user_input = st.text_area(
    "Your review:",
    value=default_text,
    height=160,
    placeholder="e.g. The pacing was slow but the visuals made up for it..."
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            label, prob, confidence = predict_sentiment(user_input)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", label)
        with col2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        st.progress(prob, text=f"Positive probability: {prob:.3f}")

        with st.expander("How this works"):
            st.markdown(
                f"""
                - The review is lowercased, stripped of punctuation, and tokenized.
                - Each word is mapped to its integer ID from the IMDB vocabulary
                  (top {VOCAB_SIZE:,} most frequent words; rare/unknown words map to `<UNK>`).
                - The sequence is padded/truncated to **{MAX_LEN} tokens**.
                - A **Bidirectional LSTM** reads the sequence in both directions and
                  outputs a probability between 0 (negative) and 1 (positive).
                """
            )

st.divider()
st.caption("Built with TensorFlow/Keras (Bidirectional LSTM) + Streamlit · Trained on the IMDB Movie Reviews dataset")
