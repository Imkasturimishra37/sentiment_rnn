# 🎬 Movie Review Sentiment Analysis (LSTM + Streamlit)

Classify movie reviews as **positive** or **negative** using a Bidirectional LSTM
RNN trained on the IMDB Movie Reviews dataset (50,000 reviews), with a live
Streamlit demo to test your own text.

## Project structure

```
sentiment_rnn/
├── train.py            # trains the LSTM model on IMDB dataset
├── app.py               # Streamlit app for live predictions
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py
```

- The IMDB dataset downloads automatically the first time (via Keras) —
  no manual dataset download needed.
- Training takes roughly **5-10 minutes on CPU**, ~1-2 minutes on GPU, for 5 epochs.
- Expected test accuracy: **~85-88%** with this architecture.
- This produces 3 files used by the app:
  - `sentiment_lstm_model.h5` — the trained model
  - `tokenizer.pickle` — word → integer index mapping
  - `config.pickle` — vocab size / max sequence length used at train time

## 3. Run the Streamlit app

```bash
streamlit run app.py
```

This opens a browser tab where you can:
- Type/paste any movie review
- Click "Analyze Sentiment"
- See the predicted label, confidence %, and a probability bar
- Try the built-in positive/negative examples

## How it works (short version)

1. **Tokenization**: each word in a review is mapped to an integer ID based on
   frequency rank in the IMDB training corpus (top 10,000 words kept; rarer
   words become `<UNK>`).
2. **Padding**: every review is padded/truncated to 200 tokens so they can be
   batched together.
3. **Embedding layer**: maps each integer ID to a dense 128-dim vector.
4. **Bidirectional LSTM**: reads the sequence forward and backward, capturing
   context from both directions (e.g. "not good" vs "good").
5. **Dense + Sigmoid**: outputs a single probability — closer to 1 means
   positive, closer to 0 means negative.

## Things you can extend this into (optional, for a stronger portfolio piece)

- Swap `LSTM` for `GRU` and compare speed/accuracy.
- Add an attention layer over the LSTM outputs and visualize which words
  drove the prediction.
- Fine-tune on a different domain (e.g. product reviews, tweets) to show
  transfer learning.
- Deploy the Streamlit app publicly via [Streamlit Community Cloud](https://streamlit.io/cloud)
  (free, just connect your GitHub repo).
- Add a confusion matrix / ROC curve to `train.py` for a more rigorous eval
  section in your writeup.

## Dataset

[IMDB Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) —
50,000 reviews (25k train / 25k test, balanced positive/negative). Loaded
automatically through `tensorflow.keras.datasets.imdb`, no manual download
required.
