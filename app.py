# Streamlit Pro-Level AI Comment Classifier (FIXED & STABLE)

import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Absolute base path (CRITICAL FIX)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "classifier_model.pkl"
VECT_PATH  = MODELS_DIR / "vectorizer.pkl"
ENC_PATH   = MODELS_DIR / "label_encoder.pkl"

# -----------------------------
# Validate model files
# -----------------------------
for p in [MODEL_PATH, VECT_PATH, ENC_PATH]:
    if not p.exists():
        st.error(f"Missing required file: {p.name}")
        st.stop()

# -----------------------------
# Safe NLTK setup
# -----------------------------
nltk_packages = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4"
}

for pkg, path in nltk_packages.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
label_encoder = joblib.load(ENC_PATH)

labels = label_encoder.classes_

# -----------------------------
# UI Metadata
# -----------------------------
st.set_page_config(page_title="AI Comment Classifier", layout="wide")

colors = {
    "Insult": "#ff4d4d",
    "Hate": "#cc0000",
    "Threat": "#800000",
    "Love": "#00b33c",
    "Harassment": "#ff9900",
    "Neutral": "#4d79ff"
}

emojis = {
    "Insult": "😡",
    "Hate": "💢",
    "Threat": "⚠️",
    "Love": "❤️",
    "Harassment": "😠",
    "Neutral": "😐"
}

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# UI
# -----------------------------
st.title("🤖 AI Comment Classifier")

comment = st.text_area(
    "Enter your comment",
    height=200,
    placeholder="Type or paste a comment..."
)

if st.button("Predict"):
    if not comment.strip():
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(comment)
        vect = vectorizer.transform([cleaned])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vect)[0]
        else:
            dec = model.decision_function(vect)[0]
            probs = np.exp(dec) / np.sum(np.exp(dec))

        idx = np.argmax(probs)
        label = labels[idx]

        st.success(f"{emojis[label]} **{label}** — Confidence: {probs[idx]*100:.1f}%")

        df = pd.DataFrame({
            "Label": labels,
            "Probability (%)": probs * 100
        }).sort_values("Probability (%)", ascending=False)

        st.bar_chart(df.set_index("Label"))
