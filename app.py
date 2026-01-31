# Streamlit Pro-Level AI Comment Classifier (Modern UI)

import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Streamlit config (FIRST)
# -----------------------------
st.set_page_config(
    page_title="AI Comment Classifier",
    layout="wide",
    page_icon="🤖"
)

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
# Load trained models (NO TRAINING HERE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
ENC_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
label_encoder = joblib.load(ENC_PATH)

labels = label_encoder.classes_

# -----------------------------
# Labels, colors & emojis
# -----------------------------
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
# Text preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
label {
    font-size: 18px !important;
    font-weight: 600;
}
div.stButton > button {
    font-size: 18px;
    padding: 12px 36px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style='background:linear-gradient(90deg, #36d1dc, #5b86e5);
            padding:25px;
            border-radius:12px;
            text-align:center;
            color:white;
            box-shadow:0 4px 20px rgba(0,0,0,0.2);'>
    <h1>🤖 AI Comment Classifier</h1>
    <p style='font-size:18px;'>
        Detect Hate, Insult, Threat, Love, Harassment, or Neutral comments instantly
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Comment input
# -----------------------------
comment_input = st.text_area(
    "Enter your comment here",
    height=260,
    placeholder="Type or paste a comment to classify..."
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):

    if not comment_input.strip():
        st.warning("Please enter a comment!")
    else:
        cleaned = clean_text(comment_input)
        vect = vectorizer.transform([cleaned])

        # Get probabilities safely
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vect)[0]
        else:
            dec = model.decision_function(vect)[0]
            probs = np.exp(dec) / np.sum(np.exp(dec))

        pred_index = np.argmax(probs)
        predicted_label = labels[pred_index]

        # -----------------------------
        # Main Prediction Card
        # -----------------------------
        st.markdown(
            f"""
            <div style='background:{colors[predicted_label]};
                        padding:25px;
                        border-radius:16px;
                        color:white;
                        text-align:center;
                        font-size:24px;
                        font-weight:bold;
                        box-shadow:0 8px 25px rgba(0,0,0,0.3);
                        margin-bottom:20px;'>
                {emojis[predicted_label]} Predicted: {predicted_label}
                <br>
                Confidence: {probs[pred_index]*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Probabilities table
        # -----------------------------
        df_probs = pd.DataFrame({
            "Label": labels,
            "Probability": probs * 100
        }).sort_values("Probability", ascending=False)

        # -----------------------------
        # Top 3 predictions
        # -----------------------------
        st.markdown("<h3 style='text-align:center;'>Top 3 Predictions</h3>", unsafe_allow_html=True)
        cols = st.columns(3)

        for i in range(3):
            row = df_probs.iloc[i]
            cols[i].markdown(
                f"""
                <div style='background:{colors[row["Label"]]};
                            padding:20px;
                            border-radius:12px;
                            color:white;
                            text-align:center;
                            font-size:18px;
                            font-weight:bold;'>
                    {emojis[row["Label"]]} {row["Label"]}
                    <br>{row["Probability"]:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # -----------------------------
        # Probability chart
        # -----------------------------
        st.markdown("<h3 style='text-align:center;margin-top:30px;'>All Class Probabilities</h3>", unsafe_allow_html=True)
        st.bar_chart(df_probs.set_index("Label"))
