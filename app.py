
# Streamlit Pro-Level AI Comment Classifier (Modern UI)
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd

@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

load_nltk()

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load model & vectorizer
# -----------------------------
model = joblib.load("models/classifier_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# -----------------------------
# Labels, colors & emojis
# -----------------------------
labels = label_encoder.classes_
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
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Comment Classifier", layout="wide")

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
# Comment input (UPDATED)
# -----------------------------
comment_input = st.text_area(
    "Enter your comment here",
    height=260,
    placeholder="Type or paste a comment to classify..."
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict", key="predict_btn"):

    if comment_input.strip() == "":
        st.warning("Please enter a comment!")
    else:
        cleaned = clean_text(comment_input)
        vect = vectorizer.transform([cleaned])

        # Predict probabilities
        if hasattr(model, "decision_function"):
            dec = model.decision_function(vect)[0]
            probs = np.exp(dec) / np.sum(np.exp(dec))
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(vect)[0]
        else:
            probs = np.zeros(len(labels))
            probs[model.predict(vect)[0]] = 1.0

        pred_index = np.argmax(probs)
        predicted_label = labels[pred_index]

        # -----------------------------
        # Main Prediction Card
        # -----------------------------
        main_color = colors[predicted_label]
        main_emoji = emojis[predicted_label]

        st.markdown(
            f"""
            <div style='background:{main_color};
                        padding:25px;
                        border-radius:16px;
                        color:white;
                        text-align:center;
                        font-size:24px;
                        font-weight:bold;
                        box-shadow:0 8px 25px rgba(0,0,0,0.3);
                        margin-bottom:20px;'>
                {main_emoji} Predicted: {predicted_label} – Confidence: {probs[pred_index]*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Probabilities DataFrame
        # -----------------------------
        df_probs = pd.DataFrame({
            "Label": labels,
            "Probability": probs * 100,
            "Color": [colors[l] for l in labels],
            "Emoji": [emojis[l] for l in labels]
        }).sort_values("Probability", ascending=False)

        # -----------------------------
        # Top 3 Predictions
        # -----------------------------
        st.markdown("<h3 style='text-align:center;'>Top 3 Predictions</h3>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i in range(3):
            row = df_probs.iloc[i]
            cols[i].markdown(
                f"""
                <div style='background:{row["Color"]};
                            padding:20px;
                            border-radius:12px;
                            color:white;
                            text-align:center;
                            font-size:18px;
                            font-weight:bold;
                            box-shadow:0 4px 12px rgba(0,0,0,0.2);'>
                    {row["Emoji"]} {row["Label"]}<br>{row["Probability"]:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # -----------------------------
        # All Class Probabilities Chart
        # -----------------------------
        st.markdown("<h3 style='text-align:center;margin-top:30px;'>All Class Probabilities</h3>", unsafe_allow_html=True)
        st.bar_chart(df_probs.set_index("Label")["Probability"], height=260)
