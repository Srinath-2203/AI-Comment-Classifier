# train.py
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib

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
# Load Dataset
# -----------------------------
df = pd.read_csv("data/comments.csv")
print("Initial dataset shape:", df.shape)

df.dropna(subset=["comment", "label"], inplace=True)

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["cleaned_comment"] = df["comment"].apply(clean_text)

# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
X = df["cleaned_comment"]

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vect = vectorizer.fit_transform(X_train)
X_val_vect = vectorizer.transform(X_val)

# -----------------------------
# Train model
# -----------------------------
svm = LinearSVC(class_weight="balanced", max_iter=5000)
svm.fit(X_train_vect, y_train)

model = CalibratedClassifierCV(svm, method="sigmoid", cv="prefit")
model.fit(X_train_vect, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_val_vect)
print(
    "Classification Report:\n",
    classification_report(y_val, y_pred, target_names=label_encoder.classes_)
)

# -----------------------------
# SAVE MODELS (CRITICAL FIX)
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/classifier_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Training complete! Model, vectorizer, and label encoder saved.")
