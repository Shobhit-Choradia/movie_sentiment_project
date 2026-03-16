import os
import pickle
import re
from typing import Tuple

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_assets")
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


def _load_pickle(filename: str):
    """
    Helper to load a pickle file from the model assets directory.
    Expects files like 'stopwords.pkl', 'svm_model.pkl', etc. to be present
    inside the 'ml_assests' folder next to this file.
    """
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model file '{filename}' not found in {MODEL_DIR}. "
            f"Make sure your .pkl files are inside the 'ml_assests' folder."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# Load resources. These must exist in the same folder as this file.
svm_model = _load_pickle("svm_model.pkl")
tfidf_vectorizer = _load_pickle("tfidf_vectorizer.pkl")


def preprocess_text(text: str) -> str:
    """
    Basic preprocessing pipeline that is compatible with a typical
    TF‑IDF + SVM sentiment workflow:
    - lowercase
    - remove non‑alphabetic characters
    - tokenize on whitespace
    - remove stopwords
    - lemmatize tokens
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()

    # Use the stopwords list and lemmatizer that were serialized from your notebook.
    processed_tokens = []
    for token in tokens:
        if token in stopwords:
            continue
        # wordnet_lemmatizer is expected to be an nltk WordNetLemmatizer or similar.
        lemma = lemmatizer.lemmatize(token)
        processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def predict_sentiment(text: str) -> Tuple[str, float, float]:
    """
    Run the full sentiment prediction pipeline.

    Returns:
    - label: "Positive" or "Negative"
    - score: raw SVM decision function score
    - probability: estimated probability of positive sentiment
    """
    processed = preprocess_text(text)
    features = tfidf_vectorizer.transform([processed])

    # Raw decision score
    score = float(svm_model.decision_function(features)[0])

    # Probability of positive class
    try:
        proba = float(svm_model.predict_proba(features)[0][1])
    except Exception:
        # Fallback: convert decision score to a pseudo‑probability using a logistic function
        proba = float(1.0 / (1.0 + np.exp(-score)))

    label = "Positive" if score >= 0 else "Negative"
    return label, score, proba


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    review_text = ""
    result = None
    error = None

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()
        if not review_text:
            error = "Please enter a review before submitting."
        else:
            try:
                label, score, proba = predict_sentiment(review_text)
                result = {
                    "label": label,
                    "score": round(score, 4),
                    "probability": round(proba, 4),
                }
            except Exception as exc:
                error = f"An error occurred while scoring the review: {exc}"

    return render_template(
        "index.html",
        review_text=review_text,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

