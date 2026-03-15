## Movie Review Sentiment Analyzer

Movie Review Sentiment Analyzer is a simple web application that predicts the sentiment of a movie review (positive or negative) using a **TF‑IDF + SVM** model trained on labeled movie review data.

The app is live here: [`https://movie-sentiments.onrender.com/`](https://movie-sentiments.onrender.com/).

---

## Features

- **Interactive web UI** built with Flask and Bootstrap.
- **Paste any movie review** (short phrase or full paragraph).
- **Binary sentiment prediction**: Positive or Negative.
- **Model confidence**: displays SVM decision score and estimated probability for the positive class.
- **Lightweight NLP pipeline** using NLTK for tokenization, stopword removal, and lemmatization.

---

## Project Structure

- `app.py` – Flask application, text preprocessing, and prediction logic.
- `templates/index.html` – Frontend template and styling for the interface.
- `ml_assets/` – Serialized model artifacts (not tracked here) such as:
  - `svm_model.pkl`
  - `tfidf_vectorizer.pkl`
- `requirements.txt` – Python dependencies.
- `movie_sentiment_analysis_model_training.ipynb` – Notebook used to train and export the model and vectorizer.

> **Note**: Ensure that all `.pkl` files required by `app.py` live inside the `ml_assets` directory in the project root.

---

## Getting Started (Local Development)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd movie_sentiment_project
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model assets

Place your trained artifacts inside `ml_assets/` (create the folder if it does not exist):

- `svm_model.pkl`
- `tfidf_vectorizer.pkl`

These must match the objects expected in the training notebook and `app.py`.

### 5. Run the app locally

```bash
python app.py
```

By default the app will start on `http://0.0.0.0:5000` (or `http://localhost:5000`).

Open this URL in your browser and you should see the Movie Review Sentiment Analyzer interface.

---

## How It Works

1. **Preprocessing** (`preprocess_text` in `app.py`):
   - Lowercases text.
   - Removes non-alphabetic characters.
   - Splits on whitespace.
   - Removes English stopwords.
   - Lemmatizes tokens using NLTK's `WordNetLemmatizer`.
2. **Vectorization**:
   - Uses a pre-trained `TfidfVectorizer` (`tfidf_vectorizer.pkl`) to convert text into numerical features.
3. **Prediction**:
   - Feeds features to an SVM classifier (`svm_model.pkl`).
   - Returns:
     - Predicted label (“Positive” or “Negative”).
     - Raw SVM decision score.
     - Probability of the positive class (via `predict_proba` or a logistic transform of the decision score).

---

## Deployment

The app is deployed on Render and can be accessed here:

[`https://movie-sentiments.onrender.com/`](https://movie-sentiments.onrender.com/)

Render runs the Flask application behind a production-grade WSGI server (`gunicorn`), using the same codebase and model artifacts as described above.

---

## Tech Stack

- **Backend**: Flask (Python)
- **ML / NLP**: scikit-learn, NLTK, NumPy
- **Frontend**: HTML, CSS, Bootstrap 5
- **Deployment**: Render (`gunicorn`)

---

## License

This project is for educational and demonstration purposes. You can adapt it for your own experiments or learning. If you use it in a public repository, consider crediting this project or linking back to the deployed demo.

