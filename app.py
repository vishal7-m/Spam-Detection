"""
Vilgax Spam Detection - Backend API

Loads the trained CountVectorizer + MultinomialNB model and exposes
a single POST /predict endpoint used by the frontend UI.

Run:
    python train.py     # one-time: builds vectorizer.pkl + model.pkl
    python app.py        # starts the API on http://localhost:5000
"""
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = Flask(__name__)
CORS(app)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def cleantext(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    cleaned = cleantext(message)
    vec = vectorizer.transform([cleaned])

    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(proba[prediction])

    label = "Spam" if prediction == 1 else "Ham"

    return jsonify({
        "label": label,
        "is_spam": bool(prediction),
        "confidence": round(confidence, 4)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
