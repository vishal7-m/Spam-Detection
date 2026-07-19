Vilgax Spam Detection
SMS spam classifier (CountVectorizer + MultinomialNB, ~98.4% test accuracy)
with a Flask backend and a dark-themed frontend.
Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```
Run
```bash
python app.py
```
Server starts at `http://localhost:5000`. Then open `index.html` in your
browser (it calls `http://localhost:5000/predict`).
Files
`SMSSpamCollection` — training dataset (label \t message)
`train.py` — retrains the model from scratch, saves `vectorizer.pkl` + `model.pkl`
`vectorizer.pkl`, `model.pkl` — already-trained model (ready to use)
`app.py` — Flask API, exposes `POST /predict`
`index.html` — frontend UI
API
`POST /predict`
```json
{ "message": "Congratulations! You won a free prize" }
```
Response:
```json
{ "label": "Spam", "is_spam": true, "confidence": 0.99 }
```
