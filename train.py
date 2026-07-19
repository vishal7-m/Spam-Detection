"""
Trains the spam detection model using the exact same pipeline as the
original notebook (spamdetection.ipynb):

    cleantext() -> lowercase, strip non-letters, remove stopwords, stem
    CountVectorizer(max_features=3000)
    MultinomialNB()

Saves vectorizer.pkl and model.pkl for the Flask backend to load.
"""
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def cleantext(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


def main():
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "msg"])
    df["cleaned"] = df["msg"].apply(cleantext)

    vectorizer = CountVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["cleaned"]).toarray()
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved vectorizer.pkl and model.pkl")


if __name__ == "__main__":
    main()
