# 📧 Spam Detection Model

## 📌 Overview
This project implements a machine learning-based SMS Spam Detection system that classifies text messages as **Spam** or **Ham (Not Spam)**. The model uses Natural Language Processing (NLP) techniques for text preprocessing and a Multinomial Naive Bayes classifier for prediction.

---

## 🚀 Features
- SMS text preprocessing
- Stopword removal and stemming
- CountVectorizer for text vectorization
- Multinomial Naive Bayes classification
- Model evaluation using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
- Spam and Ham Word Cloud visualization

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- WordCloud

---

## 📂 Dataset
The project uses the **SMS Spam Collection Dataset**, containing labeled SMS messages as:
- Ham (Not Spam)
- Spam

---

## ⚙️ Workflow
1. Load the SMS dataset.
2. Preprocess text (lowercase, remove punctuation, remove stopwords, stemming).
3. Convert text into numerical features using CountVectorizer.
4. Split the dataset into training and testing sets.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate model performance.
7. Visualize results using confusion matrix, ROC curve, precision-recall curve, and word clouds.

---

## 📊 Model Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve
- Precision-Recall Curve

---

## ▶️ How to Run
1. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
   ```

2. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

3. Update the dataset path in the code.

4. Run:
   ```bash
   python spamdetection.py
   ```

---

## 📁 Project Structure
```
SpamDetection/
│── spamdetection.py
│── SMSSpamCollection
│── README.md
```

---

## 📌 Future Improvements
- Use TF-IDF Vectorizer.
- Compare multiple machine learning algorithms.
- Build a Flask/Streamlit web application.
- Deploy the model on the cloud.
- Improve accuracy using hyperparameter tuning.

---

## 👨‍💻 Author
**Vishal M**

B.Tech Computer Science and Engineering
SASTRA Deemed University
