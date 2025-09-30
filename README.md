## Sentiment Analysis on Tweets

This project applies Natural Language Processing (NLP) techniques to classify the sentiment of tweets as positive or negative. It covers the full pipeline from text preprocessing to model training and provides an interactive Streamlit app for real-time prediction.

#### Project Overview
Preprocessing tweets (cleaning text, removing noise, handling stopwords, stemming/lemmatization).
Feature extraction using Bag of Words (BoW), TF–IDF, and Word2Vec embeddings.
Training multiple classifiers: Naive Bayes, Logistic Regression, Linear SVM.
Evaluating models based on accuracy and interpretability.
Deploying a lightweight prediction app with Streamlit.
#### Results
Logistic Regression achieved the best performance (~75% accuracy).
Stemming slightly outperformed lemmatization.
Adding linguistic features (POS counts) gave small improvements.

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

#### Project Structure
.
├── app.py                 # Flask/Streamlit web app
├── sentiment_model.pkl    # Trained ML pipeline
├── notebooks/             # Jupyter notebooks for EDA & training
├── requirements.txt       # Dependencies
└── README.md              # Project description
