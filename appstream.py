import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")

st.title("Sentiment Analysis App")

tweet = st.text_area("Enter a tweet to analyze:", "")

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred = model.predict([tweet])[0]
        st.success(f"Predicted Sentiment: {pred}")
