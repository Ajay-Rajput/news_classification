import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load trained model
model, vectorizer, X_test, y_test = joblib.load("models/news_classifier.pkl")

# Page title
st.title("📰 News Text Classification App")

st.write("Enter a news headline or article text to classify it into categories.")

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# User input
user_input = st.text_area("Enter News Text")

# Prediction button
if st.button("Predict Category"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Clean text
        cleaned = clean_text(user_input)

        # Vectorize
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)[0]

        # AG News Labels
        labels = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Science / Technology"
        }

        result = labels.get(prediction, "Unknown")

        st.success(f"Predicted Category: **{result}**")