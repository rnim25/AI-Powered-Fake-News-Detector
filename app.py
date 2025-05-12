import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
from langdetect import detect
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK tokenizer models (needed for sentence splitting)
nltk.download('punkt')

# --------------------------------------
# App Configuration
# --------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --------------------------------------
# Load Model and Tokenizer
# --------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("news_classifier_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_length = 300  # Must match training setting

# --------------------------------------
# Text Processing Functions
# --------------------------------------
def clean_text(text):
    """Remove URLs, special characters, and convert to lowercase."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def summarize_text(text):
    """Returns text statistics such as word and sentence counts."""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Character Count": len(text)
    }

def predict_label(text):
    """Cleans and tokenizes the input, and predicts the label."""
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)
    return prediction[0][0]

# --------------------------------------
# App UI
# --------------------------------------
st.title("📰 Fake News Detection App")
st.write("This application uses an LSTM-based deep learning model to classify news articles as **Fake** or **Real**.")

# File upload
uploaded_file = st.file_uploader("📁 Upload a .txt or .csv file (optional)", type=["txt", "csv"])
user_input = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("Text from uploaded file:", user_input, height=200)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.warning("CSV file previewed above. Please enter specific article text below for prediction.")

# Manual text input
user_input = st.text_area("✏️ Enter news article text here:", user_input, height=200)

if user_input:
    # Detect language
    try:
        lang = detect(user_input)
        st.info(f"🌍 Detected Language: {lang.upper()}")
    except:
        st.warning("Language detection failed.")

    # Show text summary
    stats = summarize_text(user_input)
    st.subheader("📊 Text Summary")
    st.write(stats)

    # Prediction
    st.subheader("🧠 Model Prediction")
    probability = predict_label(user_input)
    label = "Real" if probability > 0.5 else "Fake"
    st.markdown(f"### 🏷️ **Prediction: {label.upper()}**")

    # Confidence bar
    st.subheader("📈 Model Confidence")
    confidence = probability if probability > 0.5 else 1 - probability
    st.progress(float(confidence))

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(
        [probability, 1 - probability],
        labels=["Real", "Fake"],
        colors=["green", "red"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("© 2025 - AI Fake News Detection Project | Developed with ❤️ using Streamlit and TensorFlow")


