import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
from langdetect import detect
import nltk
import re
import pandas as pd

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ------------------------------
# Load Model and Tokenizer
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("news_classifier_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_length = 300

# ------------------------------
# Utility Functions
# ------------------------------
def clean_text(text):
    """Remove URLs and special characters, and convert to lowercase."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    return text.lower()

def summarize_text(text):
    """Returns word and sentence counts. Handles tokenization errors gracefully."""
    try:
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize, sent_tokenize
        words = word_tokenize(text, language='english')  # force English
        sentences = sent_tokenize(text, language='english')  # force English
        return {
            "Word Count": len(words),
            "Sentence Count": len(sentences),
            "Character Count": len(text)
        }
    except Exception as e:
        return {
            "Word Count": 0,
            "Sentence Count": 0,
            "Character Count": len(text),
            "Error": f"Tokenizer error: {str(e)}"
        }

def predict_label(text):
    """Predicts whether the input is fake or real news."""
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)
    return prediction[0][0]

# ------------------------------
# App UI
# ------------------------------
st.title("📰 Fake News Detection App")
st.write("This application uses a trained LSTM model to classify news articles as **Fake** or **Real**.")

# File upload section
uploaded_file = st.file_uploader("📁 Upload a .txt or .csv file (optional)", type=["txt", "csv"])
user_input = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("Text from file:", user_input, height=200)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.warning("CSV files are shown for inspection only. Use text input for prediction.")
else:
    user_input = st.text_area("✏️ Enter a news article:", height=200)

# Prediction section
if user_input:
    try:
        lang = detect(user_input)
        st.info(f"🌍 Detected Language: {lang.upper()}")
    except:
        st.warning("Language detection failed.")

    # Text summary
    st.subheader("📊 Text Summary")
    stats = summarize_text(user_input)
    if "Error" in stats:
        st.error("Text summary failed. Please ensure the text is in English.")
    else:
        st.write(stats)

    # Prediction
    st.subheader("🧠 Prediction")
    probability = predict_label(user_input)
    label = "Real" if probability > 0.5 else "Fake"
    st.markdown(f"### 🏷️ **Prediction: {label.upper()}**")

    # Confidence display
    st.subheader("📈 Model Confidence")
    confidence = float(probability) if probability > 0.5 else 1 - float(probability)
    st.progress(confidence)

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
st.caption("© 2025 - AI Fake News Detection Project | Made with ❤️ using Streamlit and TensorFlow")

