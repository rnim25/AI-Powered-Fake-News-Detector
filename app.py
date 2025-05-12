import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK data (only if not already available)
nltk.download('punkt', quiet=True)

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("news_classifier_model.h5")
    return model

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()
max_length = 300  # Ensure this matches what you used during training

# ------------------------------
# Utility Functions
# ------------------------------
def clean_text(text):
    """Basic cleanup: remove special characters and URLs."""
    import re
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    return text.lower()

def summarize_text(text):
    """Returns basic statistics about the input text."""
    try:
        # Simple word and sentence counting without using NLTK tokenizer
        words = text.split()  # Splitting text into words
        sentences = text.split('.')  # Splitting text into sentences using periods
        return {
            "Word Count": len(words),
            "Sentence Count": len(sentences),
            "Character Count": len(text)
        }
    except Exception as e:
        st.warning(f"⚠️ Text summary failed: {e}")
        return {
            "Word Count": 0,
            "Sentence Count": 0,
            "Character Count": 0
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

# File upload
uploaded_file = st.file_uploader("📁 Upload a .txt or .csv file (optional)", type=["txt", "csv"])
user_input = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("Text from file:", user_input, height=200)
    elif uploaded_file.type == "text/csv":
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.warning("CSV files are shown for inspection only. Use text input for prediction.")
else:
    user_input = st.text_area("✏️ Enter news article text here:", height=200)

if user_input:
    # Text statistics
    stats = summarize_text(user_input)
    st.subheader("📊 Text Summary")
    st.write(stats)

    # Prediction
    st.subheader("🧠 Prediction")
    probability = predict_label(user_input)
    label = "Real" if probability > 0.5 else "Fake"
    st.markdown(f"### 🏷️ **Prediction: {label.upper()}**")

    # Confidence gauge
    st.subheader("📈 Model Confidence")
    st.progress(float(probability) if probability > 0.5 else 1 - float(probability))

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

