import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# ✅ Streamlit page configuration must be at the top
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="wide")

# ✅ Load model and tokenizer (cached to avoid reloading)
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("news_classifier_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Parameters
vocab_size = 10000
max_length = 300

# Streamlit UI
st.title("📰 Fake News Detector - AI Powered")
st.markdown("""
Welcome to the AI-powered **Fake News Detection App** 🧠📢.  
Paste any news article below and the AI will tell you whether it is **Real** or **Fake**.
""")

st.sidebar.header("⚙️ Settings")
confidence_display = st.sidebar.checkbox("Show model confidence", True)

# Text input field
user_input = st.text_area("✍️ Enter the news content below:", height=300)

# Prediction logic
if st.button("🔍 Analyze News"):
    if not user_input.strip():
        st.warning("❗ Please enter some text before analyzing.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        label = "✅ Real News" if prediction > 0.5 else "❌ Fake News"
        color = "green" if prediction > 0.5 else "red"
        st.markdown(f"<h3 style='color:{color};text-align:center'>{label}</h3>", unsafe_allow_html=True)

        if confidence_display:
            st.info(f"🔎 Model confidence: {prediction*100:.2f} %")

# Footer
st.markdown("---")
st.markdown("© 2025 - AI Fake News Detection Project | Powered by BERT & Streamlit 🚀")
