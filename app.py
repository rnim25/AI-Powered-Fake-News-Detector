import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Chargement du modèle et du tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("news_classifier_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Paramètres
vocab_size = 10000
max_length = 300

# Interface Streamlit
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="wide")
st.title("📰 Fake News Detector - AI Powered")
st.markdown("""
Bienvenue dans l'application de détection automatique des **fake news** 🧠📢.  
Entrez un article ci-dessous et l'IA vous dira s'il est **Vrai** ou **Faux**.
""")

st.sidebar.header("⚙️ Paramètres")
confidence_display = st.sidebar.checkbox("Afficher la confiance du modèle", True)

# Champ de saisie
user_input = st.text_area("✍️ Entrez le contenu de la news :", height=300)

if st.button("🔍 Analyser la News"):
    if not user_input.strip():
        st.warning("❗ Veuillez entrer un texte avant d'analyser.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        label = "✅ Vraie News" if prediction > 0.5 else "❌ Fake News"
        color = "green" if prediction > 0.5 else "red"
        st.markdown(f"<h3 style='color:{color};text-align:center'>{label}</h3>", unsafe_allow_html=True)

        if confidence_display:
            st.info(f"🔎 Confiance du modèle : {prediction*100:.2f} %")

# Footer
st.markdown("---")
st.markdown("© 2025 - Projet IA de détection des fake news | Propulsé par BERT & Streamlit 🚀")

