# 📰 Fake News Detector – IA avec Streamlit

Une application web simple, interactive et puissante pour détecter automatiquement les fake news à l'aide d'un modèle d'intelligence artificielle basé sur LSTM (TensorFlow).

## 🚀 Fonctionnalités

- Prédiction en temps réel : détecte si une news est **vraie** ou **fausse**
- Interface utilisateur interactive (avec Streamlit)
- Affichage optionnel du **niveau de confiance** du modèle
- Déploiement facile via Streamlit Community Cloud

---

## 📦 Fichiers inclus

- `app.py` – L'application principale Streamlit
- `news_classifier_model.h5` – Le modèle entraîné
- `tokenizer.pkl` – Le tokenizer utilisé pour la vectorisation
- `requirements.txt` – Liste des bibliothèques nécessaires

---

## 🧠 Modèle IA

- Architecture : **Bi-LSTM** avec embedding
- Framework : `TensorFlow / Keras`
- Données : [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Langue : **Anglais** (pour ce modèle)

---

## 🛠️ Installation locale

```bash
# Clone le dépôt
git clone https://github.com/TON-UTILISATEUR/fake-news-detector.git
cd fake-news-detector

# Installe les dépendances
pip install -r requirements.txt

# Lance l'application
streamlit run app.py
