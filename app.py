# 🔹 Importation des bibliothèques
import os
import json
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔹 Étape 1 : Définir les identifiants API Kaggle
kaggle_json_path = "kaggle.json"  # Assurez-vous que ce fichier est chargé dans Colab
with open(kaggle_json_path, "r") as f:
    kaggle_creds = json.load(f)
os.environ["KAGGLE_USERNAME"] = kaggle_creds["username"]
os.environ["KAGGLE_KEY"] = kaggle_creds["key"]

# 🔹 Étape 2 : Installer l'outil Kaggle CLI et télécharger le dataset
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

!kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset

# 🔹 Étape 3 : Extraire le dataset ZIP
with zipfile.ZipFile("fake-and-real-news-dataset.zip", "r") as zip_ref:
    zip_ref.extractall(".")

# 🔹 Étape 4 : Charger et étiqueter les données
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

texts = data["text"].astype(str).values
labels = data["label"].values

# 🔹 Étape 5 : Prétraitement des données
vocab_size = 10000
max_length = 300

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# 🔹 Étape 6 : Construire le modèle
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 🔹 Étape 7 : Entraîner le modèle
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# 🔹 Étape 8 : Évaluer le modèle
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 🔹 Étape 9 : Graphes Accuracy & Loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 🔹 Étape 10 : Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 🔹 Étape 11 : Sauvegarder le modèle et le tokenizer
model.save("news_classifier_model.h5")
print("✅ Modèle sauvegardé sous news_classifier_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer sauvegardé sous tokenizer.pkl")

# 🔹 Étape 12 : Télécharger les fichiers vers votre machine
from google.colab import files
files.download("news_classifier_model.h5")
files.download("tokenizer.pkl")
