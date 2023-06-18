import streamlit as st
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

import joblib
import re
import string

# Load the pre-trained model and vectorizer
model = joblib.load('model/model_knn_76%.joblib')
vectorizer = joblib.load('model/tfidf.joblib')


# Function to preprocess text data
def case_folding(tweet):
    # Melakukan case folding pada tweet
    tweet = tweet.lower()

    # Hapus string 'https://' atau 'http://' beserta kata-kata yang berada di belakangnya dan menghapus kata-kata yang berada setelah hashtag
    tweet = re.sub(r'https?://\S+|#\S+', '', tweet)

    # Hapus tanda baca, karakter lain, emoji, mention, dan hashtag
    tweet = re.sub(r'[^\w\s@#]|(@[A-Za-z0-9_]+)', '', tweet)

    # Hapus angka
    tweet = re.sub(r'\d+', '', tweet)

    # Hapus whitespace yang berlebihan
    tweet = re.sub(r'\s+', ' ', tweet)

    # Kembalikan hasil cleaning tweet
    return tweet.strip()

# Import dictionary slang dari file teks
slangs = {}
with open("model/combined_slang_words.txt", "r") as f:
    for line in f:
        slang, meaning = line.strip().split(":")
        slangs[slang] = meaning

# Membuat function "slangs" yang menerima sebuah string
def slangsd(tweet):
    # Inisialisasi variabel untuk menyimpan hasil cleaning slang
    cleaned_tweet = []
    # Lakukan looping untuk setiap kata pada tweet
    for word in tweet.split():
        # Jika kata tidak kosong, cek apakah ada dalam dictionary slang. Jika ada, ganti dengan artinya.
        if word:
            cleaned_tweet.append(slangs[word] if word in slangs else word)
    # Gabungkan hasil cleaning slang menjadi string dan kembalikan sebagai hasil dari function
    return " ".join(cleaned_tweet)
    
def predict(input):
    # Preprocess the user input
    preprocessed_input = case_folding(input)
    # Convert the preprocessed input into a feature vector
    feature_vector = vectorizer.transform([slangsd(preprocessed_input)])
    # Make a prediction using the pre-trained model
    prediction = model.predict(feature_vector)[0]
    # Display the prediction to the user
    if prediction == 1:
        st.write('Sentimen Negatif')
    else:
        st.write('Sentimen Positif')
                