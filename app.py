import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# load the dataset
import csv

with open('Corona_NLP_train.csv', encoding='latin-1',mode='r') as file:
    data = csv.DictReader(file)
    data = [row for row in data]

#Convert data to DataFrame
df_train = pd.DataFrame(data)

# Remove unwanted columns
df_train.drop(["UserName","ScreenName","TweetAt","Location"], axis=1, inplace=True)

def preprocess_text(words):
    #Convert to lowercase
    words = words.lower()

    #initialize Stopwords
    stop_words = set(stopwords.words('english'))

    #Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    processed_words = []
    for word in words.split():
        if word not in stop_words:
            lemmatized_word = lemmatizer.lemmatize(word)
            processed_words.append(lemmatized_word)

    processed_words = ' '.join(processed_words)

    return processed_words



#Load Vectorizer
with open('tf_idf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

#Load tokenizer
with open('label_encoder.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

#Load encoder
with open('onehot_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)



#Load Model
model = load_model('corona_sentiment_model.h5')

#Predict function
def predict_sentiment(model, tf_idf_vectorizer, text,encoder):
    preprocessed_text = pd.Series(data=[preprocess_text(text)])
    vectorized_text = tf_idf_vectorizer.fit_transform(preprocessed_text).toarray()
    print(vectorized_text,vectorized_text.size,vectorized_text.__sizeof__(),type(vectorized_text))
    # Pad the array with zeros to match the target size
    padded_arr = np.pad(vectorized_text.flatten(), (0, 759600 - vectorized_text.size), mode='constant')

    # Reshape
    reshaped_arr = padded_arr.reshape(3798, 200)
    print(vectorized_text, vectorized_text.shape,type(vectorized_text))
    prediction = model.predict(reshaped_arr)
    predicted_label = encoder.inverse_transform(prediction)

    return predicted_label[0][0]

st.title("Sentiment Analysis of COVID-19 Tweets")

st.write("Enter a tweet to analyze its sentiment:")

# Get user input
user_input = st.text_area("Tweet")

if st.button("Predict Sentiment"):
    if user_input:
        predicted_sentiment = predict_sentiment(model, vectorizer, user_input,encoder)
        st.write(f"Predicted Sentiment: {predicted_sentiment}")
    else:
        st.write("Please enter a tweet to analyze.")




