# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:12:18 2024

@author: Fast
"""

import streamlit as st
import pickle
#import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords 
stop_words = stopwords.words('english') 

def removeHTML(text):
    text = re.sub(r'<[^>]*>', '', text)
    return(text)

def removeStopWords(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return(text)

def Lemmatizing(text):
    text = lemmatizer.lemmatize(text)
    return(text)

def removeNoise(text):
    text = re.sub(r'[^a-zA-Z\s]+', '', text) # Only keep English letters
    text = re.sub(r'(\w)\1{2,}', r'\1', text)   #Normalize Repeated Letters
    text = re.sub(r'\s{2,}', ' ', text).strip() # Removing extra spaces
    return(text)


path="C:/Users/marvi/Documents/Python Scripts/"

with open(path+"model_sentiment.pkl", "rb") as f:
    model = pickle.load(f)

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

html_temp="""
<div style="background-color:SkyBlue; padding:30px">
<h2 style="color:black";text-align:center;">Customer Review Sentiment Prediction</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html = True)
input_text=st.text_input("Enter Review: ", )

input_text = removeHTML(input_text)
input_text = removeStopWords(input_text)
input_text = Lemmatizing(input_text)
input_text = removeNoise(input_text)
text_2_vec = vectorizer.transform(np.array([input_text]))    
if st.button("Predict"):
    pred_y = model.predict(text_2_vec)
    st.write("The customer review is "+ pred_y)


