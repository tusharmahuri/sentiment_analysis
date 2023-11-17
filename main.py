# import streamlit as st
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
#
# # Load the vectorizer
# try:
#     with open('vectorizer.pkl', 'rb') as vectorizer_file:
#         vectorizer = pickle.load(vectorizer_file)
# except FileNotFoundError:
#     st.error("Vectorizer file not found. Please make sure to train and save the vectorizer before running the app.")
#
# # Load the model
# try:
#     with open('rf_model.pkl', 'rb') as model_file:
#         nb_model = pickle.load(model_file)
# except FileNotFoundError:
#     st.error("Model file not found. Please make sure to train and save the model before running the app.")
#
# # App title and description
# st.title("Airline Sentiment Analysis App")
# st.write("Enter a piece of text, and the app will predict its sentiment.")
#
# # Input form
# user_input = st.text_area("Enter your text:", "")
#
# if st.button("Predict"):
#     if not user_input:
#         st.warning("Please enter some text for prediction.")
#     else:
#         # Continue with predictions
#         user_input_vect = vectorizer.transform([user_input])
#         prediction = nb_model.predict(user_input_vect)
#         st.write("Prediction:", prediction[0])

import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the vectorizer
try:
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Vectorizer file not found. Please make sure to train and save the vectorizer before running the app.")

# Load the model
try:
    with open('nb_model.pkl', 'rb') as model_file:
        nb_model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure to train and save the model before running the app.")

# App title and description
st.title("Airline Sentiment Analysis App")
st.write("Enter a piece of text, and the app will predict its sentiment.")

# Input form
user_input = st.text_area("Enter your text:", "")

if st.button("Predict"):
    if not user_input:
        st.warning("Please enter some text for prediction.")
    else:
        # Continue with predictions
        user_input_vect = vectorizer.transform([user_input])
        prediction = nb_model.predict(user_input_vect)
        st.write("Prediction:", prediction[0])

# Add custom styling using CSS to set the background color to cyan
st.markdown(
    """
    <style>
        body {
            background-color: #00FFFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
