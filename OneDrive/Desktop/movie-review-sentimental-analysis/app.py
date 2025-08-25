import pickle as pk
import streamlit as st

# Load the whole pipeline
model = pk.load(open('model_pipeline.pkl', 'rb'))

st.title("Movie Review Sentiment Analysis")
review = st.text_input("Enter your movie review:")

if st.button("Predict"):
    # Predict sentiment directly (pipeline handles vectorizing internally)
    prediction = model.predict([review.lower()])
    proba = model.predict_proba([review.lower()])

    st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
    st.write(f"Probabilities [Negative, Positive]: {proba[0]}")
