import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")

input_text = st.text_area("Enter News Text")

if st.button("Predict"):
    vector_input = vectorizer.transform([input_text])
    prediction = model.predict(vector_input)

    if prediction[0] == 0:
        st.error("❌ This News is FAKE")
    else:
        st.success("✅ This News is REAL")