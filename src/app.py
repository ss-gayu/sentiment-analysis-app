import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model(path="models/sentiment_pipeline.joblib"):
    return joblib.load(path)

model = load_model()

st.title("Customer Review Sentiment Analysis")
review_text = st.text_area("Enter a review:", height=150)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review text!")
    else:
        pred = model.predict([review_text])[0]
        probs = model.predict_proba([review_text])[0]
        st.write(f"**Prediction:** {pred}")
        df_probs = pd.DataFrame({"Sentiment": model.classes_, "Probability": probs}).sort_values("Probability", ascending=False)
        st.table(df_probs)
